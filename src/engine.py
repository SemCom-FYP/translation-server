# coding: utf-8
"""
Translation engines.

An engine owns the models; a session owns one *context* per utterance. Every
model call goes through `Engine.run`, a single worker thread: the GPU
serialises the work anyway, and SeamlessStreaming's agents keep a little
per-call scratch on the shared agent objects (see SeamlessStreamingEngine),
so calls must not overlap. Many streams share one loaded model; each keeps
its own decoding state.

    SeamlessStreamingEngine   facebook/seamless-streaming: simultaneous
                              speech-to-speech translation from ~100 input
                              languages into 36 spoken output languages
    MockEngine                no models: echoes each utterance back one step
                              late, for testing the plumbing without a GPU
"""

import asyncio
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .audio import MODEL_RATE

log = logging.getLogger("translation.engine")

# The 36 languages SeamlessStreaming can *speak* (its vocoder_v2 speaker table).
# It understands ~100 input languages and needs no source-language hint.
SPEECH_OUTPUT_LANGS = (
    "arb", "ben", "cat", "ces", "cmn", "cym", "dan", "deu", "eng", "est", "fin",
    "fra", "hin", "ind", "ita", "jpn", "kor", "mlt", "nld", "pes", "pol", "por",
    "ron", "rus", "slk", "spa", "swe", "swh", "tel", "tgl", "tha", "tur", "ukr",
    "urd", "uzn", "vie",
)

# Two-letter codes (what webrtc_conference uses: es, fr, de, it) → model codes.
ISO_639_1 = {
    "ar": "arb", "bn": "ben", "ca": "cat", "cs": "ces", "zh": "cmn", "cy": "cym",
    "da": "dan", "de": "deu", "en": "eng", "et": "est", "fi": "fin", "fr": "fra",
    "hi": "hin", "id": "ind", "it": "ita", "ja": "jpn", "ko": "kor", "mt": "mlt",
    "nl": "nld", "fa": "pes", "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus",
    "sk": "slk", "es": "spa", "sv": "swe", "sw": "swh", "te": "tel", "tl": "tgl",
    "th": "tha", "tr": "tur", "uk": "ukr", "ur": "urd", "uz": "uzn", "vi": "vie",
}


@dataclass
class StepResult:
    audio: np.ndarray                       # new translated speech, float32 @ output_rate
    source_text: Optional[str] = None       # whole utterance so far, if the engine knows it
    target_text: Optional[str] = None
    done: bool = False                      # utterance fully translated
    compute_ms: float = 0.0


@dataclass
class Context:
    tgt_lang: str
    state: dict = field(default_factory=dict)
    source_text: str = ""
    target_text: str = ""


class Engine:
    name = "engine"
    output_rate = MODEL_RATE

    def __init__(self, target_langs=SPEECH_OUTPUT_LANGS):
        self.target_langs = list(target_langs)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")

    def resolve_lang(self, code) -> Optional[str]:
        """'es', 'ES', 'spa' → 'spa'; None if this engine cannot speak it."""
        code = str(code or "").strip().lower()
        code = ISO_639_1.get(code, code)
        return code if code in self.target_langs else None

    @property
    def segment_samples(self) -> int:
        """How much new source audio warrants another step."""
        raise NotImplementedError

    async def run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def new_context(self, tgt_lang: str) -> Context:
        raise NotImplementedError

    def step(self, ctx: Context, source: np.ndarray, finished: bool) -> StepResult:
        """Runs on the engine thread. `source` is the whole utterance so far,
        float32 mono at 16 kHz; `finished` says no more will come."""
        raise NotImplementedError

    def release(self, ctx: Context) -> None:
        ctx.state.clear()

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


# ── mock ─────────────────────────────────────────────────────────────────────

class MockEngine(Engine):
    """Echoes the source back with a one-step delay (as a simultaneous model
    would lag) and reports made-up text. Exercises every code path of the
    server and the client adapter without any model installed."""

    name = "mock"

    def __init__(self, target_langs=SPEECH_OUTPUT_LANGS, segment_ms: int = 320,
                 compute_ms: float = 30.0):
        super().__init__(target_langs)
        self._segment = segment_ms * MODEL_RATE // 1000
        self._compute_s = compute_ms / 1000

    @property
    def segment_samples(self):
        return self._segment

    def new_context(self, tgt_lang):
        return Context(tgt_lang=tgt_lang, state={"emitted": 0})

    def step(self, ctx, source, finished):
        t0 = time.perf_counter()
        time.sleep(self._compute_s)
        upto = source.size if finished else max(0, source.size - self._segment)
        out = source[ctx.state["emitted"]:upto].astype(np.float32)
        ctx.state["emitted"] = max(ctx.state["emitted"], upto)
        ctx.source_text = f"{source.size / MODEL_RATE:.1f}s of speech"
        ctx.target_text = f"[{ctx.tgt_lang}] {upto / MODEL_RATE:.1f}s translated"
        return StepResult(audio=out, source_text=ctx.source_text,
                          target_text=ctx.target_text, done=finished,
                          compute_ms=(time.perf_counter() - t0) * 1000)


# ── SeamlessStreaming ────────────────────────────────────────────────────────

# 25 ms: one fbank window. Pushed when an utterance closes with no new audio
# (and to drain a long final translation): the feature extractor re-reads its
# last chunk if handed an empty segment, which would duplicate frames.
_PAD_SAMPLES = 400
# Safety cap on the pushes that drain an utterance once its source has ended.
MAX_FINAL_PUSHES = 6


def _build_pipeline_class():
    """SeamlessStreamingS2STJointVADAgent without its SileroVADAgent, and
    with a detokenizer that reports the whole translation so far.

    * VAD: the session's Endpointer already cuts utterances (the stock VAD
      agent also loads a Silero model through torch.hub for every stream).
    * Detokenizer: the stock one emits fragments that lose word boundaries
      ("Hola a tod" + "os"), or with detokenize_only=False drops words when
      several arrive at once. This one keeps every token of the utterance,
      including across the pipeline's early-stop resets, and emits the full
      decoded text each time it grows.
    """
    from seamless_communication.streaming.agents.common import EarlyStoppingMixin
    from seamless_communication.streaming.agents.detokenizer import (
        UnitYDetokenizerAgent, UnitYDetokenizerAgentStates)
    from seamless_communication.streaming.agents.offline_w2v_bert_encoder import (
        OfflineWav2VecBertEncoderAgent)
    from seamless_communication.streaming.agents.online_feature_extractor import (
        OnlineFeatureExtractorAgent)
    from seamless_communication.streaming.agents.online_text_decoder import (
        UnitYMMATextDecoderAgent)
    from seamless_communication.streaming.agents.online_unit_decoder import (
        NARUnitYUnitDecoderAgent)
    from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
    from seamless_communication.streaming.agents.unity_pipeline import UnitYAgentTreePipeline
    from simuleval.agents.actions import ReadAction, WriteAction

    class TextSoFarStates(EarlyStoppingMixin, UnitYDetokenizerAgentStates):
        def reset(self) -> None:
            super().reset()
            self.all_tokens = []

        def reset_early(self) -> None:
            # The pipeline restarts after an early stop mid-utterance; the
            # text said so far still belongs to this utterance.
            tokens = self.all_tokens
            super().reset()
            self.all_tokens = tokens

    class TextSoFarDetokenizerAgent(UnitYDetokenizerAgent):
        def build_states(self) -> TextSoFarStates:
            return TextSoFarStates()

        def policy(self, states: TextSoFarStates):
            if states.source:
                states.all_tokens += states.source
                states.source = []
                return WriteAction(self.decode(" ".join(states.all_tokens)),
                                   finished=states.source_finished)
            if states.source_finished:
                return WriteAction(self.decode(" ".join(states.all_tokens)), finished=True)
            return ReadAction()

    class SeamlessStreamingS2STServerAgent(UnitYAgentTreePipeline):
        pipeline = {
            OnlineFeatureExtractorAgent: [OfflineWav2VecBertEncoderAgent],
            OfflineWav2VecBertEncoderAgent: [UnitYMMATextDecoderAgent],
            UnitYMMATextDecoderAgent: [TextSoFarDetokenizerAgent, NARUnitYUnitDecoderAgent],
            TextSoFarDetokenizerAgent: [],
            NARUnitYUnitDecoderAgent: [VocoderAgent],
            VocoderAgent: [],
        }

    return SeamlessStreamingS2STServerAgent


def _to_argv(config: dict) -> list:
    argv = []
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not False and value is not None:
            argv += [flag, str(value)]
    return argv


class SeamlessStreamingEngine(Engine):
    name = "seamless-streaming"

    def __init__(self, device: str = "auto", dtype: str = "fp16", segment_ms: int = 320,
                 decision_threshold: float = 0.5, min_start_frames: int = 192,
                 min_unit_chunk_size: int = 50, max_len_b: int = 100,
                 warmup: bool = True):
        super().__init__(SPEECH_OUTPUT_LANGS)
        self.segment_ms = segment_ms
        self._config = dict(
            task="s2st",
            tgt_lang="eng",                     # default only; each stream sets its own
            device=device,
            dtype=dtype,
            sample_rate=MODEL_RATE,
            source_segment_size=segment_ms,
            min_starting_wait_w2vbert=min_start_frames,
            decision_threshold=decision_threshold,
            min_unit_chunk_size=min_unit_chunk_size,
            no_early_stop=True,
            max_len_a=0,
            max_len_b=max_len_b,
            block_ngrams=True,
            detokenize_only=True,
        )
        self.system = None
        # Load on the engine thread so CUDA state lives where it is used.
        self._pool.submit(self._load, warmup).result()

    @property
    def segment_samples(self):
        return self.segment_ms * MODEL_RATE // 1000

    def _load(self, warmup: bool):
        import torch
        from simuleval import options
        torch.set_grad_enabled(False)
        if self._config["device"] == "auto":
            self._config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self._config["device"] == "cpu":
            self._config["dtype"] = "fp32"
        agent_class = _build_pipeline_class()
        # simuleval's general_parser() and cli_argument_list() read sys.argv,
        # which here holds the *server's* flags (--device, --port, ...).
        saved_argv = sys.argv
        sys.argv = saved_argv[:1]
        try:
            parser = options.general_parser()
            agent_class.add_args(parser)
            args, _ = parser.parse_known_args(_to_argv(self._config))
        finally:
            sys.argv = saved_argv
        t0 = time.perf_counter()
        self.system = agent_class.from_args(args)
        log.info("loaded SeamlessStreaming on %s (%s) in %.1f s", self._config["device"],
                 self._config["dtype"], time.perf_counter() - t0)
        if warmup:
            self._warmup()

    def _warmup(self):
        """First calls compile/allocate CUDA kernels; pay that before a caller does."""
        t0 = time.perf_counter()
        try:
            rng = np.random.default_rng(0)
            noise = (rng.standard_normal(MODEL_RATE * 2) * 0.01).astype(np.float32)
            ctx = self.new_context("eng")
            half = noise.size // 2
            self.step(ctx, noise[:half], False)
            self.step(ctx, noise, True)
            self.release(ctx)
            log.info("warm-up done in %.1f s", time.perf_counter() - t0)
        except Exception:
            log.exception("warm-up failed (the server will still try to run)")

    def new_context(self, tgt_lang: str) -> Context:
        """Runs on the engine thread (see Session)."""
        if tgt_lang not in self.target_langs:
            raise KeyError(tgt_lang)
        return Context(tgt_lang=tgt_lang, state={"states": self.system.build_states(), "fed": 0})

    def step(self, ctx: Context, source: np.ndarray, finished: bool) -> StepResult:
        from simuleval.data.segments import SpeechSegment, TextSegment
        t0 = time.perf_counter()
        st = ctx.state
        new = source[st["fed"]:]
        st["fed"] = source.size
        if not finished and new.size == 0:
            return StepResult(audio=np.zeros(0, np.float32))
        if finished and new.size < _PAD_SAMPLES:
            new = np.concatenate([new, np.zeros(_PAD_SAMPLES - new.size, np.float32)])

        pieces = []
        done = False
        content = new.astype(np.float32).tolist()   # the feature extractor concatenates lists
        for push in range(MAX_FINAL_PUSHES if finished else 1):
            if push:
                content = [0.0] * _PAD_SAMPLES
            segment = SpeechSegment(content=content, sample_rate=MODEL_RATE,
                                    finished=finished, tgt_lang=ctx.tgt_lang)
            outputs = self.system.pushpop(segment, st["states"])
            if not isinstance(outputs, list):
                outputs = [outputs]
            for out in outputs:
                if out.is_empty:
                    continue
                if isinstance(out, SpeechSegment) and len(out.content):
                    if out.sample_rate not in (-1, MODEL_RATE):
                        raise RuntimeError(f"vocoder produced {out.sample_rate} Hz, "
                                           f"expected {MODEL_RATE}")
                    pieces.append(np.asarray(out.content, dtype=np.float32))
                elif isinstance(out, TextSegment) and isinstance(out.content, str) and out.content:
                    ctx.target_text = out.content
            if not finished or all(out.finished for out in outputs):
                done = finished
                break
        if finished and not done:
            log.debug("utterance closed before the pipeline reported it finished")
            done = True

        audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        return StepResult(audio=audio, source_text=None, target_text=ctx.target_text,
                          done=done, compute_ms=(time.perf_counter() - t0) * 1000)
