# coding: utf-8
"""
Audio plumbing for a translation session: stream resampling, decoding the
conference app's EnCodec packets, and cutting the incoming voice into
utterances.

Why utterances at all: SeamlessStreaming's w2v-BERT encoder re-runs over
every frame heard so far on each step, so the cost of a step grows with the
utterance; Meta's own pipeline cuts the stream with a VAD for the same
reason. Cutting at pauses also gives the decoder sentence-sized inputs.
"""

import logging
import math
from collections import deque
from typing import Optional

import numpy as np

from . import wire

log = logging.getLogger("translation.audio")

MODEL_RATE = 16000          # model input features and vocoder output are 16 kHz
VAD_FRAME = 512             # 32 ms at 16 kHz (Silero VAD's required frame size)
VAD_FRAME_MS = VAD_FRAME * 1000 // MODEL_RATE


# ── resampling ───────────────────────────────────────────────────────────────

class StreamResampler:
    """Chunk-by-chunk resampling with state carried across chunk boundaries."""

    def __init__(self, src_rate: int, dst_rate: int):
        self.src_rate, self.dst_rate = int(src_rate), int(dst_rate)
        self._stream = None
        self.reset()

    def reset(self):
        if self.src_rate == self.dst_rate:
            self._stream = None
            return
        import soxr
        self._stream = soxr.ResampleStream(self.src_rate, self.dst_rate, 1,
                                           dtype="float32", quality="HQ")

    def __call__(self, pcm: np.ndarray, last: bool = False) -> np.ndarray:
        pcm = np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)
        if self._stream is None:
            return pcm
        return self._stream.resample_chunk(pcm, last=last)


# ── EnCodec packets from the conference app ─────────────────────────────────

class EncodecPacketDecoder:
    """Turns the app's 'A' packets back into 24 kHz PCM, the way its own
    receiver does (EnCodec decode + a 5 ms crossfade across chunk seams).
    Needs `pip install encodec` (and torch) on the server."""

    _model = None               # shared: decoding holds no per-stream state

    def __init__(self, fade_ms: int = 5):
        if EncodecPacketDecoder._model is None:
            import torch                                    # noqa: F401
            from encodec import EncodecModel
            model = EncodecModel.encodec_model_24khz().eval()
            EncodecPacketDecoder._model = model
        self.n = max(1, fade_ms * wire.ENCODEC_SAMPLE_RATE // 1000)
        self._ramp = np.linspace(0.0, 1.0, self.n, dtype=np.float32)
        self._tail: Optional[np.ndarray] = None
        self._last_seq: Optional[int] = None

    def __call__(self, packet: bytes) -> np.ndarray:
        import torch
        pkt = wire.unpack_audio(packet)
        if self._last_seq is not None and ((pkt.seq - self._last_seq) & 0xFFFF) > 0x8000:
            return np.zeros(0, dtype=np.float32)            # late duplicate / reordered
        self._last_seq = pkt.seq
        codes = torch.from_numpy(pkt.codes).long()[None]
        with torch.no_grad():
            wav = EncodecPacketDecoder._model.decode([(codes, None)])
        return self._join(wav[0, 0].float().numpy())

    def _join(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.size <= 2 * self.n:
            return pcm
        head, body, tail = pcm[:self.n], pcm[self.n:-self.n], pcm[-self.n:]
        if self._tail is None:
            out = np.concatenate([head, body])
        else:
            out = np.concatenate([self._tail * (1 - self._ramp) + head * self._ramp, body])
        self._tail = tail
        return out

    def reset(self):
        self._tail = None
        self._last_seq = None


# ── voice activity ───────────────────────────────────────────────────────────

class EnergyVAD:
    """Level above an adaptive noise floor. No model, no dependency; good
    enough for headset/close-mic conferencing audio, weaker in noisy rooms."""

    def __init__(self, margin_db: float = 10.0, min_db: float = -50.0):
        self.margin_db, self.min_db = margin_db, min_db
        self.floor_db = -65.0

    def __call__(self, frame: np.ndarray) -> float:
        rms = math.sqrt(float(np.mean(frame.astype(np.float64) ** 2)) + 1e-12)
        db = 20.0 * math.log10(rms + 1e-9)
        # Follow the floor down at once, back up slowly (~3 dB/s at 32 ms frames),
        # so sustained speech never becomes the "noise".
        self.floor_db = db if db < self.floor_db else self.floor_db + 0.1
        if db < self.min_db:
            return 0.0
        return 1.0 if db > self.floor_db + self.margin_db else 0.0

    def reset(self):
        self.floor_db = -65.0


class SileroVAD:
    """Silero VAD (`pip install silero-vad`); one instance per stream since it
    keeps a recurrent state."""

    def __init__(self):
        from silero_vad import load_silero_vad
        import torch
        self._torch = torch
        self.model = load_silero_vad()

    def __call__(self, frame: np.ndarray) -> float:
        with self._torch.no_grad():
            return float(self.model(self._torch.from_numpy(frame), MODEL_RATE).item())

    def reset(self):
        self.model.reset_states()


def make_vad(kind: str = "auto"):
    if kind in ("auto", "silero"):
        try:
            return SileroVAD()
        except Exception as exc:
            if kind == "silero":
                raise
            log.debug("silero-vad unavailable (%s); using the energy VAD", exc)
    return EnergyVAD()


# ── endpointing ──────────────────────────────────────────────────────────────

class Endpointer:
    """Cuts a 16 kHz stream into utterances.

    `push(pcm)` returns a list of events, in order:
        ("start", None)          an utterance begins
        ("audio", ndarray)       samples belonging to the open utterance
                                 (the first one includes the pre-roll)
        ("end", reason)          the utterance is over: "silence" after
                                 `end_silence_ms` of quiet, "pause" after a
                                 shorter pause once it has run `soft_max_s`,
                                 or "max_length" at `hard_max_s`
    """

    def __init__(self, vad=None, threshold: float = 0.5, start_ms: int = 96,
                 end_silence_ms: int = 600, preroll_ms: int = 320,
                 soft_max_s: float = 10.0, soft_end_silence_ms: int = 200,
                 hard_max_s: float = 15.0, min_speech_ms: int = 250):
        self.vad = vad if vad is not None else make_vad()
        self.threshold = threshold
        self.start_frames = max(1, start_ms // VAD_FRAME_MS)
        self.end_frames = max(1, end_silence_ms // VAD_FRAME_MS)
        self.soft_end_frames = max(1, soft_end_silence_ms // VAD_FRAME_MS)
        self.soft_max_frames = int(soft_max_s * 1000 / VAD_FRAME_MS)
        self.hard_max_frames = int(hard_max_s * 1000 / VAD_FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // VAD_FRAME_MS)
        self.end_silence_ms = self.end_frames * VAD_FRAME_MS
        preroll = max(self.start_frames, preroll_ms // VAD_FRAME_MS)
        self._preroll: deque = deque(maxlen=preroll)
        self.reset()

    def reset(self):
        self._pending = np.zeros(0, dtype=np.float32)
        self._preroll.clear()
        self._voiced_run = 0
        self._silent_run = 0
        self._frames = 0
        self._speech_frames = 0
        self.in_speech = False
        self.vad.reset()

    def push(self, pcm: np.ndarray) -> list:
        events: list = []
        buf = np.concatenate([self._pending, np.asarray(pcm, dtype=np.float32)])
        n = buf.size // VAD_FRAME
        for i in range(n):
            self._frame(buf[i * VAD_FRAME:(i + 1) * VAD_FRAME], events)
        self._pending = buf[n * VAD_FRAME:]
        return _merge_audio(events)

    def flush(self, reason: str = "flush") -> list:
        """End the open utterance now, if any."""
        events: list = []
        if self.in_speech:
            if self._pending.size:
                events.append(("audio", self._pending))
            self._pending = np.zeros(0, dtype=np.float32)
            self._close(events, reason)
        return events

    def _frame(self, frame: np.ndarray, events: list):
        voiced = self.vad(frame) >= self.threshold
        if not self.in_speech:
            self._preroll.append(frame)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            if self._voiced_run >= self.start_frames:
                self.in_speech = True
                self._frames = len(self._preroll)
                self._speech_frames = self._voiced_run
                self._silent_run = 0
                events.append(("start", None))
                events.append(("audio", np.concatenate(list(self._preroll))))
                self._preroll.clear()
            return

        events.append(("audio", frame))
        self._frames += 1
        if voiced:
            self._speech_frames += 1
            self._silent_run = 0
        else:
            self._silent_run += 1

        if self._silent_run >= self.end_frames:
            self._close(events, "silence")
        elif self._frames >= self.soft_max_frames and self._silent_run >= self.soft_end_frames:
            self._close(events, "pause")
        elif self._frames >= self.hard_max_frames:
            self._close(events, "max_length")

    def _close(self, events: list, reason: str):
        # A blip (cough, click, door) that never had real speech in it is
        # dropped rather than handed to the translator to hallucinate on.
        if self._speech_frames < self.min_speech_frames:
            reason = "too_short"
        events.append(("end", reason))
        self.in_speech = False
        self._voiced_run = 0
        self._silent_run = 0


def _merge_audio(events: list) -> list:
    out: list = []
    run: list = []
    for kind, val in events:
        if kind == "audio":
            run.append(val)
            continue
        if run:
            out.append(("audio", np.concatenate(run)))
            run = []
        out.append((kind, val))
    if run:
        out.append(("audio", np.concatenate(run)))
    return out
