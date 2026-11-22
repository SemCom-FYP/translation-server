# coding: utf-8
"""
One translated voice: everything between a WebSocket's binary frames and the
engine.

    input frames ─▶ decode ─▶ resample to 16 kHz ─▶ Endpointer ─▶ utterance queue
                                                                        │
    output frames ◀─ encode ◀─ resample to output rate ◀─ engine.step ◀─┘ (pump)

The receive side never waits on the model. The pump steps the oldest
utterance whenever a segment's worth of new audio has arrived (or it has
ended); if the engine falls behind, the next step simply sees more audio at
once, so delay stays bounded instead of a backlog of steps forming.
"""

import asyncio
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import numpy as np

from . import wire
from .audio import MODEL_RATE, EncodecPacketDecoder, Endpointer, StreamResampler, make_vad
from .engine import Context, Engine

log = logging.getLogger("translation.session")

_session_ids = itertools.count(1)


class ProtocolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class SessionOptions:
    vad: str = "auto"                   # auto | silero | energy
    vad_threshold: float = 0.5
    end_silence_ms: int = 600
    soft_max_s: float = 10.0
    hard_max_s: float = 15.0
    max_pending: int = 3                # utterances queued behind the one in progress


@dataclass
class _Utterance:
    id: int
    t_start: float
    chunks: list = field(default_factory=list)
    n: int = 0                          # source samples received (16 kHz)
    stepped: int = 0                    # source samples the last step saw
    ended: bool = False
    reason: str = ""
    t_end: Optional[float] = None       # estimated wall time the speech stopped
    dropped: bool = False
    announced: bool = False
    ctx: Optional[Context] = None
    t_first_out: Optional[float] = None
    t_last_out: Optional[float] = None
    out_samples: int = 0                # translated samples (16 kHz)
    source_text: str = ""
    target_text: str = ""

    def append(self, pcm: np.ndarray):
        self.chunks.append(pcm)
        self.n += pcm.size

    def source(self) -> np.ndarray:
        if len(self.chunks) > 1:
            self.chunks = [np.concatenate(self.chunks)]
        return self.chunks[0] if self.chunks else np.zeros(0, np.float32)


class Session:
    def __init__(self, engine: Engine, send_text: Callable[[str], Awaitable],
                 send_bytes: Callable[[bytes], Awaitable], options: SessionOptions):
        self.engine = engine
        self.options = options
        self.id = f"s{next(_session_ids)}"
        self.label = ""
        self._send_text_raw, self._send_bytes_raw = send_text, send_bytes
        self._send_lock = asyncio.Lock()
        self.src_lang = self.tgt_lang = ""     # as the client named them
        self.model_tgt: Optional[str] = None   # the engine's code for tgt_lang
        self._in_fmt = wire.PCM_S16LE
        self._in_rate = 24000
        self._out_fmt = wire.PCM_S16LE
        self._out_rate = 24000
        self._decoder: Optional[EncodecPacketDecoder] = None
        self._in_resampler: Optional[StreamResampler] = None
        self._out_resampler: Optional[StreamResampler] = None
        self._endpointer: Optional[Endpointer] = None
        self._queue: "deque[_Utterance]" = deque()
        self._open: Optional[_Utterance] = None
        self._next_utterance = 1
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._pump_task: Optional[asyncio.Task] = None
        self.started = False
        self.closed = False
        # stats
        self.received_s = 0.0
        self.utterances_done = 0

    # ── control messages ─────────────────────────────────────────────────────

    async def start(self, msg: dict) -> dict:
        if self.started:
            raise ProtocolError("bad_request", "session already started")
        proto = int(msg.get("protocol", wire.PROTOCOL_VERSION))
        if proto != wire.PROTOCOL_VERSION:
            raise ProtocolError("unsupported_protocol",
                                f"server speaks protocol {wire.PROTOCOL_VERSION}, client {proto}")
        inp = msg.get("input") or {}
        out = msg.get("output") or {}
        self._in_fmt = str(inp.get("format", wire.PCM_S16LE))
        self._out_fmt = str(out.get("format", wire.PCM_S16LE))
        if self._in_fmt not in wire.INPUT_FORMATS:
            raise ProtocolError("bad_format", f"input format must be one of {wire.INPUT_FORMATS}")
        if self._out_fmt not in wire.OUTPUT_FORMATS:
            raise ProtocolError("bad_format", f"output format must be one of {wire.OUTPUT_FORMATS}")
        self._in_rate = (wire.ENCODEC_SAMPLE_RATE if self._in_fmt == wire.ENCODEC_PACKET
                         else _rate(inp.get("sample_rate", 24000)))
        self._out_rate = _rate(out.get("sample_rate", 24000))
        self.label = str(msg.get("label") or "")[:64]

        if self._in_fmt == wire.ENCODEC_PACKET:
            try:
                self._decoder = await asyncio.get_running_loop().run_in_executor(
                    None, EncodecPacketDecoder)
            except ImportError as exc:
                raise ProtocolError("unsupported_format",
                                    f"encodec_packet input needs `pip install encodec` "
                                    f"on the server ({exc})")
        self._in_resampler = StreamResampler(self._in_rate, MODEL_RATE)
        self._out_resampler = StreamResampler(self.engine.output_rate, self._out_rate)
        o = self.options
        self._endpointer = Endpointer(
            vad=make_vad(o.vad), threshold=o.vad_threshold,
            end_silence_ms=o.end_silence_ms, soft_max_s=o.soft_max_s,
            hard_max_s=o.hard_max_s)
        self._set_langs(msg.get("src_lang"), msg.get("tgt_lang"))
        self.started = True
        self._pump_task = asyncio.create_task(self._pump(), name=f"pump-{self.id}")
        log.info("[%s %s] started -> %s  in=%s@%d out=%s@%d", self.id, self.label,
                 self.model_tgt, self._in_fmt, self._in_rate, self._out_fmt, self._out_rate)
        return {"protocol": wire.PROTOCOL_VERSION, "session": self.id,
                "src_lang": self.src_lang, "tgt_lang": self.tgt_lang,
                "model_tgt_lang": self.model_tgt,
                "target_langs": self.engine.target_langs, "engine": self.engine.name,
                "output": {"format": self._out_fmt, "sample_rate": self._out_rate}}

    def _set_langs(self, src, tgt):
        """`tgt_lang` is required (two- or three-letter code). `src_lang` is
        only a hint: the model understands ~100 input languages by itself."""
        tgt = str(tgt or "")
        model_tgt = self.engine.resolve_lang(tgt)
        if model_tgt is None:
            raise ProtocolError("unsupported_language",
                                f"cannot speak {tgt!r}; target languages: "
                                f"{', '.join(self.engine.target_langs)}")
        self.src_lang, self.tgt_lang, self.model_tgt = str(src or ""), tgt, model_tgt

    async def configure(self, msg: dict):
        self._require_started()
        self._set_langs(msg.get("src_lang", self.src_lang), msg.get("tgt_lang", self.tgt_lang))
        await self.reset("reconfigured")
        log.info("[%s %s] now -> %s", self.id, self.label, self.model_tgt)

    async def reset(self, reason: str = "reset"):
        self._require_started()
        for utt in self._queue:
            utt.dropped = True
            utt.reason = reason
        self._open = None
        self._endpointer.reset()
        self._in_resampler.reset()
        if self._decoder is not None:
            self._decoder.reset()
        self._wake.set()

    async def flush(self, reason: str = "flush"):
        self._require_started()
        tail = self._in_resampler(np.zeros(0, np.float32), last=True)
        self._in_resampler.reset()
        events = self._endpointer.push(tail) if tail.size else []
        events += self._endpointer.flush(reason)
        self._apply(events)

    async def stop(self):
        """Finish everything heard so far, then return."""
        if not self.started:
            return
        await self.flush("stop")
        await self._idle_wait()

    async def close(self):
        self.closed = True
        for utt in self._queue:
            utt.dropped = True
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except (asyncio.CancelledError, Exception):
                pass
        for utt in self._queue:
            if utt.ctx is not None:
                self.engine.release(utt.ctx)
        self._queue.clear()

    # ── audio in ─────────────────────────────────────────────────────────────

    async def feed(self, data: bytes):
        self._require_started()
        if self._in_fmt == wire.ENCODEC_PACKET:
            try:
                pcm = await asyncio.get_running_loop().run_in_executor(None, self._decoder, data)
            except ValueError as exc:
                await self.send_event("error", code="bad_audio", message=str(exc))
                return
        else:
            pcm = wire.bytes_to_pcm(data, self._in_fmt)
        if pcm.size == 0:
            return
        self.received_s += pcm.size / self._in_rate
        self._apply(self._endpointer.push(self._in_resampler(pcm)))

    def _apply(self, events: list):
        now = time.monotonic()
        for kind, val in events:
            if kind == "start":
                self._open = _Utterance(id=self._next_utterance, t_start=now)
                self._next_utterance += 1
                self._queue.append(self._open)
                self._idle.clear()
                self._trim_backlog()
            elif kind == "audio" and self._open is not None:
                self._open.append(val)
            elif kind == "end" and self._open is not None:
                utt, self._open = self._open, None
                utt.ended, utt.reason = True, val
                quiet_s = self._endpointer.end_silence_ms / 1000 if val == "silence" else 0.0
                utt.t_end = now - quiet_s
                if val == "too_short" and utt.out_samples == 0:
                    utt.dropped = True
        self._wake.set()

    def _trim_backlog(self):
        """If the engine cannot keep up, drop whole utterances that have not
        started translating yet rather than fall ever further behind."""
        waiting = [u for u in list(self._queue)[1:] if not u.dropped and u is not self._open]
        excess = len(waiting) - self.options.max_pending
        for utt in waiting[:max(0, excess)]:
            utt.dropped = True
            utt.reason = "backlog"
            log.warning("[%s %s] engine behind; dropping utterance %d",
                        self.id, self.label, utt.id)

    # ── the pump ─────────────────────────────────────────────────────────────

    async def _pump(self):
        seg = self.engine.segment_samples
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._queue:
                utt = self._queue[0]
                if utt.dropped:
                    await self._finish_dropped(self._queue.popleft())
                    continue
                if not (utt.ended or utt.n - utt.stepped >= seg):
                    break
                try:
                    # The utterance may end while a step is running; it is
                    # only over once a step has run with finished=True.
                    was_final = await self._step(utt)
                except Exception as exc:
                    log.exception("[%s %s] translation step failed", self.id, self.label)
                    utt.dropped = True
                    utt.reason = "engine_error"
                    await self.send_event("error", code="engine_error", message=str(exc))
                    continue
                if was_final and not utt.dropped:
                    self._queue.popleft()
                    await self._finish(utt)
            if not self._queue:
                self._idle.set()

    async def _step(self, utt: _Utterance) -> bool:
        """One engine step; True if it was the utterance's final one."""
        if utt.ctx is None:
            utt.ctx = await self.engine.run(self.engine.new_context, self.model_tgt)
        if not utt.announced:
            utt.announced = True
            await self.send_event("speech_start", utterance=utt.id)
        source, finished = utt.source(), utt.ended
        utt.stepped = source.size
        res = await self.engine.run(self.engine.step, utt.ctx, source, finished)
        if utt.dropped:
            return finished
        if finished and not res.done:
            log.debug("[%s] engine left utterance %d unfinished", self.id, utt.id)
        if res.audio.size:
            out = self._out_resampler(res.audio, last=finished)
            if finished:
                self._out_resampler.reset()
            await self._send_audio(utt, out, res.audio.size)
        elif finished:
            tail = self._out_resampler(np.zeros(0, np.float32), last=True)
            self._out_resampler.reset()
            if tail.size:
                await self._send_audio(utt, tail, 0)
        src_text = res.source_text or utt.source_text
        tgt_text = res.target_text or utt.target_text
        if (src_text, tgt_text) != (utt.source_text, utt.target_text):
            utt.source_text, utt.target_text = src_text, tgt_text
            if not finished:
                await self.send_event("text", utterance=utt.id, source_text=src_text,
                                      target_text=tgt_text, final=False)
        return finished

    async def _send_audio(self, utt: _Utterance, pcm: np.ndarray, model_samples: int):
        if pcm.size == 0:
            return
        now = time.monotonic()
        utt.t_first_out = utt.t_first_out or now
        utt.t_last_out = now
        utt.out_samples += model_samples
        await self._send_bytes(wire.pcm_to_bytes(pcm, self._out_fmt))

    async def _finish(self, utt: _Utterance):
        if utt.ctx is not None:
            self.engine.release(utt.ctx)
        self.utterances_done += 1
        now = time.monotonic()
        t_end = utt.t_end if utt.t_end is not None else now
        await self.send_event("text", utterance=utt.id, source_text=utt.source_text,
                              target_text=utt.target_text, final=True)
        await self.send_event(
            "utterance_end", utterance=utt.id, reason=utt.reason,
            source_text=utt.source_text, target_text=utt.target_text,
            speech_ms=round(utt.n * 1000 / MODEL_RATE),
            output_ms=round(utt.out_samples * 1000 / self.engine.output_rate),
            start_offset_ms=(round((utt.t_first_out - utt.t_start) * 1000)
                             if utt.t_first_out else None),
            end_offset_ms=round(((utt.t_last_out or now) - t_end) * 1000))
        log.info("[%s %s] #%d %s | %s  ->  %s", self.id, self.label, utt.id,
                 utt.reason, utt.source_text, utt.target_text)

    async def _finish_dropped(self, utt: _Utterance):
        if utt.ctx is not None:
            self.engine.release(utt.ctx)
        if utt.announced or utt.reason not in ("too_short",):
            await self.send_event("dropped", utterance=utt.id, reason=utt.reason or "dropped")

    async def _idle_wait(self):
        self._wake.set()
        await self._idle.wait()

    # ── sending ──────────────────────────────────────────────────────────────

    async def send_event(self, kind: str, **fields):
        if self.closed:
            return
        async with self._send_lock:
            await self._send_text_raw(wire.dumps(kind, **fields))

    async def _send_bytes(self, data: bytes):
        if self.closed:
            return
        async with self._send_lock:
            await self._send_bytes_raw(data)

    def _require_started(self):
        if not self.started:
            raise ProtocolError("not_started", "send a 'start' message first")


def _rate(value) -> int:
    try:
        rate = int(value)
    except (TypeError, ValueError):
        raise ProtocolError("bad_format", f"sample_rate {value!r} is not an integer")
    if not 8000 <= rate <= 192000:
        raise ProtocolError("bad_format", f"sample_rate {rate} out of range 8000-192000")
    return rate
