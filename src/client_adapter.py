# coding: utf-8
"""
Client side of the translation server, for the conference app (or anything
else). Needs only numpy and aiohttp, both already in webrtc_conference's
requirements; nothing here imports torch or any model code.

Two levels:

  TranslationClient    one WebSocket stream, run on its own thread, with a
                       thread-safe API: send_pcm / send_packet / flush / reset /
                       configure / stop / close, plus on_audio and on_event
                       callbacks. Reconnects by itself.

  RemoteTranslation    drop-in for webrtc_conference.translator.PeerTranslation:
                       same constructor shape (name, target, out, backlog_ms),
                       same feed(pcm24k) / set_target(code) / close(), same
                       lag_ms / last_text attributes for the UI. The speech is
                       translated by the server instead of on this machine.

Integration into webrtc_conference/client.py is described in README.md.
"""

import asyncio
import logging
import threading
from typing import Callable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from . import wire

log = logging.getLogger("translation.client")

DEFAULT_URL = "ws://127.0.0.1:8770" + wire.WS_PATH


def _with_token(url: str, token: Optional[str]) -> str:
    if not token:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query["token"] = token
    return urlunsplit(parts._replace(query=urlencode(query)))


class TranslationClient:
    """One translated voice over one WebSocket.

    All public methods may be called from any thread and never block (except
    `stop` and `close(wait=True)`). Audio sent while disconnected is dropped:
    a conversation cannot wait for a reconnect, and stale speech translated
    late is worse than a gap.

    Callbacks run on the client's own network thread:
      on_audio(pcm: float32 ndarray)   translated speech at `output_rate`
      on_event(msg: dict)              every JSON message from the server, plus
                                       {"type": "connected"} / {"type": "disconnected"}
    """

    def __init__(self, url: str = DEFAULT_URL, tgt_lang: str = "en", *,
                 src_lang: Optional[str] = None, input_format: str = wire.PCM_S16LE, input_rate: int = 24000,
                 output_format: str = wire.PCM_S16LE, output_rate: int = 24000,
                 label: str = "", token: Optional[str] = None,
                 on_audio: Optional[Callable[[np.ndarray], None]] = None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 reconnect: bool = True, max_queue: int = 64):
        if input_format not in wire.INPUT_FORMATS:
            raise ValueError(f"input_format must be one of {wire.INPUT_FORMATS}")
        if output_format not in wire.OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {wire.OUTPUT_FORMATS}")
        self.url = _with_token(url, token)
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.input_format, self.input_rate = input_format, int(input_rate)
        self.output_format, self.output_rate = output_format, int(output_rate)
        self.label = label
        self.on_audio = on_audio or (lambda pcm: None)
        self.on_event = on_event or (lambda msg: None)
        self.reconnect = reconnect
        self.max_queue = max_queue

        self.connected = False
        self.ready: Optional[dict] = None       # the server's last "ready" message
        self.last_error: Optional[dict] = None
        self._running = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._ws = None
        self._closing: Optional[asyncio.Event] = None
        self._stopped = threading.Event()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, daemon=True,
                                        name=f"translation-{label or tgt_lang}")
        self._thread.start()
        self._started.wait(5)

    # ── public, thread-safe ─────────────────────────────────────────────────

    def send_pcm(self, pcm: np.ndarray) -> None:
        """Mono float32 PCM at `input_rate` (pcm_* input formats)."""
        if self.input_format == wire.ENCODEC_PACKET:
            raise RuntimeError("this stream expects encodec packets; use send_packet")
        self._post(("bin", wire.pcm_to_bytes(pcm, self.input_format)))

    def send_packet(self, packet: bytes) -> None:
        """One webrtc_conference 'A' audio packet, as received (encodec_packet input)."""
        if self.input_format != wire.ENCODEC_PACKET:
            raise RuntimeError("this stream expects PCM; use send_pcm")
        self._post(("bin", bytes(packet)))

    def flush(self) -> None:
        self._post(("text", wire.dumps("flush")), control=True)

    def reset(self) -> None:
        self._post(("text", wire.dumps("reset")), control=True)

    def configure(self, tgt_lang: Optional[str] = None, src_lang: Optional[str] = None) -> None:
        """Change the output language (and/or the source-language hint).
        Speech in flight is dropped."""
        self.tgt_lang = tgt_lang or self.tgt_lang
        self.src_lang = src_lang or self.src_lang
        self._post(("text", wire.dumps("configure", **self._langs())), control=True)

    def _langs(self) -> dict:
        langs = {"tgt_lang": self.tgt_lang}
        if self.src_lang:
            langs["src_lang"] = self.src_lang
        return langs

    def stop(self, timeout: float = 30.0) -> bool:
        """Ask the server to finish everything heard so far, wait for it, and
        disconnect. True if the server confirmed."""
        self._stopped.clear()
        self._post(("text", wire.dumps("stop")), control=True)
        ok = self._stopped.wait(timeout)
        self.close()
        return ok

    def close(self, wait: bool = True) -> None:
        self._running = False
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._wake_and_close)
            except RuntimeError:
                pass
        if wait and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)

    # ── internals ───────────────────────────────────────────────────────────

    def _post(self, item, control: bool = False):
        loop = self._loop
        if loop is None or loop.is_closed() or not self._running:
            return
        try:
            loop.call_soon_threadsafe(self._enqueue, item, control)
        except RuntimeError:
            pass

    def _enqueue(self, item, control: bool):
        if not self.connected or self._queue is None:
            return          # control state (languages) is resent in the next "start"
        q = self._queue
        if q.full():
            if not control and item[0] == "bin":
                return                          # newest audio dropped: the sender is far behind
            try:
                q.get_nowait()                  # make room for a control message
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(item)

    def _wake_and_close(self):
        if self._closing is not None:
            self._closing.set()
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if self._ws is not None and not self._ws.closed:
            asyncio.ensure_future(self._ws.close())

    def _thread_main(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._started.set()
        try:
            loop.run_until_complete(self._main())
        finally:
            loop.close()

    async def _main(self):
        import aiohttp
        self._closing = asyncio.Event()
        backoff = 0.5
        async with aiohttp.ClientSession() as http:
            while self._running:
                try:
                    clean = await self._session(http)
                    backoff = 0.5 if clean else backoff
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("[%s] translation server unreachable (%s): %s",
                                self.label, urlsplit(self.url).netloc, exc)
                if self.connected:
                    self.connected = False
                    self._emit({"type": "disconnected"})
                self._queue = None
                if not (self._running and self.reconnect):
                    break
                try:
                    await asyncio.wait_for(self._closing.wait(), backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(10.0, backoff * 2)

    async def _session(self, http) -> bool:
        """One connection's lifetime. True if it got as far as 'ready'."""
        import aiohttp
        async with http.ws_connect(self.url, heartbeat=20, max_msg_size=1 << 22) as ws:
            self._ws = ws
            await ws.send_str(wire.dumps(
                "start", protocol=wire.PROTOCOL_VERSION, **self._langs(),
                input={"format": self.input_format, "sample_rate": self.input_rate},
                output={"format": self.output_format, "sample_rate": self.output_rate},
                label=self.label))
            first = await asyncio.wait_for(ws.receive(), timeout=30)
            if first.type != aiohttp.WSMsgType.TEXT:
                raise ConnectionError(f"unexpected first frame {first.type!r}")
            msg = wire.loads(first.data)
            if msg["type"] != "ready":
                self.last_error = msg
                self._emit(msg)
                if msg.get("code") in ("unsupported_language", "bad_format",
                                       "unsupported_format", "unsupported_protocol"):
                    # Retrying will not help until the caller changes something.
                    self.reconnect = False
                raise ConnectionError(f"server refused stream: {msg.get('message', msg)}")
            self.ready = msg
            self._queue = asyncio.Queue(maxsize=self.max_queue)
            self.connected = True
            self._emit({**msg, "type": "connected"})
            writer = asyncio.create_task(self._writer(ws))
            try:
                async for frame in ws:
                    if frame.type == aiohttp.WSMsgType.BINARY:
                        self._audio(frame.data)
                    elif frame.type == aiohttp.WSMsgType.TEXT:
                        event = wire.loads(frame.data)
                        if event["type"] == "error":
                            self.last_error = event
                            log.warning("[%s] server: %s", self.label, event.get("message"))
                        elif event["type"] == "stopped":
                            self._stopped.set()
                        self._emit(event)
                    elif frame.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break
            finally:
                writer.cancel()
                self._ws = None
            return True

    async def _writer(self, ws):
        while True:
            item = await self._queue.get()
            if item is None:
                return
            kind, data = item
            if kind == "bin":
                await ws.send_bytes(data)
            else:
                await ws.send_str(data)

    def _audio(self, data: bytes):
        try:
            self.on_audio(wire.bytes_to_pcm(data, self.output_format))
        except Exception:
            log.exception("[%s] on_audio callback failed", self.label)

    def _emit(self, event: dict):
        try:
            self.on_event(event)
        except Exception:
            log.exception("[%s] on_event callback failed", self.label)


class RemoteTranslation:
    """Drop-in for webrtc_conference.translator.PeerTranslation, backed by the
    translation server.

    Same constructor shape and behaviour, so client.py can build either one.
    `target` takes the app's codes ("es", "fr", "de", "it") or the model's
    ("spa", ...); the server speaks 36 languages and understands ~100, so the
    peer's own language does not have to be known. Extras:
      * `source_lang` (optional) passes the peer's language along as a hint.
      * `feed_codes()` / `feed_packet()` forward the app's EnCodec audio as-is
        (~6 kbps instead of ~384 kbps of PCM) when built with
        input_format="encodec_packet".
    """

    def __init__(self, server_url: str, name: str, target: Optional[str],
                 out: Callable[[np.ndarray], None], backlog_ms: Callable[[], float],
                 source_lang: Optional[str] = None, *, token: Optional[str] = None,
                 sample_rate: int = 24000, input_format: str = wire.PCM_S16LE,
                 on_text: Optional[Callable[[str, str, bool], None]] = None):
        self.name, self.target, self.source_lang = name, target, source_lang
        self._out, self._backlog_ms = out, backlog_ms
        self._on_text = on_text
        self._paused = target is None
        self._codes_seq = 0
        self.running = True

        # Read by the UI, like PeerTranslation's.
        self.lag_ms = 0.0               # end of an utterance → its translation audible
        self.last_text = ""             # last finished translation
        self.last_source_text = ""      # transcript, when the engine provides one
        self.partial_text = ""          # translation of the utterance in progress
        self.error: Optional[str] = None

        self._client = TranslationClient(
            server_url, target or "en", src_lang=source_lang,
            input_format=input_format,
            input_rate=wire.ENCODEC_SAMPLE_RATE if input_format == wire.ENCODEC_PACKET else sample_rate,
            output_rate=sample_rate, label=name, token=token,
            on_audio=self._audio, on_event=self._event)

    @property
    def connected(self) -> bool:
        return self._client.connected

    # PeerTranslation API ---------------------------------------------------

    def feed(self, pcm24k: np.ndarray) -> None:
        if not self._paused:
            self._client.send_pcm(pcm24k)

    def set_target(self, code: Optional[str]) -> None:
        """Switch the language heard, or pause with None. Speech in flight is
        dropped rather than finished, as with PeerTranslation."""
        self.target = code
        if code is None:
            self._paused = True
            self._client.reset()
            return
        self._paused = False
        self.error = None
        self._client.configure(tgt_lang=code)

    def close(self) -> None:
        self.running = False
        self._client.close(wait=False)

    # Extras ------------------------------------------------------------------

    def set_source(self, lang: str) -> None:
        """The peer's spoken language changed (e.g. announced in their hello)."""
        self.source_lang = lang
        if not self._paused:
            self._client.configure(src_lang=lang)

    def feed_codes(self, seq: int, t_ms: int, codes: np.ndarray) -> None:
        """EnCodec codes as the app's audio loop holds them (AudioPacket.codes)."""
        if not self._paused:
            self._client.send_packet(wire.pack_audio(seq, t_ms, codes))

    def feed_packet(self, packet: bytes) -> None:
        """A raw 'A' packet from the audio DataChannel, untouched."""
        if not self._paused:
            self._client.send_packet(packet)

    def flush(self) -> None:
        self._client.flush()

    # Callbacks (client thread) -------------------------------------------

    def _audio(self, pcm: np.ndarray):
        if not self._paused:
            self._out(pcm)

    def _event(self, ev: dict):
        kind = ev["type"]
        if kind == "text":
            self.partial_text = ev.get("target_text", "")
            if self._on_text:
                self._on_text(ev.get("source_text", ""), self.partial_text, bool(ev.get("final")))
        elif kind == "utterance_end":
            self.last_text = ev.get("target_text", "")
            self.last_source_text = ev.get("source_text", "")
            self.partial_text = ""
            end_offset = ev.get("end_offset_ms") or 0
            self.lag_ms = float(end_offset) + float(self._backlog_ms() or 0.0)
            log.info("[%s] %s -> %s  (%.0f ms after it was said)", self.name,
                     self.last_source_text, self.last_text, self.lag_ms)
        elif kind == "error":
            self.error = ev.get("message")
        elif kind == "disconnected":
            self.error = "translation server disconnected"
        elif kind == "connected":
            self.error = None


__all__ = ["TranslationClient", "RemoteTranslation", "DEFAULT_URL"]
