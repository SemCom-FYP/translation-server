# coding: utf-8
"""
Simultaneous speech-to-speech translation server (Meta SeamlessStreaming),
for the neural conference or any other client that can open a WebSocket.

    python -m translation_server.server                 # GPU; models from the fairseq2 cache
    python -m translation_server.server --prefetch      # just download the ~8 GB of models
    python -m translation_server.server --mock          # no models; echoes speech

Any of ~100 spoken languages in, any of 36 spoken languages out; every stream
picks its own output language.

Endpoints:
    GET  /            plain-text banner
    GET  /health      {"status": "ok", "engine", "target_langs", "sessions", ...}
    WS   /v1/stream   one translated voice per connection (see wire.py)

Wire protocol: wire.py. Client library for the conference app: client_adapter.py.
"""

import argparse
import asyncio
import hmac
import logging
import os
import time
from typing import Optional

from aiohttp import WSMsgType, web

from . import wire
from .room import Room, RoomError
from .session import ProtocolError, Session, SessionOptions

log = logging.getLogger("translation.server")

DEFAULT_PORT = 8770
CALL_PATH = "/v1/call"


class TranslationServer:
    def __init__(self, engine, options: Optional[SessionOptions] = None,
                 token: Optional[str] = None, max_sessions: int = 16):
        self.engine = engine
        self.options = options or SessionOptions()
        self.token = token
        self.max_sessions = max_sessions
        self.sessions: dict = {}
        self.rooms: dict = {}
        self.started_at = time.time()

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/health", self.health)
        app.router.add_get(wire.WS_PATH, self.stream)
        app.router.add_get(CALL_PATH, self.call)
        app.on_shutdown.append(self._on_shutdown)
        return app

    async def call(self, request):
        """A person in a call room: their mic in, the others' translated voices out."""
        if not self._authorised(request):
            return web.Response(status=401, text="bad or missing token\n")
        q = request.query
        hear = q.get("lang", "")
        if not hear:
            return web.Response(status=400, text="lang= (the language you want to hear) is required\n")
        room_name = q.get("room", "default")[:64]
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
        await ws.prepare(request)
        room = self.rooms.get(room_name) or self.rooms.setdefault(room_name, Room(room_name, self))
        peer = None
        try:
            name = (q.get("name") or f"guest{len(room.peers) + 1}")[:32]
            try:
                peer = await room.join(name, hear, q.get("speaks", ""), ws)
            except RoomError as exc:
                await ws.send_str(wire.dumps("error", code=exc.code, message=str(exc)))
                return ws
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await room.feed(peer, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        kind = wire.loads(msg.data)["type"]
                    except (ValueError, KeyError):
                        continue
                    if kind == "ping":
                        await peer.send_text(wire.dumps("pong"))
                    elif kind == "leave":
                        break
                elif msg.type == WSMsgType.ERROR:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            if peer is not None:
                await room.leave(peer)
            if not room.peers:
                self.rooms.pop(room_name, None)
            if not ws.closed:
                await ws.close()
        return ws

    async def index(self, request):
        return web.Response(text=(
            f"Speech translation server ({self.engine.name}). Speaks: "
            f"{', '.join(self.engine.target_langs)}. WebSocket: {wire.WS_PATH}\n"))

    async def health(self, request):
        return web.json_response({
            "status": "ok",
            "engine": self.engine.name,
            "protocol": wire.PROTOCOL_VERSION,
            "target_langs": self.engine.target_langs,
            "sessions": len(self.sessions),
            "rooms": {name: [p.name for p in r.peers.values()] for name, r in self.rooms.items()},
            "max_sessions": self.max_sessions,
            "uptime_s": round(time.time() - self.started_at),
        })

    def _authorised(self, request) -> bool:
        if not self.token:
            return True
        given = request.query.get("token", "")
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            given = auth[7:]
        return hmac.compare_digest(given.encode(), self.token.encode())

    async def stream(self, request):
        if not self._authorised(request):
            return web.Response(status=401, text="bad or missing token\n")
        if len(self.sessions) >= self.max_sessions:
            return web.Response(status=503, text="server full\n")

        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=1 << 20)
        await ws.prepare(request)
        session = Session(self.engine, ws.send_str, ws.send_bytes, self.options)
        self.sessions[session.id] = session
        peer = request.remote
        try:
            async for msg in ws:
                try:
                    if msg.type == WSMsgType.BINARY:
                        await session.feed(msg.data)
                    elif msg.type == WSMsgType.TEXT:
                        if not await self._control(session, ws, msg.data):
                            break
                    elif msg.type == WSMsgType.ERROR:
                        log.warning("[%s] websocket error: %s", session.id, ws.exception())
                        break
                except ProtocolError as exc:
                    await session.send_event("error", code=exc.code, message=str(exc))
                    if not session.started:
                        break
                except ValueError as exc:
                    await session.send_event("error", code="bad_request", message=str(exc))
        except asyncio.CancelledError:
            pass
        finally:
            await session.close()
            self.sessions.pop(session.id, None)
            log.info("[%s %s] closed (%s, %.0f s of audio, %d utterances)",
                     session.id, session.label, peer, session.received_s,
                     session.utterances_done)
            if not ws.closed:
                await ws.close()
        return ws

    async def _control(self, session: Session, ws, text: str) -> bool:
        """Handle one JSON message; False ends the connection."""
        msg = wire.loads(text)
        kind = msg["type"]
        if kind == "start":
            ready = await session.start(msg)
            await session.send_event("ready", **ready)
        elif kind == "configure":
            await session.configure(msg)
            await session.send_event("configured", src_lang=session.src_lang,
                                     tgt_lang=session.tgt_lang,
                                     model_tgt_lang=session.model_tgt)
        elif kind == "flush":
            await session.flush()
        elif kind == "reset":
            await session.reset()
        elif kind == "ping":
            await session.send_event("pong", t=msg.get("t"))
        elif kind == "stop":
            await session.stop()
            await session.send_event("stopped")
            return False
        else:
            raise ProtocolError("bad_request", f"unknown message type {kind!r}")
        return True

    async def _on_shutdown(self, app):
        for session in list(self.sessions.values()):
            await session.close()
        self.engine.close()


# ── entry point ──────────────────────────────────────────────────────────────

def build_engine(args):
    if args.mock:
        from .engine import MockEngine
        return MockEngine(segment_ms=args.segment_ms)
    from .engine import SeamlessStreamingEngine
    return SeamlessStreamingEngine(
        device=args.device, dtype=args.dtype, segment_ms=args.segment_ms,
        decision_threshold=args.decision_threshold,
        min_start_frames=args.min_start_ms // 10,
        min_unit_chunk_size=args.min_unit_chunk, warmup=not args.prefetch)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m translation_server.server",
        description="Simultaneous speech-to-speech translation server (SeamlessStreaming)")
    g = p.add_argument_group("network")
    g.add_argument("--host", default="0.0.0.0")
    g.add_argument("--port", type=int, default=DEFAULT_PORT)
    g.add_argument("--token", default=os.environ.get("TRANSLATION_TOKEN"),
                   help="shared secret clients must send (?token= or Bearer header); "
                        "default $TRANSLATION_TOKEN, unset = open")
    g.add_argument("--max-sessions", type=int, default=16)

    g = p.add_argument_group("model")
    g.add_argument("--mock", action="store_true",
                   help="no models: echo speech back one step late (for testing)")
    g.add_argument("--prefetch", action="store_true",
                   help="download and load the models, then exit (they are cached in "
                        "$FAIRSEQ2_CACHE_DIR, default ~/.cache/fairseq2)")
    g.add_argument("--device", default="auto",
                   help="auto (first GPU if there is one), cuda:0, cuda:1, cpu, ...")
    g.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                   help="fp16 halves GPU memory; CPU always runs fp32")
    g.add_argument("--segment-ms", type=int, default=320,
                   help="new audio that triggers a decoding step "
                        "(multiple of 40; 320 is Meta's default)")
    g.add_argument("--decision-threshold", type=float, default=0.5,
                   help="how sure the model must be before it speaks: higher = later "
                        "but more accurate")
    g.add_argument("--min-start-ms", type=int, default=1920,
                   help="audio heard before the first word of an utterance is attempted")
    g.add_argument("--min-unit-chunk", type=int, default=50,
                   help="speech units gathered before each burst of synthesis "
                        "(smaller = sooner but choppier)")

    g = p.add_argument_group("utterance detection")
    g.add_argument("--vad", choices=["auto", "silero", "energy"], default="auto",
                   help="auto = Silero if `silero-vad` is installed, else energy")
    g.add_argument("--vad-threshold", type=float, default=0.5)
    g.add_argument("--end-silence-ms", type=int, default=600,
                   help="pause that ends an utterance")
    g.add_argument("--max-utterance-s", type=float, default=15.0,
                   help="hard cut for speech that never pauses")
    g.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.segment_ms < 40 or args.segment_ms % 40:
        p.error("--segment-ms must be a positive multiple of 40")
    if args.mock and args.prefetch:
        p.error("--prefetch downloads the real models; drop --mock")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-20s %(message)s", datefmt="%H:%M:%S")
    engine = build_engine(args)
    if args.prefetch:
        print(f"models downloaded and loaded ({engine.name}); start without --prefetch to serve")
        engine.close()
        return
    options = SessionOptions(
        vad=args.vad, vad_threshold=args.vad_threshold,
        end_silence_ms=args.end_silence_ms,
        soft_max_s=min(10.0, args.max_utterance_s * 2 / 3),
        hard_max_s=args.max_utterance_s)
    server = TranslationServer(engine, options, token=args.token,
                               max_sessions=args.max_sessions)
    print(f"Translation server ({engine.name}, {len(engine.target_langs)} output "
          f"languages) on ws://{args.host}:{args.port}{wire.WS_PATH}")
    web.run_app(server.build_app(), host=args.host, port=args.port,
                access_log=None, print=None)


if __name__ == "__main__":
    main()
