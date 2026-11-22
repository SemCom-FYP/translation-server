# coding: utf-8
"""
Call rooms: people join a room, and the server translates each person's voice
into every OTHER person's language and sends it to them.

    WS /v1/call?room=<name>&name=<me>&lang=<language I want to hear>[&speaks=<language I speak>]

    client -> server   binary frames: my microphone, float32 little-endian mono, 24 kHz
    server -> client   binary frames: the others' voices translated into `lang`
                       (same format), and JSON text events:
                         {"type": "joined", "room", "peer", "peers": [...], "model_tgt_lang"}
                         {"type": "peer_joined" | "peer_left", "name"}
                         the usual translation events (text, utterance_end, ...)
                       Nothing of my own voice comes back.

For every ordered pair (speaker, listener) the server runs one translation
Session (speaker's mic in, listener's language out, delivered to the listener's
WebSocket). If `speaks` is given and equals the listener's language, the voice is
forwarded untouched instead of translated. A room holds up to MAX_PEERS people.
"""

import asyncio
import itertools
import logging
from dataclasses import dataclass, field

from . import wire
from .session import ProtocolError, Session

log = logging.getLogger("translation.room")

MAX_PEERS = 4
CALL_FORMAT = {"format": wire.PCM_F32LE, "sample_rate": 24000}

_peer_ids = itertools.count(1)


class RoomError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Peer:
    name: str
    hear: str                       # language this person wants to hear
    speaks: str                     # language this person speaks ("" = unknown)
    ws: object
    id: int = field(default_factory=lambda: next(_peer_ids))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    gone: bool = False              # the connection closed; drop everything sent to it

    async def send_text(self, text: str):
        if self.gone or self.ws.closed:
            return
        async with self.lock:
            try:
                await self.ws.send_str(text)
            except (ConnectionError, RuntimeError):
                self.gone = True        # listener left mid-send: not an error

    async def send_bytes(self, data: bytes):
        if self.gone or self.ws.closed:
            return
        async with self.lock:
            try:
                await self.ws.send_bytes(data)
            except (ConnectionError, RuntimeError):
                self.gone = True


class Room:
    def __init__(self, name: str, server):
        self.name = name
        self.server = server                # TranslationServer: engine, options, sessions
        self.peers: dict = {}               # peer id -> Peer
        self.links: dict = {}               # (speaker id, listener id) -> Session | None (pass-through)

    def _same_language(self, speaker: Peer, listener: Peer) -> bool:
        if not speaker.speaks:
            return False
        engine = self.server.engine
        a, b = engine.resolve_lang(speaker.speaks), engine.resolve_lang(listener.hear)
        return a is not None and a == b

    async def join(self, name: str, hear: str, speaks: str, ws) -> Peer:
        engine = self.server.engine
        if engine.resolve_lang(hear) is None:
            raise RoomError("unsupported_language",
                            f"cannot speak {hear!r}; target languages: "
                            f"{', '.join(engine.target_langs)}")
        if len(self.peers) >= MAX_PEERS:
            raise RoomError("room_full", f"room {self.name!r} already has {MAX_PEERS} people")
        peer = Peer(name=name, hear=hear, speaks=speaks, ws=ws)
        pairs = [p for other in self.peers.values() for p in ((peer, other), (other, peer))]
        needed = sum(1 for sp, li in pairs if not self._same_language(sp, li))
        if len(self.server.sessions) + needed > self.server.max_sessions:
            raise RoomError("server_full", "not enough capacity for another person")
        self.peers[peer.id] = peer
        try:
            for speaker, listener in pairs:
                await self._link(speaker, listener)
        except Exception:
            await self.leave(peer, announce=False)
            raise
        await peer.send_text(wire.dumps(
            "joined", room=self.name, peer=peer.name, model_tgt_lang=engine.resolve_lang(hear),
            peers=[p.name for p in self.peers.values() if p is not peer]))
        await self._broadcast(peer, wire.dumps("peer_joined", name=peer.name))
        log.info("[room %s] %s joined (hears %s, speaks %s); %d in room",
                 self.name, peer.name, hear, speaks or "?", len(self.peers))
        return peer

    async def _link(self, speaker: Peer, listener: Peer):
        key = (speaker.id, listener.id)
        if self._same_language(speaker, listener):
            self.links[key] = None
            return
        session = Session(self.server.engine, listener.send_text, listener.send_bytes,
                          self.server.options)
        await session.start({"tgt_lang": listener.hear, "src_lang": speaker.speaks or None,
                             "input": CALL_FORMAT, "output": CALL_FORMAT,
                             "label": f"{self.name}:{speaker.name}->{listener.name}"})
        self.links[key] = session
        self.server.sessions[session.id] = session

    async def feed(self, speaker: Peer, data: bytes):
        """Route one chunk of the speaker's microphone to every other person."""
        for (speaker_id, listener_id), session in list(self.links.items()):
            if speaker_id != speaker.id:
                continue
            try:
                if session is None:
                    await self.peers[listener_id].send_bytes(data)
                else:
                    await session.feed(data)
            except (ProtocolError, ValueError):
                pass                                    # malformed chunk: skip it
            except (ConnectionError, KeyError):
                pass                                    # that listener just left

    async def leave(self, peer: Peer, announce: bool = True):
        self.peers.pop(peer.id, None)
        for key in [k for k in self.links if peer.id in k]:
            session = self.links.pop(key)
            if session is not None:
                await session.close()
                self.server.sessions.pop(session.id, None)
        if announce and self.peers:
            await self._broadcast(peer, wire.dumps("peer_left", name=peer.name))
        log.info("[room %s] %s left; %d in room", self.name, peer.name, len(self.peers))

    async def _broadcast(self, source: Peer, text: str):
        for other in list(self.peers.values()):
            if other is not source:
                try:
                    await other.send_text(text)
                except ConnectionError:
                    pass
