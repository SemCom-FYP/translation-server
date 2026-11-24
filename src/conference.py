# coding: utf-8
"""
Conference voice relay: the audio path of the neural conference, run through
this server.

    WS /v1/conf?room=<room>&peer=<id>&name=<name>&speaks=<lang>[&hear=<lang>][&token=]

Every participant sends their microphone as Opus and receives everyone else's
voice as Opus. For each (speaker -> listener) pair the server decides:

    listener wants no translation, or speaker already speaks the listener's
    language  ->  the Opus packet is forwarded untouched ("pass-through")
    otherwise ->  the voice is decoded, translated by SeamlessStreaming into the
                  listener's language, and re-encoded as Opus

Video does not pass through here (it stays peer-to-peer). What ties the two
together is the sender's capture timestamp `t_ms`, which every voice packet
carries: pass-through packets keep the sender's own, and translated packets are
stamped with the moment their source speech began plus the offset into the
translation, so the listener's playhead tracks the speaker's timeline.

Transport. Audio frames travel over the WebSocket by default. A client may
upgrade them to an unreliable WebRTC datachannel (UDP, so a late packet is dropped
instead of stalling the stream behind it, as TCP would): it sends
{"type": "rtc_offer", "sdp": ...} on the WebSocket, the server answers with
{"type": "rtc_answer", "sdp": ...}, and once the pre-negotiated channel "voice"
(id 1) opens, audio in both directions uses it, with the same frame formats
below. If aiortc is missing or the channel never opens, the WebSocket carries on.

Binary, participant -> server    <BBHI  magic 'U', version, seq, t_ms   + Opus
Binary, server -> participant    <BBBBHI magic 'V', version, flags, id_len, seq, t_ms
                                         + speaker id (id_len bytes) + Opus
                                 flags bit 0 = this packet is translated
Text, participant -> server      {"type": "configure", "hear": "es" | null}
                                 {"type": "ping"} | {"type": "leave"}
Text, server -> participant      {"type": "joined" | "peer_joined" | "peer_left" | "pong" | "error"}
                                 and the translation events (text, utterance_end, ...),
                                 each with a "speaker" field
"""

import asyncio
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import wire
from .opus import FRAME_MS, FRAME_SAMPLES, OpusDecoder, OpusEncoder
from .session import Session

log = logging.getLogger("translation.conference")

MAGIC_UP = ord("U")
MAGIC_DOWN = ord("V")
VERSION = 1
FLAG_TRANSLATED = 1 << 0
MAX_MEMBERS = 8
CONF_PATH = "/v1/conf"
RTC_BACKPRESSURE_BYTES = 32 * 1024
RTC_CHANNEL_ID = 1                  # pre-negotiated "voice" channel, both ends
RTC_PACKET_LIFETIME_MS = 300        # ordered, but a packet older than this is dropped

_UP = struct.Struct("<BBHI")
_DOWN = struct.Struct("<BBBBHI")

_PCM = {"format": wire.PCM_F32LE, "sample_rate": 24000}


def pack_down(speaker_id: str, flags: int, seq: int, t_ms: int, opus: bytes) -> bytes:
    sid = speaker_id.encode()[:255]
    return _DOWN.pack(MAGIC_DOWN, VERSION, flags, len(sid), seq & 0xFFFF,
                      t_ms & 0xFFFFFFFF) + sid + opus


class ConferenceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class Member:
    peer_id: str
    name: str
    speaks: str                         # language this person speaks ("" = unknown)
    hear: Optional[str]                 # language they want to hear (None = originals)
    ws: object
    decoder: Optional[OpusDecoder] = None
    gone: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pc: object = None                   # aiortc RTCPeerConnection, when upgraded
    dc: object = None                   # its "voice" datachannel
    inbox: Optional[asyncio.Queue] = None   # audio frames in arrival order, either transport

    def enqueue(self, data: bytes):
        """Audio from the participant (WebSocket or datachannel); processed in order."""
        if self.inbox is None:
            return
        try:
            self.inbox.put_nowait(data)
        except asyncio.QueueFull:
            pass                                    # server behind: drop, stay live

    async def send_audio(self, data: bytes):
        """One audio frame to the participant: the datachannel if it is open
        (dropping the frame if the link is congested), else the WebSocket."""
        dc = self.dc
        if dc is not None and dc.readyState == "open":
            if dc.bufferedAmount > RTC_BACKPRESSURE_BYTES:
                return
            try:
                dc.send(data)
                return
            except Exception:
                self.dc = None
        await self.send_bytes(data)

    async def send_text(self, text: str):
        if self.gone or self.ws.closed:
            return
        async with self.lock:
            try:
                await self.ws.send_str(text)
            except (ConnectionError, RuntimeError):
                self.gone = True

    async def send_bytes(self, data: bytes):
        if self.gone or self.ws.closed:
            return
        async with self.lock:
            try:
                await self.ws.send_bytes(data)
            except (ConnectionError, RuntimeError):
                self.gone = True

    async def send_event(self, kind: str, **fields):
        await self.send_text(wire.dumps(kind, **fields))


class Packetizer:
    """Cuts a translated voice into 20 ms Opus packets for one listener, stamping
    each with the sender-clock time it corresponds to."""

    def __init__(self, speaker: Member, listener: Member, bitrate_kbps: float):
        self.speaker, self.listener = speaker, listener
        self._enc = OpusEncoder(bitrate_kbps)
        self._buf = np.zeros(0, np.float32)
        self._t: Optional[float] = None
        self._seq = 0

    async def push(self, pcm: Optional[np.ndarray], t_ms: Optional[int]):
        if pcm is None:
            await self._flush()
            return
        if self._buf.size == 0 or self._t is None:
            self._t = float(t_ms if t_ms is not None else int(time.monotonic() * 1000))
        self._buf = np.concatenate([self._buf, np.asarray(pcm, np.float32)])
        while self._buf.size >= FRAME_SAMPLES:
            frame, self._buf = self._buf[:FRAME_SAMPLES], self._buf[FRAME_SAMPLES:]
            await self._emit(frame)

    async def _flush(self):
        if self._buf.size:
            pad = np.zeros(FRAME_SAMPLES - self._buf.size, np.float32)
            frame, self._buf = np.concatenate([self._buf, pad]), self._buf[:0]
            await self._emit(frame)
        self._t = None

    async def _emit(self, frame: np.ndarray):
        packet = self._enc.encode(frame)
        t = int(self._t) & 0xFFFFFFFF
        self._t += FRAME_MS
        self._seq += 1
        await self.listener.send_audio(
            pack_down(self.speaker.peer_id, FLAG_TRANSLATED, self._seq, t, packet))


@dataclass
class Link:
    """One speaker -> one listener. `session` None means pass-through."""
    session: Optional[Session] = None


class ConferenceRoom:
    def __init__(self, name: str, server, opus_kbps: float = 24.0):
        self.name = name
        self.server = server
        self.opus_kbps = opus_kbps
        self.members: dict = {}
        self.links: dict = {}           # (speaker id, listener id) -> Link

    # ── decisions ────────────────────────────────────────────────────────────

    def _needs_translation(self, speaker: Member, listener: Member) -> bool:
        if not listener.hear:
            return False
        engine = self.server.engine
        spoken = engine.resolve_lang(speaker.speaks) if speaker.speaks else None
        return spoken is None or spoken != engine.resolve_lang(listener.hear)

    async def _make_link(self, speaker: Member, listener: Member):
        key = (speaker.peer_id, listener.peer_id)
        old = self.links.pop(key, None)
        if old is not None and old.session is not None:
            await self._close_session(old.session)
        if not self._needs_translation(speaker, listener):
            self.links[key] = Link()
            return
        if len(self.server.sessions) >= self.server.max_sessions:
            await listener.send_event("error", code="server_full", speaker=speaker.peer_id,
                                      message=f"no capacity to translate {speaker.name}; "
                                              "playing the original voice")
            self.links[key] = Link()
            return
        session = Session(self.server.engine, self._tag(listener, speaker),
                          listener.send_bytes, self.server.options)
        await session.start({"tgt_lang": listener.hear, "src_lang": speaker.speaks or None,
                             "input": _PCM, "output": _PCM,
                             "label": f"{self.name}:{speaker.name}->{listener.name}"})
        session.audio_sink = Packetizer(speaker, listener, self.opus_kbps).push
        self.server.sessions[session.id] = session
        self.links[key] = Link(session)

    @staticmethod
    def _tag(listener: Member, speaker: Member):
        """Session events as text for the listener, naming whose speech they're about."""
        async def send(text: str):
            try:
                event = json.loads(text)
                event["speaker"] = speaker.peer_id
                text = json.dumps(event, ensure_ascii=False)
            except ValueError:
                pass
            await listener.send_text(text)
        return send

    async def _close_session(self, session: Session):
        await session.close()
        self.server.sessions.pop(session.id, None)

    # ── membership ───────────────────────────────────────────────────────────

    async def join(self, member: Member):
        engine = self.server.engine
        if member.hear and engine.resolve_lang(member.hear) is None:
            raise ConferenceError("unsupported_language", f"cannot speak {member.hear!r}")
        if member.peer_id in self.members:
            raise ConferenceError("duplicate_peer", f"{member.peer_id!r} is already in the room")
        if len(self.members) >= MAX_MEMBERS:
            raise ConferenceError("room_full", f"room {self.name!r} is full")
        existing = list(self.members.values())
        self.members[member.peer_id] = member
        try:
            for other in existing:
                await self._make_link(other, member)
                await self._make_link(member, other)
        except Exception:
            await self.leave(member, announce=False)
            raise
        await member.send_event(
            "joined", room=self.name, peer=member.peer_id,
            peers=[{"peer": m.peer_id, "name": m.name} for m in existing])
        await self._broadcast(member, wire.dumps("peer_joined", peer=member.peer_id,
                                                 name=member.name))
        log.info("[conf %s] %s joined (speaks %s, hears %s); %d in room", self.name,
                 member.name, member.speaks or "?", member.hear or "originals",
                 len(self.members))

    async def accept_rtc(self, member: Member, sdp: str):
        """Answer a participant's WebRTC offer; audio moves to the datachannel once it opens."""
        try:
            from aiortc import (RTCConfiguration, RTCIceServer, RTCPeerConnection,
                                RTCSessionDescription)
        except ImportError:
            await member.send_event("error", code="rtc_unavailable",
                                    message="server has no aiortc; staying on the WebSocket")
            return
        await self.close_rtc(member)
        servers = [RTCIceServer(urls=u) for u in (self.server.ice_urls or [])]
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=servers))
        dc = pc.createDataChannel("voice", negotiated=True, id=RTC_CHANNEL_ID, ordered=True,
                                  maxPacketLifeTime=RTC_PACKET_LIFETIME_MS)
        member.pc = pc

        @dc.on("open")
        def _open():
            member.dc = dc
            log.info("[conf %s] %s: voice over WebRTC", self.name, member.name)

        @dc.on("close")
        def _closed():
            if member.dc is dc:
                member.dc = None

        @dc.on("message")
        def _message(data):
            if isinstance(data, bytes):
                member.enqueue(data)

        @pc.on("connectionstatechange")
        async def _state():
            if pc.connectionState in ("failed", "closed") and member.dc is dc:
                member.dc = None
                log.info("[conf %s] %s: WebRTC %s; back on the WebSocket",
                         self.name, member.name, pc.connectionState)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
            await pc.setLocalDescription(await pc.createAnswer())
        except Exception as exc:
            log.warning("[conf %s] %s: bad WebRTC offer: %s", self.name, member.name, exc)
            await self.close_rtc(member)
            await member.send_event("error", code="rtc_failed", message=f"WebRTC setup failed: {exc}")
            return
        await member.send_event("rtc_answer", sdp=pc.localDescription.sdp)

    async def close_rtc(self, member: Member):
        pc, member.pc, member.dc = member.pc, None, None
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                pass

    async def leave(self, member: Member, announce: bool = True):
        member.gone = True
        await self.close_rtc(member)
        self.members.pop(member.peer_id, None)
        for key in [k for k in self.links if member.peer_id in k]:
            link = self.links.pop(key)
            if link.session is not None:
                await self._close_session(link.session)
        if announce:
            await self._broadcast(member, wire.dumps("peer_left", peer=member.peer_id,
                                                     name=member.name))
        log.info("[conf %s] %s left; %d in room", self.name, member.name, len(self.members))

    async def configure(self, member: Member, hear: Optional[str]):
        """Listener changed the language they want to hear (None = originals)."""
        if hear and self.server.engine.resolve_lang(hear) is None:
            await member.send_event("error", code="unsupported_language",
                                    message=f"cannot speak {hear!r}")
            return
        member.hear = hear or None
        for other in list(self.members.values()):
            if other is not member:
                await self._make_link(other, member)
        await member.send_event("configured", hear=member.hear)
        log.info("[conf %s] %s now hears %s", self.name, member.name, member.hear or "originals")

    async def _broadcast(self, source: Member, text: str):
        for other in list(self.members.values()):
            if other is not source:
                await other.send_text(text)

    # ── audio ────────────────────────────────────────────────────────────────

    async def on_audio(self, speaker: Member, data: bytes):
        """One Opus packet from `speaker`: forward it, or translate it, per listener."""
        if len(data) <= _UP.size or data[0] != MAGIC_UP:
            return
        _, _, seq, t_ms = _UP.unpack_from(data)
        opus = data[_UP.size:]
        pcm_bytes = None
        for (speaker_id, listener_id), link in list(self.links.items()):
            if speaker_id != speaker.peer_id:
                continue
            listener = self.members.get(listener_id)
            if listener is None:
                continue
            try:
                if link.session is None:
                    await listener.send_audio(pack_down(speaker.peer_id, 0, seq, t_ms, opus))
                    continue
                if pcm_bytes is None:                       # decode once for all listeners
                    if speaker.decoder is None:
                        speaker.decoder = OpusDecoder()
                    pcm_bytes = speaker.decoder.decode(opus).astype("<f4").tobytes()
                await link.session.feed(pcm_bytes, t_ms)
            except ValueError:
                pass                                        # bad packet: skip it
            except Exception:
                log.exception("[conf %s] %s -> %s failed", self.name, speaker.name,
                              listener.name)
