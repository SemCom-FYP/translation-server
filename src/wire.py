# coding: utf-8
"""
Wire protocol of the translation server. Shared by the server and the client
adapter, so this module needs nothing beyond numpy and the standard library.

One WebSocket per translated voice, at  ws://<host>:<port>/v1/stream

    client                                  server
    ──────                                  ──────
    {"type": "start", ...}           ──▶
                                     ◀──    {"type": "ready", ...}
    binary: input audio              ──▶
    binary: input audio              ──▶    {"type": "speech_start", "utterance": 1}
    ...                              ◀──    {"type": "text", ...}        (partial)
                                     ◀──    binary: translated speech
                                     ◀──    {"type": "utterance_end", ...}
    {"type": "stop"}                 ──▶
                                     ◀──    {"type": "stopped"}

Client → server, text frames (JSON):

    start      first message, required.
               {"type": "start", "protocol": 1,
                "tgt_lang": "es",            # required: "es" or "spa" style codes
                "src_lang": "en",            # optional hint; the model detects
                                             # the spoken language by itself
                "input":  {"format": "pcm_s16le" | "pcm_f32le" | "encodec_packet",
                           "sample_rate": 24000},
                "output": {"format": "pcm_s16le" | "pcm_f32le",
                           "sample_rate": 24000},
                "label": "Alice"}                     # optional, for logs only
    configure  switch the output language mid-stream; speech in flight is dropped.
               {"type": "configure", "tgt_lang": "fr"}
    flush      treat the current utterance as finished now (push-to-talk release).
    reset      drop the current utterance and everything queued, untranslated.
    stop       finish what was heard, then the server replies "stopped" and closes.
    ping       {"type": "ping", "t": <anything>}  → {"type": "pong", "t": <same>}

Client → server, binary frames: input audio in the declared format.

    pcm_s16le / pcm_f32le   mono little-endian PCM, any number of samples.
    encodec_packet          one webrtc_conference 'A' audio packet per frame,
                            byte-for-byte as it arrived on the `audio`
                            DataChannel (see pack_audio / unpack_audio below,
                            which match webrtc_conference/protocol.py).

Server → client, text frames (JSON):

    ready          {"type": "ready", "protocol": 1, "session": "...",
                    "src_lang", "tgt_lang",          # as the client sent them
                    "model_tgt_lang": "spa",         # what the model was told
                    "target_langs": [...],           # every language it can speak
                    "output": {...}, "engine": "seamless-streaming" | "mock"}
    configured     {"type": "configured", "src_lang", "tgt_lang"}
    speech_start   {"type": "speech_start", "utterance": n}
    text           {"type": "text", "utterance": n, "source_text": "...",
                    "target_text": "...", "final": false}
                   sent whenever either text changes; both are the whole
                   utterance so far. source_text is "" when the engine does
                   not transcribe (SeamlessStreaming translates directly).
    utterance_end  {"type": "utterance_end", "utterance": n, "reason":
                    "silence" | "pause" | "max_length" | "flush" | "stop",
                    "source_text", "target_text",
                    "speech_ms":      length of the source utterance,
                    "output_ms":      length of the translated speech,
                    "start_offset_ms": speech start → first translated audio sent,
                    "end_offset_ms":   speech end   → last translated audio sent}
    dropped        {"type": "dropped", "utterance": n, "reason": "..."}
    error          {"type": "error", "code": "...", "message": "..."}
                   codes: unsupported_language, unsupported_format, bad_format,
                   unsupported_protocol, not_started, bad_request, bad_audio,
                   engine_error
    stopped        {"type": "stopped"}
    pong

Server → client, binary frames: translated speech, in the declared output
format. Frames belonging to utterance n arrive after its speech_start and
before its utterance_end, so no per-frame header is needed.
"""

import json
import struct
from dataclasses import dataclass

import numpy as np

PROTOCOL_VERSION = 1
WS_PATH = "/v1/stream"

PCM_S16LE = "pcm_s16le"
PCM_F32LE = "pcm_f32le"
ENCODEC_PACKET = "encodec_packet"
INPUT_FORMATS = (PCM_S16LE, PCM_F32LE, ENCODEC_PACKET)
OUTPUT_FORMATS = (PCM_S16LE, PCM_F32LE)

# The conference app's EnCodec model runs at 24 kHz.
ENCODEC_SAMPLE_RATE = 24000


# ── JSON messages ────────────────────────────────────────────────────────────

def dumps(kind: str, **fields) -> str:
    return json.dumps({"type": kind, **fields}, ensure_ascii=False)


def loads(text: str) -> dict:
    msg = json.loads(text)
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise ValueError("message must be a JSON object with a string 'type'")
    return msg


# ── PCM ──────────────────────────────────────────────────────────────────────

def pcm_to_bytes(pcm: np.ndarray, fmt: str) -> bytes:
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if fmt == PCM_F32LE:
        return pcm.astype("<f4").tobytes()
    if fmt == PCM_S16LE:
        return (np.clip(pcm, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()
    raise ValueError(f"unsupported PCM format {fmt!r}")


def bytes_to_pcm(data: bytes, fmt: str) -> np.ndarray:
    if fmt == PCM_F32LE:
        n = len(data) // 4
        return np.frombuffer(data, dtype="<f4", count=n).astype(np.float32)
    if fmt == PCM_S16LE:
        n = len(data) // 2
        return np.frombuffer(data, dtype="<i2", count=n).astype(np.float32) / 32768.0
    raise ValueError(f"unsupported PCM format {fmt!r}")


# ── webrtc_conference 'A' audio packets ─────────────────────────────────────
# Kept byte-compatible with webrtc_conference/protocol.py (selftest.py checks
# the two against each other when both are importable). Duplicated rather than
# imported so the server does not depend on the conference package.

_MAGIC_AUDIO = ord("A")
_APP_VERSION = 1
_AUDIO_HEADER = struct.Struct("<BBHIBH")    # magic, version, seq, t_ms, n_q, n_frames
_CODEBOOK_BITS = 10


@dataclass
class AudioPacket:
    seq: int
    t_ms: int
    codes: np.ndarray           # (n_q, n_frames) int64


def pack_audio(seq: int, t_ms: int, codes: np.ndarray) -> bytes:
    codes = np.asarray(codes)
    n_q, n_frames = codes.shape
    header = _AUDIO_HEADER.pack(_MAGIC_AUDIO, _APP_VERSION, seq & 0xFFFF,
                                t_ms & 0xFFFFFFFF, n_q, n_frames)
    v = np.ascontiguousarray(codes.T).ravel().astype(np.uint16)
    bits = np.unpackbits(v.view(np.uint8).reshape(-1, 2)[:, ::-1], axis=1)
    flat = bits[:, 16 - _CODEBOOK_BITS:].ravel()
    pad = (-flat.size) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return header + np.packbits(flat).tobytes()


def unpack_audio(data: bytes) -> AudioPacket:
    if len(data) < _AUDIO_HEADER.size:
        raise ValueError("audio packet too short")
    magic, version, seq, t_ms, n_q, n_frames = _AUDIO_HEADER.unpack_from(data, 0)
    if magic != _MAGIC_AUDIO:
        raise ValueError(f"not an audio packet (magic={magic:#x})")
    if version != _APP_VERSION:
        raise ValueError(f"unsupported audio packet version {version}")
    count = n_q * n_frames
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8, offset=_AUDIO_HEADER.size))
    need = count * _CODEBOOK_BITS
    if bits.size < need:
        raise ValueError(f"audio payload short: {bits.size} bits < {need}")
    bits = bits[:need].reshape(count, _CODEBOOK_BITS)
    full = np.concatenate([np.zeros((count, 16 - _CODEBOOK_BITS), dtype=np.uint8), bits], axis=1)
    flat = np.packbits(full, axis=1)[:, ::-1].copy().view(np.uint16).ravel()
    return AudioPacket(seq=seq, t_ms=t_ms,
                       codes=flat.reshape(n_frames, n_q).T.astype(np.int64))
