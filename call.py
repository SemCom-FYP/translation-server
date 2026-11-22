# coding: utf-8
"""
Two laptops, two people, each hears the other in their own language, with the
translation server in the middle. Run the same script on both laptops:

    pip install sounddevice numpy aiohttp

    laptop A (speaks English, hears English):
        python call.py --room demo --name Alice --lang en

    laptop B (speaks Spanish, hears Spanish):
        python call.py --room demo --name Bob --lang es

    mic ──► server ──► translated into the OTHER person's language ──► their speaker

--lang is the language YOU want to hear. Your own voice is never played back to you.
The laptops don't need to reach each other, only the server. Use headphones, or the
other person's voice from your speaker is picked up by your mic and sent back.

If both people speak the same language, add --speaks <that language>: the voice is
then forwarded untouched instead of being translated.
"""

import argparse
import asyncio
import json
import threading
import time
from urllib.parse import urlencode

import numpy as np
from aiohttp import ClientSession, WSMsgType

DEFAULT_SERVER = "ws://192.248.10.47:8770/v1/call"
RATE = 24000                 # float32 mono 24 kHz both ways
BLOCK = RATE * 40 // 1000    # 40 ms per packet
MAX_BUFFER_S = 15            # never queue more than this much unplayed audio


class Speaker:
    """Plays incoming audio from a callback, so the network loop never blocks on the sound card."""

    def __init__(self):
        import sounddevice as sd
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.stream = sd.RawOutputStream(samplerate=RATE, channels=1, dtype="float32",
                                         callback=self._fill)
        self.stream.start()

    def _fill(self, outdata, frames, time_info, status):
        need = frames * 4
        with self.lock:
            chunk = bytes(self.buf[:need])
            del self.buf[:need]
        outdata[:len(chunk)] = chunk
        if len(chunk) < need:
            outdata[len(chunk):] = bytes(need - len(chunk))

    def push(self, data: bytes):
        with self.lock:
            self.buf += data
            overflow = len(self.buf) - MAX_BUFFER_S * RATE * 4
            if overflow > 0:
                del self.buf[:overflow]

    def close(self):
        self.stream.stop()
        self.stream.close()


async def run(args):
    import sounddevice as sd
    t0 = time.monotonic()

    def log(msg):
        print(f"{time.monotonic() - t0:7.2f}s  {msg}", flush=True)

    query = {"room": args.room, "name": args.name, "lang": args.lang}
    if args.speaks:
        query["speaks"] = args.speaks
    if args.token:
        query["token"] = args.token
    url = f"{args.server}?{urlencode(query)}"

    speaker = Speaker()
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue(maxsize=100)

    def put(data):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass                                  # network behind: drop, never block the mic

    peak = [0.0]                                  # loudest mic sample since the last check

    def on_mic(indata, frames, t, status):
        peak[0] = max(peak[0], float(np.abs(indata).max()))
        loop.call_soon_threadsafe(put, indata[:, 0].astype("<f4").tobytes())

    async def mic_watch():
        """Say so when the microphone delivers silence, the usual reason nobody hears you."""
        while True:
            await asyncio.sleep(5)
            level, peak[0] = peak[0], 0.0
            if level < 0.003:
                log("WARNING: your microphone is silent. Wrong --device, muted, or blocked by "
                    "Windows privacy settings? (python client_test.py --list-devices)")
            elif args.meter:
                log(f"mic level {level:.3f}")

    device = int(args.device) if args.device and str(args.device).isdigit() else args.device
    try:
        async with ClientSession() as session:
            async with session.ws_connect(url, heartbeat=20, max_msg_size=0) as ws:
                async def sender():
                    while True:
                        await ws.send_bytes(await q.get())

                task = asyncio.create_task(sender())
                watch = asyncio.create_task(mic_watch())
                try:
                    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                                        device=device, blocksize=BLOCK, callback=on_mic):
                        log(f"in room {args.room!r} as {args.name}, hearing {args.lang}. "
                            f"Speak now (Ctrl+C to leave)")
                        async for msg in ws:
                            if msg.type == WSMsgType.BINARY:
                                speaker.push(msg.data)
                            elif msg.type == WSMsgType.TEXT:
                                ev = json.loads(msg.data)
                                kind = ev.get("type")
                                if kind == "joined":
                                    others = ", ".join(ev["peers"]) or "nobody yet"
                                    log(f"joined; already here: {others}")
                                elif kind in ("peer_joined", "peer_left"):
                                    log(f"{ev['name']} {kind.split('_')[1]}")
                                elif kind == "utterance_end":
                                    log(f"heard: {ev.get('target_text')}")
                                elif kind in ("error", "dropped"):
                                    log(f"{kind}: {ev}")
                            elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                                break
                finally:
                    task.cancel()
                    watch.cancel()
    finally:
        speaker.close()
        log("left the call")


def main():
    p = argparse.ArgumentParser(description="Translated call through the translation server")
    p.add_argument("--room", default="demo", help="same room name on both laptops")
    p.add_argument("--name", default=None, help="your name (shown to the others)")
    p.add_argument("--lang", required=True, help="language you want to HEAR (en, es, fr, ...)")
    p.add_argument("--speaks", default=None, help="language you speak (optional)")
    p.add_argument("--server", default=DEFAULT_SERVER, help="ws://host:8770/v1/call")
    p.add_argument("--token", default=None)
    p.add_argument("--device", default=None, help="microphone id or name (see client_test.py --list-devices)")
    p.add_argument("--meter", action="store_true", help="print the mic level every 5 s")
    args = p.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
