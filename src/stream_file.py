# coding: utf-8
"""
Stream a WAV file to the translation server in real time, as a call would,
and save what comes back.

    python -m translation_server.stream_file speech_en.wav --tgt es \
        --out translated_es.wav --server ws://127.0.0.1:8770/v1/stream

The input must be 16-bit PCM WAV (convert anything else with
`ffmpeg -i in.mp3 -ac 1 -ar 24000 -c:a pcm_s16le in.wav`). Stereo is mixed down.
"""

import argparse
import sys
import threading
import time
import wave

import numpy as np

from . import wire
from .client_adapter import DEFAULT_URL, TranslationClient


def read_wav(path: str):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit PCM WAV (got {8 * w.getsampwidth()}-bit)")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm, rate


def write_wav(path: str, pcm: np.ndarray, rate: int):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(wire.pcm_to_bytes(pcm, wire.PCM_S16LE))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m translation_server.stream_file")
    p.add_argument("wav")
    p.add_argument("--tgt", default="es", help="language to hear it in (es, fr, en, spa, ...)")
    p.add_argument("--src", default=None, help="optional hint: language spoken in the file")
    p.add_argument("--server", default=DEFAULT_URL)
    p.add_argument("--token", default=None)
    p.add_argument("--out", default="translated.wav")
    p.add_argument("--chunk-ms", type=int, default=40, help="send size, like a call's audio packets")
    p.add_argument("--speed", type=float, default=1.0,
                   help="1 = real time; higher streams faster than real time")
    args = p.parse_args(argv)

    pcm, rate = read_wav(args.wav)
    out_rate = 24000
    received: list = []
    connected = threading.Event()
    t0 = time.monotonic()

    def on_event(ev):
        kind = ev["type"]
        stamp = f"{time.monotonic() - t0:6.2f}s"
        if kind == "connected":
            print(f"{stamp}  connected: {ev.get('engine')} -> {ev.get('model_tgt_lang')}")
            connected.set()
        elif kind == "text" and not ev.get("final"):
            print(f"{stamp}  ... {ev.get('target_text')}")
        elif kind == "utterance_end":
            heard = f" {ev['source_text']} ->" if ev.get("source_text") else ""
            print(f"{stamp}  #{ev['utterance']} [{ev['reason']}]{heard} {ev.get('target_text')}\n"
                  f"{'':9}speech {ev['speech_ms']} ms, output {ev['output_ms']} ms, "
                  f"first audio +{ev.get('start_offset_ms')} ms, "
                  f"done {ev.get('end_offset_ms')} ms after speech ended")
        elif kind in ("error", "dropped"):
            print(f"{stamp}  {kind}: {ev}", file=sys.stderr)

    client = TranslationClient(args.server, args.tgt, src_lang=args.src, input_rate=rate,
                               output_rate=out_rate, label="stream_file",
                               on_audio=received.append, on_event=on_event, reconnect=False)
    if not connected.wait(30):
        client.close()
        raise SystemExit(f"could not start a stream at {args.server}: {client.last_error}")

    step = max(1, rate * args.chunk_ms // 1000)
    start = time.monotonic()
    for i in range(0, pcm.size, step):
        client.send_pcm(pcm[i:i + step])
        due = start + (i + step) / rate / args.speed
        time.sleep(max(0.0, due - time.monotonic()))
    print(f"{time.monotonic() - t0:6.2f}s  sent {pcm.size / rate:.1f} s of audio; finishing...")
    if not client.stop(timeout=120):
        print("server did not confirm the stop", file=sys.stderr)

    out = np.concatenate(received) if received else np.zeros(0, np.float32)
    write_wav(args.out, out, out_rate)
    print(f"wrote {args.out}: {out.size / out_rate:.1f} s at {out_rate} Hz")


if __name__ == "__main__":
    main()
