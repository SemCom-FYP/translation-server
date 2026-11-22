# coding: utf-8
"""
Two-user demo: Alice and Bob talk, each hears the other in their own language.

    python client_test.py alice_en.wav bob_es.wav
    python client_test.py alice_en.wav bob_es.wav --alice-lang en --bob-lang es \
        --server ws://localhost:8770/v1/stream

alice_en.wav is what Alice says, bob_es.wav is what Bob says (16-bit PCM WAV).
Live microphone instead of a WAV for Alice (speak, hear the translation; use
headphones so the speaker output does not feed back into the mic):

    pip install sounddevice
    python client_test.py --mic --bob-lang es          # you speak, Bob's side hears Spanish
    python client_test.py --list-devices
    python client_test.py --mic --device 1 --bob-lang fr

Both are streamed at the same time, in real time, as a call would. Two streams
are opened on the server, one per listener:

    Alice's voice --> stream(tgt=bob_lang)   --> heard_by_bob.wav
    Bob's voice   --> stream(tgt=alice_lang) --> heard_by_alice.wav

In a real call each listener's own machine opens ITS stream and feeds it the
remote person's decoded voice; here both run in one process so you can test
without two machines. Add --play to also hear the results live (needs
`pip install sounddevice`).
"""

import argparse
import sys
import threading
import time

import numpy as np

from src.client_adapter import TranslationClient
from src.stream_file import read_wav, write_wav

DEFAULT_SERVER = "ws://192.248.10.47:8770/v1/stream"   # no "http://" inside a ws:// URL
OUT_RATE = 24000


class Direction:
    """One speaker's voice -> one listener, translated into the listener's language."""

    def __init__(self, speaker: str, listener: str, wav, listener_lang: str,
                 server: str, token, play: bool):
        """wav=None means the speaker's voice comes live from the microphone."""
        self.speaker, self.listener = speaker, listener
        self.pcm, self.rate = read_wav(wav) if wav else (None, OUT_RATE)
        self.received: list = []
        self.connected = threading.Event()
        self.t0 = time.monotonic()
        self.out_path = f"heard_by_{listener.lower()}.wav"
        self.player = None
        if play:
            import sounddevice as sd
            self.player = sd.OutputStream(samplerate=OUT_RATE, channels=1, dtype="float32")
            self.player.start()
        self.client = TranslationClient(
            server, listener_lang, token=token, input_rate=self.rate, output_rate=OUT_RATE,
            label=f"{speaker}->{listener}", on_audio=self._on_audio, on_event=self._on_event,
            reconnect=False)

    def _log(self, msg: str, err: bool = False):
        print(f"{time.monotonic() - self.t0:6.2f}s  [{self.speaker} -> {self.listener}] {msg}",
              file=sys.stderr if err else sys.stdout, flush=True)

    def _on_audio(self, pcm):
        self.received.append(pcm)
        if self.player is not None:
            self.player.write(np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1, 1))

    def _on_event(self, ev):
        kind = ev["type"]
        if kind == "connected":
            self._log(f"connected, speaking {ev.get('model_tgt_lang')}")
            self.connected.set()
        elif kind == "text" and not ev.get("final"):
            self._log(f"... {ev.get('target_text')}")
        elif kind == "utterance_end":
            self._log(f"#{ev['utterance']} {ev.get('target_text')!r} "
                      f"(first audio +{ev.get('start_offset_ms')} ms)")
        elif kind in ("error", "dropped"):
            self._log(f"{kind}: {ev}", err=True)

    def wait_connected(self, timeout: float = 30.0):
        if not self.connected.wait(timeout):
            self.client.close()
            raise SystemExit(f"could not open a stream for {self.listener}: "
                             f"{self.client.last_error}")

    def run(self, chunk_ms: int = 40):
        """Send the speaker's voice in real time, then wait for the translation to finish."""
        step = max(1, self.rate * chunk_ms // 1000)
        start = time.monotonic()
        for i in range(0, self.pcm.size, step):
            self.client.send_pcm(self.pcm[i:i + step])
            time.sleep(max(0.0, start + (i + step) / self.rate - time.monotonic()))
        self._log(f"{self.speaker} finished talking ({self.pcm.size / self.rate:.1f} s)")
        if not self.client.stop(timeout=120):
            self._log("server did not confirm the stop", err=True)
        out = np.concatenate(self.received) if self.received else np.zeros(0, np.float32)
        write_wav(self.out_path, out, OUT_RATE)
        self._log(f"wrote {self.out_path} ({out.size / OUT_RATE:.1f} s)")
        if self.player is not None:
            self.player.stop()
            self.player.close()


def run_mic(d: Direction, device=None, block_ms: int = 40):
    """Feed the microphone to the stream until Ctrl+C, then let the translation finish."""
    import sounddevice as sd

    def on_mic(indata, frames, t, status):
        if status:
            d._log(f"mic: {status}", err=True)
        d.client.send_pcm(indata[:, 0].copy())          # thread-safe, never blocks

    d._log("microphone live: speak now (Ctrl+C to stop)")
    try:
        with sd.InputStream(samplerate=d.rate, channels=1, dtype="float32", device=device,
                            blocksize=d.rate * block_ms // 1000, callback=on_mic):
            while True:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    d._log("stopping, finishing what was heard...")
    if not d.client.stop(timeout=120):
        d._log("server did not confirm the stop", err=True)
    out = np.concatenate(d.received) if d.received else np.zeros(0, np.float32)
    write_wav(d.out_path, out, OUT_RATE)
    d._log(f"wrote {d.out_path} ({out.size / OUT_RATE:.1f} s)")
    if d.player is not None:
        d.player.stop()
        d.player.close()


def main():
    p = argparse.ArgumentParser(description="Two users hearing each other translated")
    p.add_argument("alice_wav", nargs="?", help="what Alice says (omit with --mic)")
    p.add_argument("bob_wav", nargs="?", help="what Bob says (optional with --mic)")
    p.add_argument("--mic", action="store_true", help="Alice's voice comes live from the microphone")
    p.add_argument("--device", default=None, help="microphone device id or name (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--alice-lang", default="en", help="language Alice speaks AND hears (default en)")
    p.add_argument("--bob-lang", default="es", help="language Bob speaks AND hears (default es)")
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--token", default=None)
    p.add_argument("--play", action="store_true", help="play translated audio live (sounddevice)")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.mic:
        # Alice = the live microphone; Bob hears her translated, on this machine's speaker.
        to_bob = Direction("Alice", "Bob", None, args.bob_lang, args.server, args.token, True)
        to_bob.wait_connected()
        device = int(args.device) if args.device and args.device.isdigit() else args.device
        extra = []
        if args.bob_wav:        # optionally Bob also "talks" from a WAV, Alice hears him
            to_alice = Direction("Bob", "Alice", args.bob_wav, args.alice_lang,
                                 args.server, args.token, True)
            to_alice.wait_connected()
            extra = [threading.Thread(target=to_alice.run, daemon=True)]
            for t in extra:
                t.start()
        run_mic(to_bob, device)
        for t in extra:
            t.join()
        return
    if not (args.alice_wav and args.bob_wav):
        p.error("give alice_wav and bob_wav, or use --mic")

    # Bob hears Alice in Bob's language; Alice hears Bob in Alice's language.
    to_bob = Direction("Alice", "Bob", args.alice_wav, args.bob_lang,
                       args.server, args.token, args.play)
    to_alice = Direction("Bob", "Alice", args.bob_wav, args.alice_lang,
                         args.server, args.token, args.play)
    to_bob.wait_connected()
    to_alice.wait_connected()

    threads = [threading.Thread(target=d.run, name=f"{d.speaker}->{d.listener}")
               for d in (to_bob, to_alice)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("done: heard_by_bob.wav, heard_by_alice.wav")


if __name__ == "__main__":
    main()
