# coding: utf-8
"""
Checks the translation server end to end without SeamlessStreaming or a GPU: the
server runs with the mock engine (which echoes each utterance back one step
late) and the real client adapter talks to it over a real socket.

    python -m translation_server.selftest

Needs numpy, aiohttp and soxr. Stages that need more (EnCodec, the
conference package) are skipped with a note when it is missing.
"""

import asyncio
import sys
import threading
import time
import traceback

import numpy as np

from . import wire
from .audio import MODEL_RATE, EnergyVAD, Endpointer, StreamResampler

RESULTS: list = []


def stage(fn):
    def run():
        t0 = time.perf_counter()
        try:
            note = fn()
            RESULTS.append((fn.__name__, "SKIP" if note and note.startswith("skip") else "ok", note or ""))
        except Exception as exc:
            traceback.print_exc()
            RESULTS.append((fn.__name__, "FAIL", f"{type(exc).__name__}: {exc}"))
        print(f"  {RESULTS[-1][1]:4}  {fn.__name__}  ({time.perf_counter() - t0:.1f} s) "
              f"{RESULTS[-1][2]}")
    return run


def synthetic_speech(rate: int, bursts=(1.6, 2.2, 1.2), gap_s: float = 1.0, seed: int = 0):
    """Voiced-looking bursts (harmonics under a syllable-rate envelope)
    separated by near-silence. Returns pcm and the burst lengths in samples."""
    rng = np.random.default_rng(seed)
    parts, lengths = [np.zeros(int(0.5 * rate), np.float32)], []
    for dur in bursts:
        t = np.arange(int(dur * rate)) / rate
        f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
        phase = 2 * np.pi * np.cumsum(f0) / rate
        voice = sum(np.sin(k * phase) / k for k in range(1, 8))
        env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2
        parts.append((0.15 * voice * env).astype(np.float32))
        lengths.append(t.size)
        parts.append((rng.standard_normal(int(gap_s * rate)) * 1e-4).astype(np.float32))
    return np.concatenate(parts), lengths


def _conference_protocol():
    """webrtc_conference.protocol, if the conference app can be found. The server
    does not depend on it; this only checks the packet format stays compatible."""
    import importlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.environ.get("CONFERENCE_DIR") or os.path.join(here, "..", "..", "LivePortrait")
    if os.path.isdir(os.path.join(root, "webrtc_conference")):
        sys.path.insert(0, os.path.abspath(root))
    try:
        return importlib.import_module("webrtc_conference.protocol")
    finally:
        if sys.path and sys.path[0] == os.path.abspath(root):
            sys.path.pop(0)


# ── stages ──────────────────────────────────────────────────────────────────

@stage
def wire_format():
    pcm = np.linspace(-1, 1, 1001, dtype=np.float32)
    for fmt, tol in ((wire.PCM_F32LE, 0), (wire.PCM_S16LE, 1 / 16000)):
        back = wire.bytes_to_pcm(wire.pcm_to_bytes(pcm, fmt), fmt)
        assert np.max(np.abs(back - pcm)) <= tol, fmt
    codes = np.random.default_rng(1).integers(0, 1024, size=(8, 18))
    pkt = wire.unpack_audio(wire.pack_audio(7, 123456, codes))
    assert pkt.seq == 7 and pkt.t_ms == 123456 and np.array_equal(pkt.codes, codes)
    try:
        app = _conference_protocol()
    except Exception:
        return ("skip: conference protocol not found (optional; set CONFERENCE_DIR to "
                "the LivePortrait folder to check byte-compatibility)")
    assert wire.pack_audio(7, 123456, codes) == app.pack_audio(7, 123456, codes)
    assert np.array_equal(wire.unpack_audio(app.pack_audio(3, 9, codes)).codes, codes)
    return "byte-identical to webrtc_conference.protocol"


@stage
def endpointing():
    pcm, lengths = synthetic_speech(MODEL_RATE)
    ep = Endpointer(vad=EnergyVAD())
    events, rng, i = [], np.random.default_rng(2), 0
    while i < pcm.size:                      # ragged chunks, as a network delivers them
        n = int(rng.integers(100, 3000))
        events += ep.push(pcm[i:i + n])
        i += n
    starts = sum(1 for k, _ in events if k == "start")
    ends = [v for k, v in events if k == "end"]
    assert starts == 3 and ends == ["silence"] * 3, (starts, ends)
    audio = sum(v.size for k, v in events if k == "audio")
    expected = sum(lengths) + 3 * ep.end_silence_ms * MODEL_RATE // 1000
    assert abs(audio - expected) < 3 * 0.35 * MODEL_RATE, (audio, expected)
    return f"3 utterances, {audio / MODEL_RATE:.2f} s kept of {pcm.size / MODEL_RATE:.2f} s"


@stage
def resampling():
    rs = StreamResampler(24000, MODEL_RATE)
    x = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
    y = np.concatenate([rs(x[i:i + 960]) for i in range(0, x.size, 960)] + [rs(x[:0], last=True)])
    assert abs(y.size - MODEL_RATE) <= 2, y.size
    ref = np.sin(2 * np.pi * 440 * np.arange(y.size) / MODEL_RATE)
    lag = int(np.argmax(np.correlate(y[2000:4000], ref[2000:2400], "valid")))
    err = np.sqrt(np.mean((y[4000:12000] - ref[4000 - lag:12000 - lag]) ** 2))
    assert err < 0.01, err


class _ServerThread:
    """The real server app on an ephemeral port, on its own loop/thread."""

    def __init__(self, token=None):
        from aiohttp import web
        from .engine import MockEngine
        from .server import TranslationServer
        from .session import SessionOptions
        self.engine = MockEngine(target_langs=("eng", "spa", "fra"), compute_ms=20)
        self.server = TranslationServer(self.engine, SessionOptions(vad="energy"), token=token)
        self.loop = asyncio.new_event_loop()
        self.port = None
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(self.loop)
            self.runner = web.AppRunner(self.server.build_app())
            self.loop.run_until_complete(self.runner.setup())
            site = web.TCPSite(self.runner, "127.0.0.1", 0)
            self.loop.run_until_complete(site.start())
            self.port = site._server.sockets[0].getsockname()[1]
            ready.set()
            self.loop.run_forever()
            self.loop.run_until_complete(self.runner.cleanup())

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        assert ready.wait(10), "server did not start"
        self.url = f"ws://127.0.0.1:{self.port}{wire.WS_PATH}"

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


def _stream(client, pcm, rate, speed=6.0, chunk_ms=40):
    step = rate * chunk_ms // 1000
    t0 = time.monotonic()
    for i in range(0, pcm.size, step):
        client.send_pcm(pcm[i:i + step])
        time.sleep(max(0.0, t0 + (i + step) / rate / speed - time.monotonic()))


def _wait(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


@stage
def server_end_to_end():
    from .client_adapter import TranslationClient
    srv = _ServerThread()
    try:
        log: list = []          # audio and events in arrival order (same thread)
        client = TranslationClient(srv.url, "es", input_rate=24000, output_rate=24000,
                                   label="selftest",
                                   on_audio=lambda pcm: log.append(("audio", pcm)),
                                   on_event=lambda ev: log.append((ev["type"], ev)))
        assert _wait(lambda: client.connected), f"no connection: {client.last_error}"
        pcm, lengths = synthetic_speech(24000)
        _stream(client, pcm, 24000)
        assert client.stop(timeout=30), "server never confirmed stop"

        kinds = [k for k, _ in log]
        ends = [ev for k, ev in log if k == "utterance_end"]
        assert kinds[0] == "connected", kinds[:3]
        assert len(ends) == 3, kinds
        # Every audio frame sits inside its utterance's speech_start … utterance_end.
        open_utt = None
        for k, ev in log:
            if k == "speech_start":
                assert open_utt is None
                open_utt = ev["utterance"]
            elif k == "utterance_end":
                assert ev["utterance"] == open_utt
                open_utt = None
            elif k == "audio":
                assert open_utt is not None, "audio outside an utterance"
        out = sum(v.size for k, v in log if k == "audio")
        expected = sum(e["speech_ms"] for e in ends) * 24
        assert abs(out - expected) < 0.05 * expected, (out, expected)
        assert all(e["source_text"] and e["target_text"] for e in ends)
        assert srv.server.sessions == {} or _wait(lambda: not srv.server.sessions, 3)
        return (f"{len(ends)} utterances, {out / 24000:.2f} s back; end offsets "
                f"{[e['end_offset_ms'] for e in ends]} ms")
    finally:
        srv.close()


@stage
def remote_translation_dropin():
    """RemoteTranslation used exactly as client.py uses PeerTranslation."""
    from .client_adapter import RemoteTranslation
    srv = _ServerThread()
    try:
        heard: list = []
        # Built exactly like client.py builds PeerTranslation: the app's
        # two-letter target code, no knowledge of what the peer speaks.
        tr = RemoteTranslation(srv.url, "Alice", "es",
                               out=heard.append, backlog_ms=lambda: 120.0)
        assert _wait(lambda: tr.connected), tr.error
        assert tr._client.ready["model_tgt_lang"] == "spa", tr._client.ready
        pcm, _ = synthetic_speech(24000, bursts=(1.5,))
        for i in range(0, pcm.size, 5760):            # 240 ms, the app's chunk size
            tr.feed(pcm[i:i + 5760])
            time.sleep(0.04)
        assert _wait(lambda: tr.last_text != ""), "no translation finished"
        assert tr.lag_ms >= 120.0 and heard
        assert tr.last_text.startswith("[spa]"), tr.last_text
        got = sum(p.size for p in heard)

        tr.set_target(None)                           # T pressed: back to original voice
        for i in range(0, pcm.size, 5760):
            tr.feed(pcm[i:i + 5760])
        time.sleep(0.8)
        assert sum(p.size for p in heard) == got, "audio arrived while paused"

        tr.set_target("kn")                           # Kannada: understood, not spoken
        assert _wait(lambda: tr.error and "cannot speak" in tr.error), tr.error

        tr.set_target("fr")                           # T pressed again: now French
        before = tr.last_text
        for i in range(0, pcm.size, 5760):
            tr.feed(pcm[i:i + 5760])
            time.sleep(0.02)
        assert _wait(lambda: tr.last_text != before), "nothing after switching language"
        assert tr.last_text.startswith("[fra]"), tr.last_text
        tr.close()
        return f"lag {tr.lag_ms:.0f} ms; es -> pause -> bad language -> fr all handled"
    finally:
        srv.close()


@stage
def refusals():
    from .client_adapter import TranslationClient
    srv = _ServerThread(token="sesame")
    try:
        bad = TranslationClient(srv.url, "es", reconnect=False)
        time.sleep(0.7)
        assert not bad.connected
        bad.close()
        good = TranslationClient(srv.url, "es", token="sesame", reconnect=False)
        assert _wait(lambda: good.connected), good.last_error
        good.close()
        nope = TranslationClient(srv.url, "xx", token="sesame")
        assert _wait(lambda: nope.last_error is not None)
        assert nope.last_error.get("code") == "unsupported_language", nope.last_error
        assert not nope.reconnect
        nope.close()
        return "token enforced; unknown language refused without retry loop"
    finally:
        srv.close()


@stage
def language_codes():
    from .engine import MockEngine
    e = MockEngine()
    try:
        cases = {"es": "spa", "ES": "spa", "spa": "spa", " fr ": "fra", "en": "eng",
                 "zh": "cmn", "it": "ita", "de": "deu", "kn": None, "kan": None,
                 "xx": None, "": None, None: None}
        for given, want in cases.items():
            assert e.resolve_lang(given) == want, (given, e.resolve_lang(given), want)
        assert len(e.target_langs) == 36
        return "36 output languages; app codes es/fr/de/it map to spa/fra/deu/ita"
    finally:
        e.close()


@stage
def seamless_engine_glue():
    """SeamlessStreamingEngine.step against a stand-in for the SimulEval
    pipeline: checks what it pushes and how it reads what comes back."""
    try:
        from simuleval.data.segments import EmptySegment, SpeechSegment, TextSegment
    except Exception:
        return "skip: needs `pip install --no-deps simuleval==1.1.4`"
    from .engine import _PAD_SAMPLES, SeamlessStreamingEngine, _to_argv

    class FakeSystem:
        """Speaks one 0.1 s burst per push; finishes `finish_after` pushes
        after the source does (as a long final translation would)."""

        def __init__(self, finish_after=1):
            self.pushed, self.finish_after, self.final_pushes = [], finish_after, 0

        def build_states(self):
            return {"tokens": []}

        def pushpop(self, seg, states):
            self.pushed.append(seg)
            if seg.finished:
                self.final_pushes += 1
            done = seg.finished and self.final_pushes >= self.finish_after
            states["tokens"].append(f"w{len(states['tokens'])}")
            return [TextSegment(content=" ".join(states["tokens"]), finished=done),
                    SpeechSegment(content=[0.1] * 1600, sample_rate=MODEL_RATE, finished=done)
                    if len(self.pushed) % 2 else EmptySegment(finished=done)]

    eng = object.__new__(SeamlessStreamingEngine)      # no model loading
    eng.system, eng.target_langs = FakeSystem(finish_after=3), ["spa"]
    ctx = eng.new_context("spa")
    src = np.random.default_rng(3).standard_normal(MODEL_RATE).astype(np.float32) * 0.1
    r1 = eng.step(ctx, src[:5120], False)
    r2 = eng.step(ctx, src[:5120], False)                 # nothing new: no push
    r3 = eng.step(ctx, src[:10240], False)
    r4 = eng.step(ctx, src[:10300], True)                 # 60 new samples: padded
    p = eng.system.pushed
    assert [len(s.content) for s in p] == [5120, 5120, _PAD_SAMPLES, _PAD_SAMPLES, _PAD_SAMPLES], \
        [len(s.content) for s in p]
    assert all(isinstance(s.content, list) and s.tgt_lang == "spa" for s in p)
    assert [s.finished for s in p] == [False, False, True, True, True]
    assert np.allclose(p[0].content, src[:5120]) and np.allclose(p[2].content[:60], src[10240:10300])
    assert r2.audio.size == 0 and not r2.done
    assert r1.audio.size == 1600 and r3.audio.size == 0 and r4.done
    assert r4.target_text == "w0 w1 w2 w3 w4", r4.target_text
    assert _to_argv({"a_b": 1, "flag": True, "off": False, "none": None}) == ["--a-b", "1", "--flag"]
    return "incremental pushes, end padding, drain loop and text all as designed"


@stage
def encodec_input():
    try:
        import encodec  # noqa: F401
        import torch
    except Exception:
        return "skip: needs `pip install encodec torch`"
    from encodec import EncodecModel
    from .client_adapter import TranslationClient
    model = EncodecModel.encodec_model_24khz().eval()
    model.set_target_bandwidth(6.0)
    pcm, _ = synthetic_speech(24000, bursts=(1.5,))
    srv = _ServerThread()
    try:
        ends: list = []
        client = TranslationClient(srv.url, "es", input_format=wire.ENCODEC_PACKET,
                                   on_event=lambda ev: ev["type"] == "utterance_end" and ends.append(ev))
        assert _wait(lambda: client.connected), client.last_error
        chunk = 5760
        for seq, i in enumerate(range(0, pcm.size - chunk, chunk)):
            with torch.no_grad():
                codes = model.encode(torch.from_numpy(pcm[i:i + chunk])[None, None])[0][0][0]
            client.send_packet(wire.pack_audio(seq, seq * 240, codes.numpy()))
            time.sleep(0.03)
        assert client.stop(30)
        assert len(ends) == 1, ends
        return f"speech {ends[0]['speech_ms']} ms recovered from EnCodec packets"
    finally:
        srv.close()


def main():
    print("translation_server selftest (mock engine)")
    for fn in (wire_format, endpointing, resampling, language_codes, seamless_engine_glue,
               server_end_to_end, remote_translation_dropin, refusals, encodec_input):
        fn()
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed" if not failed
          else f"\n{len(failed)} FAILED: {', '.join(r[0] for r in failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
