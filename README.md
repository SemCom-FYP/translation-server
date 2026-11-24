# Simultaneous speech translation server (SeamlessStreaming)

A standalone server that translates a voice **while it is still speaking**,
using Meta's [SeamlessStreaming](https://github.com/facebookresearch/seamless_communication).
It sits *beside* the neural conference, not inside it. Any client sends a voice
in over a WebSocket and gets back translated speech and the translation text as
it forms.

```
 conference client (listener)                    translation server (GPU box)
 ───────────────────────────                     ─────────────────────────────
 remote peer's decoded voice ──── WebSocket ───▶ endpointer (VAD) ─▶ SeamlessStreaming
   (PCM, or the app's own                          │  w2v-BERT encoder → monotonic text
    EnCodec 'A' packets)                           │  decoder → unit decoder → vocoder
 speaker  ◀────────── translated speech + text ◀───┘
```

Nothing in `webrtc_conference/` was changed. The client library
(`client_adapter.py`) offers `RemoteTranslation`, a drop-in for the app's
`PeerTranslation`, so hooking it up later is a few lines. See
[Integrating with the conference app](#integrating-with-the-conference-app).

## What it can do

| | |
|---|---|
| **Directions** | **Any to any, both ways.** It understands speech in ~100 languages and speaks 36, including English ↔ Spanish, French, German and Italian (all four of the app's languages) |
| Source language | detected implicitly: the server only needs to know the language to *speak* |
| Per stream | each WebSocket picks its own output language and can switch mid-call |
| Models | one shared set, ~8 GB to download, downloaded automatically on first start |
| Output voice | a fixed voice per language (vocoder_v2), not the speaker's own |
| **Licence** | the SeamlessStreaming **models are CC-BY-NC 4.0: non-commercial use only**. The seamless_communication code is MIT; this server is your own code |

Output languages (`/health` lists them too): `arb ben cat ces cmn cym dan deu
eng est fin fra hin ind ita jpn kor mlt nld pes pol por ron rus slk spa swe
swh tel tgl tha tur ukr urd uzn vie`. Two-letter codes work as well (`es` →
`spa`, `fr` → `fra`, `en` → `eng`, `zh` → `cmn`, ...), so the app's own codes
can be passed straight through.

## Setup (server machine)

SeamlessStreaming runs on **fairseq2**, which ships **no Windows build**. Use
Linux, or WSL2 on this Windows machine (WSL2 sees the NVIDIA GPU with the
standard Windows driver). It needs Python 3.10 and PyTorch 2.2.2, which is the
version the fairseq2n 0.2.1 wheel is built against. That conflicts with the
conference app's environment, which is why the server is a separate process.

```bash
# Ubuntu / WSL2, from this folder. Creates ./.venv, private to this server:
./setup_wsl.sh
./run_server.sh --prefetch      # download (~3.6 + 4.3 + 0.2 GB) and test-load the models once
```

(Manual equivalent: `python3.10 -m venv .venv`, install torch 2.2.2 + torchaudio,
`seamless_communication` from GitHub, then `pip install -r requirements.txt`.)

**Isolation.** This folder is a standalone project: its own virtualenv, its own
dependencies, no imports from LivePortrait or `webrtc_conference`. The conference
connects to it only over the WebSocket. To use the client library from the
conference app, copy `translation_server/wire.py` and `client_adapter.py`
(numpy + aiohttp only) or `pip install` this folder in the app's environment.

The models are cached by fairseq2 under `$FAIRSEQ2_CACHE_DIR`, or
`~/.cache/fairseq2` by default. They are public, so no Hugging Face login is
needed.

## Run

```bash
./run_server.sh
# → Translation server (seamless-streaming, 36 output languages) on ws://0.0.0.0:8770/v1/stream
```

Then, from any machine that can reach it:

```bash
curl http://<host>:8770/health
python -m translation_server.stream_file speech_en.wav --tgt es \
    --server ws://<host>:8770/v1/stream --out heard_es.wav
```

`stream_file` sends the WAV in real time, as a call would, prints the
translation as it forms, and saves the translated speech. It also prints each
utterance's delay figures (below).

**No GPU or Linux box yet?** `./run_mock.ps1` (Windows) or `python -m translation_server.server --mock` runs
the whole server with a fake engine that echoes each utterance back one step
late. It needs only `aiohttp numpy soxr`, and works on Windows too. That is
enough to develop and test the client integration.

```bash
python -m translation_server.selftest      # protocol, server + client adapter end to end (mock engine)
```

### Options worth knowing

| flag | default | effect |
|---|---|---|
| `--device` | `auto` | first GPU if any, else CPU; or `cuda:1`, `cpu`, ... |
| `--dtype` | `fp16` | fp16 halves GPU memory; CPU always runs fp32 (and is far slower than real time) |
| `--decision-threshold` | `0.5` | how sure the model must be before it speaks: higher means later but more accurate |
| `--min-start-ms` | `1920` | audio heard before the first word of an utterance is attempted |
| `--min-unit-chunk` | `50` | speech units gathered per burst of synthesis: smaller means sooner but choppier |
| `--segment-ms` | `320` | new audio that triggers a decoding step |
| `--vad` | `auto` | `silero` if installed, else `energy` (fine for headsets, weaker in noisy rooms) |
| `--end-silence-ms` | `600` | pause that closes an utterance |
| `--max-utterance-s` | `15` | hard cut for someone who never pauses (it cuts earlier at a short pause after ⅔ of this) |
| `--token` | `$TRANSLATION_TOKEN` | shared secret; clients send `?token=` or `Authorization: Bearer` |
| `--max-sessions` | `16` | concurrent streams |

The defaults are the values Meta's own streaming demo uses.

## How it works

**One stream per voice.** Each WebSocket carries one person's speech into one
output language. A listener in a 4-person call opens 3 streams. All streams
share one loaded model, and each keeps its own decoding state
(`system.build_states()` per utterance). SimulEval's pipeline is built to take
explicit per-stream state, so nothing is swapped in and out.

**Utterances.** The w2v-BERT encoder re-runs over every frame heard so far on
each step, so the server cuts the incoming voice at pauses, just as Meta's demo
does. Within an utterance, translation is *simultaneous*: about every 320 ms of
new audio, the monotonic decoder decides whether it has heard enough to say the
next words, and translated speech is streamed back as soon as it is
synthesised.

**Sharing the GPU.** All model calls run on one worker thread, because the
agents keep a little per-call scratch state on the shared model objects, so
calls must not overlap. Streams wait only for each other's ~320 ms steps, never
for each other's sentences. If the GPU falls behind, the next step simply sees
more audio at once. Utterances that have not started translating are dropped
(with a `dropped` event) rather than letting the delay grow without bound.

**What differs from Meta's `SeamlessStreamingS2STJointVADAgent`:**
- **No VAD agent.** The server's endpointer cuts utterances instead. The stock
  agent loads a Silero model through `torch.hub` for every stream.
- **Its own detokenizer.** The stock one either emits fragments that lose word
  boundaries ("Hola a tod" + "os"), or drops words when several arrive at
  once. This one reports the whole translation so far, and keeps it across the
  pipeline's early-stop resets.
- SimulEval's `general_parser()` reads `sys.argv`, which holds the *server's*
  flags. The engine builds the model's arguments itself and hides `sys.argv`
  while it parses.

**Delay.** Each utterance reports `start_offset_ms` (speech began → first
translated audio) and `end_offset_ms` (speech ended → last translated audio).
They depend on the GPU, the options above, and how many streams share it.
Measure on your own hardware with `stream_file`.

## Conference voice relay (`/v1/conf`)

The neural conference sends all its audio here as Opus. Participants join a room; for every
speaker/listener pair the server either forwards the Opus packet untouched or, when the
listener asked for another language than the speaker speaks, translates it and re-encodes
it as Opus. Packets keep the speaker's capture timestamp (translated ones are stamped with
when their source speech began), so the client can keep video in sync. Needs `pip install
PyOgg` and, on Linux/WSL, `sudo apt install libopus0`. Audio starts on the WebSocket and moves to an unreliable WebRTC datachannel (UDP, via aiortc;
`--stun` sets the STUN servers) when the client offers one; without aiortc or UDP it stays on the
WebSocket. Protocol: `src/conference.py`.

```
ws://<host>:8770/v1/conf?room=demo&peer=<id>&name=<name>&speaks=en[&hear=es][&token=...]
```

## Protocol

One WebSocket per voice at `ws://<host>:8770/v1/stream`. The full reference is
the docstring of [wire.py](wire.py). In short:

```jsonc
// client → server, first message
{"type": "start", "protocol": 1, "tgt_lang": "es",          // required; "es" or "spa"
 "src_lang": "en",                                          // optional hint
 "input":  {"format": "pcm_s16le", "sample_rate": 24000},   // or pcm_f32le, or encodec_packet
 "output": {"format": "pcm_s16le", "sample_rate": 24000},
 "label": "Alice"}
// then binary frames of audio; optionally
// {"type": "configure", "tgt_lang": "fr"} | "flush" | "reset" | "stop" | "ping"

// server → client
{"type": "ready", "session": "s1", "model_tgt_lang": "spa", "target_langs": [...], ...}
{"type": "speech_start", "utterance": 1}
{"type": "text", "utterance": 1, "source_text": "", "target_text": "hola a todos", "final": false}
<binary: translated speech>
{"type": "utterance_end", "utterance": 1, "reason": "silence", "target_text": "hola a todos",
 "speech_ms": 2240, "output_ms": 1980, "start_offset_ms": 2310, "end_offset_ms": 820}
```

`source_text` is empty: SeamlessStreaming translates speech directly and does
not transcribe it. Both sample rates are free, because the server resamples.
`24000` matches the conference app's `SAMPLE_RATE`, so translated audio can go
straight into its `SpeakerMixer`.

Input format `encodec_packet` takes the app's own `'A'` audio packets, byte for
byte as they come off the `audio` DataChannel (~6 kbps instead of ~384 kbps of
PCM), and decodes them server-side the same way the app does. The self-test
checks that `wire.pack_audio` / `unpack_audio` are byte-identical to
`webrtc_conference/protocol.py`.

Any language can be a client: browsers, Node and Go only need a WebSocket. The
Python client library needs only `numpy` and `aiohttp`:

```python
from translation_server.client_adapter import TranslationClient

client = TranslationClient("ws://gpu-box:8770/v1/stream", "es",   # hear it in Spanish
                           input_rate=24000, output_rate=24000,
                           on_audio=lambda pcm: speaker.play(pcm),
                           on_event=lambda ev: print(ev))
client.send_pcm(float32_chunk)   # thread-safe, never blocks
client.configure(tgt_lang="fr")  # switch language mid-call
client.stop()                    # finish what was heard, then disconnect
```

## Integrating with the conference app

Not applied: these are the changes to make when you want it.
`RemoteTranslation` takes the same arguments as `PeerTranslation` and has the
same `feed` / `set_target` / `close` / `lag_ms` / `last_text`. It also accepts
the app's language codes (`es`, `fr`, `de`, `it`) as they are. So the **T** key,
`--translate`, `--translate-langs`, the speaker mixing, the lag display and the
A/V-sync handling in `client.py` all stay as they are.

The client needs `translation_server` importable (see Isolation above). No new
packages are needed.

**1. A flag**: in `parse_args()`, translation group:

```python
g.add_argument("--translate-server", default=None, metavar="URL",
               help="Translate on a SeamlessStreaming server (ws://host:8770/v1/stream) "
                    "instead of local Whisper/Opus-MT/Piper")
g.add_argument("--translate-token", default=os.environ.get("TRANSLATION_TOKEN"))
```

To offer English (or any of the 36) as a target too, add it to `--translate`
and `--translate-langs` `choices`, and give it an entry in `LANGUAGES` in
`translator.py`, e.g. `Language("en", "English", "", "")`, so the notices can
name it.

**2. Build a `RemoteTranslation` instead of a `PeerTranslation`**: in
`_feed_translation()`, the only change is which class is constructed:

```python
from translation_server.client_adapter import RemoteTranslation   # top of file

if remote.translation is None:
    key = remote.peer_id + VOICE_SUFFIX
    out = lambda p, k=key: self.speaker.push(k, p, cap_s=VOICE_BUFFER_S)
    backlog = lambda k=key: self.speaker.backlog_ms(k)
    if self.args.translate_server:
        remote.translation = RemoteTranslation(
            self.args.translate_server, remote.name, self.translate_to,
            out=out, backlog_ms=backlog,
            token=self.args.translate_token, sample_rate=SAMPLE_RATE)
    else:
        remote.translation = PeerTranslation(
            self.translator, remote.name, self.translate_to, out=out, backlog_ms=backlog,
            speak_fn=self.pocket_voice.speak if self.pocket_voice is not None else None)
```

**3. Skip local model loading**: at the top of `_load_translation()`:

```python
if self.args.translate_server:
    self._translate_loading = None
    self._set_translation(code)      # nothing to load; the server has the models
    return True
```

and in `cycle_translation()`, treat the server as always ready:
`elif self.args.translate_server or (self.translator is not None and
self.translator.is_ready(code)):`.

That is the whole integration. Run with
`--translate-server ws://<gpu-box>:8770/v1/stream` and press **T**. Everyone
else is heard in Spanish, then French, whatever language they speak, and the
tile subtitle shows `→ es lag 2.1 s` as before.

*Optional extras:*
- **Skip same-language peers.** Add `"lang": <your language>` to the `hello` in
  `_greet()`, store it on `RemoteState` in `on_ctrl`, and play a peer's
  original voice when `remote.lang == self.translate_to`. Otherwise a Spanish
  speaker heard "in Spanish" is re-voiced by the model rather than passed
  through.
- **Save bandwidth to a remote server.** Build `RemoteTranslation(...,
  input_format="encodec_packet")` and, in `_audio_decode_loop()`, call
  `remote.translation.feed_codes(pkt.seq, pkt.t_ms, pkt.codes)` instead of
  `feed(pcm)`. The server then needs `pip install encodec`.

## Files

| file | |
|---|---|
| [server.py](server.py) | aiohttp app and CLI (`/health`, `/v1/stream`, `--prefetch`, `--mock`) |
| [session.py](session.py) | one stream: decode → resample → endpoint → step scheduling → output |
| [engine.py](engine.py) | `SeamlessStreamingEngine` (pipeline, per-stream states, language codes) and `MockEngine` |
| [audio.py](audio.py) | stream resampler, EnCodec packet decoder, VADs, `Endpointer` |
| [wire.py](wire.py) | protocol reference + PCM / app-packet codecs (numpy only) |
| [client_adapter.py](client_adapter.py) | `TranslationClient`, `RemoteTranslation` (numpy + aiohttp only) |
| [stream_file.py](stream_file.py) | stream a WAV in real time, save the translation |
| [selftest.py](selftest.py) | end-to-end checks with the mock engine, plus the engine glue against a stand-in pipeline |
