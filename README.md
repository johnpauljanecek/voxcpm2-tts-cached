# `voxcpm2-tts-cached` — Serverless TTS (VoxCPM2) with RunPod cached models

This is a **drop-in API replacement** for the existing `voxcpm2-tts` endpoint
on this RunPod account (id `d6wx79l48172qr`, image
`nopa90/voxcpm2-runpod:weights`, weights baked into a ~7 GB image). Same
request shape, same response shape, all four modes preserved — but the
weights live **outside the image** on RunPod's network volume, delivered by
the endpoint's Model field.

| | |
|---|---|
| **GitHub repo** | https://github.com/johnpauljanecek/voxcpm2-tts-cached (public) |
| **Replaces** | endpoint `d6wx79l48172qr` (image `nopa90/voxcpm2-runpod:weights`, ~7 GB, weights baked) |
| **Worker image** | ~2 GB compressed (CUDA base + voxcpm, **zero weights**) |
| **Model** | `openbmb/VoxCPM2` only (public, apache-2.0, ~4.9 GB) |
| **Deploy path** | RunPod console → *Import Git Repository* + set Model field |

## Files

```
voxcpm2-tts-cached/
├── README.md            ← this file
├── handler.py           ← the entire worker (~360 lines)
├── Dockerfile           ← runpod/pytorch:2.4.0 + voxcpm, no weights stage
├── requirements.txt     ← pinned (see "Dependency pinning")
├── .dockerignore
└── .git/                ← own repo, origin → johnpauljanecek/voxcpm2-tts-cached
```

## Why this exists

The old endpoint baked VoxCPM2's 4.6 GB `model.safetensors` + 360 MB
`audiovae.pth` into the Docker image via a two-stage build
(`Dockerfile.weights` + a GitHub Actions job that downloads and verifies the
weights on every push). That made the image ~7 GB, made every code change
re-download 4.9 GB at build time, and required Docker Hub credentials.

This worker:

- Carries **zero weights** in the image.
- Resolves the cached `openbmb/VoxCPM2` snapshot at first job time from
  `/runpod-volume/huggingface-cache/hub/`.
- Verifies the two weight files' sizes and `model.safetensors`'s SHA-256
  once per cold start (cheap insurance against a truncated cache).
- Fails fast with a clear error if the cache is missing — **never** falls
  back to a silent download.
- Keeps every mode (`tts`, `voice_design`, `clone`, `ultimate_clone`) and
  every output field identical to the old endpoint.

## API contract — identical to the old endpoint

**Input** (`job["input"]`):

```json
{
  "mode": "tts",                     // optional, default "tts"
  "text": "Hello world",             // required
  "reference_wav_b64": "<base64 wav>",   // required for clone / optional for ultimate_clone
  "prompt_wav_b64": "<base64 wav>",      // required for ultimate_clone
  "prompt_text": "transcript of prompt", // required for ultimate_clone
  "cfg_value": 2.0,                  // optional
  "inference_timesteps": 10,         // optional
  "seed": 42,                        // optional
  "format": "wav"                    // optional: "wav" (default) or "mp3"
}
```

**Modes:**

| mode | text format | extra inputs |
|---|---|---|
| `tts` | plain text | — |
| `voice_design` | `"(voice description)Text to speak."` — **parens REQUIRED**, else the model silently does plain TTS (no error) | — |
| `clone` | plain text | `reference_wav_b64` |
| `ultimate_clone` | plain text | `prompt_wav_b64` + `prompt_text` (+ optional `reference_wav_b64`) |

**Output** (always a completed job):

```json
{
  "wav_b64": "<base64 wav>",
  "sample_rate": 48000,
  "duration_seconds": 3.5,
  "file_size_bytes": 84000
}
```

Errors never escape the handler — they come back as a completed job with
`{"error": "<message>", "type": "<ExceptionName>"}`.

## Deploy

New endpoint via RunPod console (GitHub-import path — same as
`examples/runpod-github-test` and `examples/whisper-large-v3`):

1. console.runpod.io/serverless → **New Endpoint** → **Import Git Repository**
2. Repo `https://github.com/johnpauljanecek/voxcpm2-tts-cached`, branch `main`
   (root `Dockerfile`, no path override)
3. **GPU** — pick `AMPERE_24`, then **CRITICAL** in the "Enabled GPU types"
   section **uncheck `PRO 6000 MIG 24GB` (NVIDIA RTX PRO 6000 Blackwell
   Server Edition MIG 1g.24gb)** and keep **only `L4` checked**.
   - This cu12.4 image + PyTorch supports up to `sm_90`; Blackwell is
     `sm_120` and crashes the worker at boot (`no kernel image is available`,
     exit code 1). Full write-up: `reference/deploy-gotchas.md` §18.
4. **Model** field: set to `openbmb/VoxCPM2`.
   - ⚠️ The console placeholder reads *"Paste in a link from Hugging Face or
     type your model name."* when unset — a **blank** Model field is a silent
     failure: the worker starts fine but every job errors with
     `snapshots directory not found`. It must be set at endpoint creation
     (see `reference/deploy-gotchas.md` §17).
5. **Deploy Endpoint**. If the pre-deploy scanner complains ("could not find
   handler"), **force through** and verify with a run — the scanner is a
   known false alarm (§1).
6. Wait for `HEALTHY`, then run the smoke suite below.

First job pays the cold start: RunPod stages the ~4.9 GB repo into the
volume once, then the worker boots, imports torch + voxcpm, and
`VoxCPM.from_pretrained()` does its `torch.compile` warm-up. Expect the
first request to take ~1–3 min; warm workers are near-realtime
(~10–13 s of speech in ~10 s wall, L4, 10 timesteps).

## Smoke suite (parity vs old endpoint)

From the old project's `TEST-RESULTS.md` (2026-09-02). Six Russian voices
verified against the old endpoint; run the same payloads here and compare
duration/audibility.

**tts** (plain):

```json
{ "input": { "mode": "tts", "text": "Флот, говорит командир, сегодня ночью мы держим рубеж." } }
```

**voice_design** — six distinct voices, all Russian speech, English
descriptions (voice description prefix REQUIRED):

| Label | Description | Speech dur (old) |
|---|---|---|
| commander | Deep authoritative male commander, aged 50, low pitch, gravelly texture, calm commanding, slow deliberate | 12.5 s |
| young-woman | Bright young female, aged 20, clear, energetic, light cheerful, fast lively | 9.8 s |
| elder-narrator | Elderly male narrator, aged 70, warm weathered, slow deliberate, gentle wise | 12.5 s |
| soldier | Young male soldier, aged 25, breathless intense, mid pitch, urgent stressed, short clipped | 11.8 s |
| anchor | Calm professional female news anchor, aged 35, smooth polished, neutral standard Russian, measured | 10.9 s |
| villain | Harsh menacing male, low pitch, slow, cold cruel, whispery edge | 13.4 s |

Example payload (commander):

```json
{ "input": { "mode": "voice_design",
             "text": "(Deep authoritative male commander, aged 50, low pitch, gravelly texture, calm commanding, slow deliberate)Флот, говорит командир, сегодня ночью мы держим рубеж." } }
```

**clone / ultimate_clone** — need a reference WAV. Local artifacts from the
old project (not in git): `tests/audio/voxcpm-seg-01-commander.wav` etc.

## Dependency pinning

- **Hard-pinned** (behavior-relevant pure-Python deps): `voxcpm==2.0.3`,
  `transformers==4.51.3`, `runpod==1.12.0`.
  - `transformers==4.51.3` is **sacred**: voxcpm declares `>=4.36.2` with no
    upper bound, and newer transformers break `LlamaTokenizerFast`
    (`TypeError: Input must be a List[Union[str, AddedToken]]`).
  - `voxcpm==2.0.3` pinned so a future release can't change
    `from_pretrained` local-dir behavior.
- **Floors only** for the torch family (`torch>=2.5.0`,
  `torchaudio>=2.5.0`, `torchcodec`): the base image ships torch 2.4.0 and
  pip upgrades to the latest cu12.4-compatible wheel at build time. Do NOT
  hard-pin `torch==X.Y.Z+cuNNN` — the `+cu` suffix is index-time and
  brittle. The base image's CUDA 12.4 constrains the wheel family.

## Handler invariants (carried from the old project)

1. **Never load the model at import time** — an import crash kills the
   worker before the runtime reports the error and jobs hang IN_QUEUE
   forever (worker crash-loop). Load lazily inside `handler()`.
2. **Entire handler body inside one try/except** returning
   `{"error":…, "type":…}` — a thrown exception can mark jobs FAILED and
   churn the queue; a returned dict always completes cleanly.
3. **Offline env flags set before any HF/transformers import** — even
   `runpod` pulls in huggingface_hub. `HF_HUB_OFFLINE=1` +
   `TRANSFORMERS_OFFLINE=1` mean a missing cache is a clear per-job error,
   never a silent network fallback.
4. `voxcpm.VoxCPM.from_pretrained(<dir>)` on a **local dir** is fully
   offline (`os.path.isdir()` branch in `core.py`); it only calls
   `snapshot_download()` when given a hub id. Pointing it at the resolved
   volume snapshot is sufficient — no hub involvement at all.

## Gotchas carried forward

- Voice-design text **must** have the `(description)` prefix — without
  parens the model silently does plain TTS (no error).
- Endpoint `gpuIds`/GPU types are **GPU pool selectors**, not model names;
  the pool *label* is a pricing tier, not an architecture (§18).
- First request after deploy may need `executionTimeout` headroom — cold
  start (torch import + voxcpm + `torch.compile` warm-up) is the slow part,
  not the 4.9 GB volume read (RunPod stages it before the worker starts).
- Wave stitching (multi-segment): read each PCM via stdlib `wave`, concat
  with zero-pad silences, write mono/16-bit/48 kHz. `wave` errors like
  `# channels not specified` vanish when params are set explicitly.
