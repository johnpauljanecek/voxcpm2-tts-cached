"""
voxcpm2-tts-cached — VoxCPM2 TTS served via RunPod's cached-model feature.

Drop-in API replacement for the old `voxcpm2-tts` endpoint
(id `d6wx79l48172qr`, image `nopa90/voxcpm2-runpod:weights`). Same input
field set, same output shape, every mode preserved — but the weights live
**outside the image** on RunPod's network volume, attached automatically
when the endpoint's Model field is set to `openbmb/VoxCPM2`.

The handler enforces offline mode (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)
so a missing cache produces a clear error per-job instead of a silent
fallback download. `voxcpm`'s own `VoxCPM.from_pretrained()` already treats
a local directory as fully offline (it only calls `snapshot_download()` when
given a hub id), so pointing it at the resolved snapshot dir is enough.

Modes (identical to the old endpoint):
  tts            — basic text-to-speech
  voice_design   — natural-language voice description, text prefixed as
                   "(voice description)Text to speak." (parens REQUIRED,
                   else the model silently does plain TTS — no error)
  clone          — clone voice from reference audio (base64 WAV)
  ultimate_clone — highest-fidelity clone (prompt audio + transcript)
"""

import os

# Set offline flags BEFORE any HF/transformers import (including `runpod`,
# which pulls in huggingface_hub). Doc:
# https://docs.runpod.io/tutorials/serverless/model-caching-text
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import base64
import time
import tempfile

import numpy as np
import runpod
import soundfile as sf


# ---------- configuration ------------------------------------------------

# RunPod mounts the cached model here, same layout as the standard HF hub
# cache (`models--{org}--{name}/snapshots/{hash}/`). The snapshot dir holds
# the flat files VoxCPM needs (config.json, model.safetensors,
# audiovae.pth, tokenizer files) — identical to the layout the old image
# baked at /app/models/VoxCPM2, but WITHOUT tokenizer_config.json missing
# (the cached snapshot has the full repo).
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"
MODEL_ID = "openbmb/VoxCPM2"

# Expected sizes (bytes) of the two big weight files, used as a cheap
# first-load integrity check. From builder/download-weights.sh in the old
# project (verified at image build time).
EXPECT_MODEL_SHA = "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d"
EXPECT_MODEL_SIZE = 4580080592
EXPECT_VAE_SIZE = 376951122


# ---------- snapshot path resolver (from the cached-models tutorial) --------

def resolve_snapshot_path(model_id: str) -> str:
    """
    Navigate `/runpod-volume/huggingface-cache/hub/models--{org}--{name}/`
    to find the snapshot directory holding the actual model files.
    Prefer `refs/main` so we get exactly the commit HF considers "main";
    fall back to the only/oldest snapshot if `refs/main` is missing.
    """
    if "/" not in model_id:
        raise ValueError(f"MODEL_ID '{model_id}' is not in 'org/name' format")
    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    # Preferred: use the commit hash from refs/main (matches what
    # from_pretrained() would resolve to with network access).
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            snap = f.read().strip()
        candidate = os.path.join(snapshots_dir, snap)
        if os.path.isdir(candidate):
            print(f"[voxcpm2] snapshot from refs/main: {candidate}", flush=True)
            return candidate

    # Fallback: list snapshots and take the first alphabetically.
    if not os.path.isdir(snapshots_dir):
        raise RuntimeError(
            f"[cached-models] snapshots directory not found: {snapshots_dir} "
            f"— is the Model field set on this endpoint?"
        )
    versions = sorted(
        d for d in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, d))
    )
    if not versions:
        raise RuntimeError(
            f"[cached-models] no snapshot subdirectories under {snapshots_dir}"
        )
    chosen = os.path.join(snapshots_dir, versions[0])
    print(f"[voxcpm2] snapshot (fallback): {chosen}", flush=True)
    return chosen


def verify_weights(snapshot_dir: str) -> None:
    """
    Cheap first-load integrity check: sizes of model.safetensors and
    audiovae.pth must match what the old project verified at build time.
    Raises if missing/short — the job fails fast with a clear error instead
    of the model silently loading garbage or a truncated file.
    """
    import hashlib
    import os

    model_path = os.path.join(snapshot_dir, "model.safetensors")
    vae_path = os.path.join(snapshot_dir, "audiovae.pth")

    size = os.path.getsize(model_path)
    if size != EXPECT_MODEL_SIZE:
        raise RuntimeError(
            f"[voxcpm2] model.safetensors size {size} != expected "
            f"{EXPECT_MODEL_SIZE} — cached snapshot incomplete?"
        )
    vae_size = os.path.getsize(vae_path)
    if vae_size != EXPECT_VAE_SIZE:
        raise RuntimeError(
            f"[voxcpm2] audiovae.pth size {vae_size} != expected "
            f"{EXPECT_VAE_SIZE} — cached snapshot incomplete?"
        )

    # SHA-256 over 4.58 GB takes a few seconds on first load; do it once
    # per worker cold start. Skip if already verified in this process.
    if not getattr(verify_weights, "_verified", False):
        print("[voxcpm2] verifying model.safetensors SHA-256 ...", flush=True)
        t0 = time.time()
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
        elapsed = time.time() - t0
        if digest != EXPECT_MODEL_SHA:
            raise RuntimeError(
                f"[voxcpm2] model.safetensors SHA-256 {digest} != expected "
                f"{EXPECT_MODEL_SHA} — corrupted cache?"
            )
        verify_weights._verified = True
        print(f"[voxcpm2] model.safetensors verified OK in {elapsed:.0f}s", flush=True)


# ---------- lazy model load (invariant #1) --------------------------------

# Lazy-loaded on first request. NEVER load at import time: an import failure
# kills the worker before the serverless runtime reports the error, and the
# job hangs IN_QUEUE forever while the worker crash-loops.
model = None
SAMPLE_RATE = 48000  # default until the model reports its real rate on load


def load_model():
    """Load VoxCPM2 from the cached-model volume (no download, ever)."""
    global model, SAMPLE_RATE
    if model is not None:
        return model

    snapshot_dir = resolve_snapshot_path(MODEL_ID)
    verify_weights(snapshot_dir)  # sizes + one-time SHA of model.safetensors
    print(f"[voxcpm2] Loading model from {snapshot_dir} ...", flush=True)
    t0 = time.time()

    from voxcpm import VoxCPM

    # os.path.isdir() branch → local load, fully offline. Never downloads.
    model = VoxCPM.from_pretrained(
        snapshot_dir,
        load_denoiser=False,
    )
    SAMPLE_RATE = model.tts_model.sample_rate  # typically 48000

    elapsed = time.time() - t0
    print(f"[voxcpm2] Model loaded in {elapsed:.1f}s", flush=True)
    return model


# ---------- helpers -------------------------------------------------------


def decode_b64_wav(b64_str: str) -> str:
    """Decode a base64 WAV string into a temporary file and return its path."""
    raw = base64.b64decode(b64_str)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def encode_wav_b64(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Encode a numpy array to a base64 WAV string."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    sf.write(out_path, wav, sample_rate)

    with open(out_path, "rb") as f:
        audio_bytes = f.read()

    os.unlink(out_path)

    duration = len(wav) / sample_rate
    return {
        "wav_b64": base64.b64encode(audio_bytes).decode("utf-8"),
        "sample_rate": sample_rate,
        "duration_seconds": round(duration, 2),
        "file_size_bytes": len(audio_bytes),
    }


def cleanup_file(path: str):
    """Safely remove a temporary file."""
    try:
        os.unlink(path)
    except OSError:
        pass


# ---------- handler -------------------------------------------------------


def handler(job):
    """RunPod entry point. The ENTIRE body is wrapped so no exception can ever
    escape the handler: an uncaught exception marks the job FAILED and can
    trigger retry/queue churn. Returning a dict (even an error) always
    completes the job cleanly with a readable payload."""
    temp_files = []
    try:
        data = job.get("input") or {}

        text = data.get("text", "").strip()
        if not text:
            return {"error": "No 'text' provided in input."}

        mode = data.get("mode", "tts")

        # Lazy model load: cached on the global after the first success.
        # Inside the try so a load failure is REPORTED, never thrown.
        load_model()

        # Common generation params
        gen_kwargs = {}
        if "cfg_value" in data:
            gen_kwargs["cfg_value"] = float(data["cfg_value"])
        if "inference_timesteps" in data:
            gen_kwargs["inference_timesteps"] = int(data["inference_timesteps"])
        if "seed" in data:
            gen_kwargs["seed"] = int(data["seed"])

        # ---- Mode routing ----

        if mode == "voice_design":
            # Voice design: text must be formatted as
            # "(voice description)Text to speak." (README-confirmed)
            wav = model.generate(text=text, **gen_kwargs)

        elif mode == "clone":
            ref_b64 = data.get("reference_wav_b64")
            if not ref_b64:
                return {"error": "Clone mode requires 'reference_wav_b64'."}
            ref_path = decode_b64_wav(ref_b64)
            temp_files.append(ref_path)
            wav = model.generate(
                text=text,
                reference_wav_path=ref_path,
                **gen_kwargs,
            )

        elif mode == "ultimate_clone":
            prompt_b64 = data.get("prompt_wav_b64")
            prompt_text = data.get("prompt_text")
            if not prompt_b64:
                return {
                    "error": "Ultimate clone mode requires 'prompt_wav_b64'."
                }
            if not prompt_text:
                return {
                    "error": "Ultimate clone mode requires 'prompt_text'."
                }
            prompt_path = decode_b64_wav(prompt_b64)
            temp_files.append(prompt_path)

            # Optional: use a separate reference audio for better similarity
            ref_path = None
            ref_b64 = data.get("reference_wav_b64")
            if ref_b64:
                ref_path = decode_b64_wav(ref_b64)
                temp_files.append(ref_path)
            else:
                ref_path = prompt_path

            wav = model.generate(
                text=text,
                prompt_wav_path=prompt_path,
                prompt_text=prompt_text,
                reference_wav_path=ref_path,
                **gen_kwargs,
            )

        else:
            # Basic TTS
            wav = model.generate(text=text, **gen_kwargs)

        # Encode result
        return encode_wav_b64(wav, SAMPLE_RATE)

    except Exception as e:
        # Report, never throw. Worker stays healthy; job completes with error.
        return {"error": str(e), "type": type(e).__name__}

    finally:
        for f in temp_files:
            cleanup_file(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
