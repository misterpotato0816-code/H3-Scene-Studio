# -*- coding: utf-8 -*-
"""Download the Phase 1 model set (plan 3) and verify sizes byte-for-byte.

Filenames/repos/sizes were verified against the HuggingFace API before this
script was written - see docs/05_models_verification.md. Nothing here is a
guess, and no similar-looking file is substituted.
"""
import os, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from huggingface_hub import hf_hub_download

# Target model store: pass it as the first argument or set H3_MODELS_DIR
# (the ComfyUI models folder that comfy_paths.yaml points at).
MODELS = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("H3_MODELS_DIR", "")).strip()
if not MODELS:
    sys.exit("usage: download_models.py <models_dir>  (or set H3_MODELS_DIR)")

# (repo_id, filename_in_repo, local_dir, expected_bytes)
JOBS = [
    ("Comfy-Org/MiniMax-H3",
     "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
     MODELS, 20_970_379_616),

    ("Comfy-Org/MiniMax-H3",
     "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
     MODELS, 27_141_342_152),

    ("larryvrh/MiniMax-H3-Turbo-Lora",
     "minimax_h3_turbo_v4_step600_ema.safetensors",
     os.path.join(MODELS, "loras"), 779_849_816),

    ("drbaph/MiniMax-H3-Turbo-Lora-ComfyUI",
     "minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors",
     os.path.join(MODELS, "loras"), 620_285_592),
]


def human(n):
    return "%.2f GiB" % (n / 1024 ** 3)


def main():
    results = []
    for repo, fname, local_dir, expect in JOBS:
        target = os.path.join(local_dir, fname.replace("/", os.sep))
        print("=" * 78)
        print("REPO : %s" % repo)
        print("FILE : %s" % fname)
        print("DEST : %s" % target)
        print("SIZE : %s (%d bytes)" % (human(expect), expect))

        if os.path.exists(target) and os.path.getsize(target) == expect:
            print("STATUS: already present with exact size - skipping")
            results.append((fname, "skipped", os.path.getsize(target)))
            continue

        t0 = time.time()
        path = hf_hub_download(repo_id=repo, filename=fname,
                               local_dir=local_dir, resume_download=True)
        dt = time.time() - t0
        got = os.path.getsize(path)
        ok = (got == expect)
        print("STATUS: %s  got=%d bytes  in %.1fs  (%.1f MiB/s)"
              % ("OK" if ok else "SIZE MISMATCH", got, dt,
                 (got / 1024 ** 2) / dt if dt > 0 else 0))
        if not ok:
            print("  !! expected %d, got %d" % (expect, got))
        results.append((fname, "ok" if ok else "MISMATCH", got))

    print("=" * 78)
    print("SUMMARY")
    bad = 0
    for fname, status, size in results:
        print("  %-10s %-14s %s" % (status, human(size), fname))
        if status == "MISMATCH":
            bad += 1
    print("done. mismatches=%d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
