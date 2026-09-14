# -*- coding: utf-8 -*-
"""PRERUN hardware snapshot for H3 benchmarks (read-only, no GPU work).

Captures: GPU temp/clock/power/memory (nvidia-smi), system RAM free,
pagefile usage, and the app/bench code identity. Writes one JSON file
per invocation into benchmarks/. Benchmark-only; never used in production.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_DIR = PROJECT_ROOT / "benchmarks"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"UNAVAILABLE: {exc}"


def snapshot(tag: str) -> dict:
    gpu_csv = _run(["nvidia-smi",
                    "--query-gpu=index,name,temperature.gpu,clocks.sm,"
                    "clocks.mem,power.draw,memory.used,memory.total,"
                    "utilization.gpu",
                    "--format=csv,noheader,nounits"])
    snap = {
        "tag": tag,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nvidia_smi_gpu": gpu_csv,
        "commit_comfyui": _run(["git", "-C",
                                r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI",
                                "rev-parse", "--short", "HEAD"]),
        "comfyui_status": _run(["git", "-C",
                                r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI",
                                "status", "--porcelain"]),
    }
    try:
        import psutil  # noqa: PLC0415 - optional, fallback below
        vm = psutil.virtual_memory()
        snap["ram_free_mb"] = round(vm.available / 1048576, 1)
        snap["ram_total_mb"] = round(vm.total / 1048576, 1)
    except Exception:  # noqa: BLE001
        snap["ram_free_mb"] = snap["ram_total_mb"] = "UNAVAILABLE-no-psutil"
    out = BENCH_DIR / f"prerun_{tag}.json"
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    snap = snapshot(args.tag)
    print(json.dumps(snap, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
