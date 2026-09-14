# -*- coding: utf-8 -*-
"""Sample GPU/CPU memory during a ComfyUI run and report peaks.

Usage: python vram_sampler.py <out.json> [interval_s]
Stops when the file <out.json>.stop appears.
"""
import subprocess, sys, time, json, os, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

out_path = sys.argv[1]
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
stop_flag = out_path + ".stop"

Q = ["nvidia-smi",
     "--query-gpu=index,memory.used,memory.total,utilization.gpu",
     "--format=csv,noheader,nounits"]

peak = {}       # idx -> peak MiB
util_sum = {}
samples = 0
ram_peak = 0

try:
    import psutil
    have_psutil = True
except ImportError:
    have_psutil = False

t0 = time.time()
while not os.path.exists(stop_flag):
    try:
        raw = subprocess.check_output(Q, text=True, timeout=10)
        for line in raw.strip().splitlines():
            idx, used, total, util = [x.strip() for x in line.split(",")]
            idx, used, total, util = int(idx), int(used), int(total), int(util)
            if used > peak.get(idx, 0):
                peak[idx] = used
            util_sum[idx] = util_sum.get(idx, 0) + util
        samples += 1
    except Exception:
        pass
    if have_psutil:
        vm = psutil.virtual_memory()
        ram_peak = max(ram_peak, vm.total - vm.available)
    time.sleep(interval)

res = {
    "duration_s": round(time.time() - t0, 1),
    "samples": samples,
    "gpu_peak_mib": peak,
    "gpu_avg_util_pct": {k: round(v / samples, 1) for k, v in util_sum.items()} if samples else {},
    "ram_peak_gib": round(ram_peak / 1024 ** 3, 2) if ram_peak else None,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
