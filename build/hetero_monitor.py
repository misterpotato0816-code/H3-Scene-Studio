# -*- coding: utf-8 -*-
"""Per-GPU + system monitor for the HeteroVRAM smoke test.

Records utilization / dedicated VRAM / temperature per GPU, plus system RAM,
committed (pagefile-backed) bytes, and flags the abort conditions.

Usage: python hetero_monitor.py <out.json> [interval_s]
Stops when <out.json>.stop appears.
"""
import subprocess, sys, time, json, os, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

out_path = sys.argv[1]
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
stop_flag = out_path + ".stop"

Q = ["nvidia-smi",
     "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
     "--format=csv,noheader,nounits"]

try:
    import psutil
except ImportError:
    psutil = None

peak = {}          # idx -> dict of peaks
util_hist = {}     # idx -> list
ram_peak = 0
commit_peak = 0
samples = 0
gpu1_active_samples = 0
t0 = time.time()

while not os.path.exists(stop_flag):
    try:
        raw = subprocess.check_output(Q, text=True, timeout=10)
        for line in raw.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            idx = int(parts[0]); used = int(parts[1]); total = int(parts[2])
            util = int(parts[3]); temp = int(parts[4])
            try:
                power = float(parts[5])
            except Exception:
                power = 0.0
            d = peak.setdefault(idx, {"mem": 0, "util": 0, "temp": 0, "power": 0.0, "total": total})
            d["mem"] = max(d["mem"], used)
            d["util"] = max(d["util"], util)
            d["temp"] = max(d["temp"], temp)
            d["power"] = max(d["power"], power)
            util_hist.setdefault(idx, []).append(util)
            if idx == 1 and util > 0:
                gpu1_active_samples += 1
        samples += 1
    except Exception:
        pass
    if psutil:
        vm = psutil.virtual_memory()
        ram_peak = max(ram_peak, vm.total - vm.available)
        try:
            sm = psutil.swap_memory()
            commit_peak = max(commit_peak, sm.used)
        except Exception:
            pass
    time.sleep(interval)

res = {
    "duration_s": round(time.time() - t0, 1),
    "samples": samples,
    "gpu": {str(k): {"peak_mib": v["mem"], "total_mib": v["total"],
                     "peak_util_pct": v["util"], "peak_temp_c": v["temp"],
                     "peak_power_w": round(v["power"], 1),
                     "avg_util_pct": round(sum(util_hist.get(k, [0])) / max(1, len(util_hist.get(k, [1]))), 1)}
            for k, v in sorted(peak.items())},
    "ram_peak_gib": round(ram_peak / 1024 ** 3, 2) if ram_peak else None,
    "pagefile_peak_gib": round(commit_peak / 1024 ** 3, 2) if commit_peak else None,
    # "GPU1 actually computed" evidence: samples where GPU1 utilization > 0
    "gpu1_active_samples": gpu1_active_samples,
    "gpu1_active_pct_of_run": round(100.0 * gpu1_active_samples / max(1, samples), 1),
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2, ensure_ascii=False))
