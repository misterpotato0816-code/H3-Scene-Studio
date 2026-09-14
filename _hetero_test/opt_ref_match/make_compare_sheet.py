from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"E:\AI-Projects\H3\_hetero_test\opt_ref_match")
OUTPUT = Path(r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\output\H3")
BASE_DIR = OUTPUT / "20260814_FINAL"
CAND_DIR = OUTPUT / "20260814_OPT"


def latest(folder: Path, prefix: str) -> Path:
    files = sorted(folder.glob(prefix + "*.mp4"), key=lambda p: p.stat().st_mtime_ns)
    if not files:
        raise FileNotFoundError(f"No {prefix}*.mp4 found in {folder}")
    return files[-1]


def decode(path: Path) -> list[np.ndarray]:
    frames = []
    with av.open(str(path)) as c:
        for frame in c.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise RuntimeError(f"No video frames decoded: {path}")
    return frames


def main() -> None:
    baseline = latest(BASE_DIR, "gpu1_live_e2e")
    candidate = latest(CAND_DIR, "ref_match")
    a = decode(baseline)
    b = decode(candidate)
    n = min(len(a), len(b))
    if n < 2:
        raise RuntimeError("Videos are too short for A/B")
    picks = sorted(set(round(i * (n - 1) / 5) for i in range(6)))
    thumb_w, header = 288, 30
    rows = []
    metrics = []
    for idx in picks:
        x, y = a[idx], b[idx]
        if x.shape != y.shape:
            raise RuntimeError(f"Frame shape mismatch at {idx}: {x.shape} vs {y.shape}")
        d = x.astype(np.float64) - y.astype(np.float64)
        metrics.append({
            "frame": idx,
            "pixel_mae_0_255": float(np.abs(d).mean()),
            "pixel_rmse_0_255": float(np.sqrt(np.mean(d*d))),
        })
        ia, ib = Image.fromarray(x), Image.fromarray(y)
        size = (thumb_w, max(1, round(ia.height * thumb_w / ia.width)))
        ia = ia.resize(size, Image.Resampling.LANCZOS)
        ib = ib.resize(size, Image.Resampling.LANCZOS)
        row = Image.new("RGB", (thumb_w * 2, size[1] + header), "white")
        row.paste(ia, (0, header))
        row.paste(ib, (thumb_w, header))
        draw = ImageDraw.Draw(row)
        draw.text((6, 8), f"CURRENT max  frame {idx}", fill="black")
        draw.text((thumb_w + 6, 8), f"CANDIDATE match  frame {idx}", fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (thumb_w * 2, sum(r.height for r in rows)), "white")
    yy = 0
    for row in rows:
        sheet.paste(row, (0, yy))
        yy += row.height
    sheet_path = ROOT / "current_max_left_candidate_match_right.jpg"
    sheet.save(sheet_path, quality=92)
    report = {
        "current_video": str(baseline),
        "candidate_video": str(candidate),
        "current_frames": len(a),
        "candidate_frames": len(b),
        "selected_frames": picks,
        "metrics": metrics,
        "quality_gate": "Human face/identity, motion, lips/audio and temporal stability decide adoption; pixel metrics do not.",
    }
    (ROOT / "compare_frame_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "contact_sheet": str(sheet_path),
        "current_video": str(baseline),
        "candidate_video": str(candidate),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
