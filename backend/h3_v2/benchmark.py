# -*- coding: utf-8 -*-
"""Benchmark engine. Pure record + JSONL store. No speed judged by feel.

Every generation logs: preset/model/LoRA(+hash)/sampler/scheduler/steps/
shifts, resolution/frames/refs/seed, per-stage timings, peak VRAM per GPU,
output path, success/failure. A/B compares identical prompt/seed/refs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BenchmarkRecord:
    preset: str = ""
    model: str = ""
    lora: str | None = None
    lora_hash: str = ""
    lora_strength: float = 0.0
    lora_bindable_keys: int = 0
    sampler: str = ""
    scheduler: str = ""
    steps: int = 0
    shift_video: float | None = None
    shift_audio: float | None = None
    width: int = 0
    height: int = 0
    frames: int = 0
    ref_count: int = 0
    seed: int = 0
    t_conditioning: float = 0.0
    t_model_load: float = 0.0
    t_sampling: float = 0.0
    t_vae_decode: float = 0.0
    t_audio_decode: float = 0.0
    t_merge: float = 0.0
    t_other: float = 0.0       # executing time on nodes with no stage mapping
    t_relay: float = 0.0       # tail-chain prep (prev/load/tail/trim nodes)
    t_orchestration: float = 0.0  # harness wall outside ComfyUI submits
    t_total: float = 0.0
    gpu0_peak_vram_mb: float = 0.0
    gpu1_peak_vram_mb: float = 0.0
    ram_peak_mb: float = 0.0
    output_path: str = ""
    success: bool = False
    error: str = ""
    backend: str = "h3_v2"
    comfyui_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


def append_jsonl(store_path: str | Path, record: BenchmarkRecord) -> Path:
    p = Path(store_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return p


def load_jsonl(store_path: str | Path) -> list[BenchmarkRecord]:
    p = Path(store_path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(BenchmarkRecord.from_dict(json.loads(line)))
    return out


def ab_table(records: list[BenchmarkRecord]) -> list[dict]:
    """One row per preset: time + VRAM + output path. No auto quality verdict;
    the human judges quality from the saved videos side by side."""
    rows = []
    for r in records:
        rows.append({
            "preset": r.preset,
            "total_sec": round(r.t_total, 1),
            "sampling_sec": round(r.t_sampling, 1),
            "t_model_load": round(r.t_model_load, 1),
            "t_conditioning": round(r.t_conditioning, 1),
            "t_vae_decode": round(r.t_vae_decode, 1),
            "t_audio_decode": round(r.t_audio_decode, 1),
            "t_merge": round(r.t_merge, 1),
            "t_other": round(r.t_other, 1),
            "t_relay": round(r.t_relay, 1),
            "t_orchestration": round(r.t_orchestration, 1),
            "gpu0_peak_mb": round(r.gpu0_peak_vram_mb),
            "gpu1_peak_mb": round(r.gpu1_peak_vram_mb),
            "success": r.success,
            "output": r.output_path,
            "note": r.error if not r.success else "",
        })
    return rows
