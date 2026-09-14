# -*- coding: utf-8 -*-
"""GPU detection and allocation planning (pure logic + nvidia-smi wrapper).

Never imports torch in the app process: torch is loaded only inside the
ComfyUI child process. Everything here is testable with injected/fake data.

Terminology used throughout: "video" GPU runs the DiT sampler + video/audio
VAE (always cuda:0 by construction, see H3CudaPhaseRestore target_gpu=0);
"aux" is either a second GPU (dual, text encoder via SelectCLIPDevice gpu:1)
or the CPU (single GPU, text encoder on main memory). 12GB + 12GB is never
treated as 24GB anywhere in this module.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

MiB = 1024 * 1024
GiB = 1024 * MiB

DEFAULT_GPU_SETTINGS = {
    "mode": "auto",
    "video_gpu": "",
    "aux_device": "",
    "reserve_vram_gb": 1.0,
}

_DETECT_CACHE: dict[str, Any] = {"data": None, "at": 0.0}
_DETECT_TTL_SECONDS = 5.0


# --------------------------------------------------------------- detection ---
def _nvidia_smi_path() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = Path(system_root) / "System32" / "nvidia-smi.exe"
    if candidate.is_file():
        return str(candidate)
    return None


def _run_nvidia_smi(fields: list[str]) -> tuple[int, str, str]:
    exe = _nvidia_smi_path()
    if not exe:
        return 127, "", "nvidia-smi not found"
    args = [exe, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "nvidia-smi timeout"
    except Exception as exc:                                       # noqa: BLE001
        return 1, "", str(exc)


_FULL_FIELDS = ["index", "uuid", "name", "pci.bus_id", "memory.total",
                "memory.free", "memory.used", "compute_cap"]
_NO_CC_FIELDS = ["index", "uuid", "name", "pci.bus_id", "memory.total",
                 "memory.free", "memory.used"]


def _parse_int(s: str) -> int:
    s = (s or "").strip()
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


def _parse_rows(stdout: str, has_cc: bool) -> list[dict]:
    gpus: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        needed = 8 if has_cc else 7
        if len(parts) < needed:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        row = {
            "index": index,
            "uuid": parts[1],
            "name": parts[2],
            "pci_bus_id": parts[3],
            "total_mib": _parse_int(parts[4]),
            "free_mib": _parse_int(parts[5]),
            "used_mib": _parse_int(parts[6]),
            "compute_cap": parts[7] if has_cc and len(parts) > 7 else "",
        }
        if not row["uuid"]:
            continue
        gpus.append(row)
    return gpus


def _ram_total_mib() -> int | None:
    if os.name != "nt":
        return None
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys // MiB)
    except Exception:                                                # noqa: BLE001
        return None
    return None


def detect(refresh: bool = False, *, runner=None, now: float | None = None) -> dict:
    """Detect installed NVIDIA GPUs via nvidia-smi. Cached for 5 seconds.

    `runner`/`now` are injection points for tests (runner replaces the
    subprocess call, now replaces time.monotonic()).
    """
    clock = now if now is not None else time.monotonic()
    if not refresh and _DETECT_CACHE["data"] is not None:
        if clock - _DETECT_CACHE["at"] < _DETECT_TTL_SECONDS:
            return copy_dict(_DETECT_CACHE["data"])

    if runner is not None:
        rc, out, err = runner(_FULL_FIELDS)
        has_cc = True
        if rc != 0:
            rc2, out2, err2 = runner(_NO_CC_FIELDS)
            if rc2 == 0:
                rc, out, err, has_cc = rc2, out2, err2, False
    else:
        rc, out, err = _run_nvidia_smi(_FULL_FIELDS)
        has_cc = True
        if rc != 0:
            rc2, out2, err2 = _run_nvidia_smi(_NO_CC_FIELDS)
            if rc2 == 0:
                rc, out, err, has_cc = rc2, out2, err2, False

    detected_at = time.time()
    if rc != 0:
        result = {"ok": False, "error": err.strip() or "nvidia-smiの実行に失敗しました。",
                  "source": "nvidia-smi", "gpus": [],
                  "ram_total_mib": _ram_total_mib(), "detected_at": detected_at}
    else:
        gpus = _parse_rows(out, has_cc)
        result = {"ok": True, "error": "", "source": "nvidia-smi", "gpus": gpus,
                  "ram_total_mib": _ram_total_mib(), "detected_at": detected_at}

    _DETECT_CACHE["data"] = copy_dict(result)
    _DETECT_CACHE["at"] = clock
    return result


def copy_dict(d: dict) -> dict:
    import copy as _copy
    return _copy.deepcopy(d)


# ---------------------------------------------------------------- settings ---
def normalize_settings(raw: dict | None, *, legacy_launch_args: list | None = None) -> dict:
    """Normalize the `gpu` config block. Migrates --reserve-vram when `raw` is
    absent/empty (first run after this feature is added)."""
    out = dict(DEFAULT_GPU_SETTINGS)
    if not raw:
        reserve = _reserve_from_legacy_args(legacy_launch_args or [])
        if reserve is not None:
            out["reserve_vram_gb"] = reserve
        return out
    mode = str(raw.get("mode") or "auto").strip().lower()
    if mode not in ("auto", "single", "dual"):
        mode = "auto"
    out["mode"] = mode
    out["video_gpu"] = str(raw.get("video_gpu") or "").strip()
    out["aux_device"] = str(raw.get("aux_device") or "").strip()
    try:
        reserve = float(raw.get("reserve_vram_gb", DEFAULT_GPU_SETTINGS["reserve_vram_gb"]))
    except (TypeError, ValueError):
        reserve = DEFAULT_GPU_SETTINGS["reserve_vram_gb"]
    out["reserve_vram_gb"] = reserve
    return out


def _reserve_from_legacy_args(args: list) -> float | None:
    args = [str(a) for a in (args or [])]
    for i, a in enumerate(args):
        if a == "--reserve-vram" and i + 1 < len(args):
            try:
                return float(args[i + 1])
            except ValueError:
                return None
        if a.startswith("--reserve-vram="):
            try:
                return float(a.split("=", 1)[1])
            except ValueError:
                return None
    return None


def strip_device_flags(args: list) -> list[str]:
    """Remove --cuda-device / --default-device / --reserve-vram (both
    `--flag value` and `--flag=value` forms) from a launch-args list."""
    flags = {"--cuda-device", "--default-device", "--reserve-vram"}
    out: list[str] = []
    args = [str(a) for a in (args or [])]
    i = 0
    while i < len(args):
        a = args[i]
        eq = a.split("=", 1)
        if eq[0] in flags and len(eq) == 2:
            i += 1
            continue
        if a in flags:
            i += 2
            continue
        out.append(a)
        i += 1
    return out


# ------------------------------------------------------------------ ranking --
def _rank_key(g: dict) -> tuple:
    try:
        cc = float(g.get("compute_cap") or 0)
    except (TypeError, ValueError):
        cc = 0.0
    total_gib_rounded = round((g.get("total_mib") or 0) / 1024)
    return (-total_gib_rounded, -cc, str(g.get("pci_bus_id") or ""))


def _ranked(gpus: list[dict]) -> list[dict]:
    return sorted(gpus, key=_rank_key)


def _by_uuid(gpus: list[dict], uuid: str) -> dict | None:
    for g in gpus:
        if g.get("uuid") == uuid:
            return g
    return None


ROLE_VIDEO = "動画生成（DiT・VAE）"
ROLE_AUX = "テキスト・参照の条件付け（text encoder）"
ROLE_GEMMA = "ComfyUI内ローカルGemma（llama.cpp: H3用ComfyUIに見えるGPUへ自動分割、個別指定なし）"


def _fmt_reserve(r: float) -> str:
    s = f"{r:.1f}"
    return s


def build_plan(settings: dict, detected: dict, *, text_encoder_bytes: int | None = None) -> dict:
    """Compute the GPU allocation plan. Never raises."""
    errors: list[str] = []
    warnings: list[str] = []
    mode = str((settings or {}).get("mode") or "auto").strip().lower()
    if mode not in ("auto", "single", "dual"):
        mode = "auto"
    try:
        reserve = float((settings or {}).get("reserve_vram_gb", 1.0))
    except (TypeError, ValueError):
        reserve = 1.0

    plan = {
        "ok": False, "errors": errors, "warnings": warnings, "mode": mode,
        "effective_mode": None, "video": None, "aux": None,
        "reserve_vram_gb": reserve, "launch_args": [], "clip_device": "",
        "measured": False, "visible_uuids": [], "roles": [],
    }

    if not detected.get("ok"):
        errors.append("NVIDIAのGPUを検出できません。")
        return plan
    gpus = detected.get("gpus") or []
    if not gpus:
        errors.append("NVIDIAのGPUを検出できません。")
        return plan

    if not (0.0 <= reserve <= 4.0):
        errors.append("予約VRAM（GB）は0〜4の範囲で指定してください。")
        return plan

    ranked = _ranked(gpus)
    video_uuid = str((settings or {}).get("video_gpu") or "").strip()
    aux_raw = str((settings or {}).get("aux_device") or "").strip()

    effective_mode = mode
    video_gpu: dict | None = None
    aux: dict | None = None

    if mode == "auto":
        if len(ranked) >= 2:
            effective_mode = "dual"
            video_gpu = ranked[0]
            aux = ranked[1]
        else:
            effective_mode = "single"
            video_gpu = ranked[0]
            aux = None  # cpu

    elif mode == "dual":
        effective_mode = "dual"
        if len(ranked) < 2:
            errors.append(f"デュアルGPUには2枚以上のGPUが必要です（検出: {len(ranked)}枚）。")
            return plan
        if video_uuid:
            video_gpu = _by_uuid(ranked, video_uuid)
            if video_gpu is None:
                errors.append(f"保存済みのGPU（{video_uuid}）が見つかりません。GPUを選び直してください。")
                return plan
        else:
            video_gpu = ranked[0]
        if aux_raw == "cpu":
            errors.append(
                "デュアルGPUでは別処理側にGPUを指定してください。"
                "CPUで処理する場合はシングルGPUを選んでください。")
            return plan
        if aux_raw:
            aux = _by_uuid(ranked, aux_raw)
            if aux is None:
                errors.append(f"保存済みのGPU（{aux_raw}）が見つかりません。GPUを選び直してください。")
                return plan
        else:
            aux = next((g for g in ranked if g["uuid"] != video_gpu["uuid"]), None)
        if aux is not None and video_gpu is not None and aux["uuid"] == video_gpu["uuid"]:
            errors.append(
                "動画生成側と別処理側に同じGPUは指定できません。"
                "12GB+12GBは1枚の24GBとしては使えません。")
            return plan

    elif mode == "single":
        effective_mode = "single"
        if video_uuid:
            video_gpu = _by_uuid(ranked, video_uuid)
            if video_gpu is None:
                errors.append(f"保存済みのGPU（{video_uuid}）が見つかりません。GPUを選び直してください。")
                return plan
        else:
            video_gpu = ranked[0]
        if aux_raw not in ("", "cpu"):
            errors.append(
                "シングルGPUでは別処理側にGPUを指定できません"
                "（text encoderはCPU＝メインメモリで処理します）。")
            return plan
        aux = None
        ram_mib = detected.get("ram_total_mib")
        if text_encoder_bytes and ram_mib:
            need_mib = (text_encoder_bytes / MiB) + 8192
            if ram_mib < need_mib:
                have_gb = ram_mib / 1024
                need_gb = need_mib / 1024
                errors.append(
                    "メインメモリ（RAM）が不足しています。"
                    f"text encoder（約{text_encoder_bytes / GiB:.1f}GB）をCPUで処理するには"
                    f"約{need_gb:.1f}GB必要ですが、検出されたRAMは約{have_gb:.1f}GBです。")
                return plan

    if video_gpu is None:
        errors.append("NVIDIAのGPUを検出できません。")
        return plan

    # Reserve-vram must leave >= 2 GiB of each used GPU's total.
    used_gpus = [video_gpu] + ([aux] if aux is not None else [])
    for g in used_gpus:
        total_gib = (g.get("total_mib") or 0) / 1024
        if total_gib - reserve < 2.0:
            errors.append(
                f"{g.get('name')}の予約VRAM（{_fmt_reserve(reserve)}GB）が大きすぎます。"
                "各GPUにつき最低2GBの空きを残してください。")
            return plan

    # ---- warnings -----------------------------------------------------
    if effective_mode == "dual" and video_gpu is not None and aux is not None:
        measured = ("4070 Ti" in (video_gpu.get("name") or "")
                    and "3060" in (aux.get("name") or "")
                    and abs(reserve - 1.0) < 1e-9)
        plan["measured"] = bool(measured)
        if not measured:
            warnings.append(
                "この割り当ては実測済みの構成（RTX 4070 Ti＝動画生成、RTX 3060＝テキスト条件付け、"
                "予約1.0GB）と異なります。速度とVRAM使用量は未計測です。")
    elif effective_mode == "single":
        warnings.append("text encoderをCPUで処理します（過去の参考値は約555秒/本、この環境では未再計測）。")

    for g in used_gpus:
        if (g.get("total_mib") or 0) < 11 * 1024:
            warnings.append(f"{g.get('name')}は実測環境（12GB級）より小さく、未検証の構成です。")
        total = g.get("total_mib") or 0
        free = g.get("free_mib") or 0
        if total > 0 and free < total * 0.5:
            warnings.append(f"{g.get('name')}の空きVRAMが少なくなっています（他のアプリが使用中の可能性があります）。")

    # ---- launch args / clip device -------------------------------------
    reserve_str = _fmt_reserve(reserve)
    if effective_mode == "dual":
        launch_args = ["--cuda-device", f"{video_gpu['uuid']},{aux['uuid']}",
                       "--reserve-vram", reserve_str]
        clip_device = "gpu:1"
        visible_uuids = [video_gpu["uuid"], aux["uuid"]]
        roles = [{"role": ROLE_VIDEO, "device": f"cuda:0 ({video_gpu['name']})"},
                 {"role": ROLE_AUX, "device": f"gpu:1 ({aux['name']})"},
                 {"role": ROLE_GEMMA, "device": "自動分割"}]
    else:
        launch_args = ["--cuda-device", video_gpu["uuid"], "--reserve-vram", reserve_str]
        clip_device = "cpu"
        visible_uuids = [video_gpu["uuid"]]
        roles = [{"role": ROLE_VIDEO, "device": f"cuda:0 ({video_gpu['name']})"},
                 {"role": ROLE_AUX, "device": "cpu"},
                 {"role": ROLE_GEMMA, "device": "自動分割"}]

    plan.update({
        "ok": True, "effective_mode": effective_mode,
        "video": video_gpu,
        "aux": ({"kind": "gpu", **aux} if aux is not None else {"kind": "cpu"}),
        "reserve_vram_gb": reserve,
        "launch_args": launch_args, "clip_device": clip_device,
        "visible_uuids": visible_uuids, "roles": roles,
    })
    return plan


# ---------------------------------------------------------------- applied ----
def _strip_torch_device_name(name: str) -> str:
    """'cuda:0 NVIDIA GeForce RTX 4070 Ti : native' -> 'NVIDIA GeForce RTX 4070 Ti'."""
    n = str(name or "")
    if ":" in n:
        # remove leading "cuda:N " prefix
        head, _, rest = n.partition(" ")
        if head.startswith("cuda:"):
            n = rest
    # drop trailing " : <alloc>"
    if " : " in n:
        n = n.rsplit(" : ", 1)[0]
    return n.strip()


def applied_from_stats(stats: dict, detected: dict) -> dict:
    """Parse ComfyUI /system_stats into what GPU mapping is actually applied."""
    system = (stats or {}).get("system") or {}
    argv = [str(a) for a in (system.get("argv") or [])]
    devices_raw = [d for d in ((stats or {}).get("devices") or []) if d.get("type") == "cuda"]
    devices = []
    for d in devices_raw:
        total = int(d.get("vram_total") or 0)
        free = int(d.get("vram_free") or 0)
        devices.append({
            "index": d.get("index"),
            "name": d.get("name", ""),
            "vram_total_mib": round(total / MiB),
            "vram_free_mib": round(free / MiB),
            "uuid": "",
        })

    launch_args: list[str] = []
    reserve_vram_gb: float | None = None
    for i, a in enumerate(argv):
        if a in ("--cuda-device", "--default-device", "--reserve-vram"):
            launch_args.append(a)
            if i + 1 < len(argv):
                launch_args.append(argv[i + 1])
        m = a.split("=", 1)
        if m[0] in ("--cuda-device", "--default-device", "--reserve-vram") and len(m) == 2:
            launch_args.append(a)
    for i, a in enumerate(argv):
        if a == "--reserve-vram" and i + 1 < len(argv):
            try:
                reserve_vram_gb = float(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith("--reserve-vram="):
            try:
                reserve_vram_gb = float(a.split("=", 1)[1])
            except ValueError:
                pass

    cuda_device_val = None
    for i, a in enumerate(argv):
        if a == "--cuda-device" and i + 1 < len(argv):
            cuda_device_val = argv[i + 1]
        elif a.startswith("--cuda-device="):
            cuda_device_val = a.split("=", 1)[1]

    known = False
    video_uuid = ""
    aux_uuid = ""
    visible_uuids: list[str] = []
    note = ""

    detected_gpus = (detected or {}).get("gpus") or []

    if cuda_device_val:
        parts = [p.strip() for p in cuda_device_val.split(",") if p.strip()]
        if parts and all(p.upper().startswith("GPU-") for p in parts):
            visible_uuids = parts
            known = True
            if len(parts) >= 1:
                video_uuid = parts[0]
            if len(parts) >= 2:
                aux_uuid = parts[1]
        else:
            known, note = _map_by_name(devices, detected_gpus)
            if known:
                visible_uuids = [d["uuid"] for d in devices]
                if visible_uuids:
                    video_uuid = visible_uuids[0]
                if len(visible_uuids) > 1:
                    aux_uuid = visible_uuids[1]
    else:
        # --default-device or nothing: numeric mapping via device names.
        known, note = _map_by_name(devices, detected_gpus)
        if known:
            visible_uuids = [d["uuid"] for d in devices]
            if visible_uuids:
                video_uuid = visible_uuids[0]
            if len(visible_uuids) > 1:
                aux_uuid = visible_uuids[1]

    if not known and not note:
        note = "起動引数から適用中のGPU割り当てを特定できませんでした。"

    return {
        "known": bool(known),
        "launch_args": launch_args,
        "video_uuid": video_uuid,
        "aux_uuid": aux_uuid,
        "visible_uuids": visible_uuids,
        "reserve_vram_gb": reserve_vram_gb,
        "devices": devices,
        "note": note,
    }


def _map_by_name(devices: list[dict], detected_gpus: list[dict]) -> tuple[bool, str]:
    """Map torch device names (in `devices`, mutated in place with uuid) to
    detected GPUs by exact stripped name. Ambiguous/unmatched -> False."""
    by_name: dict[str, list[dict]] = {}
    for g in detected_gpus:
        by_name.setdefault(g.get("name", ""), []).append(g)

    ok = True
    note = ""
    for d in devices:
        clean = _strip_torch_device_name(d.get("name", ""))
        candidates = by_name.get(clean) or []
        if len(candidates) == 1:
            d["uuid"] = candidates[0]["uuid"]
        elif len(candidates) == 0:
            ok = False
            note = f"GPU名「{clean}」を検出結果と対応付けできませんでした。"
        else:
            ok = False
            note = f"GPU名「{clean}」が複数一致するため一意に特定できません。"
    return ok, note


# ---------------------------------------------------------------- convenience --
def _loras_dir(cfg) -> Path:
    """Mirrors Pipeline._loras_dir: shared loras dir from comfy_paths.yaml,
    falling back to the install-local models/loras. Duplicated (not
    imported) to avoid a gpu.py <-> pipeline.py import cycle."""
    import re as _re
    try:
        text = Path(cfg.paths_yaml).read_text(encoding="utf-8")
        m = _re.search(r"base_path:\s*['\"]([^'\"]+)['\"]", text)
        if m:
            cand = Path(m.group(1)) / "loras"
            if cand.is_dir():
                return cand
    except Exception:                                              # noqa: BLE001
        pass
    return cfg.comfy_dir / "models" / "loras"


def text_encoder_bytes(cfg) -> int | None:
    try:
        loras_dir = _loras_dir(cfg)
        shared = Path(loras_dir).parent
        p = shared / "text_encoders" / cfg.models["clip"]
        if p.is_file():
            return p.stat().st_size
    except Exception:                                              # noqa: BLE001
        return None
    return None


def current_plan(cfg, *, refresh: bool = False) -> dict:
    """Convenience: normalize cfg's saved `gpu` settings, detect, build plan."""
    settings = normalize_settings(cfg.get("gpu"),
                                  legacy_launch_args=cfg.get("comfy_launch_args"))
    detected = detect(refresh=refresh)
    te_bytes = text_encoder_bytes(cfg)
    return build_plan(settings, detected, text_encoder_bytes=te_bytes)
