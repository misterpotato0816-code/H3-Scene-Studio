# -*- coding: utf-8 -*-
"""H3_Media export helpers: user-facing copies of finals.

Internal working files (ComfyUI input/output, project/clip working copies)
stay exactly where they are. These helpers only ADD best-effort export
copies under the media root, so generation logic, pointers and parity are
untouched. Every helper never raises: export must not fail a generation.
"""
from __future__ import annotations

from pathlib import Path
import errno
import os
import re
import shutil
import unicodedata


def _unique_dest(dest_dir: Path, stem: str, suffix: str,
                 *, exists_ok) -> tuple[Path | None, str]:
    """Find dest_dir/<stem or stem_N><suffix> that satisfies exists_ok(path).

    exists_ok(path) is called only for paths that already exist; it should
    return True when that existing file may be reused as-is (identical
    content already there), else False to force trying the next suffix.
    Returns (path_to_use_or_None, "reuse"|"new"|"exhausted").
    """
    candidate = dest_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate, "new"
    if exists_ok(candidate):
        return candidate, "reuse"
    for n in range(2, 1000):
        candidate = dest_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate, "new"
        if exists_ok(candidate):
            return candidate, "reuse"
    return None, "exhausted"


def export_copy(src: str | Path | None, dest_dir: Path, name: str, *,
                media_root: str | Path | None = None,
                media_type: str = "", project_id: str = "") -> str:
    """Copy src into dest_dir/name (or dest_dir/name_2, ... on conflict).

    Returns the dest path or "" on failure. Never overwrites a different
    existing file; an identical-size existing file is reused unchanged.
    When media_root is given, a manifest row is recorded (never raises).
    """
    try:
        if not src:
            return ""
        src_path = Path(str(src))
        if not src_path.is_file():
            return ""
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(name).stem
        suffix = Path(name).suffix
        src_size = src_path.stat().st_size

        def _ok(p: Path) -> bool:
            try:
                if p.resolve() == src_path.resolve():
                    return True
                return p.stat().st_size == src_size
            except Exception:                                # noqa: BLE001
                return False

        dest, mode = _unique_dest(dest_dir, stem, suffix, exists_ok=_ok)
        if dest is None:
            return ""
        if mode == "new" and dest.resolve() != src_path.resolve():
            shutil.copy2(src_path, dest)
        if media_root is not None:
            record_export(media_root, source=src_path, exported=dest,
                          media_type=media_type, project_id=project_id)
        return str(dest)
    except Exception:                                        # noqa: BLE001
        return ""


_SAFE_NAME_RE = re.compile(r"[^\w\-\(\)]", re.UNICODE)


def sanitize_export_stem(text: str, fallback: str = "h3") -> str:
    """Keep Unicode letters/digits, -, _, (, ); replace others with _.

    Max 80 chars (after NFC normalization).
    """
    text = unicodedata.normalize("NFC", str(text or ""))
    cleaned = _SAFE_NAME_RE.sub("_", text).strip("_")
    cleaned = cleaned[:80].strip("_")
    return cleaned or fallback


def _short_reason(exc: Exception) -> str:
    if isinstance(exc, PermissionError):
        return "書き込み権限がありません"
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return "ディスク容量が不足しています"
    if isinstance(exc, FileNotFoundError):
        return "元の動画が見つかりません"
    msg = f"{type(exc).__name__}: {exc}"
    return msg[:200]


def export_video(src: str | Path | None, *, dest_dir: Path, filename: str,
                 media_root: str | Path | None = None, kind: str,
                 project_id: str = "") -> dict:
    """Copy a generated video into the user-facing Videos folder.

    Never raises. Returns {"ok", "kind", "path", "name", "error"} where
    "name" is the path relative to the Videos folder (e.g.
    "完成動画\\abc_s01.mp4"). No-overwrite: on a name collision, tries
    "<stem>_2.mp4", "_3.mp4", ... A failed attempt unlinks ONLY the file
    this call created (never an existing file).
    """
    result = {"ok": False, "kind": kind, "path": "", "name": "", "error": ""}
    created: Path | None = None
    try:
        if not src:
            result["error"] = "元の動画が見つかりません"
            return result
        src_path = Path(str(src))
        if not src_path.is_file():
            result["error"] = "元の動画が見つかりません"
            return result
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize_export_stem(Path(filename).stem or filename)

        dest: Path | None = None
        for n in range(1, 1000):
            candidate_stem = stem if n == 1 else f"{stem}_{n}"
            candidate = dest_dir / f"{candidate_stem}.mp4"
            try:
                fd = os.open(str(candidate),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_BINARY)
            except FileExistsError:
                continue
            except OSError as exc:
                result["error"] = _short_reason(exc)
                return result
            dest = candidate
            created = candidate
            break
        if dest is None:
            result["error"] = "保存先の空きファイル名が見つかりません"
            return result

        try:
            with os.fdopen(fd, "wb") as out, open(src_path, "rb") as inp:
                while True:
                    chunk = inp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
        except OSError as exc:
            try:
                if created is not None:
                    created.unlink(missing_ok=True)
            except Exception:                                # noqa: BLE001
                pass
            result["error"] = _short_reason(exc)
            return result

        if not dest.is_file() or dest.stat().st_size != src_path.stat().st_size:
            try:
                dest.unlink(missing_ok=True)
            except Exception:                                # noqa: BLE001
                pass
            result["error"] = "コピーの検証に失敗しました（サイズが一致しません）"
            return result

        try:
            rel_root = dest_dir.parent if dest_dir.name in ("完成動画", "クリップ") else dest_dir
            name = str(dest.relative_to(rel_root))
        except Exception:                                    # noqa: BLE001
            name = dest.name
        result["ok"] = True
        result["path"] = str(dest)
        result["name"] = name
        if media_root is not None:
            record_export(media_root, source=src_path, exported=dest,
                          media_type=kind, project_id=project_id)
        return result
    except Exception as exc:                                 # noqa: BLE001
        try:
            if created is not None:
                created.unlink(missing_ok=True)
        except Exception:                                    # noqa: BLE001
            pass
        result["error"] = _short_reason(exc)
        return result


def safe_stem(text: str, fallback: str = "h3") -> str:
    """Filesystem-safe short stem for export filenames."""
    cleaned = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in
                      (text or "").strip())[:48].strip("_")
    return cleaned or fallback


# ------------------------------------------------------------ manifest ------
MANIFEST_NAME = "export_manifest.jsonl"

MEDIA_EXTS = {".mp4", ".m4v", ".mov", ".webm",
              ".png", ".jpg", ".jpeg", ".webp", ".bmp",
              ".wav", ".mp3", ".aac", ".ogg", ".flac"}

# Never scanned, never deleted, no exceptions.
PROTECTED_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf",
                      ".json", ".jsonl", ".csv", ".log", ".md", ".txt",
                      ".py", ".ps1", ".bat", ".cmd", ".yaml", ".yml",
                      ".toml", ".db", ".sqlite"}
PROTECTED_DIR_HINTS = ("benchmarks", "docs", "_backup", "projects",
                       "story_projects", "custom_nodes", ".venv", ".git")


def manifest_path(media_root: str | Path) -> Path:
    return Path(media_root) / MANIFEST_NAME


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> float:
    import time
    return time.time()


def record_export(media_root: str | Path, *, source: str | Path,
                  exported: str | Path, media_type: str = "",
                  project_id: str = "") -> dict:
    """Append one manifest row. Never raises; returns the row ({} on failure)."""
    try:
        import json as _json
        src = Path(str(source))
        dst = Path(str(exported))
        if not src.is_file() or not dst.is_file():
            return {}
        row = {
            "source_path": str(src),
            "exported_path": str(dst),
            "size": dst.stat().st_size,
            "sha256": _sha256(dst),
            "exported_at": _now(),
            "media_type": media_type,
            "project_id": project_id,
            "safe_to_cleanup": True,
        }
        mp = manifest_path(media_root)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, "a", encoding="utf-8") as f:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")
        return row
    except Exception:                                        # noqa: BLE001
        return {}


def read_manifest(media_root: str | Path) -> list[dict]:
    """All manifest rows (duplicates kept; dedupe happens at match time)."""
    import json as _json
    mp = manifest_path(media_root)
    rows: list[dict] = []
    try:
        if not mp.is_file():
            return rows
        for line in mp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            if isinstance(row, dict) and row.get("source_path"):
                rows.append(row)
    except Exception:                                        # noqa: BLE001
        pass
    return rows


def _snapshot(path: Path) -> dict:
    """Size/mtime/hash identity used for scan→delete revalidation."""
    try:
        st = path.stat()
        return {"size": st.st_size, "mtime": st.st_mtime,
                "sha256": _sha256(path) if st.st_size <= (1 << 30) else ""}
    except Exception:                                        # noqa: BLE001
        return {}


def _is_protected_path(path: Path) -> str:
    """Return a reason when the path must never be a candidate, else ""."""
    name = path.name
    if path.suffix.lower() in PROTECTED_SUFFIXES:
        return f"protected extension {path.suffix.lower()}"
    lowered = str(path).replace("\\", "/").lower()
    for hint in PROTECTED_DIR_HINTS:
        if f"/{hint}/" in lowered or lowered.endswith(f"/{hint}"):
            return f"inside {hint}"
    if name.startswith("export_manifest"):
        return "manifest itself"
    return ""


def scan_candidates(*, media_root: str | Path,
                    scan_roots: list[str | Path],
                    media_dirs: list[str | Path] | None = None,
                    protected_paths: set[str] | None = None,
                    min_age_days: float = 1.0,
                    now: float | None = None) -> dict:
    """Scan-only: list deletable internal duplicates. Never deletes.

    A file becomes a candidate only when its H3_Media twin exists with equal
    size (plus equal hash when a manifest row carries one). Returns counts,
    per-folder rollup and an opaque token binding the exact snapshot.
    """
    import hashlib as _hashlib
    import json as _json
    import time as _time
    now = now if now is not None else _time.time()
    cutoff = now - float(min_age_days) * 86400.0
    mroot = Path(str(media_root))
    protected = {str(Path(p)) for p in (protected_paths or set())}
    by_source: dict[str, list[dict]] = {}
    for row in read_manifest(mroot):
        by_source.setdefault(str(Path(row["source_path"])), []).append(row)

    def media_twin(path: Path, size: int) -> Path | None:
        """A same-name, same-size file under the media dirs (legacy match)."""
        for md in (media_dirs or []):
            cand = Path(md) / path.name
            try:
                if cand.is_file() and cand.stat().st_size == size:
                    return cand
            except Exception:                                # noqa: BLE001
                continue
        return None

    candidates: list[dict] = []
    excluded = 0
    scanned = 0
    folders: dict[str, dict] = {}
    for root in scan_roots:
        rpath = Path(str(root))
        if not rpath.is_dir():
            continue
        try:
            files = [p for p in rpath.rglob("*") if p.is_file()]
        except Exception:                                    # noqa: BLE001
            continue
        for path in files:
            scanned += 1
            protected_reason = _is_protected_path(path)
            if protected_reason:
                excluded += 1
                continue
            if path.suffix.lower() not in MEDIA_EXTS:
                excluded += 1
                continue
            try:
                st = path.stat()
            except Exception:                                # noqa: BLE001
                excluded += 1
                continue
            if st.st_size <= 0:
                excluded += 1
                continue
            if st.st_mtime > cutoff:
                excluded += 1
                continue
            if str(path) in protected or str(path.resolve()) in protected:
                excluded += 1
                continue
            # Twin proof: manifest row first, legacy same-name+size second.
            twin: Path | None = None
            rows = [r for r in by_source.get(str(path), [])
                    if r.get("safe_to_cleanup", True)]
            for r in rows:
                try:
                    tp = Path(r["exported_path"])
                    if tp.is_file() and tp.stat().st_size == st.st_size and \
                            (not r.get("sha256") or
                             _snapshot(tp).get("sha256") == r["sha256"]):
                        twin = tp
                        break
                except Exception:                            # noqa: BLE001
                    continue
            if twin is None:
                twin = media_twin(path, st.st_size)
            if twin is None:
                excluded += 1
                continue
            snap = _snapshot(path)
            if not snap:
                excluded += 1
                continue
            candidates.append({
                "path": str(path),
                "size": snap["size"],
                "mtime": snap["mtime"],
                "sha256": snap.get("sha256", ""),
                "twin": str(twin),
            })
            key = str(rpath)
            slot = folders.setdefault(key, {"files": 0, "bytes": 0})
            slot["files"] += 1
            slot["bytes"] += snap["size"]
    blob = _json.dumps([c["path"] for c in candidates] +
                       [c["sha256"] for c in candidates], sort_keys=True)
    token = _hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {
        "candidate_count": len(candidates),
        "candidate_bytes": sum(c["size"] for c in candidates),
        "excluded_count": excluded,
        "scanned_count": scanned,
        "folders": [{"folder": k, "files": v["files"], "bytes": v["bytes"]}
                    for k, v in sorted(folders.items())],
        "candidates": candidates,
        "token": token,
    }


def clean_candidates(*, candidates: list[dict], token: str,
                     expected_token: str,
                     recycle: bool = True) -> dict:
    """Delete scan-approved candidates with TOCTOU revalidation.

    Every file is re-checked (exists, size, mtime, hash) against the scan
    snapshot; anything that changed is skipped, never deleted. Never raises.
    """
    removed = 0
    recovered = 0
    failed: list[dict] = []
    skipped: list[dict] = []
    if token != expected_token:
        return {"removed": 0, "recovered_bytes": 0, "failed": [],
                "skipped": [{"path": "(all)", "reason": "scan token mismatch"}],
                "method": "none"}
    for cand in candidates:
        path = Path(str(cand.get("path") or ""))
        snap = _snapshot(path)
        if not snap or snap.get("size") != cand.get("size") or \
                snap.get("mtime") != cand.get("mtime") or \
                (cand.get("sha256") and snap.get("sha256") != cand.get("sha256")):
            skipped.append({"path": str(path),
                            "reason": "changed or missing since scan"})
            continue
        ok, method = _remove(path, recycle=recycle)
        if ok:
            removed += 1
            recovered += int(cand.get("size") or 0)
        else:
            failed.append({"path": str(path), "reason": method})
    return {"removed": removed, "recovered_bytes": recovered,
            "failed": failed, "skipped": skipped,
            "method": "recycle" if recycle else "permanent"}


def _remove(path: Path, *, recycle: bool) -> tuple[bool, str]:
    """Best-effort single-file removal. Returns (ok, method-or-error)."""
    if recycle:
        try:
            import ctypes
            from ctypes import wintypes
            gchar = wintypes.WCHAR if hasattr(wintypes, "WCHAR") else ctypes.c_wchar

            class _Op(ctypes.Structure):
                _fields_ = [("hwnd", ctypes.c_void_p),
                            ("wFunc", ctypes.c_uint),
                            ("pFrom", ctypes.c_void_p),
                            ("pTo", ctypes.c_void_p),
                            ("fFlags", ctypes.c_ushort),
                            ("fAnyOperationsAborted", ctypes.c_int),
                            ("hNameMappings", ctypes.c_void_p),
                            ("lpszProgressTitle", ctypes.c_void_p)]

            buf = (gchar * (len(str(path)) + 2))()
            buf.value = str(path)
            op = _Op()
            op.wFunc = 3  # FO_DELETE
            op.pFrom = ctypes.cast(buf, ctypes.c_void_p)
            op.fFlags = 0x40 | 0x10 | 0x04  # ALLOWUNDO|NOCONFIRMATION|NOERRORUI
            rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
            if rc == 0 and not path.exists():
                return True, "recycle"
        except Exception as exc:                             # noqa: BLE001
            pass
    try:
        path.unlink()
        return True, ("permanent-fallback" if recycle else "permanent")
    except Exception as exc:                                 # noqa: BLE001
        return False, str(exc)[:200]

