from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


LEGACY_SCHEMA = "h3_gold_conditioning_cache"
LEGACY_FORMAT_VERSION = 1
EXECUTION_SCHEMA = "h3_gold_execution_cache"
EXECUTION_FORMAT_VERSION = 2

# Backward-compatible names used by the conditioning-only API.
SCHEMA = LEGACY_SCHEMA
SCHEMA_VERSION = LEGACY_FORMAT_VERSION
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class CacheError(RuntimeError):
    pass


class CacheMissError(CacheError):
    pass


class CacheCorruptError(CacheError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_model_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError("text_encoder_file_sha256 must be exactly 64 hexadecimal characters")
    return normalized


def parse_settings_json(settings_json: str) -> Any:
    try:
        value = json.loads(settings_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tokenize_encoder_settings_json is not valid JSON: {exc}") from exc
    _canonical_json_bytes(value)
    return value


def build_cache_identity(
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
) -> dict[str, Any]:
    model_name = text_encoder_model_name.strip()
    if not model_name:
        raise ValueError("text_encoder_model_name must not be empty")
    settings = parse_settings_json(tokenize_encoder_settings_json)
    return {
        "gold_prompt_sha256": _sha256_bytes(gold_prompt.encode("utf-8")),
        "text_encoder_model_name": model_name,
        "text_encoder_file_sha256": _validated_model_sha256(text_encoder_file_sha256),
        "tokenize_encoder_settings": settings,
    }


def build_cache_key(
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
) -> str:
    identity = build_cache_identity(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    return _sha256_bytes(_canonical_json_bytes(identity))


def build_execution_cache_identity(
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
) -> dict[str, Any]:
    identity = build_cache_identity(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    return {
        "cache_format": {
            "schema": EXECUTION_SCHEMA,
            "format_version": EXECUTION_FORMAT_VERSION,
            "roots": ["conditioning", "latent"],
        },
        **identity,
    }


def build_execution_cache_key(
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
) -> str:
    identity = build_execution_cache_identity(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    return _sha256_bytes(_canonical_json_bytes(identity))


def cache_paths(cache_directory: str | os.PathLike[str], cache_key: str) -> tuple[Path, Path]:
    directory = Path(cache_directory).expanduser().resolve()
    return directory / f"{cache_key}.manifest.json", directory / f"{cache_key}.safetensors"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def _encode_float(value: float) -> dict[str, Any]:
    if math.isnan(value):
        encoded = "nan"
    elif math.isinf(value):
        encoded = "+inf" if value > 0 else "-inf"
    else:
        encoded = value
    return {"type": "float", "value": encoded}


def _dict_value_path(path: str, key: Any, index: int) -> str:
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{path}.{key}"
    return f"{path}.dict_value[{index}]"


def _is_comfy_nested_tensor(value: Any) -> bool:
    value_type = type(value)
    return value_type.__module__ == "comfy.nested_tensor" and value_type.__qualname__ == "NestedTensor"


def _encode_structure(
    value: Any,
    tensors: dict[str, torch.Tensor],
    tensor_meta: dict[str, Any],
    *,
    root: str | None = None,
    path: str = "$",
) -> Any:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized or value.is_complex():
            raise TypeError(
                f"unsupported tensor for safetensors cache: layout={value.layout}, "
                f"quantized={value.is_quantized}, complex={value.is_complex()}"
            )
        name = f"tensor_{len(tensors):06d}"
        cpu_tensor = value.detach().to(device="cpu").contiguous().clone()
        tensors[name] = cpu_tensor
        tensor_meta[name] = {
            "dtype": str(cpu_tensor.dtype),
            "shape": list(cpu_tensor.shape),
            "sha256": _sha256_bytes(_tensor_bytes(cpu_tensor)),
            "requires_grad": bool(value.requires_grad),
        }
        if root is not None:
            tensor_meta[name]["root"] = root
            tensor_meta[name]["path"] = path
        return {"type": "tensor", "name": name}
    if _is_comfy_nested_tensor(value):
        nested_tensors = getattr(value, "tensors", None)
        if not isinstance(nested_tensors, list):
            raise TypeError("comfy NestedTensor.tensors must be a list")
        return {
            "type": "comfy_nested_tensor",
            "tensors": [
                _encode_structure(
                    tensor,
                    tensors,
                    tensor_meta,
                    root=root,
                    path=f"{path}.tensors[{index}]",
                )
                for index, tensor in enumerate(nested_tensors)
            ],
        }
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return _encode_float(value)
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _encode_structure(v, tensors, tensor_meta, root=root, path=f"{path}[{index}]")
                for index, v in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _encode_structure(v, tensors, tensor_meta, root=root, path=f"{path}[{index}]")
                for index, v in enumerate(value)
            ],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": [
                [
                    _encode_structure(
                        k,
                        tensors,
                        tensor_meta,
                        root=root,
                        path=f"{path}.dict_key[{index}]",
                    ),
                    _encode_structure(
                        v,
                        tensors,
                        tensor_meta,
                        root=root,
                        path=_dict_value_path(path, k, index),
                    ),
                ]
                for index, (k, v) in enumerate(value.items())
            ],
        }
    raise TypeError(f"unsupported cache metadata value: {type(value).__module__}.{type(value).__qualname__}")


def _decode_float(value: Any) -> float:
    if value == "nan":
        return float("nan")
    if value == "+inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise CacheCorruptError(f"invalid encoded float: {value!r}")


def _decode_structure(node: Any, tensors: dict[str, torch.Tensor], tensor_meta: dict[str, Any]) -> Any:
    if not isinstance(node, dict) or not isinstance(node.get("type"), str):
        raise CacheCorruptError("invalid structure node")
    kind = node["type"]
    if kind == "tensor":
        name = node.get("name")
        if name not in tensors or name not in tensor_meta:
            raise CacheCorruptError(f"missing tensor referenced by structure: {name!r}")
        tensor = tensors[name]
        meta = tensor_meta[name]
        if str(tensor.dtype) != meta.get("dtype"):
            raise CacheCorruptError(f"dtype mismatch for {name}")
        if list(tensor.shape) != meta.get("shape"):
            raise CacheCorruptError(f"shape mismatch for {name}")
        if _sha256_bytes(_tensor_bytes(tensor)) != meta.get("sha256"):
            raise CacheCorruptError(f"tensor value hash mismatch for {name}")
        requires_grad = meta.get("requires_grad")
        if not isinstance(requires_grad, bool):
            raise CacheCorruptError(f"invalid requires_grad metadata for {name}")
        if requires_grad:
            if not (tensor.is_floating_point() or tensor.is_complex()):
                raise CacheCorruptError(f"invalid requires_grad tensor dtype for {name}")
            tensor.requires_grad_(True)
        return tensor
    if kind == "comfy_nested_tensor":
        encoded_tensors = node.get("tensors")
        if not isinstance(encoded_tensors, list):
            raise CacheCorruptError("invalid comfy NestedTensor tensor list")
        restored_tensors = [_decode_structure(item, tensors, tensor_meta) for item in encoded_tensors]
        if not all(isinstance(tensor, torch.Tensor) for tensor in restored_tensors):
            raise CacheCorruptError("comfy NestedTensor may contain only tensors")
        try:
            from comfy.nested_tensor import NestedTensor
        except (ImportError, ModuleNotFoundError) as exc:
            raise CacheCorruptError("cannot import comfy.nested_tensor.NestedTensor") from exc
        return NestedTensor(restored_tensors)
    if kind == "none":
        return None
    if kind == "bool" and isinstance(node.get("value"), bool):
        return node["value"]
    if kind == "int" and isinstance(node.get("value"), int) and not isinstance(node.get("value"), bool):
        return node["value"]
    if kind == "float":
        return _decode_float(node.get("value"))
    if kind == "str" and isinstance(node.get("value"), str):
        return node["value"]
    if kind == "bytes" and isinstance(node.get("base64"), str):
        try:
            return base64.b64decode(node["base64"], validate=True)
        except ValueError as exc:
            raise CacheCorruptError("invalid base64 bytes value") from exc
    if kind in {"list", "tuple"} and isinstance(node.get("items"), list):
        values = [_decode_structure(v, tensors, tensor_meta) for v in node["items"]]
        return values if kind == "list" else tuple(values)
    if kind == "dict" and isinstance(node.get("items"), list):
        result = {}
        for pair in node["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise CacheCorruptError("invalid encoded dict item")
            key = _decode_structure(pair[0], tensors, tensor_meta)
            try:
                if key in result:
                    raise CacheCorruptError(f"duplicate decoded dict key: {key!r}")
                result[key] = _decode_structure(pair[1], tensors, tensor_meta)
            except TypeError as exc:
                raise CacheCorruptError(f"unhashable decoded dict key: {key!r}") from exc
        return result
    raise CacheCorruptError(f"invalid or unsupported structure node type: {kind!r}")


def _manifest_payload_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _bundle_integrity_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("bundle_integrity_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _execution_manifest_summary(identity: dict[str, Any]) -> dict[str, Any]:
    settings = identity.get("tokenize_encoder_settings")
    if not isinstance(settings, dict):
        settings = {}
    geometry = settings.get("conditioning_geometry")
    if not isinstance(geometry, dict):
        geometry = {}
    references = settings.get("references")
    if not isinstance(references, list):
        references = []
    reference_hashes = []
    for reference in references:
        if isinstance(reference, dict):
            reference_hashes.append(
                {
                    "slot": reference.get("slot"),
                    "sha256": reference.get("sha256"),
                }
            )
    return {
        "prompt_sha256": identity.get("gold_prompt_sha256"),
        "text_encoder_sha256": identity.get("text_encoder_file_sha256"),
        "video_vae_sha256": settings.get("video_vae_file_sha256"),
        "audio_vae_sha256": settings.get("audio_vae_file_sha256"),
        "reference_hashes": reference_hashes,
        "width": geometry.get("width"),
        "height": geometry.get("height"),
        "length": geometry.get("length"),
        "ref_image_size": geometry.get("ref_image_size"),
    }


def save_execution_cache(
    conditioning: Any,
    latent: Any,
    cache_directory: str | os.PathLike[str],
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
    overwrite: bool = False,
) -> tuple[str, Path]:
    identity = build_execution_cache_identity(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    key = _sha256_bytes(_canonical_json_bytes(identity))
    manifest_path, tensor_path = cache_paths(cache_directory, key)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() or tensor_path.exists():
        if not overwrite:
            if manifest_path.exists() and tensor_path.exists():
                load_execution_cache(
                    cache_directory,
                    gold_prompt,
                    text_encoder_model_name,
                    text_encoder_file_sha256,
                    tokenize_encoder_settings_json,
                )
                return key, manifest_path
            raise CacheCorruptError(
                f"partial execution cache already exists for key {key}; use overwrite only after inspection"
            )

    tensors: dict[str, torch.Tensor] = {}
    tensor_meta: dict[str, Any] = {}
    roots = {
        "conditioning": _encode_structure(
            conditioning,
            tensors,
            tensor_meta,
            root="conditioning",
            path="roots.conditioning",
        ),
        "latent": _encode_structure(
            latent,
            tensors,
            tensor_meta,
            root="latent",
            path="roots.latent",
        ),
    }
    root_tensor_counts = {
        root: sum(1 for meta in tensor_meta.values() if meta.get("root") == root)
        for root in ("conditioning", "latent")
    }
    if root_tensor_counts["conditioning"] == 0 or root_tensor_counts["latent"] == 0:
        raise ValueError("execution cache requires at least one tensor in both conditioning and latent roots")

    token = uuid.uuid4().hex
    tensor_tmp = tensor_path.with_name(f".{tensor_path.name}.{token}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    try:
        save_file(
            tensors,
            str(tensor_tmp),
            metadata={
                "schema": EXECUTION_SCHEMA,
                "format_version": str(EXECUTION_FORMAT_VERSION),
                "cache_key": key,
            },
        )
        manifest = {
            "schema": EXECUTION_SCHEMA,
            "format_version": EXECUTION_FORMAT_VERSION,
            "cache_key": key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            **_execution_manifest_summary(identity),
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": _sha256_file(tensor_tmp),
            "tensor_count": len(tensors),
            "total_tensor_bytes": sum(_tensor_nbytes(tensor) for tensor in tensors.values()),
            "tensors": tensor_meta,
            "roots": roots,
        }
        manifest["bundle_integrity_sha256"] = _bundle_integrity_hash(manifest)
        manifest_tmp.write_bytes(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        )
        os.replace(tensor_tmp, tensor_path)
        os.replace(manifest_tmp, manifest_path)
    finally:
        tensor_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return key, manifest_path


def load_execution_cache(
    cache_directory: str | os.PathLike[str],
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
) -> tuple[Any, Any, str, Path]:
    identity = build_execution_cache_identity(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    key = _sha256_bytes(_canonical_json_bytes(identity))
    manifest_path, expected_tensor_path = cache_paths(cache_directory, key)
    if not manifest_path.is_file() or not expected_tensor_path.is_file():
        legacy_key = build_cache_key(
            gold_prompt,
            text_encoder_model_name,
            text_encoder_file_sha256,
            tokenize_encoder_settings_json,
        )
        legacy_manifest, legacy_tensor = cache_paths(cache_directory, legacy_key)
        if legacy_manifest.exists() or legacy_tensor.exists():
            raise CacheCorruptError(
                "legacy format v1 conditioning-only cache exists and cannot be used as an execution cache"
            )
        raise CacheMissError(f"execution cache MISS for key {key}: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheCorruptError(f"cannot read execution cache manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CacheCorruptError("execution cache manifest root must be an object")
    if manifest.get("schema") != EXECUTION_SCHEMA:
        raise CacheCorruptError("execution cache schema mismatch")
    if manifest.get("format_version") != EXECUTION_FORMAT_VERSION:
        raise CacheCorruptError(
            f"execution cache format version mismatch: expected {EXECUTION_FORMAT_VERSION}, "
            f"got {manifest.get('format_version')!r}"
        )
    if manifest.get("cache_key") != key or manifest.get("identity") != identity:
        raise CacheCorruptError("execution cache identity/key mismatch")
    if manifest.get("bundle_integrity_sha256") != _bundle_integrity_hash(manifest):
        raise CacheCorruptError("execution cache bundle integrity check failed")
    for field, expected in _execution_manifest_summary(identity).items():
        if manifest.get(field) != expected:
            raise CacheCorruptError(f"execution cache summary mismatch: {field}")
    if manifest.get("tensor_file") != expected_tensor_path.name:
        raise CacheCorruptError("execution cache tensor filename mismatch")
    if manifest.get("tensor_file_sha256") != _sha256_file(expected_tensor_path):
        raise CacheCorruptError("execution cache safetensors integrity check failed")
    roots = manifest.get("roots")
    if not isinstance(roots, dict) or set(roots) != {"conditioning", "latent"}:
        raise CacheCorruptError("execution cache must contain exactly conditioning and latent roots")
    tensor_meta = manifest.get("tensors")
    if not isinstance(tensor_meta, dict) or not tensor_meta:
        raise CacheCorruptError("execution cache tensor metadata is missing")
    for name, meta in tensor_meta.items():
        if not isinstance(meta, dict):
            raise CacheCorruptError(f"invalid tensor metadata for {name}")
        if meta.get("root") not in {"conditioning", "latent"}:
            raise CacheCorruptError(f"invalid tensor root for {name}")
        if not isinstance(meta.get("path"), str) or not meta["path"].startswith(f"roots.{meta['root']}"):
            raise CacheCorruptError(f"invalid tensor path for {name}")
    if not any(meta.get("root") == "conditioning" for meta in tensor_meta.values()):
        raise CacheCorruptError("execution cache conditioning root has no tensors")
    if not any(meta.get("root") == "latent" for meta in tensor_meta.values()):
        raise CacheCorruptError("execution cache latent root has no tensors")
    try:
        tensors = load_file(str(expected_tensor_path), device="cpu")
    except Exception as exc:
        raise CacheCorruptError(f"cannot load execution cache safetensors: {exc}") from exc
    if set(tensors) != set(tensor_meta):
        raise CacheCorruptError("execution cache tensor set does not match manifest")
    if manifest.get("tensor_count") != len(tensors):
        raise CacheCorruptError("execution cache tensor_count mismatch")
    total_tensor_bytes = sum(_tensor_nbytes(tensor) for tensor in tensors.values())
    if manifest.get("total_tensor_bytes") != total_tensor_bytes:
        raise CacheCorruptError("execution cache total_tensor_bytes mismatch")
    conditioning = _decode_structure(roots["conditioning"], tensors, tensor_meta)
    latent = _decode_structure(roots["latent"], tensors, tensor_meta)
    return conditioning, latent, key, manifest_path


def execution_cache_fingerprint(
    cache_directory: str | os.PathLike[str],
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
) -> str:
    key = build_execution_cache_key(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    manifest_path, tensor_path = cache_paths(cache_directory, key)
    if not manifest_path.is_file() or not tensor_path.is_file():
        return f"missing:v{EXECUTION_FORMAT_VERSION}:{key}"
    tensor_stat = tensor_path.stat()
    return (
        f"present:v{EXECUTION_FORMAT_VERSION}:{key}:{_sha256_file(manifest_path)}:"
        f"{tensor_stat.st_size}:{tensor_stat.st_mtime_ns}"
    )


def save_conditioning_cache(
    conditioning: Any,
    cache_directory: str | os.PathLike[str],
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
    overwrite: bool = False,
) -> tuple[str, Path]:
    identity = build_cache_identity(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    key = _sha256_bytes(_canonical_json_bytes(identity))
    manifest_path, tensor_path = cache_paths(cache_directory, key)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() or tensor_path.exists():
        if not overwrite:
            if manifest_path.exists() and tensor_path.exists():
                load_conditioning_cache(
                    cache_directory,
                    gold_prompt,
                    text_encoder_model_name,
                    text_encoder_file_sha256,
                    tokenize_encoder_settings_json,
                )
                return key, manifest_path
            raise CacheCorruptError(f"partial cache already exists for key {key}; use overwrite only after inspection")

    tensors: dict[str, torch.Tensor] = {}
    tensor_meta: dict[str, Any] = {}
    structure = _encode_structure(conditioning, tensors, tensor_meta)
    if not tensors:
        raise ValueError("CONDITIONING cache contains no tensors")

    token = uuid.uuid4().hex
    tensor_tmp = tensor_path.with_name(f".{tensor_path.name}.{token}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    try:
        save_file(tensors, str(tensor_tmp), metadata={"schema": SCHEMA, "cache_key": key})
        manifest = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "cache_key": key,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": _sha256_file(tensor_tmp),
            "tensors": tensor_meta,
            "structure": structure,
        }
        manifest["manifest_payload_sha256"] = _manifest_payload_hash(manifest)
        manifest_tmp.write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        os.replace(tensor_tmp, tensor_path)
        os.replace(manifest_tmp, manifest_path)
    finally:
        tensor_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
    return key, manifest_path


def load_conditioning_cache(
    cache_directory: str | os.PathLike[str],
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
) -> tuple[Any, str, Path]:
    identity = build_cache_identity(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    key = _sha256_bytes(_canonical_json_bytes(identity))
    manifest_path, expected_tensor_path = cache_paths(cache_directory, key)
    if not manifest_path.is_file() or not expected_tensor_path.is_file():
        raise CacheMissError(f"CONDITIONING cache MISS for key {key}: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheCorruptError(f"cannot read cache manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CacheCorruptError("cache manifest root must be an object")
    if manifest.get("schema") != SCHEMA or manifest.get("schema_version") != SCHEMA_VERSION:
        raise CacheCorruptError("cache schema or version mismatch")
    if manifest.get("cache_key") != key or manifest.get("identity") != identity:
        raise CacheCorruptError("cache identity/key mismatch")
    if manifest.get("manifest_payload_sha256") != _manifest_payload_hash(manifest):
        raise CacheCorruptError("cache manifest integrity check failed")
    if manifest.get("tensor_file") != expected_tensor_path.name:
        raise CacheCorruptError("cache tensor filename mismatch")
    if manifest.get("tensor_file_sha256") != _sha256_file(expected_tensor_path):
        raise CacheCorruptError("cache safetensors integrity check failed")
    tensor_meta = manifest.get("tensors")
    if not isinstance(tensor_meta, dict) or not tensor_meta:
        raise CacheCorruptError("cache tensor metadata is missing")
    try:
        tensors = load_file(str(expected_tensor_path), device="cpu")
    except Exception as exc:
        raise CacheCorruptError(f"cannot load cache safetensors: {exc}") from exc
    if set(tensors) != set(tensor_meta):
        raise CacheCorruptError("cache tensor set does not match manifest")
    conditioning = _decode_structure(manifest.get("structure"), tensors, tensor_meta)
    return conditioning, key, manifest_path


def cache_fingerprint(
    cache_directory: str | os.PathLike[str],
    gold_prompt: str,
    text_encoder_model_name: str,
    text_encoder_file_sha256: str,
    tokenize_encoder_settings_json: str,
) -> str:
    key = build_cache_key(
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    )
    manifest_path, tensor_path = cache_paths(cache_directory, key)
    if not manifest_path.is_file() or not tensor_path.is_file():
        return f"missing:{key}"
    tensor_stat = tensor_path.stat()
    return (
        f"present:{key}:{_sha256_file(manifest_path)}:"
        f"{tensor_stat.st_size}:{tensor_stat.st_mtime_ns}"
    )
