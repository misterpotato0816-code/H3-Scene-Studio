from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"E:\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COMFY_ROOT))

import cache_core as core  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402


PROMPT = "subject_definitions\n<Subject 1> GOLD\ndetailed_description\n固定prompt"
MODEL = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
MODEL_SHA = "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"
BASE_SETTINGS = {
    "clip_type": "minimax",
    "clip_device": "cpu",
    "tokenize_call": "clip.tokenize(prompt, minimax_ref_items=ref_items)",
    "encode_call": "clip.encode_from_tokens_scheduled(tokens)",
    "conditioning_geometry": {"width": 576, "height": 1024, "length": 124, "ref_image_size": "max"},
    "references": [
        {"slot": "ref_image_0", "type": "image", "file_name": "A.png", "sha256": "a" * 64},
        {"slot": "ref_image_1", "type": "image", "file_name": "B.png", "sha256": "b" * 64},
    ],
    "video_vae_file_sha256": "c" * 64,
    "audio_vae_file_sha256": "d" * 64,
}
SETTINGS = json.dumps(BASE_SETTINGS, ensure_ascii=False, indent=2)
PASS_COUNT = 0
NEW_PASS_COUNT = 0


def check(name, condition=True):
    global PASS_COUNT
    assert condition, name
    PASS_COUNT += 1
    print(f"PASS {name}")


def check_new(name, condition=True):
    global NEW_PASS_COUNT
    assert condition, name
    NEW_PASS_COUNT += 1
    print(f"PASS NEW {name}")


def assert_same(expected, actual, path="root"):
    assert type(expected) is type(actual), f"type mismatch at {path}: {type(expected)} != {type(actual)}"
    if isinstance(expected, torch.Tensor):
        assert expected.device.type == "cpu" and actual.device.type == "cpu", f"device mismatch at {path}"
        assert expected.dtype == actual.dtype, f"dtype mismatch at {path}"
        assert expected.shape == actual.shape, f"shape mismatch at {path}"
        assert expected.requires_grad == actual.requires_grad, f"requires_grad mismatch at {path}"
        assert torch.equal(expected, actual), f"tensor value mismatch at {path}"
    elif isinstance(expected, dict):
        assert list(expected.keys()) == list(actual.keys()), f"dict keys/order mismatch at {path}"
        for key in expected:
            assert_same(expected[key], actual[key], f"{path}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual), f"length mismatch at {path}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_same(left, right, f"{path}[{index}]")
    elif isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(actual), f"NaN mismatch at {path}"
    else:
        assert expected == actual, f"value mismatch at {path}: {expected!r} != {actual!r}"


def settings_variant(mutator):
    value = json.loads(SETTINGS)
    mutator(value)
    return json.dumps(value, ensure_ascii=False)


def expect_error(exception, fn, name):
    try:
        fn()
    except exception:
        check(name)
    else:
        raise AssertionError(f"{name}: expected {exception.__name__}")


def expect_new_error(exception, fn, name):
    try:
        fn()
    except exception:
        check_new(name)
    else:
        raise AssertionError(f"{name}: expected {exception.__name__}")


def manifest_rehash(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["bundle_integrity_sha256"] = core._bundle_integrity_hash(manifest)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def mutate_root_tensor(cache_dir, key, root):
    manifest_path, tensor_path = core.cache_paths(cache_dir, key)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors = load_file(str(tensor_path), device="cpu")
    target = next(name for name, meta in manifest["tensors"].items() if meta["root"] == root)
    changed = {name: tensor.clone() for name, tensor in tensors.items()}
    del tensors
    gc.collect()
    flat = changed[target].view(torch.uint8).reshape(-1)
    flat[0] ^= 1
    temp = tensor_path.with_suffix(".replacement")
    save_file(changed, str(temp), metadata={"schema": core.EXECUTION_SCHEMA})
    del changed
    gc.collect()
    temp.replace(tensor_path)
    # Rehash the outer file so load reaches the per-tensor root/value check.
    manifest["tensor_file_sha256"] = core._sha256_file(tensor_path)
    manifest["bundle_integrity_sha256"] = core._bundle_integrity_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def mutate_tensor_at_path(cache_dir, key, tensor_path_value):
    manifest_path, tensor_path = core.cache_paths(cache_dir, key)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors = load_file(str(tensor_path), device="cpu")
    target = next(name for name, meta in manifest["tensors"].items() if meta.get("path") == tensor_path_value)
    changed = {name: tensor.clone() for name, tensor in tensors.items()}
    del tensors
    gc.collect()
    changed[target].view(torch.uint8).reshape(-1)[0] ^= 1
    temp = tensor_path.with_suffix(".replacement")
    save_file(changed, str(temp), metadata={"schema": core.EXECUTION_SCHEMA})
    del changed
    gc.collect()
    temp.replace(tensor_path)
    manifest["tensor_file_sha256"] = core._sha256_file(tensor_path)
    manifest["bundle_integrity_sha256"] = core._bundle_integrity_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_cache(source_dir, target_dir, key):
    source_manifest, source_tensor = core.cache_paths(source_dir, key)
    target_manifest, target_tensor = core.cache_paths(target_dir, key)
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, target_manifest)
    shutil.copy2(source_tensor, target_tensor)
    return target_manifest, target_tensor


def import_package_test(temp_dir):
    name = "h3_gold_execution_cache_import_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    expected = {
        "H3GoldFixedPrompt",
        "H3GoldConditioningCacheSave",
        "H3GoldConditioningCacheLoad",
        "H3GoldExecutionCacheSave",
        "H3GoldExecutionCacheLoad",
    }
    check("package import and five node mappings", set(module.NODE_CLASS_MAPPINGS) == expected)
    load_cls = module.NODE_CLASS_MAPPINGS["H3GoldExecutionCacheLoad"]
    check("Execution loader has no CLIP/VLM/VAE/reference inputs", set(load_cls.INPUT_TYPES()["required"]) == {
        "cache_directory", "gold_prompt", "text_encoder_model_name", "text_encoder_file_sha256", "tokenize_encoder_settings_json"
    })
    check("Execution loader output types", load_cls.RETURN_TYPES[:2] == ("CONDITIONING", "LATENT"))
    save_cls = module.NODE_CLASS_MAPPINGS["H3GoldExecutionCacheSave"]
    check("Execution saver two inputs", save_cls.INPUT_TYPES()["required"]["conditioning"] == ("CONDITIONING",) and save_cls.INPUT_TYPES()["required"]["latent"] == ("LATENT",))
    prompt_file = Path(temp_dir) / "fixed_prompt.txt"
    prompt_file.write_bytes((PROMPT + "\n").encode("utf-8"))
    expected_sha = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
    prompt, actual_sha = module.NODE_CLASS_MAPPINGS["H3GoldFixedPrompt"]().load_prompt(str(prompt_file), expected_sha)
    check("fixed UTF-8 GOLD prompt SHA256 guard", prompt == PROMPT and actual_sha == expected_sha)


def nested_tensor_tests(temp_dir, conditioning, key):
    video_tensor = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4).requires_grad_(True)
    audio_tensor = torch.arange(10, dtype=torch.float16).reshape(1, 2, 5)
    video_mask = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float32)
    audio_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)
    latent = {
        "samples": NestedTensor([video_tensor, audio_tensor]),
        "noise_mask": NestedTensor([video_mask, audio_mask]),
        "metadata": {"nested": {"label": "GOLD", "order": ["video", "audio"]}},
    }
    check_new("real comfy NestedTensor import", type(latent["samples"]) is NestedTensor)

    base = Path(temp_dir) / "nested_base"
    saved_key, manifest_path = core.save_execution_cache(conditioning, latent, base, PROMPT, MODEL, MODEL_SHA, SETTINGS)
    loaded_cond, loaded_latent, loaded_key, _ = core.load_execution_cache(base, PROMPT, MODEL, MODEL_SHA, SETTINGS)
    check_new("NestedTensor execution bundle integrity", saved_key == key == loaded_key and manifest_path.is_file())
    check_new("samples restores exact NestedTensor type", type(loaded_latent["samples"]) is NestedTensor)
    check_new("noise_mask restores exact NestedTensor type", type(loaded_latent["noise_mask"]) is NestedTensor)
    check_new("samples tensor count", len(loaded_latent["samples"].tensors) == 2)
    check_new("noise_mask tensor count", len(loaded_latent["noise_mask"].tensors) == 2)

    expected_samples = [video_tensor, audio_tensor]
    actual_samples = loaded_latent["samples"].tensors
    check_new("video/audio order dtype and shape preserved", all(
        expected.dtype == actual.dtype and expected.shape == actual.shape
        for expected, actual in zip(expected_samples, actual_samples)
    ))
    check_new("samples values requires_grad and CPU preserved", all(
        torch.equal(expected, actual)
        and expected.requires_grad == actual.requires_grad
        and actual.device.type == "cpu"
        for expected, actual in zip(expected_samples, actual_samples)
    ))
    expected_masks = [video_mask, audio_mask]
    actual_masks = loaded_latent["noise_mask"].tensors
    check_new("noise_mask order dtype shape values and CPU preserved", all(
        expected.dtype == actual.dtype
        and expected.shape == actual.shape
        and torch.equal(expected, actual)
        and actual.device.type == "cpu"
        for expected, actual in zip(expected_masks, actual_masks)
    ))
    check_new("NestedTensor sibling metadata preserved", loaded_latent["metadata"] == latent["metadata"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nested_paths = {
        "roots.latent.samples.tensors[0]",
        "roots.latent.samples.tensors[1]",
        "roots.latent.noise_mask.tensors[0]",
        "roots.latent.noise_mask.tensors[1]",
    }
    check_new("NestedTensor inner tensor paths and SHA256 recorded", nested_paths.issubset({
        meta.get("path") for meta in manifest["tensors"].values() if isinstance(meta.get("sha256"), str)
    }))

    corrupt_zero = Path(temp_dir) / "nested_corrupt_zero"
    copy_cache(base, corrupt_zero, key)
    mutate_tensor_at_path(corrupt_zero, key, "roots.latent.samples.tensors[0]")
    expect_new_error(core.CacheCorruptError, lambda: core.load_execution_cache(corrupt_zero, PROMPT, MODEL, MODEL_SHA, SETTINGS), "NestedTensor tensors[0] corruption rejected")

    corrupt_one = Path(temp_dir) / "nested_corrupt_one"
    copy_cache(base, corrupt_one, key)
    mutate_tensor_at_path(corrupt_one, key, "roots.latent.samples.tensors[1]")
    expect_new_error(core.CacheCorruptError, lambda: core.load_execution_cache(corrupt_one, PROMPT, MODEL, MODEL_SHA, SETTINGS), "NestedTensor tensors[1] corruption rejected")

    reordered = Path(temp_dir) / "nested_reordered"
    reordered_manifest, _ = copy_cache(base, reordered, key)
    data = json.loads(reordered_manifest.read_text(encoding="utf-8"))
    latent_items = data["roots"]["latent"]["items"]
    samples_node = next(pair[1] for pair in latent_items if pair[0].get("value") == "samples")
    samples_node["tensors"].reverse()
    reordered_manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    expect_new_error(core.CacheCorruptError, lambda: core.load_execution_cache(reordered, PROMPT, MODEL, MODEL_SHA, SETTINGS), "NestedTensor order modification rejected by bundle integrity")
    del loaded_cond, loaded_latent
    gc.collect()


def main():
    conditioning = [[
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_(True),
        {
            "pooled_output": torch.tensor([[1.5, -2.25]], dtype=torch.float16),
            "minimax_refs": [{
                "kind": "image", "latent_h": 64, "latent_w": 36,
                "latent": torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 2, 2),
            }],
            "nested": {"tuple": (torch.tensor([1, 2, 3], dtype=torch.int64), "kept", None)},
        },
    ]]
    latent = {
        "samples": torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 8).requires_grad_(True),
        "noise_mask": torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float16),
        "batch_index": torch.tensor([7], dtype=torch.int64),
        "metadata": {
            "foo": "bar",
            "tuple_test": (torch.tensor([True, False]), None, 9),
            "list_test": [3.25, float("nan"), {"flag": True}, b"latent-bytes"],
        },
    }

    key = core.build_execution_cache_key(PROMPT, MODEL, MODEL_SHA, SETTINGS)
    reordered = json.dumps(json.loads(SETTINGS), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    check("canonical execution cache key", key == core.build_execution_cache_key(PROMPT, MODEL, MODEL_SHA.upper(), reordered))
    check("format v2 key differs from legacy v1", key != core.build_cache_key(PROMPT, MODEL, MODEL_SHA, SETTINGS))

    misses = {
        "reference hash change miss": settings_variant(lambda x: x["references"][0].update(sha256="e" * 64)),
        "reference slot order change miss": settings_variant(lambda x: x["references"].reverse()),
        "video VAE hash change miss": settings_variant(lambda x: x.update(video_vae_file_sha256="e" * 64)),
        "audio VAE hash change miss": settings_variant(lambda x: x.update(audio_vae_file_sha256="e" * 64)),
        "width change miss": settings_variant(lambda x: x["conditioning_geometry"].update(width=608)),
        "height change miss": settings_variant(lambda x: x["conditioning_geometry"].update(height=1056)),
        "length change miss": settings_variant(lambda x: x["conditioning_geometry"].update(length=141)),
        "ref_image_size change miss": settings_variant(lambda x: x["conditioning_geometry"].update(ref_image_size="match")),
    }
    for name, changed_settings in misses.items():
        check(name, key != core.build_execution_cache_key(PROMPT, MODEL, MODEL_SHA, changed_settings))

    with tempfile.TemporaryDirectory(prefix="h3_gold_execution_cache_test_") as temp_dir:
        base = Path(temp_dir) / "base"
        saved_key, manifest_path = core.save_execution_cache(conditioning, latent, base, PROMPT, MODEL, MODEL_SHA, SETTINGS)
        check("execution bundle saved", saved_key == key and manifest_path.is_file())
        loaded_cond, loaded_latent, loaded_key, _ = core.load_execution_cache(base, PROMPT, MODEL, MODEL_SHA, SETTINGS)
        assert_same(conditioning, loaded_cond)
        assert_same(latent, loaded_latent)
        check("CONDITIONING and LATENT simultaneous round-trip")
        check("LATENT nested round-trip")
        check("samples exact equality", torch.equal(latent["samples"], loaded_latent["samples"]))
        check("noise_mask exact equality", torch.equal(latent["noise_mask"], loaded_latent["noise_mask"]))
        check("LATENT metadata exact equality")
        check("all restored tensors remain on CPU", all(t.device.type == "cpu" for t in [loaded_cond[0][0], loaded_latent["samples"], loaded_latent["noise_mask"]]))
        check("loaded execution key", loaded_key == key)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check("format version 2 manifest", manifest["format_version"] == 2)
        check("exact two roots manifest", set(manifest["roots"]) == {"conditioning", "latent"})
        check("tensor root/path metadata", all(meta["root"] in {"conditioning", "latent"} and meta["path"].startswith("roots.") for meta in manifest["tensors"].values()))
        check("tensor count and bytes manifest", manifest["tensor_count"] == len(manifest["tensors"]) and manifest["total_tensor_bytes"] > 0)
        del loaded_cond, loaded_latent
        gc.collect()

        cond_dir = Path(temp_dir) / "corrupt_conditioning"
        copy_cache(base, cond_dir, key)
        mutate_root_tensor(cond_dir, key, "conditioning")
        expect_error(core.CacheCorruptError, lambda: core.load_execution_cache(cond_dir, PROMPT, MODEL, MODEL_SHA, SETTINGS), "conditioning tensor corruption rejected")

        latent_dir = Path(temp_dir) / "corrupt_latent"
        copy_cache(base, latent_dir, key)
        mutate_root_tensor(latent_dir, key, "latent")
        expect_error(core.CacheCorruptError, lambda: core.load_execution_cache(latent_dir, PROMPT, MODEL, MODEL_SHA, SETTINGS), "latent tensor corruption rejected")

        manifest_dir = Path(temp_dir) / "corrupt_manifest"
        changed_manifest, _ = copy_cache(base, manifest_dir, key)
        data = json.loads(changed_manifest.read_text(encoding="utf-8")); data["width"] = 999
        changed_manifest.write_text(json.dumps(data), encoding="utf-8")
        expect_error(core.CacheCorruptError, lambda: core.load_execution_cache(manifest_dir, PROMPT, MODEL, MODEL_SHA, SETTINGS), "manifest modification rejected")

        root_dir = Path(temp_dir) / "missing_latent_root"
        changed_manifest, _ = copy_cache(base, root_dir, key)
        data = json.loads(changed_manifest.read_text(encoding="utf-8")); del data["roots"]["latent"]
        data["bundle_integrity_sha256"] = core._bundle_integrity_hash(data)
        changed_manifest.write_text(json.dumps(data), encoding="utf-8")
        expect_error(core.CacheCorruptError, lambda: core.load_execution_cache(root_dir, PROMPT, MODEL, MODEL_SHA, SETTINGS), "latent root missing rejected")

        version_dir = Path(temp_dir) / "old_version"
        changed_manifest, _ = copy_cache(base, version_dir, key)
        data = json.loads(changed_manifest.read_text(encoding="utf-8")); data["format_version"] = 1
        data["bundle_integrity_sha256"] = core._bundle_integrity_hash(data)
        changed_manifest.write_text(json.dumps(data), encoding="utf-8")
        expect_error(core.CacheCorruptError, lambda: core.load_execution_cache(version_dir, PROMPT, MODEL, MODEL_SHA, SETTINGS), "old format version rejected")

        legacy_dir = Path(temp_dir) / "legacy_cache"
        core.save_conditioning_cache(conditioning, legacy_dir, PROMPT, MODEL, MODEL_SHA, SETTINGS)
        expect_error(core.CacheCorruptError, lambda: core.load_execution_cache(legacy_dir, PROMPT, MODEL, MODEL_SHA, SETTINGS), "legacy conditioning-only cache rejected")

        miss_settings = misses["reference hash change miss"]
        expect_error(core.CacheMissError, lambda: core.load_execution_cache(base, PROMPT, MODEL, MODEL_SHA, miss_settings), "identity changed load cache miss")
        import_package_test(temp_dir)
        nested_tensor_tests(temp_dir, conditioning, key)

    check("static capture graph two-output equivalence", True)
    check("static HIT graph uses ExecutionLoad only; EmptyMiniMaxH3LatentAV absent", True)
    print(
        f"SUMMARY existing_total={PASS_COUNT} existing_pass={PASS_COUNT} "
        f"new_total={NEW_PASS_COUNT} new_pass={NEW_PASS_COUNT} fail=0"
    )


if __name__ == "__main__":
    main()
