from __future__ import annotations

import hashlib
from pathlib import Path

from .cache_core import (
    cache_fingerprint,
    execution_cache_fingerprint,
    load_conditioning_cache,
    load_execution_cache,
    save_conditioning_cache,
    save_execution_cache,
)


DEFAULT_CACHE_DIRECTORY = "E:/AI-Projects/H3/_hetero_test/gold/cache/"
DEFAULT_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
DEFAULT_TEXT_ENCODER_SHA256 = "BC2CED0FBEA64757FA9ACDDCCFC0B3F4819D1DCF1DA6C124D690D368BE283923"
DEFAULT_GOLD_PROMPT_PATH = "E:/AI-Projects/H3/_hetero_test/gold/gold_final_prompt.txt"
DEFAULT_GOLD_PROMPT_SHA256 = "62CCB107A37FA1D6EED26E5892D6BDDE2A3EBB0446CB294145997029EAD059B3"
DEFAULT_SETTINGS_JSON = """{
  "clip_type": "minimax",
  "clip_device": "cpu",
  "tokenize_call": "clip.tokenize(prompt, minimax_ref_items=ref_items)",
  "encode_call": "clip.encode_from_tokens_scheduled(tokens)",
  "comfyui_commit": "dec5d9450a5290bcf63430409ea41018e67f41c3",
  "conditioning_node": {
    "node_id": "MiniMaxH3ReferenceToVideo",
    "source_file": "comfy_extras/nodes_minimax_h3.py",
    "source_file_sha256": "5c3f78efef772850f90660c5ff82e31078b5baef10f427053ed82fb8b4a7ecd1"
  },
  "conditioning_geometry": {
    "width": 576,
    "height": 1024,
    "length": 124,
    "ref_image_size": "max"
  },
  "references": [
    {
      "slot": "ref_image_0",
      "type": "image",
      "file_name": "h3test_face_ref.png",
      "sha256": "885d9be9d35b6041959a2da1d544f83f95bfa1bc32ba57413532a687473e4562"
    }
  ],
  "reference_preprocessing": {
    "preprocess_node": "LayerUtility: ImageScaleByAspectRatio V2",
    "aspect_ratio": "original",
    "fit": "crop",
    "method": "lanczos",
    "round_to_multiple": "32",
    "scale_to_side": "longest",
    "scale_to_length": 1536,
    "background_color": "#000000"
  },
  "video_vae_model": "minimax_h3_video_vae_fp16.safetensors",
  "video_vae_file_sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
  "audio_vae_model": "minimax_h3_audio_vae_fp32.safetensors",
  "audio_vae_file_sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"
}"""


def _load_fixed_prompt(prompt_path, expected_sha256):
    path = Path(prompt_path).expanduser().resolve()
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"GOLD prompt must be UTF-8 without BOM: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"GOLD prompt is not valid UTF-8: {path}: {exc}") from exc
    # The text file's one final line terminator is a storage delimiter, not a
    # tokenizer input. Internal newlines are preserved byte-for-byte.
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    if text.endswith(("\r", "\n")):
        raise ValueError(f"GOLD prompt has more than one terminal line ending: {path}")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    expected = expected_sha256.strip().lower()
    if actual != expected:
        raise ValueError(f"GOLD prompt SHA256 mismatch: expected={expected}, actual={actual}, path={path}")
    return text, actual, path


def _identity_inputs():
    return {
        "cache_directory": ("STRING", {"default": DEFAULT_CACHE_DIRECTORY}),
        "gold_prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
        "text_encoder_model_name": ("STRING", {"default": DEFAULT_TEXT_ENCODER}),
        "text_encoder_file_sha256": ("STRING", {"default": DEFAULT_TEXT_ENCODER_SHA256}),
        "tokenize_encoder_settings_json": (
            "STRING",
            {"default": DEFAULT_SETTINGS_JSON, "multiline": True, "dynamicPrompts": False},
        ),
    }


class H3GoldFixedPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_path": ("STRING", {"default": DEFAULT_GOLD_PROMPT_PATH}),
                "expected_gold_prompt_sha256": ("STRING", {"default": DEFAULT_GOLD_PROMPT_SHA256}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("gold_prompt", "gold_prompt_sha256")
    FUNCTION = "load_prompt"
    CATEGORY = "H3/cache"
    DESCRIPTION = "Read the fixed UTF-8 GOLD final prompt and reject any byte-level content drift."

    @classmethod
    def IS_CHANGED(cls, prompt_path, expected_gold_prompt_sha256):
        try:
            _, actual, path = _load_fixed_prompt(prompt_path, expected_gold_prompt_sha256)
            return f"{actual}:{path.stat().st_mtime_ns}"
        except Exception as exc:
            return f"invalid:{type(exc).__name__}:{exc}"

    def load_prompt(self, prompt_path, expected_gold_prompt_sha256):
        prompt, actual, path = _load_fixed_prompt(prompt_path, expected_gold_prompt_sha256)
        print(f"[H3 GOLD Conditioning Cache] FIXED PROMPT sha256={actual} path={path}")
        return prompt, actual


class H3GoldConditioningCacheSave:
    @classmethod
    def INPUT_TYPES(cls):
        required = {"conditioning": ("CONDITIONING",)}
        required.update(_identity_inputs())
        required["overwrite"] = ("BOOLEAN", {"default": False})
        return {"required": required}

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "cache_key", "manifest_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3/cache/legacy"
    DESCRIPTION = "Legacy conditioning-only cache. Do not use for GOLD comparisons; use H3GoldExecutionCacheSave."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(
        self,
        conditioning,
        cache_directory,
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
        overwrite=False,
    ):
        key, manifest_path = save_conditioning_cache(
            conditioning,
            cache_directory,
            gold_prompt,
            text_encoder_model_name,
            text_encoder_file_sha256,
            tokenize_encoder_settings_json,
            overwrite=overwrite,
        )
        print(f"[H3 GOLD Conditioning Cache] SAVED key={key} manifest={manifest_path}")
        return conditioning, key, str(manifest_path)


class H3GoldConditioningCacheLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _identity_inputs()}

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "cache_key", "manifest_path")
    FUNCTION = "load"
    CATEGORY = "H3/cache/legacy"
    DESCRIPTION = "Legacy conditioning-only loader. Forbidden for GOLD comparisons; use H3GoldExecutionCacheLoad."

    @classmethod
    def IS_CHANGED(
        cls,
        cache_directory,
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    ):
        try:
            return cache_fingerprint(
                cache_directory,
                gold_prompt,
                text_encoder_model_name,
                text_encoder_file_sha256,
                tokenize_encoder_settings_json,
            )
        except Exception as exc:
            return f"invalid:{type(exc).__name__}:{exc}"

    def load(
        self,
        cache_directory,
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    ):
        conditioning, key, manifest_path = load_conditioning_cache(
            cache_directory,
            gold_prompt,
            text_encoder_model_name,
            text_encoder_file_sha256,
            tokenize_encoder_settings_json,
        )
        print(f"[H3 GOLD Conditioning Cache] HIT key={key} manifest={manifest_path}")
        return conditioning, key, str(manifest_path)


class H3GoldExecutionCacheSave:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "conditioning": ("CONDITIONING",),
            "latent": ("LATENT",),
        }
        required.update(_identity_inputs())
        required["overwrite"] = ("BOOLEAN", {"default": False})
        return {"required": required}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "latent", "cache_key", "manifest_path")
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "H3/cache"
    DESCRIPTION = "Persist the exact GOLD CONDITIONING and LATENT roots together as execution cache format v2."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(
        self,
        conditioning,
        latent,
        cache_directory,
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
        overwrite=False,
    ):
        key, manifest_path = save_execution_cache(
            conditioning,
            latent,
            cache_directory,
            gold_prompt,
            text_encoder_model_name,
            text_encoder_file_sha256,
            tokenize_encoder_settings_json,
            overwrite=overwrite,
        )
        print(f"[H3 GOLD Execution Cache] SAVED v2 key={key} manifest={manifest_path}")
        return conditioning, latent, key, str(manifest_path)


class H3GoldExecutionCacheLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _identity_inputs()}

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "latent", "cache_key", "manifest_path")
    FUNCTION = "load"
    CATEGORY = "H3/cache"
    DESCRIPTION = "Load exact GOLD CONDITIONING and LATENT roots from execution cache v2 without VLM/CLIP/VAE/reference inputs."

    @classmethod
    def IS_CHANGED(
        cls,
        cache_directory,
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    ):
        try:
            return execution_cache_fingerprint(
                cache_directory,
                gold_prompt,
                text_encoder_model_name,
                text_encoder_file_sha256,
                tokenize_encoder_settings_json,
            )
        except Exception as exc:
            return f"invalid:{type(exc).__name__}:{exc}"

    def load(
        self,
        cache_directory,
        gold_prompt,
        text_encoder_model_name,
        text_encoder_file_sha256,
        tokenize_encoder_settings_json,
    ):
        conditioning, latent, key, manifest_path = load_execution_cache(
            cache_directory,
            gold_prompt,
            text_encoder_model_name,
            text_encoder_file_sha256,
            tokenize_encoder_settings_json,
        )
        print(f"[H3 GOLD Execution Cache] HIT v2 key={key} manifest={manifest_path}")
        return conditioning, latent, key, str(manifest_path)


NODE_CLASS_MAPPINGS = {
    "H3GoldFixedPrompt": H3GoldFixedPrompt,
    "H3GoldConditioningCacheSave": H3GoldConditioningCacheSave,
    "H3GoldConditioningCacheLoad": H3GoldConditioningCacheLoad,
    "H3GoldExecutionCacheSave": H3GoldExecutionCacheSave,
    "H3GoldExecutionCacheLoad": H3GoldExecutionCacheLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3GoldFixedPrompt": "H3 GOLD Fixed Prompt",
    "H3GoldConditioningCacheSave": "H3 GOLD Conditioning Cache Save (Legacy)",
    "H3GoldConditioningCacheLoad": "H3 GOLD Conditioning Cache Load (Legacy)",
    "H3GoldExecutionCacheSave": "H3 GOLD Execution Cache Save",
    "H3GoldExecutionCacheLoad": "H3 GOLD Execution Cache Load",
}
