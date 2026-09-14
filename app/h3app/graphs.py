# -*- coding: utf-8 -*-
"""API-format graph builders.

Pure: no file I/O, no network, no clock. Everything the builders need is passed
in, so the whole module is testable from server.py --selftest without ComfyUI.

Three graphs are submitted per generation:
  A  profile   - reference sheet -> CHARACTER PROFILE (VLM)
  B  director  - Japanese request -> six-section English H3 prompt (VLM), then
                 unload + purge so llama.cpp's ~11 GiB is gone before H3 loads
  C  generate  - the measured H3_HETERO_V1 chain

Node ids are descriptive strings rather than numbers: the pipeline maps websocket
"executing" events back to user-facing stages by id.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from .config import FPS, MAX_PIXEL_AREA

Graph = dict[str, dict[str, Any]]
Link = list

# ------------------------------------------------------------------ stage ids --
# Graph C ids the pipeline needs to know about by name.
NODE_H3 = "h3"
NODE_SAMPLER = "sampler"
NODE_SAVE = "save_video"
NODE_SAVE_SEG = "save_video_seg"
NODE_PROFILE_PREVIEW = "preview"
NODE_DIRECTOR_PREVIEW = "preview"


class GraphError(RuntimeError):
    pass


# ------------------------------------------------------------------- helpers --
def frames_for_seconds(seconds: float) -> int:
    """H3 accepts frame counts where n % 17 == 5, trained range 124..362.

    5s -> 124 (the v1 fixed value), 10s -> 243, 15s -> 362.
    """
    n = round(FPS * float(seconds))
    n = n + (5 - n % 17) % 17
    return min(362, max(124, n))


def seconds_for_frames(frames: int) -> float:
    return round(frames / float(FPS), 3)


def validate_resolution(width: int, height: int) -> None:
    if width % 32 or height % 32:
        raise GraphError(f"解像度は32の倍数で指定してください: {width}x{height}")
    if width * height > MAX_PIXEL_AREA:
        raise GraphError(
            f"解像度が H3 の面積上限を超えています: {width}x{height} "
            f"({width * height} > {MAX_PIXEL_AREA})")


def _ref_chain(graph: Graph, images: list[str], ref_longest: list[int]) -> list[str]:
    """LoadImage -> ImageScaleByAspectRatio V2, one pair per reference slot.

    Widget values are the ones the measured graph used (node "3" of
    candidate_prompt.json); only scale_to_length varies per slot.
    """
    if not images:
        raise GraphError("参照画像が1枚もありません。少なくとも1枚必要です。")
    if len(images) > 4:
        raise GraphError("参照画像は最大4枚までです。")

    scale_ids: list[str] = []
    for i, name in enumerate(images):
        load_id, scale_id = f"load_{i}", f"scale_{i}"
        longest = ref_longest[i] if i < len(ref_longest) else ref_longest[-1]
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        graph[scale_id] = {
            "class_type": "LayerUtility: ImageScaleByAspectRatio V2",
            "inputs": {
                "aspect_ratio": "original",
                "proportional_width": 1,
                "proportional_height": 1,
                "fit": "crop",
                "method": "lanczos",
                "round_to_multiple": "32",
                "scale_to_side": "longest",
                "scale_to_length": int(longest),
                "background_color": "#000000",
                "image": [load_id, 0],
            },
        }
        scale_ids.append(scale_id)
    return scale_ids


def _sheet_chain(graph: Graph, scale_ids: list[str]) -> str:
    """Lay the (already scaled) references out as one contact sheet for the
    VLM: two columns, 4 px black gutters, using only the core ImageStitch
    node (comfy_extras/nodes_images.py). Previously "CR Image Grid Panel"
    (Comfyroll) did this; that pack publishes no licence, so the public
    release must not depend on it. Node ids: sheet_row0 / sheet_row1 / sheet
    (only the ones needed for the number of references are emitted).
    """
    def stitch(node_id: str, first: str, second: str, direction: str) -> str:
        graph[node_id] = {
            "class_type": "ImageStitch",
            "inputs": {
                "image1": [first, 0],
                "direction": direction,
                "match_image_size": True,
                "spacing_width": 4,
                "spacing_color": "black",
                "image2": [second, 0],
            },
        }
        return node_id

    n = len(scale_ids)
    if n == 1:
        return scale_ids[0]
    row0 = stitch("sheet_row0", scale_ids[0], scale_ids[1], "right")
    if n == 2:
        return row0
    row1 = (stitch("sheet_row1", scale_ids[2], scale_ids[3], "right")
            if n >= 4 else scale_ids[2])
    return stitch("sheet", row0, row1, "down")


def _vlm_loader(graph: Graph, vlm: dict) -> str:
    """Identical widgets in graph A and graph B so ComfyUI reuses the loaded model.

    vram_limit MUST stay -1. A partial cap forces a CPU/GPU layer split that trips
      GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)
    inside llama.cpp, which abort()s the whole ComfyUI process (measured).
    """
    graph["vlm"] = {
        "class_type": "llama_cpp_model_loader",
        "inputs": {
            "model": vlm["model"],
            "mmproj": vlm["mmproj"],
            "chat_handler": vlm["chat_handler"],
            "n_ctx": int(vlm.get("n_ctx", 16384)),
            "vram_limit": -1,
            "image_min_tokens": 0,
            "image_max_tokens": 1568,
        },
    }
    return "vlm"


# ------------------------------------------------------------------ graph A ---
def build_profile_graph(images: list[str], vlm: dict, ref_longest: list[int],
                        system_prompt: str, user_prompt: str) -> Graph:
    """Reference sheet -> CHARACTER PROFILE text."""
    graph: Graph = {}
    scale_ids = _ref_chain(graph, images, ref_longest)
    sheet = _sheet_chain(graph, scale_ids)
    _vlm_loader(graph, vlm)

    # Low temperature: the profile is a description, not a creative act.
    graph["params"] = {
        "class_type": "llama_cpp_parameters",
        "inputs": {
            "max_tokens": 1024, "top_k": 30, "top_p": 0.9, "min_p": 0.05,
            "typical_p": 1.0, "temperature": 0.35, "repeat_penalty": 1.05,
            "frequency_penalty": 0.0, "present_penalty": 0.0,
            "mirostat_mode": 0, "mirostat_eta": 0.1, "mirostat_tau": 5.0,
            "state_uid": -1,
        },
    }
    graph["instruct"] = {
        "class_type": "llama_cpp_instruct_adv",
        "inputs": {
            "llama_model": ["vlm", 0],
            "preset_prompt": "Empty - Nothing",
            "custom_prompt": user_prompt,
            "system_prompt": system_prompt,
            "inference_mode": "images",
            "max_frames": 8,
            "max_size": 1024,
            "seed": 1,                 # profile extraction is deterministic on purpose
            "force_offload": False,    # graph B reuses the same loaded model
            "save_states": False,
            "parameters": ["params", 0],
            "images": [sheet, 0],
        },
    }
    # PreviewAny is an OUTPUT_NODE returning {"ui": {"text": (value,)}}, so the
    # result is readable from /history without a custom node.
    graph["preview"] = {"class_type": "PreviewAny", "inputs": {"source": ["instruct", 0]}}
    return graph


# ------------------------------------------------------------------ graph B ---
def build_director_graph(images: list[str], vlm: dict, ref_longest: list[int],
                         system_prompt: str, user_prompt: str, seed: int) -> Graph:
    """Japanese request -> six-section English H3 prompt, then free the VLM VRAM."""
    graph: Graph = {}
    scale_ids = _ref_chain(graph, images, ref_longest)
    sheet = _sheet_chain(graph, scale_ids)
    _vlm_loader(graph, vlm)

    graph["params"] = {
        "class_type": "llama_cpp_parameters",
        "inputs": {
            "max_tokens": 3072, "top_k": 40, "top_p": 0.92, "min_p": 0.05,
            "typical_p": 1.0, "temperature": 0.7, "repeat_penalty": 1.05,
            "frequency_penalty": 0.0, "present_penalty": 0.0,
            "mirostat_mode": 0, "mirostat_eta": 0.1, "mirostat_tau": 5.0,
            "state_uid": -1,
        },
    }
    graph["instruct"] = {
        "class_type": "llama_cpp_instruct_adv",
        "inputs": {
            "llama_model": ["vlm", 0],
            "preset_prompt": "Empty - Nothing",
            "custom_prompt": user_prompt,
            "system_prompt": system_prompt,
            "inference_mode": "images",
            "max_frames": 8,
            # The director sees the sheet too, at a smaller size than the profile
            # pass: it only needs enough to keep <Subject 1> anchored.
            "max_size": 768,
            "seed": int(seed) & 0xFFFFFFFFFFFFFFFF,
            "force_offload": False,    # the explicit unload below does this instead
            "save_states": False,
            "parameters": ["params", 0],
            "images": [sheet, 0],
        },
    }
    graph["preview"] = {"class_type": "PreviewAny", "inputs": {"source": ["instruct", 0]}}

    # Unload + purge are wired as a DATA dependency off the instruct output, not
    # left dangling: ComfyUI is otherwise free to start loading the 25 GiB text
    # encoder while llama.cpp still holds ~11 GiB, which OOMs a 12 GiB card.
    graph["unload"] = {
        "class_type": "llama_cpp_unload_model",
        "inputs": {"any": ["instruct", 0]},
    }
    graph["purge"] = {
        "class_type": "LayerUtility: PurgeVRAM V2",
        "inputs": {"anything": ["unload", 0], "purge_cache": True, "purge_models": True},
    }
    return graph


# ------------------------------------------------- device restore probe ---
def build_device_restore_graph(
        label: str = "app-h3-restore") -> Graph:
    """Reset torch current_device to cuda:0 between director and generate.

    Overnight 2026-09-07 W1 root cause: the director graph's VLM cycle leaves
    torch current_device=cuda:1. The generate graph's VAELoader nodes execute
    BEFORE the in-graph H3CudaPhaseRestore and capture the polluted device,
    so the video VAE decodes on GPU1 (124f vae 61s -> 159s, worse at 362f).
    Submitting this tiny graph first makes generate immune. No pixels, no
    weights, no sampler state are touched - placement only.
    """
    return {
        "seed": {"class_type": "PrimitiveInt", "inputs": {"value": 0}},
        "restore": {
            "class_type": "H3CudaPhaseRestore",
            "inputs": {
                "anything": ["seed", 0],
                "target_gpu": 0,
                "label": label,
                "synchronize": True,
            },
        },
        # H3CudaPhaseRestore is not an OUTPUT_NODE; terminate on PreviewAny
        # (same pattern as the VLM-unload helper in comfy.py).
        "done": {"class_type": "PreviewAny", "inputs": {"source": ["restore", 0]}},
    }


# ------------------------------------------------------------------ graph C ---
def build_vlm_unload_graph() -> Graph:
    """Free the llama.cpp VLM + purge VRAM (same tail as graph B).

    Used by the AI Director handoff before video generation: Director calls
    keep the VLM resident, and this restores the exact pre-generate state the
    verified single/story flows rely on.
    """
    return {
        "seed": {"class_type": "PrimitiveInt", "inputs": {"value": 0}},
        "unload": {
            "class_type": "llama_cpp_unload_model",
            "inputs": {"any": ["seed", 0]},
        },
        "purge": {
            "class_type": "LayerUtility: PurgeVRAM V2",
            "inputs": {"anything": ["unload", 0], "purge_cache": True,
                       "purge_models": True},
        },
        # Mirror comfy.py: terminate on the unload output (purge still runs
        # as part of the prompt; verified pattern).
        "done": {"class_type": "PreviewAny",
                 "inputs": {"source": ["unload", 0]}},
    }


def build_generate_graph(images: list[str], en_prompt: str, settings: dict,
                         models: dict, ref_longest: list[int], output_prefix: str,
                         continuation: dict | None = None, *,
                         concat_previous: bool = True,
                         voice_master: str | None = None,
                         clip_device: str = "gpu:1") -> Graph:
    """The verified H3_HETERO_V1 chain.

    Modelled node-for-node on _hetero_test/opt_ref_match/candidate_prompt.json.
    Deliberate differences:
      - the fixed-prompt node is gone; the director's English text is passed as
        MiniMaxH3ReferenceToVideo.prompt directly
      - up to 4 reference images instead of 1
      - CLIPLoader -> H3CudaPhaseRestore + H3GatedCLIPLoader + device reports,
        because a VLM ran earlier in this same process and left
        torch.cuda.current_device() pointing at llama.cpp's GPU
      - optional Tail Relay inputs and previous-clip concatenation

    `continuation` (None for a fresh clip):
      {"video": <filename in comfy input>, "prev_total_frames": int,
       "tail_frames": int, "tail_audio": bool (optional, default True),
       "voice_master": <audio filename in comfy input> (optional),
       "last_frame_image": <image filename in comfy input> (optional)}

      "tail_frames" 0 = 0f relay: no tail-video nodes; needs last_frame_image.
      "voice_master" set = fixed voice as ref_audio_0; combining it with
      tail audio (ref_video_audios) is a GraphError.

      "tail_audio" False keeps the tail VIDEO conditioning and drops only the
      tail AUDIO conditioning. Nothing in the app passes False today; it exists so
      that a future "silent segments must not inherit the previous voice" is a
      one-line change rather than a graph rewrite.

    `concat_previous` (continuation only):
      True  - the verified single-shot behaviour. The previous clip is decoded,
              concatenated in-graph (ImageBatch + AudioConcat) and the MAIN save
              is the full video so far, with an extra `_seg` save of just the new
              segment.
      False - tail relay only. LoadVideo / GetVideoComponents / ImageFromBatch /
              TrimAudioDuration are still built (the tail still conditions the H3
              node), but nothing is concatenated and there is exactly ONE
              SaveVideo holding just the new segment. Story mode needs this:
              concatenating N clips of 124 frames at 576x1024 in-graph would blow
              up host RAM by segment 5.
    """
    width = int(settings["width"])
    height = int(settings["height"])
    frames = int(settings["frames"])
    validate_resolution(width, height)

    graph: Graph = {}
    scale_ids = _ref_chain(graph, images, ref_longest)

    # ---- CUDA phase barrier ------------------------------------------------
    # ComfyUI captures the text encoder's base load/offload device from
    # torch.cuda.current_device() at loader time, and SelectCLIPDevice retargets
    # relative to that base. Hanging the loader off the restore is what orders
    # them; a dangling reset node would not be guaranteed to run first.
    graph["restore"] = {
        "class_type": "H3CudaPhaseRestore",
        "inputs": {
            "anything": [scale_ids[0], 0],
            "target_gpu": 0,
            "label": "app-h3-restore",
            "synchronize": True,
        },
    }
    graph["clip"] = {
        "class_type": "H3GatedCLIPLoader",
        "inputs": {
            "barrier": ["restore", 0],
            "clip_name": models["clip"],
            "type": "minimax",
            "device": "default",     # SelectCLIPDevice below does the retargeting
        },
    }
    graph["clip_report_base"] = {
        "class_type": "H3ClipDeviceReport",
        "inputs": {"clip": ["clip", 0], "label": "clip-base"},
    }
    # THE heterogeneous-GPU mechanism (ComfyUI core, comfy_extras/nodes_multigpu.py).
    # Removing it does not just change speed - the 217.61 s profile stops existing.
    graph["clip_select"] = {
        "class_type": "SelectCLIPDevice",
        "inputs": {"clip": ["clip_report_base", 0], "device": clip_device},
    }
    graph["clip_report_target"] = {
        "class_type": "H3ClipDeviceReport",
        "inputs": {"clip": ["clip_select", 0], "label": "clip-target"},
    }

    # ---- models ------------------------------------------------------------
    graph["vae_video"] = {"class_type": "VAELoader", "inputs": {"vae_name": models["vae_video"]}}
    graph["vae_audio"] = {"class_type": "VAELoader", "inputs": {"vae_name": models["vae_audio"]}}
    graph["unet"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": models["unet"], "weight_dtype": "default"},
    }
    # v1 parity: the RAW turbo LoRA whose 518 keys fail to bind here. Kept, at
    # strength 1.0, because that is the state the measurement was taken in.
    graph["lora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "lora_name": models["lora"],
            "strength_model": float(models.get("lora_strength", 1.0)),
            "model": ["unet", 0],
        },
    }

    # ---- H3 reference to video --------------------------------------------
    h3_inputs: dict[str, Any] = {
        "clip": ["clip_report_target", 0],
        "vae": ["vae_video", 0],
        "audio_vae": ["vae_audio", 0],
        "prompt": en_prompt,
        "width": width,
        "height": height,
        "length": frames,
        "ref_image_size": settings["ref_image_size"],
    }
    # Every Character Master reference stays connected, including in continuation
    # mode: reference count was not what caused the GPU1 OOM, and relying on
    # <Video 1> alone drifts the identity within a few clips.
    for i, sid in enumerate(scale_ids):
        h3_inputs[f"ref_images.ref_image_{i}"] = [sid, 0]

    if continuation:
        tail = int(continuation["tail_frames"])
        prev_total = int(continuation.get("prev_total_frames", 0))
        # Continuation-carried VM wins; otherwise the fresh-clip VM applies.
        voice_master = continuation.get("voice_master") or voice_master
        if tail > 0:
            graph["prev_video"] = {
                "class_type": "LoadVideo",
                "inputs": {"file": continuation["video"]},
            }
            graph["prev_components"] = {
                "class_type": "GetVideoComponents",
                "inputs": {"video": ["prev_video", 0]},
            }
            # batch_index / start_index are computed here because the app already
            # knows prev_total_frames. No GetImageSizeAndCount / MathExpression nodes.
            graph["tail_images"] = {
                "class_type": "ImageFromBatch",
                "inputs": {
                    "image": ["prev_components", 0],
                    "batch_index": max(0, prev_total - tail),
                    "length": tail,
                },
            }
            h3_inputs["ref_videos.ref_video_0"] = ["tail_images", 0]
        elif continuation.get("last_frame_image"):
            # 0f relay: no tail video nodes at all; the previous last frame
            # rides as one extra reference image (motion anchor, no audio path).
            n_ref = len(scale_ids)
            if n_ref >= 9:
                raise GraphError("参照画像が多すぎます（last frame併用で上限9枚）。")
            graph["last_frame"] = {
                "class_type": "LoadImage",
                "inputs": {"image": continuation["last_frame_image"]},
            }
            h3_inputs[f"ref_images.ref_image_{n_ref}"] = ["last_frame", 0]
        # `tail_audio` defaults to True everywhere and is threaded through from the
        # story runner, so a future "silent segments must not inherit the previous
        # voice" is one flag - not a graph rewrite. LONG_FAST sets it False.
        if continuation.get("tail_audio", True):
            if tail <= 0:
                raise GraphError("tail_audio=True に tail video（0f）は組み合わせられません。")
            graph["tail_audio"] = {
                "class_type": "TrimAudioDuration",
                "inputs": {
                    "audio": ["prev_components", 1],
                    "start_index": -(tail / float(FPS)),   # negative = from the end
                    "duration": tail / float(FPS),
                },
            }
            h3_inputs["ref_video_audios.ref_video_audio_0"] = ["tail_audio", 0]
        # Voice Master: fixed external voice as <Audio 1>. NEVER combined with
        # the tail soundtrack - that double-audio path is a GraphError, not a
        # silent choice.
        if voice_master:
            if "ref_video_audios.ref_video_audio_0" in h3_inputs:
                raise GraphError(
                    "VOICE_MASTER_CONFLICT: ref_video_audios が接続されたままです。"
                    "Voice Master 使用時は tail audio を外してください。")
            graph["voice_master"] = {
                "class_type": "LoadAudio",
                "inputs": {"audio": voice_master},
            }
            h3_inputs["ref_audios.ref_audio_0"] = ["voice_master", 0]

    # Fresh-clip external Voice Master (no continuation): same <Audio 1> slot.
    if not continuation and voice_master:
        graph["voice_master"] = {
            "class_type": "LoadAudio",
            "inputs": {"audio": voice_master},
        }
        h3_inputs["ref_audios.ref_audio_0"] = ["voice_master", 0]

    graph[NODE_H3] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": h3_inputs}

    # ---- sampling ----------------------------------------------------------
    graph["noise"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(settings["seed"])}}
    graph["guider"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": ["lora", 0], "conditioning": [NODE_H3, 0]},
    }
    graph["sampler_select"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": settings["sampler"]},
    }
    graph["scheduler"] = {
        "class_type": "BasicScheduler",
        "inputs": {
            "scheduler": settings["scheduler"],
            "steps": int(settings["steps"]),
            "denoise": 1.0,
            "model": ["lora", 0],
        },
    }
    graph[NODE_SAMPLER] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["noise", 0],
            "guider": ["guider", 0],
            "sampler": ["sampler_select", 0],
            "sigmas": ["scheduler", 0],
            "latent_image": [NODE_H3, 1],
        },
    }
    graph["decode_video"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [NODE_SAMPLER, 0], "vae": ["vae_video", 0]},
    }
    graph["decode_audio"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": [NODE_SAMPLER, 0], "vae": ["vae_audio", 0]},
    }

    # ---- output ------------------------------------------------------------
    if continuation and concat_previous:
        # The saved main file is the FULL video so far, so "continue" can always
        # feed the previous result straight back in.
        graph["concat_images"] = {
            "class_type": "ImageBatch",
            "inputs": {"image1": ["prev_components", 0], "image2": ["decode_video", 0]},
        }
        graph["concat_audio"] = {
            "class_type": "AudioConcat",
            "inputs": {
                "audio1": ["prev_components", 1],
                "audio2": ["decode_audio", 0],
                "direction": "after",
            },
        }
        video_src = ["concat_images", 0]
        audio_src = ["concat_audio", 0]
    else:
        video_src = ["decode_video", 0]
        audio_src = ["decode_audio", 0]

    graph["create_video"] = {
        "class_type": "CreateVideo",
        "inputs": {"fps": float(FPS), "bit_depth": 8, "images": video_src, "audio": audio_src},
    }
    graph[NODE_SAVE] = {
        "class_type": "SaveVideo",
        "inputs": {
            "filename_prefix": output_prefix,
            "format": "auto",
            "codec": "auto",
            "video": ["create_video", 0],
        },
    }

    if continuation and concat_previous:
        # Also save the new segment on its own: if the concat fails or produces a
        # bad file, the 5 seconds that cost 3.6 minutes of GPU time still exist.
        graph["create_video_seg"] = {
            "class_type": "CreateVideo",
            "inputs": {
                "fps": float(FPS), "bit_depth": 8,
                "images": ["decode_video", 0], "audio": ["decode_audio", 0],
            },
        }
        graph[NODE_SAVE_SEG] = {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": output_prefix + "_seg",
                "format": "auto",
                "codec": "auto",
                "video": ["create_video_seg", 0],
            },
        }

    return graph


# ------------------------------------------------------------ parity gate ----
def _find(graph: Graph, class_type: str) -> list[tuple[str, dict]]:
    return [(nid, n) for nid, n in graph.items() if n.get("class_type") == class_type]


def assert_v1_parity(graph: Graph, settings: dict, models: dict,
                     manifest: dict, defaults: dict | None = None, *,
                     clip_device: str = "gpu:1") -> None:
    """Hard gate run before every generate submit.

    Raises GraphError unless the graph is still the H3_HETERO_V1 chain. When the
    user has not moved anything off the shipped defaults, the fixed values are
    compared against H3_HETERO_V1_MANIFEST.json itself rather than numbers typed
    in here, so this check cannot drift away from the frozen profile.
    """
    h3 = _find(graph, "MiniMaxH3ReferenceToVideo")
    if len(h3) != 1:
        raise GraphError("v1整合性: MiniMaxH3ReferenceToVideo が1個ではありません。")
    h3_inputs = h3[0][1]["inputs"]
    if h3_inputs.get("ref_image_size") != settings["ref_image_size"]:
        raise GraphError(
            "v1整合性: ref_image_size がグラフと設定で一致しません "
            f"({h3_inputs.get('ref_image_size')} != {settings['ref_image_size']})")

    sel = _find(graph, "SelectCLIPDevice")
    if len(sel) != 1 or sel[0][1]["inputs"].get("device") != clip_device:
        raise GraphError(
            f"v1整合性: SelectCLIPDevice(device={clip_device}) がありません。"
            "GPU1 conditioning を外すと 217.61 秒の構成ではなくなります。")

    gated = _find(graph, "H3GatedCLIPLoader")
    if len(gated) != 1:
        raise GraphError("v1整合性: CLIP ローダーが H3GatedCLIPLoader ではありません。")
    barrier = gated[0][1]["inputs"].get("barrier")
    if not isinstance(barrier, list) or len(barrier) != 2:
        raise GraphError("v1整合性: H3GatedCLIPLoader の barrier が接続されていません。")
    src = graph.get(barrier[0], {})
    if src.get("class_type") != "H3CudaPhaseRestore":
        raise GraphError("v1整合性: barrier の供給元が H3CudaPhaseRestore ではありません。")
    if int(src["inputs"].get("target_gpu", -1)) != 0:
        raise GraphError("v1整合性: H3CudaPhaseRestore.target_gpu が 0 ではありません。")
    if gated[0][1]["inputs"].get("clip_name") != models["clip"]:
        raise GraphError("v1整合性: text encoder のファイル名が設定と一致しません。")

    unet = _find(graph, "UNETLoader")
    if len(unet) != 1 or unet[0][1]["inputs"].get("unet_name") != models["unet"]:
        raise GraphError("v1整合性: UNET のファイル名が設定と一致しません。")
    lora = _find(graph, "LoraLoaderModelOnly")
    if len(lora) != 1 or lora[0][1]["inputs"].get("lora_name") != models["lora"]:
        raise GraphError("v1整合性: LoRA のファイル名が設定と一致しません。")
    vaes = {n["inputs"].get("vae_name") for _, n in _find(graph, "VAELoader")}
    if vaes != {models["vae_video"], models["vae_audio"]}:
        raise GraphError("v1整合性: VAE のファイル名が設定と一致しません。")

    ks = _find(graph, "KSamplerSelect")
    if len(ks) != 1 or ks[0][1]["inputs"].get("sampler_name") != settings["sampler"]:
        raise GraphError("v1整合性: sampler が設定と一致しません。")
    sch = _find(graph, "BasicScheduler")
    if len(sch) != 1:
        raise GraphError("v1整合性: BasicScheduler が1個ではありません。")
    sch_inputs = sch[0][1]["inputs"]
    if sch_inputs.get("scheduler") != settings["scheduler"]:
        raise GraphError("v1整合性: scheduler が設定と一致しません。")
    if int(sch_inputs.get("steps", -1)) != int(settings["steps"]):
        raise GraphError("v1整合性: steps が設定と一致しません。")
    if float(sch_inputs.get("denoise", -1)) != 1.0:
        raise GraphError("v1整合性: denoise が 1.0 ではありません。")

    noise = _find(graph, "RandomNoise")
    if len(noise) != 1 or int(noise[0][1]["inputs"].get("noise_seed", -1)) != int(settings["seed"]):
        raise GraphError("v1整合性: seed が設定と一致しません。")
    for key in ("width", "height"):
        if int(h3_inputs.get(key, -1)) != int(settings[key]):
            raise GraphError(f"v1整合性: {key} が設定と一致しません。")
    if int(h3_inputs.get("length", -1)) != int(settings["frames"]):
        raise GraphError("v1整合性: frames が設定と一致しません。")

    # Only when nothing has been moved off the shipped defaults does the manifest
    # apply. A user who chose 10 seconds is knowingly outside the measured point.
    if defaults is not None and _is_default_profile(settings, defaults):
        fixed = manifest["fixed_generation_settings"]
        expected = {
            "width": int(fixed["width"]),
            "height": int(fixed["height"]),
            "frames": int(fixed["frames"]),
            "steps": int(fixed["steps"]),
            "seed": int(fixed["seed"]),
            "sampler": fixed["sampler"],
            "scheduler": fixed["scheduler"],
            "ref_image_size": fixed["ref_image_size"],
        }
        actual = {
            "width": int(h3_inputs["width"]),
            "height": int(h3_inputs["height"]),
            "frames": int(h3_inputs["length"]),
            "steps": int(sch_inputs["steps"]),
            "seed": int(noise[0][1]["inputs"]["noise_seed"]),
            "sampler": ks[0][1]["inputs"]["sampler_name"],
            "scheduler": sch_inputs["scheduler"],
            "ref_image_size": h3_inputs["ref_image_size"],
        }
        bad = {k: (actual[k], v) for k, v in expected.items() if actual[k] != v}
        if bad:
            detail = ", ".join(f"{k}: {a} != manifest {e}" for k, (a, e) in sorted(bad.items()))
            raise GraphError(f"v1整合性: 既定値のはずの設定がマニフェストと一致しません ({detail})")
        for key, model_key in (("model", "unet"), ("text_encoder", "clip"), ("turbo_lora", "lora")):
            if fixed[key] != models[model_key]:
                raise GraphError(
                    f"v1整合性: {key} がマニフェストと一致しません "
                    f"({models[model_key]} != {fixed[key]})")


_REF_LABEL_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>")


def assert_reference_labels(graph: Graph, en_prompt: str) -> None:
    """Phase 1 item 6: no phantom <Picture N>/<Video N>/<Audio N> labels.

    The MiniMax H3 tokenizer numbers references strictly by the order they
    are actually connected to MiniMaxH3ReferenceToVideo (see
    comfy_extras/nodes_minimax_h3.py execute() and
    comfy/text_encoders/minimax.py tokenize_with_weights): every
    ref_images.ref_image_i becomes <Picture i+1>, in order; every
    ref_videos.ref_video_k becomes <Video k+1>; and <Audio j> ordinals are
    consumed in ref_items order - a video's own paired soundtrack
    (ref_video_audios with the same index) is counted first (before its
    <Video k> label), then any standalone ref_audios entries.

    Extra CONNECTED references that the prompt never mentions are fine
    (harmless slack). The reverse - the prompt naming a reference that is
    not actually wired up - is exactly the LONG_FAST "<Video 1>"/"soundtrack
    of <Video 1>" bug this gate exists to catch, so it is a hard GraphError.
    """
    h3_nodes = _find(graph, "MiniMaxH3ReferenceToVideo")
    if len(h3_nodes) != 1:
        raise GraphError(
            "参照ラベル検証: MiniMaxH3ReferenceToVideo が1個ではありません。")
    inputs = h3_nodes[0][1].get("inputs", {})

    def _count(prefix: str) -> int:
        return len([k for k in inputs if k.startswith(prefix) and
                   inputs.get(k) is not None])

    n_pictures = _count("ref_images.ref_image_")
    n_videos = _count("ref_videos.ref_video_")
    n_video_audio = _count("ref_video_audios.ref_video_audio_")
    n_standalone_audio = _count("ref_audios.ref_audio_")
    n_audios = n_video_audio + n_standalone_audio

    available = {"Picture": n_pictures, "Video": n_videos, "Audio": n_audios}
    used: dict[str, set[int]] = {"Picture": set(), "Video": set(), "Audio": set()}
    for kind, num in _REF_LABEL_RE.findall(str(en_prompt or "")):
        used[kind].add(int(num))

    problems: list[str] = []
    for kind, ordinals in used.items():
        limit = available[kind]
        bad = sorted(n for n in ordinals if n < 1 or n > limit)
        if bad:
            problems.append(
                f"<{kind} {'/'.join(str(b) for b in bad)}>"
                f"（接続数 {limit}）")
    if problems:
        raise GraphError(
            "参照ラベル検証: プロンプトが実際には接続されていない参照を"
            "名指ししています: " + ", ".join(problems))


def _is_default_profile(settings: dict, defaults: dict) -> bool:
    keys = ("width", "height", "frames", "steps", "seed", "sampler", "scheduler",
            "ref_image_size")
    for k in keys:
        if k not in defaults:
            return False
        if str(settings.get(k)) != str(defaults[k]):
            return False
    return True


def default_settings(defaults: dict) -> dict:
    """The shipped v1 profile as a settings dict."""
    s = {k: copy.deepcopy(v) for k, v in defaults.items()}
    s["frames"] = int(defaults["frames"])
    return s
