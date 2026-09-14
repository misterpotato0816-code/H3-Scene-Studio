# -*- coding: utf-8 -*-
"""Build H3_phase1_core.json and H3_phase1_full.json.

Run:  python build_phase1.py <object_info.json> <out_dir>
"""
import sys, os, io, json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wfbuild import Schema, Workflow          # noqa: E402
import prompts as P                            # noqa: E402

# ---------------------------------------------------------------- constants --
BYPASS = 4

GREEN = ("#232", "#353")
RED = ("#322", "#533")
BLUE = ("#223", "#335")
PALE = ("#2a363b", "#3f5159")
YELLOW = ("#432", "#653")
PURPLE = ("#323", "#535")
TEAL = ("#233", "#355")

# H3 weights. Names/sizes verified against the HF API - docs/05_models_verification.md.
# Primary is the pruned int8_convrot safetensors: it matches the article workflow's
# download metadata, and pairs with the *_pruned_comfyui turbo LoRA.
M_UNET_INT8 = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
M_UNET_GGUF = "MiniMax-H3-Ref2VA-Q4_K_S.gguf"
M_CLIP_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
M_CLIP_GGUF = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
M_CLIP_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"   # installed
M_VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"           # installed
M_VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"           # installed
# The raw larryvrh file (minimax_h3_turbo_v4_step600_ema.safetensors) uses bare
# "blocks.*" keys. In THIS environment ComfyUI's LoRA mapper expects the
# "diffusion_model." prefix, and all 518/518 keys failed to bind (measured).
# The *_pruned_comfyui build is the same LoRA with that prefix applied.
M_LORA_T1 = "minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors"
M_LORA_T2 = "minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors"
# H3_HETERO_V1 fixes the RAW file above. Its 518/518 keys fail to bind, so the
# measured 217.61 s build effectively ran WITHOUT the turbo LoRA. The Stable
# preset keeps that exact state on purpose; do not "fix" it here.
M_LORA_V1_RAW = "minimax_h3_turbo_v4_step600_ema.safetensors"
M_VLM = "gemma-4-E4B-it-ultra-uncensored-heretic-Q8_0.gguf"     # installed
M_VLM_MMPROJ = "gemma-4-E4B-it-mmproj-BF16.gguf"                # installed
M_UPSCALE = "RealESRGAN_x4plus.pth"                             # installed

FPS = 24
# H3_HETERO_V1 fixed profile: 576x1024 / 124 frames. DEF_SECONDS is chosen so
# FRAME_EXPR lands exactly on 124 (24 * 5.0 = 120 -> +4 -> 124).
DEF_W, DEF_H = 576, 1024
DEF_SECONDS = 5.0
DEF_TAIL = 56          # 56 % 17 == 5 -> survives H3's ref-video truncation intact

# Frames must satisfy n % 17 == 5 (see docs). Round up onto the grid, clamp to
# the documented training range, then let H3 do its own final alignment.
FRAME_EXPR = "min(362, max(124, round(a*b) + (5 - round(a*b) % 17) % 17))"


def col(x, y0=0, step=0):
    """Tiny helper: returns a cursor object for laying out a column."""
    class C:
        def __init__(self):
            self.x, self.y = x, y0

        def at(self, h=0, gap=40):
            p = (self.x, self.y)
            self.y += h + gap
            return p
    return C()


def build(sch, full):
    wf = Workflow(sch)
    G = {}   # group name -> node list

    def g(name, node):
        G.setdefault(name, []).append(node)
        return node

    # =====================================================================
    # 00  README
    # =====================================================================
    c = col(-700, 0)
    readme = wf.add("MarkdownNote", c.at(760, 60), size=(560, 760),
                    title="00 ここから読む / START HERE",
                    color=GREEN[0], bgcolor=GREEN[1],
                    widgets={"md": README_MD_FULL if full else README_MD_CORE})
    g("00 README", readme)

    # =====================================================================
    # 10  Character Master
    # =====================================================================
    c = col(0, 0)
    cast_defs = [
        ("① 顔寄り / FACE", 1536),
        ("② 上半身 / BUST", 1152),
        ("③ 全身 / FULL BODY", 1152),
        ("④ 衣装・小物 / OUTFIT", 896),
    ]
    loads, scales = [], []
    for i, (title, longest) in enumerate(cast_defs):
        ld = wf.add("LoadImage", c.at(320), size=(320, 320), title=title,
                    color=PALE[0], bgcolor=PALE[1],
                    widgets={"image": "example.png"},
                    mode=BYPASS if i >= 2 else 0)
        sc = wf.add("LayerUtility: ImageScaleByAspectRatio V2", c.at(220), size=(320, 220),
                    title="resize (longest %d px)" % longest,
                    widgets={"aspect_ratio": "original", "fit": "crop",
                             "method": "lanczos", "round_to_multiple": "32",
                             "scale_to_side": "longest", "scale_to_length": longest,
                             "background_color": "#000000"},
                    mode=BYPASS if i >= 2 else 0)
        wf.link(ld, "IMAGE", sc, "image")
        loads.append(ld)
        scales.append(sc)
        g("10 Character Master", ld)
        g("10 Character Master", sc)

    c2 = col(500, 0)
    batch = wf.add("BatchImagesNode", c2.at(160), size=(300, 160),
                   title="参照をバッチ化",
                   autogrow=[{"label": "image%d" % i, "name": "images.image%d" % i,
                              "type": "IMAGE"} for i in range(4)])
    for i, sc in enumerate(scales):
        wf.link(sc, "image", batch, "images.image%d" % i)
    grid = wf.add("CR Image Grid Panel", c2.at(220), size=(300, 220),
                  title="VLMに見せるキャラシート",
                  widgets={"max_columns": 2, "border_thickness": 4,
                           "border_color": "custom", "border_color_hex": "#000000"})
    wf.link(batch, "IMAGE", grid, "images")
    prev_grid = wf.add("PreviewImage", c2.at(300), size=(300, 300),
                       title="シート確認")
    wf.link(grid, "image", prev_grid, "images")
    for n in (batch, grid, prev_grid):
        g("10 Character Master", n)

    # =====================================================================
    # 20  Character Profile (VLM extraction)
    # =====================================================================
    c = col(900, 0)
    vlm = wf.add("llama_cpp_model_loader", c.at(210), size=(400, 210),
                 title="VLM (Gemma-4)", color=PURPLE[0], bgcolor=PURPLE[1],
                 # vram_limit = -1 (use whatever fits). Do NOT cap this: a partial
                 # cap forces a CPU/GPU layer split that trips
                 #   GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS)
                 # inside llama.cpp, which abort()s the whole ComfyUI process
                 # (measured). Device isolation is handled by launching ComfyUI
                 # with --cuda-device 0, not by this widget.
                 widgets={"model": M_VLM, "mmproj": M_VLM_MMPROJ,
                          "chat_handler": "Gemma4", "n_ctx": 16384,
                          "vram_limit": -1, "image_min_tokens": 0,
                          "image_max_tokens": 1568})
    par_prof = wf.add("llama_cpp_parameters", c.at(360), size=(400, 360),
                      title="params: profile (low temp)",
                      widgets={"max_tokens": 1024, "top_k": 30, "top_p": 0.9,
                               "min_p": 0.05, "typical_p": 1.0, "temperature": 0.35,
                               "repeat_penalty": 1.05, "mirostat_mode": 0,
                               "state_uid": -1})
    sys_prof = wf.add("ttN text", c.at(320), size=(400, 320),
                      title="SYS: Character Profile 抽出",
                      widgets={"text": P.SYS_CHARACTER_PROFILE})
    run_prof = wf.add("llama_cpp_instruct_adv", c.at(400), size=(400, 400),
                      title="① Character Profile を抽出",
                      color=PURPLE[0], bgcolor=PURPLE[1],
                      widgets={"preset_prompt": "Empty - Nothing",
                               "custom_prompt": P.USER_CHARACTER_PROFILE,
                               "inference_mode": "images", "max_frames": 8,
                               "max_size": 1024, "seed": 1,
                               "__control_after_generate__": "fixed",
                               "force_offload": True, "save_states": False})
    wf.link(vlm, "llama_model", run_prof, "llama_model")
    wf.link(par_prof, "parameters", run_prof, "parameters")
    wf.link(grid, "image", run_prof, "images")
    wf.wlink(sys_prof, "text", run_prof, "system_prompt")

    show_prof = wf.add("ShowText|pysssss", c.at(320), size=(400, 320),
                       title="Character Profile (ここで直接編集可)",
                       color=YELLOW[0], bgcolor=YELLOW[1])
    wf.link(run_prof, "output", show_prof, "text")
    man_prof = wf.add("ttN text", c.at(240), size=(400, 240), mode=BYPASS,
                      title="Profile 手動上書き (使うなら Ctrl+B で有効化)",
                      widgets={"text": ""})
    sw_prof = wf.add("Any Switch (rgthree)", c.at(80), size=(400, 80),
                     title="Profile: 手動 > 自動",
                     manual_inputs=[{"name": "any_01", "type": "*", "link": None},
                                    {"name": "any_02", "type": "*", "link": None}],
                     manual_outputs=[{"name": "*", "type": "*", "links": []}])
    wf.link(man_prof, "text", sw_prof, "any_01")
    wf.link(show_prof, "STRING", sw_prof, "any_02")
    for n in (vlm, par_prof, sys_prof, run_prof, show_prof, man_prof, sw_prof):
        g("20 Character Profile", n)

    # =====================================================================
    # 30  Japanese director input
    # =====================================================================
    c = col(1400, 0)
    jp_scene = wf.add("ttN text", c.at(300), size=(430, 300),
                      title="① シーン指示（日本語）",
                      color=GREEN[0], bgcolor=GREEN[1],
                      widgets={"text": P.JP_SCENE_DEFAULT})
    jp_dlg = wf.add("ttN text", c.at(280), size=(430, 280),
                    title="② セリフ確定（1行1発話 / 空ならAIが創作）",
                    color=GREEN[0], bgcolor=GREEN[1],
                    widgets={"text": P.JP_DIALOGUE_DEFAULT})
    jp_prev = wf.add("ttN text", c.at(220), size=(430, 220),
                     title="③ 前クリップ情報（続き生成時のみ書き換え）",
                     widgets={"text": P.JP_CONTINUATION_DEFAULT})
    jp_neg = wf.add("ttN text", c.at(260), size=(430, 260),
                    title="④ 避けたい要素",
                    widgets={"text": P.NEGATIVE_DEFAULT})
    for n in (jp_scene, jp_dlg, jp_prev, jp_neg):
        g("30 日本語入力", n)

    # length maths
    sec = wf.add("PrimitiveFloat", c.at(80), size=(430, 80),
                 title="尺（秒） 5.2 ～ 15.0 推奨",
                 color=TEAL[0], bgcolor=TEAL[1], widgets={"value": DEF_SECONDS})
    fps_n = wf.add("PrimitiveInt", c.at(80), size=(430, 80), title="fps (H3 は 24 固定)",
                   widgets={"value": FPS})
    frames = wf.add("MathExpression|pysssss", c.at(140), size=(430, 140),
                    title="フレーム数 (n %% 17 == 5 に揃える)",
                    color=TEAL[0], bgcolor=TEAL[1],
                    widgets={"expression": FRAME_EXPR})
    wf.link(fps_n, "INT", frames, "a")
    wf.link(sec, "FLOAT", frames, "b")
    disp = wf.add("Display Any (rgthree)", c.at(90), size=(430, 90),
                  title="→ 実際のフレーム数")
    wf.link(frames, "INT", disp, "source")
    for n in (sec, fps_n, frames, disp):
        g("30 日本語入力", n)

    # =====================================================================
    # 40  Director prompt (JP -> H3 six-section EN)
    # =====================================================================
    c = col(1950, 0)
    frames_s = wf.add("CR Integer To String", c.at(80), size=(380, 80),
                      title="frames → string")
    wf.link(frames, "INT", frames_s, "int_")

    lit = {}
    for key, text, h in (("prof", P.HDR_PROFILE, 80), ("scene", P.HDR_SCENE, 80),
                         ("dlg", P.HDR_DIALOGUE, 130), ("dlgEnd", P.HDR_DIALOGUE_END, 90),
                         ("prev", P.HDR_PREV, 80),
                         ("neg", P.HDR_NEG, 80), ("lenA", P.HDR_LEN_A, 90),
                         ("lenB", P.HDR_LEN_B, 110)):
        lit[key] = wf.add("ttN text", c.at(h), size=(380, h),
                          title="hdr:%s" % key, collapsed=True,
                          widgets={"text": text})
        g("40 H3 Director Prompt", lit[key])

    g("40 H3 Director Prompt", frames_s)

    # KJNodes' JoinStringMulti rebuilds its dynamic string_N inputs on load, which
    # silently drops hand-declared slots. Only string_1/string_2 are schema-declared
    # and therefore stable, so fold the pieces with a chain of 2-input joins.
    # Dialogue goes LAST - see prompts.HDR_DIALOGUE for why.
    pieces = [(lit["prof"], "text"), (sw_prof, "*"),
              (lit["scene"], "text"), (jp_scene, "text"),
              (lit["prev"], "text"), (jp_prev, "text"),
              (lit["neg"], "text"), (jp_neg, "text"),
              (lit["lenA"], "text"), (frames_s, "STRING"),
              (lit["lenB"], "text"),
              (lit["dlg"], "text"), (jp_dlg, "text"), (lit["dlgEnd"], "text")]
    acc, acc_out = pieces[0]
    for idx, (src, out) in enumerate(pieces[1:], start=1):
        j = wf.add("JoinStringMulti", c.at(60), size=(380, 60), collapsed=True,
                   title="join %02d" % idx,
                   widgets={"inputcount": 2, "delimiter": "\n",
                            "return_list": False})
        wf.link(acc, acc_out, j, "string_1")
        wf.link(src, out, j, "string_2")
        g("40 H3 Director Prompt", j)
        acc, acc_out = j, "string"
    join2, join2_out = acc, acc_out

    c = col(2450, 0)
    sys_new = wf.add("ttN text", c.at(360), size=(430, 360),
                     title="SYS-A: 新規クリップ",
                     color=BLUE[0], bgcolor=BLUE[1],
                     widgets={"text": P.SYS_DIRECTOR_NEW})
    sys_cont = wf.add("ttN text", c.at(360), size=(430, 360), mode=BYPASS,
                      title="SYS-B: 続き生成（Ctrl+B で有効化）",
                      color=RED[0], bgcolor=RED[1],
                      widgets={"text": P.SYS_DIRECTOR_CONTINUATION})
    sw_sys = wf.add("Any Switch (rgthree)", c.at(80), size=(430, 80),
                    title="MODE: 続き > 新規",
                    manual_inputs=[{"name": "any_01", "type": "*", "link": None},
                                   {"name": "any_02", "type": "*", "link": None}],
                    manual_outputs=[{"name": "*", "type": "*", "links": []}])
    wf.link(sys_cont, "text", sw_sys, "any_01")
    wf.link(sys_new, "text", sw_sys, "any_02")

    par_dir = wf.add("llama_cpp_parameters", c.at(360), size=(430, 360),
                     title="params: director",
                     widgets={"max_tokens": 3072, "top_k": 40, "top_p": 0.92,
                              "min_p": 0.05, "typical_p": 1.0, "temperature": 0.7,
                              "repeat_penalty": 1.05, "mirostat_mode": 0,
                              "state_uid": -1})
    run_dir = wf.add("llama_cpp_instruct_adv", c.at(400), size=(430, 400),
                     title="② H3 プロンプト生成",
                     color=PURPLE[0], bgcolor=PURPLE[1],
                     widgets={"preset_prompt": "Empty - Nothing",
                              "custom_prompt": P.DIRECTOR_CUSTOM_PROMPT,
                              "inference_mode": "images", "max_frames": 8,
                              "max_size": 768, "seed": 1,
                              "__control_after_generate__": "randomize",
                              "force_offload": True, "save_states": False})
    wf.link(vlm, "llama_model", run_dir, "llama_model")
    wf.link(par_dir, "parameters", run_dir, "parameters")
    wf.link(grid, "image", run_dir, "images")
    # Diagnostic only: sits on the system-prompt path so it is guaranteed to run
    # before the director VLM call.
    probe_vlm_pre = wf.add("H3CudaProbe", c.at(90), size=(430, 90),
                           title="診断: VLM 開始前 current_device",
                           widgets={"label": "1-vlm-pre"})
    wf.wlink(sw_sys, "*", probe_vlm_pre, "anything")
    wf.wlink(probe_vlm_pre, "anything", run_dir, "system_prompt")
    wf.wlink(join2, join2_out, run_dir, "custom_prompt")

    clean = wf.add("CR Text Replace", c.at(220), size=(430, 220),
                   title="余分なタグを除去",
                   widgets={"find1": "```", "replace1": "",
                            "find2": "<think>", "replace2": "",
                            "find3": "</think>", "replace3": ""})
    wf.link(run_dir, "output", clean, "text")
    show_dir = wf.add("ShowText|pysssss", c.at(420), size=(430, 420),
                      title="H3 Director Prompt (ここで直接編集可)",
                      color=YELLOW[0], bgcolor=YELLOW[1])
    wf.link(clean, "STRING", show_dir, "text")
    man_dir = wf.add("ttN text", c.at(300), size=(430, 300), mode=BYPASS,
                     title="プロンプト手動入力（VLMを飛ばす）",
                     widgets={"text": ""})
    sw_dir = wf.add("Any Switch (rgthree)", c.at(80), size=(430, 80),
                    title="PROMPT: 手動 > 自動",
                    manual_inputs=[{"name": "any_01", "type": "*", "link": None},
                                   {"name": "any_02", "type": "*", "link": None}],
                    manual_outputs=[{"name": "*", "type": "*", "links": []}])
    wf.link(man_dir, "text", sw_dir, "any_01")
    wf.link(show_dir, "STRING", sw_dir, "any_02")

    # The prompt is routed THROUGH the unload/purge pair on purpose: it creates a
    # hard data dependency so llama.cpp has released its VRAM before H3 loads.
    # Without this, ComfyUI is free to start loading the 25 GiB text encoder while
    # the VLM still holds ~11 GiB, which OOMs on a 12 GiB card (observed in Test 3).
    probe_vlm_post = wf.add("H3CudaProbe", c.at(90), size=(430, 90),
                            title="診断: VLM 実行後 current_device",
                            widgets={"label": "2-vlm-post"})
    wf.link(sw_dir, "*", probe_vlm_post, "anything")

    unload = wf.add("llama_cpp_unload_model", c.at(80), size=(430, 80),
                    title="VLM をアンロード（H3 の前に必ず実行）")
    wf.link(probe_vlm_post, "anything", unload, "any")

    probe_unload = wf.add("H3CudaProbe", c.at(90), size=(430, 90),
                          title="診断: unload 直後 current_device",
                          widgets={"label": "3-unload-post"})
    wf.link(unload, "any", probe_unload, "anything")

    purge = wf.add("LayerUtility: PurgeVRAM V2", c.at(110), size=(430, 110),
                   title="VRAM 解放",
                   widgets={"purge_cache": True, "purge_models": True})
    wf.link(probe_unload, "anything", purge, "anything")

    # ---- CUDA phase barrier -------------------------------------------------
    # llama.cpp picks its own CUDA device. ComfyUI captures a text encoder's
    # base load/offload device from torch.cuda.current_device() when CLIPLoader
    # runs, and SelectCLIPDevice retargets relative to that base. So the current
    # device must be back on GPU0 *before* the H3 CLIP loader executes.
    # Everything downstream (the H3 prompt AND the gated CLIP loader) hangs off
    # this node, which is what orders it - a bare "reset" node with no consumers
    # would not be guaranteed to run first.
    restore = wf.add("H3CudaPhaseRestore", c.at(150), size=(430, 150),
                     title="★ CUDA を cuda:0 へ復元（H3 フェーズの境界）",
                     color=RED[0], bgcolor=RED[1],
                     widgets={"target_gpu": 0, "label": "4-restore",
                              "synchronize": True})
    wf.link(purge, "any", restore, "anything")

    for n in (sys_new, sys_cont, sw_sys, par_dir, run_dir, clean, show_dir,
              man_dir, sw_dir, probe_vlm_pre, probe_vlm_post, unload,
              probe_unload, purge, restore):
        g("40 H3 Director Prompt", n)

    # =====================================================================
    # 45  Tail relay (continuation source)
    # =====================================================================
    # Its own column. Sharing a column with group 50 made the two group
    # bounding boxes overlap, and LiteGraph decides group membership purely by
    # containment - so bypassing this group also bypassed the 2nd turbo LoRA,
    # SageAttention, the H3 memory patch and SigmaShift. See the overlap guard
    # at the end of build().
    c = col(6300, 0)
    tail_note = wf.add("MarkdownNote", c.at(300), size=(430, 300),
                       title="45 続き生成の使い方",
                       color=RED[0], bgcolor=RED[1], widgets={"md": TAIL_MD})
    tail_len = wf.add("PrimitiveInt", c.at(80), size=(430, 80), mode=BYPASS,
                      title="tail フレーム数 (39/56/73/90 推奨)",
                      widgets={"value": DEF_TAIL})
    vid_in = wf.add("LoadVideo", c.at(120), size=(430, 120), mode=BYPASS,
                    title="前クリップの動画",
                    widgets={"file": ""})
    vid_comp = wf.add("GetVideoComponents", c.at(120), size=(430, 120), mode=BYPASS,
                      title="分解")
    wf.link(vid_in, "VIDEO", vid_comp, "video")
    vid_count = wf.add("GetImageSizeAndCount", c.at(120), size=(430, 120), mode=BYPASS,
                       title="総フレーム数を取得")
    wf.link(vid_comp, "images", vid_count, "image")
    tail_start = wf.add("MathExpression|pysssss", c.at(120), size=(430, 120), mode=BYPASS,
                        title="開始位置 = count - tail",
                        widgets={"expression": "max(0, a - b)"})
    wf.link(vid_count, "count", tail_start, "a")
    wf.link(tail_len, "INT", tail_start, "b")
    tail_img = wf.add("ImageFromBatch", c.at(140), size=(430, 140), mode=BYPASS,
                      title="末尾フレームを切り出し",
                      color=RED[0], bgcolor=RED[1],
                      widgets={"batch_index": 0, "length": DEF_TAIL})
    wf.link(vid_count, "image", tail_img, "image")
    wf.wlink(tail_start, "INT", tail_img, "batch_index")
    wf.wlink(tail_len, "INT", tail_img, "length")

    tail_secs = wf.add("MathExpression|pysssss", c.at(120), size=(430, 120), mode=BYPASS,
                       title="tail 秒数",
                       widgets={"expression": "a / 24.0"})
    wf.link(tail_len, "INT", tail_secs, "a")
    tail_negs = wf.add("MathExpression|pysssss", c.at(120), size=(430, 120), mode=BYPASS,
                       title="tail 秒数（負）= 末尾から",
                       widgets={"expression": "0.0 - (a / 24.0)"})
    wf.link(tail_len, "INT", tail_negs, "a")
    tail_aud = wf.add("TrimAudioDuration", c.at(140), size=(430, 140), mode=BYPASS,
                      title="末尾音声を切り出し",
                      color=RED[0], bgcolor=RED[1],
                      widgets={"start_index": -(DEF_TAIL / 24.0),
                               "duration": DEF_TAIL / 24.0})
    wf.link(vid_comp, "audio", tail_aud, "audio")
    wf.wlink(tail_negs, "FLOAT", tail_aud, "start_index")
    wf.wlink(tail_secs, "FLOAT", tail_aud, "duration")
    for n in (tail_note, tail_len, vid_in, vid_comp, vid_count, tail_start,
              tail_img, tail_secs, tail_negs, tail_aud):
        g("45 Tail Relay 続き生成", n)

    # =====================================================================
    # 50  H3 models - STABLE preset, strict H3_HETERO_V1 parity
    #
    # This group reproduces the measured 217.61 s build exactly:
    #   UNETLoader -> LoraLoaderModelOnly(raw) -> guider/scheduler
    #   CLIPLoader(int8_convrot) -> SelectCLIPDevice(gpu:1) -> H3 Ref2VA
    # Everything unverified lives in group 55 and is bypassed by default.
    # See docs/H3_HETERO_V1_FINAL.md and H3_HETERO_V1_MANIFEST.json.
    # =====================================================================
    c = col(3000, 0)
    mdl_note = wf.add("MarkdownNote", c.at(300), size=(430, 300),
                      title="50 モデルの選び方",
                      color=GREEN[0], bgcolor=GREEN[1], widgets={"md": MODEL_MD})
    unet = wf.add("UNETLoader", c.at(110), size=(430, 110),
                  title="H3 Ref2VA pruned int8_convrot ★v1固定",
                  color=RED[0], bgcolor=RED[1],
                  widgets={"unet_name": M_UNET_INT8, "weight_dtype": "default"})
    unet_gguf = wf.add("UnetLoaderGGUF", c.at(90), size=(430, 90), mode=BYPASS,
                       title="代替: H3 Ref2VA GGUF Q4_K_S (18.5GiB/未導入)",
                       widgets={"unet_name": M_UNET_GGUF})
    # H3_HETERO_V1 runs the int8_convrot text encoder on GPU1 (RTX 3060) via
    # SelectCLIPDevice. Measured conditioning time there: 16.42 s.
    # device="default" is correct - SelectCLIPDevice below does the retargeting.
    # Gated CLIPLoader: ComfyUI captures the text encoder's base load/offload
    # device here, from torch.cuda.current_device(). The barrier input forces
    # this to happen AFTER H3CudaPhaseRestore, so the base is cuda:0 even though
    # llama.cpp ran earlier. Loading itself is ComfyUI's own CLIPLoader.
    clip = wf.add("H3GatedCLIPLoader", c.at(170), size=(430, 170),
                  title="TextEnc: Qwen3-VL-32B int8_convrot ★v1固定（barrier付）",
                  color=RED[0], bgcolor=RED[1],
                  widgets={"clip_name": M_CLIP_INT8, "type": "minimax",
                           "device": "default"})
    wf.link(restore, "anything", clip, "barrier")
    rep_base = wf.add("H3ClipDeviceReport", c.at(90), size=(430, 90),
                      title="診断: SelectCLIPDevice 適用前 (base)",
                      widgets={"label": "6-clip-base"})
    wf.link(clip, "CLIP", rep_base, "clip")
    # GPU1 conditioning. This single node is the heterogeneous-GPU mechanism;
    # it ships with ComfyUI core (comfy_extras/nodes_multigpu.py).
    selclip = wf.add("SelectCLIPDevice", c.at(100), size=(430, 100),
                     title="★ GPU1 (RTX 3060) で conditioning",
                     color=RED[0], bgcolor=RED[1],
                     widgets={"device": "gpu:1"})
    wf.link(rep_base, "CLIP", selclip, "clip")
    rep_target = wf.add("H3ClipDeviceReport", c.at(90), size=(430, 90),
                        title="診断: SelectCLIPDevice 適用後 (target)",
                        widgets={"label": "7-clip-target"})
    wf.link(selclip, "CLIP", rep_target, "clip")
    # Test 3B measured nvfp4 on GPU0 at 46 s encode; it is NOT the v1 encoder.
    clip_nvfp4 = wf.add("CLIPLoader", c.at(140), size=(430, 140), mode=BYPASS,
                        title="代替1: nvfp4_awq (Test 3B 基準 / v1 では未使用)",
                        widgets={"clip_name": M_CLIP_NVFP4, "type": "minimax",
                                 "device": "default"})
    clip_gguf = wf.add("CLIPLoaderGGUF", c.at(110), size=(430, 110), mode=BYPASS,
                       title="代替2: TextEnc GGUF Q4_K_M (未導入)",
                       widgets={"clip_name": M_CLIP_GGUF, "type": "minimax"})
    vae_v = wf.add("VAELoader", c.at(90), size=(430, 90), title="H3 video VAE",
                   color=GREEN[0], bgcolor=GREEN[1],
                   widgets={"vae_name": M_VAE_VIDEO})
    vae_a = wf.add("VAELoader", c.at(90), size=(430, 90), title="H3 audio VAE",
                   color=GREEN[0], bgcolor=GREEN[1],
                   widgets={"vae_name": M_VAE_AUDIO})
    # v1 parity: the RAW turbo LoRA. All 518 keys fail to bind, so this is a
    # no-op on the weights - that is exactly the state the 217.61 s run and its
    # quality check were made in. Do not swap this for the pruned build here.
    lora_v1 = wf.add("LoraLoaderModelOnly", c.at(130), size=(430, 130),
                     title="Turbo LoRA (v1 raw / 518キー未bind = 実質無効)",
                     color=RED[0], bgcolor=RED[1],
                     widgets={"lora_name": M_LORA_V1_RAW, "strength_model": 1.0})
    wf.link(unet, "MODEL", lora_v1, "model")
    for n in (mdl_note, unet, unet_gguf, clip, rep_base, selclip, rep_target,
              clip_nvfp4, clip_gguf, vae_v, vae_a, lora_v1):
        g("50 H3 Models (Stable=v1)", n)

    # =====================================================================
    # 55  Experimental - NOT measured with the v1 route. Bypassed by default.
    #
    # These sit between the Stable LoRA and the guider/scheduler, so bypassing
    # the whole group leaves the exact v1 chain. Un-bypass the group to try the
    # SageAttention path plus a turbo LoRA that actually binds.
    # =====================================================================
    c = col(3550, 0)
    exp_note = wf.add("MarkdownNote", c.at(360), size=(430, 360),
                      title="55 Experimental（未測定・既定OFF）",
                      color=YELLOW[0], bgcolor=YELLOW[1], widgets={"md": EXP_MD})
    lora_bind = wf.add("LoraLoaderModelOnly", c.at(130), size=(430, 130), mode=BYPASS,
                       title="Turbo LoRA 1 (pruned_comfyui / 正しくbind)",
                       widgets={"lora_name": M_LORA_T1, "strength_model": 1.0})
    wf.link(lora_v1, "MODEL", lora_bind, "model")
    lora2 = wf.add("LoraLoaderModelOnly", c.at(120), size=(430, 120), mode=BYPASS,
                   title="Turbo LoRA 2 (pruned_comfyui)",
                   widgets={"lora_name": M_LORA_T2, "strength_model": 0.5})
    wf.link(lora_bind, "MODEL", lora2, "model")
    sage = wf.add("PathchSageAttentionKJ", c.at(110), size=(430, 110), mode=BYPASS,
                  title="SageAttention",
                  widgets={"sage_attention": "auto", "allow_compile": False})
    wf.link(lora2, "MODEL", sage, "model")
    # KJNodes >= 1.4.9. Trims peak VRAM when sageattention is active.
    h3mem = wf.add("MiniMaxH3MemoryEfficientSageAttentionPatch", c.at(90), size=(430, 90),
                   mode=BYPASS, title="H3 省メモリ Attention")
    wf.link(sage, "MODEL", h3mem, "model")
    shift = wf.add("MiniMaxH3SigmaShift", c.at(120), size=(430, 120), mode=BYPASS,
                   title="Sigma Shift",
                   widgets={"shift_video": 12.0, "shift_audio": 6.0})
    wf.link(h3mem, "model", shift, "model")
    for n in (exp_note, lora_bind, lora2, sage, h3mem, shift):
        g("55 Experimental (未測定)", n)

    # =====================================================================
    # 60  H3 Reference to Video
    # =====================================================================
    c = col(4100, 0)
    res_note = wf.add("MarkdownNote", c.at(330), size=(430, 330),
                      title="60 解像度と尺のプリセット",
                      color=GREEN[0], bgcolor=GREEN[1], widgets={"md": RES_MD})
    voice = wf.add("LoadAudio", c.at(140), size=(430, 140), mode=BYPASS,
                   title="声質リファレンス（任意）",
                   widgets={"audio": ""})

    ag = ([{"label": "ref_image_%d" % i, "name": "ref_images.ref_image_%d" % i,
            "type": "IMAGE"} for i in range(5)]
          + [{"label": "ref_video_0", "name": "ref_videos.ref_video_0", "type": "IMAGE"},
             {"label": "ref_video_audio_0",
              "name": "ref_video_audios.ref_video_audio_0", "type": "AUDIO"},
             {"label": "ref_audio_0", "name": "ref_audios.ref_audio_0", "type": "AUDIO"},
             {"label": "ref_audio_1", "name": "ref_audios.ref_audio_1", "type": "AUDIO"}])
    h3 = wf.add("MiniMaxH3ReferenceToVideo", c.at(560), size=(430, 560),
                title="③ MiniMax H3 Reference to Video",
                color=RED[0], bgcolor=RED[1], autogrow=ag,
                # ref_image_size="match" is the v1 adopted profile: it saved
                # 90.75 s (29.43%) versus "max" with no visible quality loss.
                widgets={"prompt": "", "width": DEF_W, "height": DEF_H,
                         "length": 124, "ref_image_size": "match"})
    wf.link(rep_target, "CLIP", h3, "clip")   # GPU1 conditioning - see group 50
    wf.link(vae_v, "VAE", h3, "vae")
    wf.link(vae_a, "VAE", h3, "audio_vae")
    # Reference count is NOT what caused the GPU1 OOM: 2 references peaked at
    # 10959 MiB and 1 reference at 10991 MiB, both failing at the same point.
    # The Stable wiring therefore keeps every Character Master reference.
    for i, sc in enumerate(scales):
        wf.link(sc, "image", h3, "ref_images.ref_image_%d" % i)
    wf.link(tail_img, "IMAGE", h3, "ref_videos.ref_video_0")
    wf.link(tail_aud, "AUDIO", h3, "ref_video_audios.ref_video_audio_0")
    wf.link(voice, "AUDIO", h3, "ref_audios.ref_audio_0")
    wf.wlink(restore, "anything", h3, "prompt")   # via unload/purge/restore - group 40
    wf.wlink(frames, "INT", h3, "length")

    noise = wf.add("RandomNoise", c.at(110), size=(430, 110), title="noise",
                   widgets={"noise_seed": 123456789,
                            "__control_after_generate__": "fixed"})
    guider = wf.add("BasicGuider", c.at(90), size=(430, 90), title="guider")
    wf.link(shift, "MODEL", guider, "model")
    wf.link(h3, "positive", guider, "conditioning")
    samp = wf.add("KSamplerSelect", c.at(90), size=(430, 90), title="sampler",
                  widgets={"sampler_name": "res_multistep"})
    sched = wf.add("BasicScheduler", c.at(140), size=(430, 140), title="scheduler",
                   widgets={"scheduler": "simple", "steps": 8, "denoise": 1.0})
    wf.link(shift, "MODEL", sched, "model")
    sca = wf.add("SamplerCustomAdvanced", c.at(160), size=(430, 160),
                 title="sampling", color=RED[0], bgcolor=RED[1])
    wf.link(noise, "NOISE", sca, "noise")
    wf.link(guider, "GUIDER", sca, "guider")
    wf.link(samp, "SAMPLER", sca, "sampler")
    wf.link(sched, "SIGMAS", sca, "sigmas")
    wf.link(h3, "LATENT", sca, "latent_image")
    dec_v = wf.add("VAEDecode", c.at(90), size=(430, 90), title="decode video")
    wf.link(sca, "output", dec_v, "samples")
    wf.link(vae_v, "VAE", dec_v, "vae")
    dec_a = wf.add("VAEDecodeAudio", c.at(90), size=(430, 90), title="decode audio")
    wf.link(sca, "output", dec_a, "samples")
    wf.link(vae_a, "VAE", dec_a, "vae")
    for n in (res_note, voice, h3, noise, guider, samp, sched, sca, dec_v, dec_a):
        g("60 H3 Reference to Video", n)

    # =====================================================================
    # 70  Audio layer
    # =====================================================================
    c = col(4650, 0)
    aud_note = wf.add("MarkdownNote", c.at(280), size=(430, 280),
                      title="70 音声レイヤー",
                      color=GREEN[0], bgcolor=GREEN[1], widgets={"md": AUDIO_MD})
    g("70 Audio", aud_note)
    final_audio = dec_a
    final_audio_out = "AUDIO"
    if full:
        bgm = wf.add("LoadAudio", c.at(140), size=(430, 140), mode=BYPASS,
                     title="BGM ソース", widgets={"audio": ""})
        bgm_trim = wf.add("TrimAudioDuration", c.at(130), size=(430, 130), mode=BYPASS,
                          title="BGM を尺に合わせて切る",
                          widgets={"start_index": 0.0, "duration": DEF_SECONDS})
        wf.link(bgm, "AUDIO", bgm_trim, "audio")
        bgm_vol = wf.add("AudioAdjustVolume", c.at(100), size=(430, 100), mode=BYPASS,
                         title="BGM 音量 (dB)", widgets={"volume": -14})
        wf.link(bgm_trim, "AUDIO", bgm_vol, "audio")
        bgm_eq = wf.add("AudioEqualizer3Band", c.at(250), size=(430, 250), mode=BYPASS,
                        title="BGM EQ（中域を下げて声を通す）",
                        widgets={"low_gain_dB": 0.0, "low_freq": 100,
                                 "mid_gain_dB": -4.0, "mid_freq": 1200,
                                 "mid_q": 0.9, "high_gain_dB": 0.0,
                                 "high_freq": 6000})
        wf.link(bgm_vol, "AUDIO", bgm_eq, "audio")
        mix = wf.add("AudioMerge", c.at(110), size=(430, 110), mode=BYPASS,
                     title="H3音声 + BGM", color=YELLOW[0], bgcolor=YELLOW[1],
                     widgets={"merge_method": "add"})
        wf.link(dec_a, "AUDIO", mix, "audio1")
        wf.link(bgm_eq, "AUDIO", mix, "audio2")
        prev_aud = wf.add("PreviewAudio", c.at(110), size=(430, 110), title="音声確認")
        wf.link(mix, "AUDIO", prev_aud, "audio")
        mp3 = wf.add("SaveAudioMP3", c.at(120), size=(430, 120), mode=BYPASS,
                     title="音声のみ書き出し",
                     widgets={"filename_prefix": "H3/audio/h3", "quality": "320k"})
        wf.link(mix, "AUDIO", mp3, "audio")
        for n in (bgm, bgm_trim, bgm_vol, bgm_eq, mix, prev_aud, mp3):
            g("70 Audio", n)
        final_audio, final_audio_out = mix, "AUDIO"
    else:
        prev_aud = wf.add("PreviewAudio", c.at(110), size=(430, 110), title="音声確認")
        wf.link(dec_a, "AUDIO", prev_aud, "audio")
        g("70 Audio", prev_aud)

    # =====================================================================
    # 80  Output
    # =====================================================================
    c = col(5200, 0)
    cv = wf.add("CreateVideo", c.at(110), size=(430, 110),
                title="動画化 (fps=24 固定)",
                color=GREEN[0], bgcolor=GREEN[1],
                widgets={"fps": float(FPS), "bit_depth": 8})
    wf.link(dec_v, "IMAGE", cv, "images")
    wf.link(final_audio, final_audio_out, cv, "audio")
    sv = wf.add("SaveVideo", c.at(140), size=(430, 140),
                title="保存（続き生成の入力になる）",
                color=GREEN[0], bgcolor=GREEN[1],
                widgets={"filename_prefix": "H3/%date:yyyyMMdd%/h3_%date:hhmmss%",
                         "format": "auto", "codec": "auto"})
    wf.link(cv, "VIDEO", sv, "video")
    for n in (cv, sv):
        g("80 Output", n)

    if full:
        # vertical gap so group 80's and group 85's bounding boxes do not touch
        c.at(0, 260)
        up_ld = wf.add("UpscaleModelLoader", c.at(90), size=(430, 90), mode=BYPASS,
                       title="RealESRGAN", widgets={"model_name": M_UPSCALE})
        up = wf.add("ImageUpscaleWithModel", c.at(100), size=(430, 100), mode=BYPASS,
                    title="4x 拡大")
        wf.link(up_ld, "UPSCALE_MODEL", up, "upscale_model")
        wf.link(dec_v, "IMAGE", up, "image")
        down = wf.add("LayerUtility: ImageScaleByAspectRatio V2", c.at(230),
                      size=(430, 230), mode=BYPASS,
                      title="1080x1920 (9:16) に整形",
                      widgets={"aspect_ratio": "custom", "proportional_width": 9,
                               "proportional_height": 16, "fit": "crop",
                               "method": "lanczos", "round_to_multiple": "None",
                               "scale_to_side": "height", "scale_to_length": 1920,
                               "background_color": "#000000"})
        wf.link(up, "IMAGE", down, "image")
        cv2 = wf.add("CreateVideo", c.at(110), size=(430, 110), mode=BYPASS,
                     title="動画化（高画質）",
                     widgets={"fps": float(FPS), "bit_depth": 8})
        wf.link(down, "image", cv2, "images")
        wf.link(final_audio, final_audio_out, cv2, "audio")
        sv2 = wf.add("SaveVideo", c.at(140), size=(430, 140), mode=BYPASS,
                     title="保存（1080x1920）",
                     widgets={"filename_prefix": "H3/%date:yyyyMMdd%/h3_hq_%date:hhmmss%",
                              "format": "auto", "codec": "auto"})
        wf.link(cv2, "VIDEO", sv2, "video")
        for n in (up_ld, up, down, cv2, sv2):
            g("85 Upscale (1080x1920)", n)

        # Whisper verification
        c = col(5750, 0)
        wsp_note = wf.add("MarkdownNote", c.at(240), size=(430, 240),
                          title="90 検証",
                          color=GREEN[0], bgcolor=GREEN[1], widgets={"md": VERIFY_MD})
        wsp = wf.add("Apply Whisper", c.at(160), size=(430, 160),
                     title="生成結果のセリフを文字起こし",
                     widgets={"model": "large-v3-turbo", "language": "Japanese",
                              "prompt": ""})
        wf.link(dec_a, "AUDIO", wsp, "audio")
        wsp_show = wf.add("ShowText|pysssss", c.at(260), size=(430, 260),
                          title="→ 実際に何と言ったか",
                          color=YELLOW[0], bgcolor=YELLOW[1])
        wf.link(wsp, "text", wsp_show, "text")
        for n in (wsp_note, wsp, wsp_show):
            g("90 検証", n)

        # memory cleanup
        clean1 = wf.add("easy cleanGpuUsed", c.at(90), size=(430, 90),
                        title="GPU クリーン")
        wf.link(dec_v, "IMAGE", clean1, "anything")
        clean2 = wf.add("easy clearCacheAll", c.at(90), size=(430, 90),
                        title="キャッシュクリア")
        wf.link(clean1, "output", clean2, "anything")
        for n in (clean1, clean2):
            g("90 検証", n)

    # ---- groups ----
    colors = {"00 README": "#3f789e", "10 Character Master": "#3f789e",
              "20 Character Profile": "#a1309b", "30 日本語入力": "#8A8",
              "40 H3 Director Prompt": "#a1309b",
              "45 Tail Relay 続き生成": "#b06634",
              "50 H3 Models (Stable=v1)": "#88A",
              "55 Experimental (未測定)": "#653",
              "60 H3 Reference to Video": "#b06634",
              "70 Audio": "#8A8", "80 Output": "#3f789e",
              "85 Upscale (1080x1920)": "#88A", "90 検証": "#8A8"}
    for name, nodes in G.items():
        wf.group(name, nodes, color=colors.get(name, "#3f789e"))

    # --- guard: group bounding boxes must not overlap ---------------------
    # LiteGraph assigns a node to a group by containment, so two overlapping
    # groups silently share nodes and "bypass this group" hits the neighbour.
    gs = wf.groups
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            ax, ay, aw, ah = gs[i]["bounding"]
            bx, by, bw, bh = gs[j]["bounding"]
            if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                raise AssertionError(
                    "group bounding boxes overlap: %r %s  vs  %r %s"
                    % (gs[i]["title"], gs[i]["bounding"],
                       gs[j]["title"], gs[j]["bounding"]))
    return wf


# ------------------------------------------------------------------ notes --
README_MD_CORE = """# H3 Phase 1 CORE

MiniMax H3 専用。**人物保持 / 続き生成 / 日本語→英語自動化** の3つが中核です。

## 手順

1. **10 Character Master**
   `①顔寄り` にキャラクター画像を読み込みます。
   ②③④ は任意。使うならノードを選んで **Ctrl+B** でバイパスを解除。
   高解像度の参照ほど同一性が上がります（その分遅くなります）。

2. **20 Character Profile**
   実行すると VLM が参照画像から顔立ち・髪・肌・体格・衣装・素材感を抽出します。
   結果は黄色の ShowText に出ます。**その場で編集できます**。
   気に入ったらコピーして保存しておくと、以降のクリップで使い回せます。

3. **30 日本語入力**
   ①シーン指示と②セリフを日本語で書きます。
   **②に書いた行は一字一句そのまま** `<d>[Japanese] ...</d>` に入ります。
   ②を空にすれば AI がセリフを創作します。

4. **40 H3 Director Prompt**
   VLM が H3 公式の6セクション英語プロンプトを生成します。
   結果は黄色の ShowText で直接編集できます。
   完成したプロンプトを使い回したいときは
   「プロンプト手動入力」を Ctrl+B で有効化すると VLM を飛ばせます。

5. **60** で実行。**80 Output** に mp4 が保存されます。

## 続きを作るとき

**45 Tail Relay** グループのノードを全部選択して Ctrl+B（バイパス解除）。
同じく **SYS-B: 続き生成** と **③前クリップ情報** も有効化します。
詳しくは 45 グループのノートを見てください。

## ※ モデルが未導入だと 50 のローダーが赤くなります
20〜40（VLM部分）は現状のモデルだけで動きます。
"""

README_MD_FULL = README_MD_CORE + """
## FULL 版で追加されるもの

- **70 Audio**: BGM を別トラックで合成（音量・EQ・差し替え可）
- **85 Upscale**: RealESRGAN → 1080x1920 に整形
- **90 検証**: Whisper で生成結果のセリフを文字起こしして確認

これらはすべて **初期状態ではバイパス** されています。
使うグループだけ Ctrl+B で有効化してください。
"""

TAIL_MD = """# 続き生成（Tail Relay）

前クリップの**末尾数秒**を `<Video 1>` として H3 に戻します。

## 手順
1. まず普通に1本目を生成し、**80 Output** に保存された mp4 を確認
2. その mp4 を ComfyUI の input フォルダに置く（またはアップロード）
3. このグループのノードを全選択して **Ctrl+B**（バイパス解除）
4. `前クリップの動画` でその mp4 を選択
5. **SYS-B: 続き生成** を Ctrl+B で有効化
6. **③前クリップ情報** を Ctrl+B で有効化し、前の内容を日本語で要約
7. 実行

## tail フレーム数
H3 は参照動画のフレーム数を `n %% 17 == 5` に**切り下げ**ます。
下記ならロスなしです。

| frames | 秒 | 向き |
|---|---|---|
| 39 | 1.63 | ドリフト最小 |
| 56 | 2.33 | **推奨** |
| 73 | 3.04 | 会話の途中でつなぐ |
| 90 | 3.75 | 長回し感を維持 |

## 重要
続き生成中も **10 Character Master は接続したまま**にしてください。
`<Video 1>` だけだと数クリップで人物が崩れていきます。
"""

MODEL_MD = """# モデル

## 本体（どちらか1つ）
- **★基準: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`**（19.53 GiB / 導入済）
  記事WFのDLメタデータと一致。`_pruned_comfyui` の Turbo LoRA と組み合わせる前提。
- 代替: `MiniMax-H3-Ref2VA-Q4_K_S.gguf`（**18.49 GiB** / 未導入）
  非pruned本体の量子化なので 12GB には載りません。GGUFにする利点は限定的です。

※ 導入済の `minimax_h3_fl2va_*` は **FL2VA（初・終フレーム用）**で、
参照画像による人物保持には使えません。

## テキストエンコーダ
- **★v1固定: `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`**
  H3_HETERO_V1 が実測で採用している値です。**変更しないでください。**
- 代替1: `nvfp4_awq`（Test 3B の基準。v1 経路では未使用）
- 代替2: GGUF `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf`（未導入）

## ★ GPU1 conditioning（この構成の高速化の本体）
`SelectCLIPDevice(device=gpu:1)` が text/reference conditioning を
**RTX 3060 側**へ寄せます。DiT sampling と Video/Audio VAE は
起動引数 `--default-device 0` により **RTX 4070 Ti** で走ります。

実測（v1 / 217.61秒）:

| 段 | GPU | 時間 |
|---|---|---:|
| conditioning | GPU1 3060 | 16.42 s |
| sampling (8 steps) | GPU0 4070 Ti | 164.65 s |
| video decode | GPU0 | 27.98 s |
| audio decode | GPU0 | 1.52 s |

**この経路を外すと 217.61 秒は再現しません。**

## VAE（導入済）
video / audio の2つともロード済みです。

## Turbo LoRA
既定は v1 と同じ **raw** ファイルです。
このファイルは本機で 518/518 キーが bind に失敗するため、
**実質 LoRA 無効の状態**で 8 steps を回しています。
217.61 秒の実測と目視品質確認はこの状態で行われたものなので、
Stable 側では意図的にそのまま維持しています。

正しく bind する `_pruned_comfyui` 版は **55 Experimental** に置いてあります。
"""

EXP_MD = """# 55 Experimental（未測定・既定OFF）

このグループは **H3_HETERO_V1 の 217.61 秒構成には含まれていません。**
既定で全ノードがバイパスされており、その状態が **v1 厳密一致**です。

## 中身
| ノード | 内容 |
|---|---|
| Turbo LoRA 1 (pruned_comfyui) | 警告 0 件で**正しく bind する** Turbo LoRA |
| Turbo LoRA 2 (pruned_comfyui) | 2段目 strength 0.5 |
| SageAttention | Test 3B で sampler を 13.93 s/step にしていた経路 |
| H3 省メモリ Attention | 上の SageAttention 使用時のピーク VRAM 削減 |
| Sigma Shift | shift_video 12.0 / shift_audio 6.0 |

## 使うとどうなるか（未検証）
Test 3B（単GPU・Sage有効）の sampler は **13.93 s/step**、
v1（GPU1 conditioning・Sage無し）は **20.58 s/step** でした。
生成条件は両者とも 576x1024 / 124f / 8 steps で同一です。

したがって「GPU1 conditioning + match + SageAttention」を併用すれば
v1 より速くなる**可能性**がありますが、**この組み合わせは未測定です。**
断定できません。

また Turbo LoRA が実際に効くようになるため、
**同じ seed でも絵が変わります。**v1 の目視品質判定の前提から外れます。

## 使い方
グループを右クリック →「Bypass Group Nodes」で ON/OFF します。
比較するときは seed・プロンプト・参照画像を固定し、
**1回に1つだけ**変えてください。

戻すときは再度バイパスすれば v1 厳密一致に戻ります。
"""

RES_MD = """# 解像度と尺

## 縦型（9:16）プリセット
| 用途 | width | height | 備考 |
|---|---|---|---|
| 高速確認 | 288 | 512 | 厳密に 9:16 |
| **★v1固定** | **576** | **1024** | 217.61 秒の実測条件 |
| 高解像 | 768 | 1344 | 面積上限。**v1 未測定** |

H3 の面積上限は 768x1344 = 1,032,192 px。幅・高さとも **32 の倍数**。

## 尺（length）
`n %% 17 == 5` に揃える必要があり、**fps は 24 固定**。
30 グループの「尺（秒）」を変えれば自動で揃えます。

**★v1固定は 124 フレーム**（尺 5.0 秒 → 124）。
ノードの tooltip 上の学習済レンジは **124〜362 フレーム（約 5.2〜15.1 秒）**です。
この外も入力自体は可能ですが未検証領域になります。

## ref_image_size
- `max`  : 短辺 2048 まで使う。参照トークンが毎ステップ乗るので遅い
- **★v1固定 `match`**: 生成面積に合わせて縮小

v1 では `max` → `match` で **90.75 秒（29.43%）短縮**し、
実機目視で顔・identity・画質・動き・口の動きに差が出ないことを確認済みです。

どちらも**拡大はしません**。参照画像が小さいと max にしても差が出ません。

## 固定すべき値 / 変えてよい値
**v1 性能維持のため固定**（変えると 217.61 秒は再現しません）:
width 576 / height 1024 / frames 124 / steps 8 / seed 123456789 /
sampler `res_multistep` / scheduler `simple` / ref_image_size `match` /
GPU1 conditioning / モデル・TE・LoRA の各ファイル名

**UI で自由に変えてよい**:
参照画像、日本語のシーン・セリフ、ネガティブ、
BGM と音量、出力ファイル名、Upscale の ON/OFF、Tail Relay の ON/OFF
"""

AUDIO_MD = """# 音声レイヤー

H3 は映像と音声を**1本の混ざったステレオ**として出します。
後から BGM だけ差し替えるには、**混ぜる前に分ける**必要があります。

## やり方
1. プロンプト側で `non_diegetic_music: N/A` を強制
   → SYS プロンプトの R2 ルールで自動的にかかります
   → H3 の出力は「セリフ + 環境音」だけになります
2. BGM は `LoadAudio` から別トラックで読み込み
3. 音量（-12〜-18dB）→ EQ（中域 -3〜-6dB）→ `AudioMerge (add)`

## BGM を差し替えるとき
`LoadAudio` のファイルを変えて **70 以降だけ再実行**すれば済みます。
H3 の生成をやり直す必要はありません。

## 注意
`AudioMerge` は **audio1 の長さに audio2 を合わせます**。
必ず audio1 = H3 音声 / audio2 = BGM にしてください。
"""

VERIFY_MD = """# 検証

Whisper で**生成結果の音声**を文字起こしします。
30 グループの「②セリフ確定」と見比べてください。

| 結果 | 判断 |
|---|---|
| ほぼ一致 | OK |
| 単語が違う | seed を振り直して再生成 |
| 聞き取れない | セリフが長すぎ。1発話 10〜15 文字に割る |
| 英語になる | `<d>[Japanese]` タグが入っていない。プロンプトを確認 |

Whisper モデルは初回実行時に自動ダウンロードされます（large-v3-turbo で約 1.6GB）。
"""


if __name__ == "__main__":
    oi_path, out_dir = sys.argv[1], sys.argv[2]
    sch = Schema(oi_path)
    for name, full in (("H3_phase1_core.json", False), ("H3_phase1_full.json", True)):
        wf = build(sch, full)
        p = wf.save(os.path.join(out_dir, name))
        d = wf.to_dict()
        print("wrote %-24s nodes=%3d links=%3d groups=%d"
              % (name, len(d["nodes"]), len(d["links"]), len(d["groups"])))
