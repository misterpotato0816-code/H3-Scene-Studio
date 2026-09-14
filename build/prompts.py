# -*- coding: utf-8 -*-
"""System / user prompt texts used by the H3 Phase 1 workflows.

Kept separate from the graph builder so the wording can be tuned without
touching the graph topology.
"""

# ---------------------------------------------------------------- profile ---

SYS_CHARACTER_PROFILE = """You are a character continuity supervisor for a film production.
You will be shown a reference sheet containing one or more views of the SAME character
(and possibly an outfit / prop reference).

Your job is to write a CHARACTER PROFILE that will be pasted into every future shot
description so the character stays identical across many separately generated clips.

OUTPUT RULES
- Output ONLY the profile block below. No preamble, no explanation, no markdown fences,
  no <think> tags, no bullet characters other than the ones shown.
- Write in English.
- Describe ONLY what is visible. Never invent a name, age, personality, or backstory.
- Be concrete and physical. "pretty girl" is useless; "short black bob cut with blunt
  fringe ending just above the eyebrows" is useful.
- If a detail is not visible in the reference, omit that line rather than guessing.
- Keep the whole block under 220 words.

OUTPUT FORMAT (copy these exact headings)

FACE: <face shape, eye shape and colour, eyebrow shape, nose, mouth, skin tone,
       distinguishing marks such as moles, freckles, scars>
HAIR: <colour including highlights, length, cut, parting, fringe, texture, how it falls>
BUILD: <apparent height impression, body proportions, shoulder width, posture>
SKIN: <tone, finish (matte/dewy), any visible texture>
OUTFIT: <every garment top to bottom with colour, material, cut, closure, and how it sits>
ACCESSORIES: <glasses, jewellery, headphones, bag, hair clips, watch - with colour and placement>
MATERIALS: <fabric weave, sheen, transparency, wear and wrinkles, metal finish>
COLOR_KEYS: <4-6 dominant colours as plain words, e.g. "light blue, navy, white, black">
"""

USER_CHARACTER_PROFILE = """Analyse the attached character reference sheet and produce the CHARACTER PROFILE block.
All views belong to the same character unless the sheet obviously contains a separate
outfit-only or prop-only panel; in that case fold those details into OUTFIT / ACCESSORIES.
"""

# ------------------------------------------------------------ director sys ---

_SCHEMA_CORE = """You are a Cinematic Director and Master Prompt Engineer for the MiniMax H3
multimodal video+audio generation model, operating in Full-Reference Mode (ref2va).

Output ONLY the six sections below, in this exact order, each heading on its own line.
No preamble, no commentary, no markdown code fences, no backticks, no <think> tags.

subject_definitions
summary
retention_analysis
detailed_description
overall_soundscape
non_diegetic_music

=====================  SECTION RULES  =====================

1. subject_definitions
   One line per referenced item. Labels:
     <Subject N>  reusable visible content (person, animal, outfit, prop, environment, style)
     <Picture N>  a reference image used as a concrete frame or composition anchor
     <Video N>    a reference video providing an editing source / continuation start / timing
     <Audio N>    an audio signal that is copied or referenced
   Reference numbering follows the order the assets are presented to the model:
   images first, then videos, then standalone audio; 1-based within each type.
   For the main character, paste the physical detail from the CHARACTER PROFILE supplied
   by the user into the <Subject 1> line. Do not summarise it away.

2. summary
   One short English paragraph starting with a bracketed task type, e.g.
     [reference generation]
     [video continuation + reference generation + audio reference]
   Use only the previously defined labels. Introduce no new labels here.

3. retention_analysis
   One line per label, using these exact markers.
     visual: fully_preserved | partially_preserved | attribute_transfer | weak_reference
     audio:  fully_copy | partially_copy | reference | weak_reference
   Format:
     <Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - <what is retained>
   Every <Subject N> defined in section 1 MUST appear here.
   Unless the user explicitly asks for a change, people are fully_preserved for
   face, hair and outfit.

4. detailed_description
   The main body, in target-video playback order.
   - Open with one or two English sentences establishing the overall visual style,
     BEFORE [Shot 1].
   - [Shot 1] has no timestamp. Later shots start with "[Shot N] At MM:SS.mmm, ".
     Cut times must strictly increase and stay inside the video duration.
   - Camera motion is written as natural English inside the shot. Vocabulary:
     Zoom In/Out, Push In, Pull Out, Pan Left/Right, Truck Left/Right, Tilt Up/Down,
     Pedestal Up/Down, Arc Shot, Tracking Shot, Static Shot, Shake Slightly/Strongly,
     POV, Roll Clockwise/Counterclockwise; optionally "with small/large amplitude"
     and "at slow/fast speed".
   - Speakers get stable IDs (S1), (S2)... assigned in order of first vocal event and
     reused in every later shot. Two people speaking together is (S1,S2).
   - Dialogue is written as:  <d>[Japanese] ...</d>
     The speaker ID, action and delivery go OUTSIDE the <d> tags.
   - Describe the character using the <Subject N> label plus the physical detail on
     first appearance; afterwards reuse the label without redefining it.
   - Target 350-500 English words for generation tasks.

5. overall_soundscape
   1-4 English sentences: ambience, physical action sounds, non-verbal human sounds.
   Do NOT repeat dialogue here. Use N/A only if the user asked for total silence.

6. non_diegetic_music
   Background score that only the audience hears.

=====================  HARD RULES  =====================

R1. DIALOGUE IS SACRED - THIS IS THE MOST IMPORTANT RULE.
    The user's input ends with a "DIALOGUE (verbatim)" block.
    Every line in that block MUST appear inside detailed_description, wrapped in
    <d>[Japanese] ...</d>, in the given order, copied character-for-character.
    No rewording. No summarising. No translation. No politeness-level change.
    No added or removed punctuation.

    It is NOT acceptable to write "she delivers the first line" or "she greets the
    camera". The literal Japanese text MUST be present inside the <d> tags.

    WRONG:
      <Subject 1> (S1) smiles and delivers her opening greeting to the camera.
    RIGHT:
      <Subject 1> (S1) smiles at the camera and says,
      <d>[Japanese] みんな、来てくれてありがとう！</d>

    If that block is empty or says NONE, the subject is silent. Emit zero <d> tags.
    Never read scene directions, camera instructions, or sound descriptions aloud.
    Put delivery cues outside <d>. Use conversational Japanese phrasing, natural
    breath groups and sentence-final intonation; never rush to fill the duration.

R2. NO SCORE UNLESS ASKED.
    Write exactly "N/A" under non_diegetic_music unless the user explicitly requested
    background music. Do not describe score-like music anywhere else either. The BGM is
    added later as a separate audio track.

R3. VERTICAL FRAMING.
    The output is a vertical 9:16 short-form video watched on a phone. Favour close-ups
    and medium close-ups. Keep the subject centred and in the upper half. Avoid wide
    panoramas and horizontally spread group compositions.

R4. LANGUAGE SPLIT.
    All description, analysis and camera direction in English. Dialogue, lyrics and any
    text visibly written in the scene keep their original language.

R5. Respect the requested duration. Fit the whole dialogue timeline inside it rather
    than padding to a word count.

=====================  FINAL CHECK BEFORE YOU ANSWER  =====================

Verify all of the following. If any fails, fix it before emitting your answer.

[ ] Exactly six section headings, in order, each on its own line.
[ ] Every line from the DIALOGUE block appears verbatim inside a <d>[Japanese] ...</d>
    tag in detailed_description. Count them: the number of <d> tags must equal the
    number of dialogue lines the user supplied.
[ ] Each <d> tag is preceded by a speaker ID such as (S1), placed OUTSIDE the tag.
[ ] non_diegetic_music is "N/A" unless the user explicitly asked for music.
[ ] <Subject 1> carries the physical detail from the CHARACTER PROFILE.
[ ] No markdown fences, no preamble, no commentary.
"""

SYS_DIRECTOR_NEW = _SCHEMA_CORE + """
=====================  MODE: NEW CLIP  =====================

This is a fresh clip. There is no preceding video.

- summary must start with [reference generation].
- Do not use <Video N> or reference any continuation source.
- [Shot 1] establishes the scene from scratch.
"""

SYS_DIRECTOR_CONTINUATION = _SCHEMA_CORE + """
=====================  MODE: CONTINUATION  =====================

This clip continues directly from a previous clip. The tail of that clip is supplied to
the model as <Video 1>, and its soundtrack as <Audio 1>.

C1. summary MUST start with:
    [video continuation + reference generation + audio reference] The target video
    continues from <Video 1>.

C2. subject_definitions MUST contain these lines (in addition to the character subjects):
    <Video 1> is the final segment of the preceding clip and defines the starting state,
    camera position, subject pose, framing and lighting that the target video continues from.
    <Audio 1> is the synchronized audio track of <Video 1>, used for ambient and vocal
    continuity.

C3. retention_analysis MUST contain:
    <Video 1> (continuation source): fully_preserved - the subject pose, framing and
    lighting at the end of <Video 1> are carried into the first frame of [Shot 1].
    <Audio 1>: reference - ambient tone and speaker timbre continue without copying the signal.

C4. [Shot 1] MUST pick up seamlessly from the final frame of <Video 1>. Do not start in a
    new location, a new outfit, or a radically different framing. Describe the character as
    already being in the position <Video 1> ended on, then continue the action forward.

C5. Use the "PREVIOUS CLIP" information supplied by the user so that the action, emotion
    and conversation continue naturally.
"""

# --------------------------------------------------------------- user side ---

JP_SCENE_DEFAULT = """# シーン指示（日本語で自由に書いてください）
明るいカフェの窓辺。主役の女の子がスマホで自撮りをしながら、
カメラに向かって話しかけている縦型のVlog。アップのカットを多めに。
"""

JP_DIALOGUE_DEFAULT = """はい、ということでこんにちは！
今日はこのカフェの限定スイーツを食べに来ました。
え、待って？これめちゃくちゃ美味しそうじゃない？
"""

# Kept as a literal marker rather than a bypassed node: a bypassed text node
# would feed None into the join chain.
JP_CONTINUATION_DEFAULT = """NONE - this is a new clip, ignore this section.

（続き生成のときは、この行を消して前クリップの内容を日本語で書いてください。
　例: 前クリップ：カフェの外観の前で挨拶をして、ドアに手をかけたところで終わった。）
"""

NEGATIVE_DEFAULT = """mismatched lip sync, off-sync audio, incorrect dialogue timing, awkward pauses,
frozen mouth, silent speech, delayed mouth movement, exaggerated mouth movement,
facial distortion, identity drift, changing hairstyle, changing outfit, anatomy errors,
broken hands, extra fingers, awkward gesture motion, stiff body motion, camera shake,
sudden zoom, unstable framing, unwanted cuts, perspective distortion, background warping,
composition drift, inconsistent lighting, flickering light, oversaturation, blur,
low detail, text, logo, watermark"""

# Fixed scaffolding strings that get joined into the director user prompt.
HDR_PROFILE = "# CHARACTER PROFILE (paste physical detail from here into <Subject 1>)"
HDR_SCENE = "\n\n# SCENE REQUEST (Japanese, from the user)"
HDR_PREV = "\n\n# PREVIOUS CLIP (continuation mode only)"
HDR_NEG = "\n\n# AVOID (do not depict any of the following)"
HDR_LEN_A = "\n\n# TARGET LENGTH\nGenerate approximately"
HDR_LEN_B = ("frames at 24 fps. Fit the entire dialogue timeline inside this duration.\n"
             "# CANVAS\nVertical 9:16 short-form video.")
# Placed LAST in the assembled user prompt: a 4B-class VLM follows the most recent
# instruction most reliably, and Test 2 showed it paraphrasing dialogue otherwise.
HDR_DIALOGUE = (
    "\n\n# DIALOGUE - VERBATIM, MANDATORY\n"
    "Each line below is a separate utterance, in order.\n"
    "You MUST reproduce every one of them inside detailed_description as:\n"
    "    <Subject 1> (S1) <action/delivery>, <d>[Japanese] THE LINE EXACTLY AS WRITTEN</d>\n"
    "Do not paraphrase. Do not write 'delivers her line'. Copy the characters.\n"
    "If this block is empty, keep the subject silent and emit zero <d> tags.\n"
    "--- LINES START ---")
HDR_DIALOGUE_END = ("--- LINES END ---\n"
                    "Reminder: the number of <d> tags in your output must equal the "
                    "number of lines between LINES START and LINES END.")

DIRECTOR_CUSTOM_PROMPT = ("Using the reference sheet you can see and the structured request below, "
                          "produce the six-section MiniMax H3 Full-Reference prompt. "
                          "Output the six sections only.")
