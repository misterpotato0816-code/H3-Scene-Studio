"""One contract at the video submission boundary, including resume and retry."""
import re

from . import compiler, director_speech
from .errors import PipelineError


DELIVERIES = {
    "natural": "Relaxed conversational Japanese, clear breath groups, natural sentence-final intonation.",
    "warm": "Warm conversational Japanese, gentle expressive pitch, unhurried breath pauses.",
    "calm": "Calm conversational Japanese, restrained pitch variation, clear phrase endings and relaxed pauses.",
    "bright": "Bright conversational Japanese, lively but unhurried phrasing, distinct sentence endings.",
}


def prepare(prompt: str, speech: str, *, allow_s2: bool = False,
            delivery: str = "natural") -> tuple[str, dict]:
    if delivery not in DELIVERIES:
        raise PipelineError("話し方の指定が不正です。", kind="prompt_rejected")
    expected = director_speech.dialogue_lines(speech)
    # Detect before sanitation: neutralizing a speaking verb must not disguise
    # quoted, unscripted English as scenery. Ordinary object names stay valid.
    prose = director_speech._strip_d_tags(prompt)
    unsafe_quote = re.search(
        r"\b(?:says?|speaks?|whispers?|shouts?|narrates?|replies|asks?)\b[^\n<>]{0,50}[\"“「]",
        prose, flags=re.I)
    cleaned, report = director_speech.final_gate(prompt, dialogue=expected, allow_s2=allow_s2)
    if unsafe_quote:
        report["hard"].append("発話タグ外に引用された台詞があります。")
    if compiler.hygiene_violations(cleaned):
        report["hard"].append("生成指示にAIの思考過程が残っています。")
    if report["hard"]:
        raise PipelineError("発話チェック: " + " / ".join(report["hard"][:3]),
                            kind="prompt_rejected")
    if expected:
        # Place performance in the shot section; never inside the dialogue.
        cleaned = re.sub(r"(?m)^Speech performance: [^\n]*\n?", "", cleaned)
        cleaned = re.sub(r"(?m)^(detailed_description:?)\s*$",
                         lambda m: m[1] + "\nSpeech performance: " + DELIVERIES[delivery],
                         cleaned, count=1)
    report["delivery"] = delivery
    return cleaned, report
