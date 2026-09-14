"""Optional VOICEVOX HTTP integration, loopback only, no media persistence.

Engine: https://github.com/VOICEVOX/voicevox_engine
API: https://voicevox.github.io/voicevox_engine/api/
Preview is not a prediction of H3's voice or lip timing.
"""
import copy
import math

import aiohttp

BASE = "http://127.0.0.1:50021"
LIMIT = 16 * 1024 * 1024


class RehearsalError(ValueError):
    pass


def speaker_id(value) -> int:
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise RehearsalError("話者を選択してください。") from None
    if result < 0 or result > 100000:
        raise RehearsalError("話者IDが不正です。")
    return result


def normalize_query(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise RehearsalError("先に読みを解析してください。")
    q = copy.deepcopy(raw)
    phrases = q.get("accent_phrases")
    if not isinstance(phrases, list) or not 1 <= len(phrases) <= 200:
        raise RehearsalError("アクセント句が不正です。")
    for phrase in phrases:
        if not isinstance(phrase, dict) or not isinstance(phrase.get("moras"), list):
            raise RehearsalError("アクセント句が不正です。")
        if not 1 <= len(phrase["moras"]) <= 200:
            raise RehearsalError("読みが長すぎます。")
        try:
            accent = int(phrase["accent"])
        except (ValueError, TypeError, KeyError):
            raise RehearsalError("アクセント位置が不正です。") from None
        if not 1 <= accent <= len(phrase["moras"]):
            raise RehearsalError("アクセント位置が読みの範囲外です。")
        phrase["accent"] = accent
    for key, default, low, high in (
        ("speedScale", 1.0, 0.5, 2.0), ("pitchScale", 0.0, -0.15, 0.15),
        ("intonationScale", 1.0, 0.0, 2.0), ("volumeScale", 1.0, 0.0, 2.0),
        ("prePhonemeLength", 0.1, 0.0, 1.0), ("postPhonemeLength", 0.1, 0.0, 1.0),
    ):
        try:
            number = float(q.get(key, default))
        except (ValueError, TypeError):
            raise RehearsalError("音声の調整値が不正です。") from None
        if not math.isfinite(number) or not low <= number <= high:
            raise RehearsalError("音声の調整値が範囲外です。")
        q[key] = number
    q["outputSamplingRate"] = 24000
    q["outputStereo"] = False
    q.pop("kana", None)  # Edited accents supersede the original kana notation.
    return q


class VoiceRehearsal:
    async def _request(self, method, path, *, params=None, data=None, binary=False):
        # No caller-supplied URL, no redirect, no proxy/env credential use.
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60), trust_env=False) as session:
                async with session.request(method, BASE + path, params=params,
                                           json=data, allow_redirects=False) as response:
                    if response.status != 200:
                        raise RehearsalError("VOICEVOXが処理を拒否しました。読みや話者を確認してください。")
                    chunks, size = [], 0
                    async for chunk in response.content.iter_chunked(65536):
                        size += len(chunk)
                        if size > LIMIT:
                            raise RehearsalError("試聴音声が長すぎます。台詞を短くしてください。")
                        chunks.append(chunk)
                    payload = b"".join(chunks)
                    if binary:
                        if not payload.startswith(b"RIFF") or payload[8:12] != b"WAVE":
                            raise RehearsalError("音声データを取得できませんでした。")
                        return payload
                    import json
                    return json.loads(payload)
        except RehearsalError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError):
            raise RehearsalError("VOICEVOXを起動してください（ローカルの50021番ポート）。") from None

    async def speakers(self):
        return await self._request("GET", "/speakers")

    async def query(self, text, speaker):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000 or "<" in text or ">" in text:
            raise RehearsalError("台詞を1〜2000文字で入力してください。制御タグは使えません。")
        return await self._request("POST", "/audio_query",
                                   params={"text": text, "speaker": speaker_id(speaker)})

    async def synthesize(self, query, speaker):
        q = normalize_query(query)
        params = {"speaker": speaker_id(speaker)}
        # Accent edits need fresh mora pitches/durations, not just a new integer.
        q["accent_phrases"] = await self._request("POST", "/mora_data", params=params,
                                                 data=q["accent_phrases"])
        return await self._request("POST", "/synthesis", params=params, data=q, binary=True)
