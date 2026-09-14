# -*- coding: utf-8 -*-
"""AI Director provider interface + local Gemma (llama-cpp) adapter.

Director logic talks only to DirectorProvider. Gemma specifics (node types,
widgets, graph shape, unload discipline) live in GemmaDirectorProvider.
External providers (OpenAI/Gemini/Claude) can implement the same interface
later without touching director logic, spec handling or UI.
"""
from __future__ import annotations

import json as _json
from typing import Callable


class DirectorError(RuntimeError):
    """Typed failure: message is user-facing Japanese."""

    def __init__(self, message: str, *, kind: str = ""):
        super().__init__(message)
        self.kind = kind


class DirectorProvider:
    """Interface. generate() returns the parsed JSON object from the model."""

    async def generate(self, *, system: str, user: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.7) -> dict:
        raise NotImplementedError


def parse_json_answer(text: str) -> dict:
    """Parse a model answer that should be a JSON object (fences tolerated).

    The model may emit reasoning (including inline {...} examples and
    <channel|> markers) before the real answer, so the LAST balanced
    top-level {...} object wins - never the first brace in the text.
    """
    import re as _re
    cleaned = (text or "").strip()
    cleaned = _re.sub(r"<\|?channel\|?>", " ", cleaned)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    decoder = _json.JSONDecoder()
    # Scan fenced and unfenced answers alike: an earlier fenced example must
    # not override the model's final object.
    best: dict | None = None
    cursor = 0
    while cursor < len(cleaned):
        pos = cleaned.find("{", cursor)
        if pos < 0:
            break
        try:
            obj, end = decoder.raw_decode(cleaned[pos:])
        except Exception:                                    # noqa: BLE001
            cursor = pos + 1
            continue
        cursor = pos + end
        if isinstance(obj, dict):
            best = obj
    if best is not None:
        return best
    _dump_raw(text)
    raise DirectorError("AIの回答をJSONとして読めませんでした。再生成してください。")


def _dump_raw(text: str) -> None:
    """Keep the unparseable answer for diagnosis (best-effort)."""
    try:
        import time as _time
        from pathlib import Path as _Path
        debug = _Path(__file__).resolve().parent.parent / "_debug"
        debug.mkdir(parents=True, exist_ok=True)
        stamp = int(_time.time())
        (_Path(debug) / f"director_raw_{stamp}.txt").write_text(
            text or "", encoding="utf-8", errors="replace")
    except Exception:                                        # noqa: BLE001
        pass



class GemmaDirectorProvider(DirectorProvider):
    """Local Gemma VLM via the ComfyUI llama-cpp nodes."""

    def __init__(self, client, vlm: dict,
                  on_event: Callable[[dict], None] | None = None):
        self._client = client
        self._vlm = dict(vlm)
        self._on_event = on_event or (lambda ev: None)
        self.last_info: dict = {}

    def _graph(self, *, system: str, user: str, max_tokens: int,
               temperature: float) -> dict:
        from . import graphs as graphs_mod
        graph: dict = {}
        graphs_mod._vlm_loader(graph, self._vlm)
        graph["params"] = {
            "class_type": "llama_cpp_parameters",
            "inputs": {
                "max_tokens": int(max_tokens), "top_k": 40, "top_p": 0.92,
                "min_p": 0.05, "typical_p": 1.0,
                "temperature": float(temperature),
                "repeat_penalty": 1.05, "frequency_penalty": 0.0,
                "present_penalty": 0.0, "mirostat_mode": 0,
                "mirostat_eta": 0.1, "mirostat_tau": 5.0, "state_uid": -1,
            },
        }
        instruct_inputs: dict = {
            "llama_model": ["vlm", 0],
            "preset_prompt": "Empty - Nothing",
            "custom_prompt": user,
            "system_prompt": system,
            "inference_mode": "images",
            "max_frames": 8,
            "max_size": 768,
            "seed": 1,
            "force_offload": False,
            "save_states": False,
            "parameters": ["params", 0],
        }
        graph["instruct"] = {
            "class_type": "llama_cpp_instruct_adv",
            "inputs": instruct_inputs,
        }
        graph["preview"] = {"class_type": "PreviewAny",
                            "inputs": {"source": ["instruct", 0]}}
        return graph

    async def generate(self, *, system: str, user: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.7) -> dict:
        from . import comfy as comfy_mod
        import time as _time
        started = _time.monotonic()
        graph = self._graph(system=system, user=user,
                            max_tokens=max_tokens, temperature=temperature)
        try:
            # VLM traffic is flag-agnostic: never trigger a profile restart.
            history = await self._client.run_graph(
                graph, on_event=self._on_event, profile=None)
        except Exception as exc:                                 # noqa: BLE001
            raise DirectorError(f"Directorの生成に失敗しました: {exc}") from exc
        try:
            text = comfy_mod.text_output(history, "preview")
        except Exception as exc:                                 # noqa: BLE001
            raise DirectorError("Directorの回答を取得できませんでした。") from exc
        self.last_info = {
            "http_status": 0,
            "elapsed_ms": int((_time.monotonic() - started) * 1000),
            "retry_after": "",
            "usage": {"input_tokens": 0, "output_tokens": 0,
                      "reasoning_tokens": 0},
            "incomplete": "",
            "endpoint": "local-gemma",
        }
        return parse_json_answer(text)

    async def unload(self) -> None:
        """Free the VLM before video generation (same tail as graph B)."""
        from . import graphs as graphs_mod
        graph = graphs_mod.build_vlm_unload_graph()
        try:
            await self._client.run_graph(graph, on_event=self._on_event,
                                         profile=None)
        except Exception:                                        # noqa: BLE001
            pass


class AdapterDirectorProvider(DirectorProvider):
    """Generic Director adapter over any WP-A llm_providers adapter.

    Replaces the former OpenAICompatProvider / OpenCodeGoProvider pair:
    callers resolve the effective provider via ai_settings.role_adapter()
    and wrap it here. Gemma keeps its own GemmaDirectorProvider (ComfyUI
    graph path, untouched).
    """

    def __init__(self, adapter, model: str = ""):
        self._adapter = adapter
        if model:
            self._adapter.model = str(model)
        self.last_info: dict = {}

    @property
    def provider_id(self) -> str:
        spec = getattr(self._adapter, "spec", None)
        return str(getattr(spec, "id", "") or "")

    async def generate(self, *, system: str, user: str,
                       max_tokens: int = 2048,
                       temperature: float = 0.7) -> dict:
        try:
            from .llm_providers import LLMError as _LLMError
        except Exception:                                    # noqa: BLE001
            _LLMError = None
        try:
            text, info = await self._adapter.chat(
                system=system, user=user, images=[],
                max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:                                 # noqa: BLE001
            kind = str(getattr(exc, "kind", "") or "")
            self.last_info = {
                "http_status": 0, "elapsed_ms": 0, "retry_after": "",
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "reasoning_tokens": 0},
                "incomplete": "", "endpoint": self.provider_id,
            }
            raise DirectorError(str(exc), kind=kind) from exc
        self.last_info = dict(info)
        try:
            return parse_json_answer(text)
        except DirectorError as exc:
            raise DirectorError(str(exc)) from exc

    async def unload(self) -> None:
        """Nothing resident on the local GPU process for API/server
        providers; LM Studio-style unload goes through adapter.release()
        in pipeline._release_local_llm, never here."""
        return None


class OpenAICompatProvider(AdapterDirectorProvider):
    """Deprecated alias: build an LM Studio adapter instead.

    Kept so old imports keep working; new code uses AdapterDirectorProvider
    with ai_settings.role_adapter().
    """

    def __init__(self, *, base_url: str, model: str, token: str = ""):
        from .llm_providers.lmstudio import LMStudioAdapter
        super().__init__(LMStudioAdapter(base_url=base_url, model=model,
                                         token=token))


class OpenCodeGoProvider(AdapterDirectorProvider):
    """Deprecated alias: build an OpenCode Go adapter instead.

    Kept so old imports keep working; new code uses AdapterDirectorProvider
    with ai_settings.role_adapter().
    """

    def __init__(self, *, api_key: str, model: str, endpoint: str = "",
                 session_id: str = ""):
        from .llm_providers.opencode_go import OpenCodeGoAdapter
        super().__init__(OpenCodeGoAdapter(
            model=model, token=api_key,
            extra={"endpoint": endpoint or "responses",
                   "session_id": session_id or "h3-director"}))
