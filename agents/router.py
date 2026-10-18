"""Local OpenAI-compatible provider routing with ordered error fallback."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _routes() -> List[Tuple[str, str, str]]:
    """Return configured (provider, model, key) routes in fallback order."""
    routes: List[Tuple[str, str, str]] = []

    # 1. NVIDIA API routes
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        nvidia_models: List[str] = [
            "nvidia/nemotron-3-ultra-550b-a55b",
            "nvidia/nemotron-3-super-120b-a12b",
            "meta/llama-3.3-70b-instruct",
        ]
        env_nvidia = os.getenv("NVIDIA_MODEL")
        if env_nvidia and env_nvidia not in nvidia_models:
            nvidia_models.insert(0, env_nvidia)
        routes.extend(("nvidia", model, nvidia_key) for model in nvidia_models)

    # 2. OpenRouter free models
    openrouter_models: List[str] = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "cohere/north-mini-code:free",
        "poolside/laguna-s-2.1:free",
    ]
    env_openrouter = os.getenv("OPENROUTER_MODEL") or os.getenv("ANTHROPIC_MODEL")
    if env_openrouter and env_openrouter not in openrouter_models:
        openrouter_models.insert(0, env_openrouter)

    for key_name in ("OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"):
        key = os.getenv(key_name)
        if key:
            routes.extend(("openrouter", model, key) for model in openrouter_models)

    # 3. Gemini models (include 2.5, 2.0, 1.5 flash for high-demand 503 resilience)
    gemini_models: List[str] = [
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    ]
    env_gemini = os.getenv("GEMINI_MODEL")
    if env_gemini and env_gemini not in gemini_models:
        gemini_models.insert(0, env_gemini)

    for key_name in ("GEMINI_API_KEY_1", "GEMINI_API_KEY_2"):
        key = os.getenv(key_name)
        if key:
            routes.extend(("gemini", model, key) for model in gemini_models)

    # Retain compatibility with deployments using unnumbered environment variables
    for key_name, provider, model_list in (
        ("OPENROUTER_API_KEY", "openrouter", openrouter_models),
        ("GEMINI_API_KEY", "gemini", gemini_models),
    ):
        key = os.getenv(key_name)
        if key:
            routes.extend((provider, model, key) for model in model_list)

    # Deduplicate routes while strictly preserving order
    seen = set()
    deduped_routes: List[Tuple[str, str, str]] = []
    for r in routes:
        if r not in seen:
            seen.add(r)
            deduped_routes.append(r)
    return deduped_routes


class _Completions:
    def create(self, **params: Any) -> Any:
        routes = _routes()
        if not routes:
            raise RuntimeError("No provider API keys configured for local routing.")

        errors = []
        params.pop("timeout", None)

        for provider, model, key in routes:
            request: Dict[str, Any] = dict(params)
            request["model"] = model
            request.setdefault("temperature", 0.1)

            # CRITICAL: Always pop 'reasoning' so it is never passed as a top-level kwarg
            # to client.chat.completions.create(), which raises:
            # TypeError: Completions.create() got an unexpected keyword argument 'reasoning'
            reasoning = request.pop("reasoning", None)

            if provider == "openrouter":
                base_url = OPENROUTER_BASE_URL
                extra_headers = dict(request.get("extra_headers") or {})
                extra_headers.setdefault("HTTP-Referer", "https://github.com/AnalystGPT")
                extra_headers.setdefault("X-Title", "AnalystGPT")
                request["extra_headers"] = extra_headers

                # For OpenRouter, reasoning belongs in extra_body, not top-level kwargs
                if reasoning is not None:
                    extra_body = dict(request.get("extra_body") or {})
                    extra_body["reasoning"] = reasoning
                    request["extra_body"] = extra_body
                elif any(k in model.lower() for k in ("nemotron", "r1", "reasoning", "deepseek")):
                    extra_body = dict(request.get("extra_body") or {})
                    extra_body.setdefault("reasoning", {"effort": "low"})
                    request["extra_body"] = extra_body
            elif provider == "nvidia":
                base_url = NVIDIA_BASE_URL
                extra_body = request.get("extra_body")
                if isinstance(extra_body, dict):
                    extra_body = dict(extra_body)
                    extra_body.pop("reasoning", None)
                    if extra_body:
                        request["extra_body"] = extra_body
                    else:
                        request.pop("extra_body", None)
            else:
                # Gemini
                base_url = GEMINI_BASE_URL
                extra_body = request.get("extra_body")
                if isinstance(extra_body, dict):
                    extra_body = dict(extra_body)
                    extra_body.pop("reasoning", None)
                    if extra_body:
                        request["extra_body"] = extra_body
                    else:
                        request.pop("extra_body", None)
                request.setdefault("max_tokens", 7000)

            try:
                client = OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=None)
                return client.chat.completions.create(**request)
            except Exception as exc:
                # If OpenRouter rejects reasoning in extra_body with HTTP 400, retry once without reasoning
                if provider == "openrouter" and "extra_body" in request and "reasoning" in request.get("extra_body", {}):
                    try:
                        retry_req = dict(request)
                        retry_eb = dict(retry_req["extra_body"])
                        retry_eb.pop("reasoning", None)
                        if retry_eb:
                            retry_req["extra_body"] = retry_eb
                        else:
                            retry_req.pop("extra_body", None)
                        return client.chat.completions.create(**retry_req)
                    except Exception as retry_exc:
                        exc = retry_exc

                errors.append(f"{provider}/{model}: {type(exc).__name__}: {exc}")

        raise RuntimeError("All local LLM routes failed: " + " | ".join(errors))


class _Chat:
    def __init__(self) -> None:
        self.completions = _Completions()


class LocalRouterClient:
    """Small client facade matching the OpenAI chat-completions call surface."""

    def __init__(self, agent_name: str = "LLM") -> None:
        self.chat = _Chat()
        self.agent_name = agent_name


def create_local_router(agent_name: str = "LLM") -> LocalRouterClient:
    return LocalRouterClient(agent_name)
