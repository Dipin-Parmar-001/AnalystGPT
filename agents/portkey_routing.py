"""Portkey request routing helpers shared by the planner and reporter."""

import json
from typing import Any, Dict


THINKING_PORTKEY_CONFIG = {
    "strategy": {
        "mode": "fallback",
        "on_status_codes": [429, 500, 502, 503, 504],
    },
    "targets": [
        {
            "virtual_key": "analystgpt-nvidia",
            "override_params": {
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "temperature": 0.1,
                "max_tokens": 16384,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": True}
                },
            },
        }
    ]
}


def create_with_thinking_route(client: Any, **request_params: Any) -> Any:
    """Try the dedicated Nvidia thinking route, then the client's normal route.

    The client's default Portkey header remains intact for fallback, including
    deployments that use PORTKEY_CONFIG_ID instead of the inline config.
    """
    try:
        return client.chat.completions.create(
            **request_params,
            extra_headers={
                "x-portkey-config": json.dumps(THINKING_PORTKEY_CONFIG)
            },
        )
    except Exception as primary_error:
        print(
            "[Portkey] Dedicated Nvidia thinking route failed; "
            f"retrying with the configured fallback route: {primary_error}"
        )
        return client.chat.completions.create(**request_params)


def extract_reasoning_text(message: Any) -> str:
    """Extract provider reasoning fields without making them workflow-critical."""
    try:
        extra = getattr(message, "model_extra", None)
        if not isinstance(extra, dict):
            extra = {}

        details = getattr(message, "reasoning_details", None)
        if details is None:
            details = extra.get("reasoning_details")
        pieces = []
        if isinstance(details, dict):
            details = [details]
        if isinstance(details, list):
            for item in details:
                if isinstance(item, str):
                    value = item
                elif isinstance(item, dict):
                    value = item.get("text") or item.get("content") or item.get("summary")
                else:
                    value = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(value, str) and value.strip():
                    pieces.append(value.strip())

        for key in ("reasoning_content", "reasoning"):
            value = getattr(message, key, None)
            if value is None:
                value = extra.get(key)
            if isinstance(value, str) and value.strip():
                pieces.append(value.strip())
            elif isinstance(value, dict):
                text = value.get("text") or value.get("content") or value.get("summary")
                if isinstance(text, str) and text.strip():
                    pieces.append(text.strip())

        # Preserve order while avoiding duplicate provider fields.
        return "\n\n".join(dict.fromkeys(pieces))
    except Exception:
        # Reasoning is optional metadata: malformed or unknown formats must not
        # interrupt planning, reporting, or any other part of the workflow.
        return ""
