"""Shared helpers retained for callers while routing locally."""

from typing import Any, Dict


def create_with_thinking_route(client: Any, **request_params: Any) -> Any:
    """Use local provider routing; retain the legacy argument for compatibility."""
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
