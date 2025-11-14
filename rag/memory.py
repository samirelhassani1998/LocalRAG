"""Conversation memory helpers (rolling summary)."""
from __future__ import annotations

from typing import Any, Sequence

from adapters import to_chat_messages


def llm_call_simple(client, model: str, prompt: str) -> str:
    """Fire-and-forget helper around the Chat Completions API."""

    messages = to_chat_messages([
        {"role": "system", "content": "Tu résumes des conversations de manière factuelle."},
        {"role": "user", "content": prompt},
    ])
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
        stream=False,
    )
    if not response.choices:
        return ""
    message = getattr(response.choices[0], "message", None)
    content = getattr(message, "content", None)
    return content or ""


def summarize_history(
    client,
    model: str,
    messages: Sequence[dict[str, Any]],
    max_chars: int = 1500,
) -> str:
    """Return a rolling summary of the full conversation."""

    history_lines = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, dict) and item.get("type") in {"input_image", "image_url"}:
                    parts.append("[image]")
            content_str = " ".join(part for part in parts if part)
        else:
            content_str = str(content or "")
        if not content_str:
            continue
        history_lines.append(f"{role}: {content_str}")

    if not history_lines:
        return ""

    joined = "\n".join(history_lines)
    window = joined[-12000:]
    prompt = (
        "Résume en 10 lignes max, factuel, sans perdre d'info clé:\n\n" + window
    )

    try:
        summary = llm_call_simple(client, "gpt-5.1-mini", prompt)
    except Exception:
        return window[:max_chars]

    return (summary or window)[:max_chars]


__all__ = ["summarize_history", "llm_call_simple"]
