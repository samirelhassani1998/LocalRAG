from __future__ import annotations

from typing import Any, Dict, List


def to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convertit des messages 'chat' (OpenAI Chat Completions) en 'input' pour Responses API.
    - msg["content"] peut être str (texte) OU list (parts multimodales).
    - Remplace tous les 'text' (chat) par 'input_text' (user/system) ou 'output_text' (assistant).
    """
    converted: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts: List[Dict[str, Any]] = []

        if isinstance(content, str):
            if role == "assistant":
                parts = [{"type": "output_text", "text": content}]
            else:
                parts = [{"type": "input_text", "text": content}]
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    if p is None:
                        continue
                    text_value = str(p)
                    if role == "assistant":
                        parts.append({"type": "output_text", "text": text_value})
                    else:
                        parts.append({"type": "input_text", "text": text_value})
                    continue

                t = p.get("type")
                if t in ("input_image", "computer_screenshot", "input_file"):
                    parts.append(p)
                elif t in ("text", "input_text"):
                    text_value = p.get("text", "")
                    if role == "assistant":
                        parts.append({"type": "output_text", "text": text_value})
                    else:
                        parts.append({"type": "input_text", "text": text_value})
                elif t in ("output_text", "summary_text", "refusal"):
                    parts.append(p)
                else:
                    if "text" in p and t is None:
                        text_value = p["text"]
                        if role == "assistant":
                            parts.append({"type": "output_text", "text": text_value})
                        else:
                            parts.append({"type": "input_text", "text": text_value})
            if not parts:
                parts = [{"type": "input_text", "text": ""}]
        else:
            text_value = str(content)
            if role == "assistant":
                parts = [{"type": "output_text", "text": text_value}]
            else:
                parts = [{"type": "input_text", "text": text_value}]

        converted.append({"role": role, "content": parts})
    return converted
