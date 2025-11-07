from typing import List, Dict


def to_chat_messages(messages: List[Dict]) -> List[Dict]:
    """
    Convertit messages app -> schéma Chat Completions:
    - user/system: parts {type:"text", text:"..."} et {type:"image_url", image_url:{"url": "..."}}
    - assistant: string simple (ou parts 'text' uniquement)
    """
    out = []
    for m in messages:
        role = m.get("role","user")
        content = m.get("content","")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts = []
            for p in content:
                t = p.get("type")
                if t in ("text","input_text"):  # texte -> 'text'
                    parts.append({"type":"text","text": p.get("text","")})
                elif t in ("input_image","image_url"):
                    # normaliser en image_url {url: ...}
                    iu = p.get("image_url") or p.get("image_url".replace("image_url","image_url"))
                    if isinstance(iu, str):
                        iu = {"url": iu}
                    elif isinstance(iu, dict) and "url" not in iu and "image_url" in iu:
                        iu = {"url": iu["image_url"]}
                    if not iu:
                        iu = {"url": p.get("url") or p.get("data_uri")}
                    parts.append({"type":"image_url","image_url": iu})
                else:
                    # assistant 'output_text' -> texte
                    if t in ("output_text","summary_text","refusal"):
                        parts.append({"type":"text","text": p.get("text","")})
            if role == "assistant" and len(parts)==1 and parts[0].get("type")=="text":
                # string simple OK aussi pour assistant
                out.append({"role": role, "content": parts[0]["text"]})
            else:
                out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": str(content)})
    return out


def to_responses_input(messages: List[Dict]) -> List[Dict]:
    """
    Convertit messages app -> schéma Responses API:
    - user/system: 'input_text' / 'input_image'
    - assistant: 'output_text'
    """
    out = []
    for m in messages:
        role = m.get("role","user")
        content = m.get("content","")
        parts = []
        if isinstance(content, str):
            parts = [{"type":"output_text","text":content}] if role=="assistant" else [{"type":"input_text","text":content}]
        elif isinstance(content, list):
            for p in content:
                t = p.get("type")
                if t in ("text","input_text"):
                    parts.append({"type": "input_text" if role!="assistant" else "output_text", "text": p.get("text","")})
                elif t in ("input_image","image_url"):
                    iu = p.get("image_url")
                    if isinstance(iu, dict) and "url" in iu:
                        data_uri = iu["url"]
                    else:
                        data_uri = iu if isinstance(iu, str) else p.get("url") or p.get("data_uri")
                    parts.append({"type":"input_image","image_url": data_uri})
                elif t in ("output_text","summary_text","refusal"):
                    parts.append(p)
            if not parts:
                parts = [{"type":"input_text","text":""}]
        else:
            parts = [{"type":"input_text","text": str(content)}] if role!="assistant" else [{"type":"output_text","text": str(content)}]
        out.append({"role": role, "content": parts})
    return out
