"""High-level orchestration for the upgraded RAG pipeline."""
from __future__ import annotations

import time
from typing import Any, Dict, Sequence

from adapters import to_chat_messages
from config import PerfConfig

from .memory import summarize_history
from .prompts import IMPROVE_TEMPLATE, SYSTEM_BASE, USER_TEMPLATE
from .retriever import build_context, retrieve


def _extract_text_from_choice(choice: Any) -> str:
    if choice is None:
        return ""
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return ""


def llm_chat_or_responses(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 2000,
    streaming: bool = True,
) -> tuple[str, Dict[str, Any]]:
    """Call the Chat Completions API with optional streaming support."""

    payload_messages = to_chat_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    def _usage_dict(usage_obj: Any | None) -> Dict[str, Any]:
        if usage_obj is None:
            return {}
        data = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage_obj, key, None)
            if value is not None:
                data[key] = value
        return data

    try:
        if streaming:
            stream = client.chat.completions.create(
                model=model,
                messages=payload_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            parts: list[str] = []
            usage_info: Dict[str, Any] = {}
            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                delta = getattr(choice.delta, "content", None) if choice else None
                if delta:
                    parts.append(delta)
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage_info = _usage_dict(chunk_usage)
            return "".join(parts), usage_info

        completion = client.chat.completions.create(
            model=model,
            messages=payload_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=False,
        )
        text = _extract_text_from_choice(completion.choices[0]) if completion.choices else ""
        return text, _usage_dict(getattr(completion, "usage", None))
    except Exception:
        if streaming:
            return llm_chat_or_responses(
                client,
                model,
                system_prompt,
                user_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                streaming=False,
            )
        raise


def run_rag_pipeline(
    client,
    model: str,
    vectorstore,
    query: str,
    messages: Sequence[dict[str, Any]],
    *,
    cfg: PerfConfig,
) -> Dict[str, Any]:
    """Execute the RAG workflow using the provided performance configuration."""

    diagnostics: Dict[str, Any] = {"model": model, "k": cfg.rag_k, "mode": "performance"}

    history_start = time.perf_counter()
    history_summary = summarize_history(client, model, messages)
    diagnostics["history_time_s"] = round(time.perf_counter() - history_start, 3)
    diagnostics["history_chars"] = len(history_summary)

    retrieval_start = time.perf_counter()
    docs = retrieve(vectorstore, query, k=cfg.rag_k, state=cfg)
    diagnostics["retrieval_time_s"] = round(time.perf_counter() - retrieval_start, 3)
    diagnostics["retrieved_docs"] = len(docs)
    diagnostics["rerank_used"] = bool(cfg.use_reranker and not cfg.use_mmr)

    context, sources = build_context(docs)

    system_prompt = SYSTEM_BASE
    user_prompt = USER_TEMPLATE.format(
        k=cfg.rag_k,
        context=context or "(aucun contexte)",
        history=history_summary or "(aucun historique)",
        query=query,
    )

    pass1_start = time.perf_counter()
    draft, usage_pass1 = llm_chat_or_responses(
        client,
        model,
        system_prompt,
        user_prompt,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        streaming=cfg.streaming,
    )
    diagnostics["pass1_time_s"] = round(time.perf_counter() - pass1_start, 3)
    diagnostics["pass1_usage"] = usage_pass1

    final_answer = draft
    usage_pass2: Dict[str, Any] | None = None

    if cfg.use_multipass:
        improve_prompt = IMPROVE_TEMPLATE.format(draft=draft)
        pass2_start = time.perf_counter()
        final_answer, usage_pass2 = llm_chat_or_responses(
            client,
            model,
            system_prompt,
            improve_prompt,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            streaming=cfg.streaming,
        )
        diagnostics["pass2_time_s"] = round(time.perf_counter() - pass2_start, 3)
        diagnostics["pass2_usage"] = usage_pass2
    else:
        diagnostics["pass2_skipped"] = True

    if "### Sources" not in (final_answer or ""):
        source_lines = "\n".join([f"- [{idx}] {src}" for idx, src in sources]) if sources else "- Aucun contexte exploitable"
        final_answer = (final_answer or "").rstrip() + f"\n\n### Sources\n{source_lines}"


    result = {
        "answer": final_answer,
        "draft": draft,
        "sources": sources,
        "context": context,
        "history": history_summary,
        "diagnostics": diagnostics,
        "documents": docs,
    }

    if usage_pass2 is None:
        result["usage"] = usage_pass1
    else:
        merged_usage = dict(usage_pass1 or {})
        for key, value in (usage_pass2 or {}).items():
            if value is None:
                continue
            previous = merged_usage.get(key)
            if isinstance(previous, (int, float)) and isinstance(value, (int, float)):
                merged_usage[key] = previous + value
            else:
                merged_usage[key] = value
        result["usage"] = merged_usage

    return result


__all__ = ["llm_chat_or_responses", "run_rag_pipeline"]
