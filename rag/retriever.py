"""Enhanced retrieval and reranking utilities for the Streamlit RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol, Sequence, Tuple

@dataclass
class Document:
    """Simple container mimicking LangChain's Document interface."""

    page_content: str
    metadata: dict[str, Any] | None = None


_CROSS_ENCODER_MODEL = None
_CROSS_ENCODER_ERROR: Exception | None = None


def _get_cross_encoder():
    """Lazy-load the HuggingFace cross-encoder when available."""

    global _CROSS_ENCODER_MODEL, _CROSS_ENCODER_ERROR
    if _CROSS_ENCODER_MODEL is not None:
        return _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_ERROR is not None:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER_MODEL = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation="softmax"
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        _CROSS_ENCODER_ERROR = exc
        return None
    return _CROSS_ENCODER_MODEL


def _bm25_rerank(query: str, docs: Sequence[Document], k: int) -> List[Document]:
    try:
        from rank_bm25 import BM25Okapi
    except Exception:  # pragma: no cover - optional dependency
        return list(docs[:k])

    tokenized_corpus = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]


class RetrieverState(Protocol):
    use_mmr: bool
    mmr_fetch_k: int
    mmr_lambda: float
    use_reranker: bool


def retrieve(
    vectorstore,
    query: str,
    k: int,
    state: RetrieverState | Any,
) -> List[Document]:
    """Return the top-k documents using configuration-driven logic."""

    if vectorstore is None:
        return []

    target_k = max(int(k or 0), 1)
    docs: Sequence[Document] = []

    use_mmr = bool(getattr(state, "use_mmr", False))
    if use_mmr:
        fetch_k = max(int(getattr(state, "mmr_fetch_k", target_k * 6)), target_k * 6)
        lambda_mult = float(getattr(state, "mmr_lambda", 0.5))
        try:
            docs = vectorstore.max_marginal_relevance_search(
                query,
                k=target_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            ) or []
            docs = [doc for doc in docs if doc and getattr(doc, "page_content", None)]
            if docs:
                return list(docs[:target_k])
        except Exception:
            pass

    fetch_k = max(target_k * 3, target_k)
    try:
        docs = vectorstore.similarity_search(query, k=fetch_k) or []
    except Exception:
        docs = []
    docs = [doc for doc in docs if doc and getattr(doc, "page_content", None)]
    if not docs:
        return []

    if not bool(getattr(state, "use_reranker", False)):
        return list(docs[:target_k])

    model = _get_cross_encoder()
    if model is not None:
        pairs = [(query, doc.page_content) for doc in docs]
        try:
            scores = model.predict(pairs)
        except Exception:  # pragma: no cover - inference failure
            reranked = _bm25_rerank(query, docs, target_k)
        else:
            ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
            reranked = [doc for doc, _ in ranked[:target_k]]
    else:
        reranked = _bm25_rerank(query, docs, target_k)

    return reranked


def build_context(docs: Iterable[Document], with_sources: bool = True) -> Tuple[str, List[Tuple[int, str]]]:
    """Format retrieved documents into a numbered context block and collect sources."""

    context_lines: List[str] = []
    sources: List[Tuple[int, str]] = []
    for i, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", None) or {}
        context_lines.append(f"[{i}] {doc.page_content}")
        source = (
            metadata.get("source")
            or metadata.get("file")
            or metadata.get("path")
            or metadata.get("name")
            or f"doc_{i}"
        )
        sources.append((i, str(source)))
    return "\n\n".join(context_lines), sources if with_sources else []


__all__ = ["Document", "retrieve", "build_context"]
