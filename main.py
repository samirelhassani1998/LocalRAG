import os
import re
import urllib.parse as _url
from html import escape
from typing import Any, Dict, List, Optional, Sequence
from types import SimpleNamespace
from dataclasses import replace

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from adapters import to_chat_messages, to_responses_input

from image_utils import to_image_part
from rag_utils import (
    ALLOW_LARGE_FILES,
    DEFAULT_MAX_FILE_MB,
    FAISS_IMPORT_ERROR,
    add_embeddings_to_index,
    embed_texts,
    clear_status_callbacks,
    configure_status_callbacks,
    format_context,
    format_source_badge,
    format_bytes,
    human_readable_size,
    index_files_from_chat,
    load_file_to_chunks,
)
from token_utils import (
    count_tokens_chat,
    truncate_context_text,
    truncate_messages_to_budget,
)
from utils.rendering import extract_code_block, try_extract_dbml_heuristic
from rag.pipeline import run_rag_pipeline
from rag.retriever import Document
from quality.escalation import need_quality_escalation, output_is_poor, expects_dbml
from config import PerfConfig

load_dotenv()

st.set_page_config(page_title="ChatGPT-like Chatbot", layout="wide")


def _init_perf_state() -> None:
    """Initialize high-performance defaults once per session."""

    st.session_state.setdefault("selected_model", "gpt-4o-mini")

    # RAG performance tuning
    st.session_state.setdefault("rag_k", 4)
    st.session_state.setdefault("use_mmr", True)
    st.session_state.setdefault("mmr_fetch_k", 24)
    st.session_state.setdefault("mmr_lambda", 0.5)

    # No reranker / multipass in perf mode
    st.session_state.setdefault("use_reranker", False)
    st.session_state.setdefault("use_multipass", False)

    # Generation parameters tuned for speed
    st.session_state.setdefault("gen_temperature", 0.6)
    st.session_state.setdefault("gen_top_p", 0.9)
    st.session_state.setdefault("gen_max_tokens", 900)
    st.session_state.setdefault("gen_streaming", True)

    st.session_state.setdefault("mode", "performance")
    st.session_state.setdefault("quality_escalated", False)


_init_perf_state()

if "app_config" not in st.session_state:
    base_cfg = PerfConfig()
    if os.getenv("QUALITY_ESCALATION", "0") == "1":
        base_cfg = PerfConfig(quality_escalation=True)
    st.session_state.app_config = base_cfg

CFG: PerfConfig = st.session_state.app_config

AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-5",
]


PREFERRED_MODELS = ["gpt-5", "gpt-4o", "gpt-4o-mini"]


EMBEDDING_MODEL = "text-embedding-3-large"
VISION_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-5"}
MAX_INPUT_TOKENS = 300_000
RESERVE_OUTPUT_TOKENS = 1_000
MAX_RAG_CONTEXT_TOKENS = 30_000
MAX_FILES = 5
MAX_FILE_BYTES = DEFAULT_MAX_FILE_MB * 1024 * 1024
CHUNK_MAX_CHARS = 4000
CHUNK_OVERLAP = 400
MAX_IMAGE_ATTACHMENTS = 5

SIDEBAR_UPLOAD_KEY = "sidebar_uploaded_files"
CHAT_FILE_UPLOAD_KEY = "chat_file_uploader"
CHAT_IMAGE_UPLOAD_KEY = "chat_image_uploader"


try:  # pragma: no cover - optional dependency check
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch absent
    TORCH_AVAILABLE = False
else:  # pragma: no cover - torch present
    TORCH_AVAILABLE = True


DEFAULT_RETRIEVAL_K = 8
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 2000
MAX_TOKENS_MIN = 512
MAX_TOKENS_MAX = 8192

PERFORMANCE_DEFAULT_K = 4
PERFORMANCE_DEFAULT_MAX_TOKENS = 900
PERFORMANCE_DEFAULT_TEMPERATURE = 0.6
PERFORMANCE_DEFAULT_TOP_P = 0.9


BASE_GLOBAL_SYSTEM_PROMPT = (
    "Tu es un assistant conversationnel utile et professionnel. Réponds en français lorsque c'est pertinent."
)
FORMAT_ENFORCEMENT_BLOCK = """
FORMAT DE SORTIE OBLIGATOIRE
1. Un titre H3.
2. Un résumé en puces (3–5 bullets max).
3. Un bloc de code unique correctement délimité par des backticks, avec la bonne étiquette :
   - DBML → ```dbml
   - SQL  → ```sql
   - Python → ```python
   - JSON → ```json
Ne jamais mettre de texte après le bloc de code.
Si l’utilisateur demande un modèle DBML, produire exclusivement un bloc ```dbml.
"""

GLOBAL_SYSTEM_PROMPT = BASE_GLOBAL_SYSTEM_PROMPT + "\n\n" + FORMAT_ENFORCEMENT_BLOCK
GLOBAL_SYSTEM_MESSAGE = {"role": "system", "content": GLOBAL_SYSTEM_PROMPT}

DBML_ENFORCEMENT_PROMPT = (
    "Quand l’utilisateur mentionne DBML/dbdiagram, renvoie UNIQUEMENT un bloc ```dbml sans explication après."
)


def effective_params_from_mode() -> PerfConfig:
    return PerfConfig(
        default_model=st.session_state.get("selected_model", CFG.default_model),
        rag_k=int(st.session_state.get("rag_k", CFG.rag_k)),
        use_mmr=bool(st.session_state.get("use_mmr", CFG.use_mmr)),
        mmr_fetch_k=int(st.session_state.get("mmr_fetch_k", CFG.mmr_fetch_k)),
        mmr_lambda=float(st.session_state.get("mmr_lambda", CFG.mmr_lambda)),
        use_reranker=bool(st.session_state.get("use_reranker", CFG.use_reranker)),
        use_multipass=bool(st.session_state.get("use_multipass", CFG.use_multipass)),
        temperature=float(st.session_state.get("gen_temperature", CFG.temperature)),
        top_p=float(st.session_state.get("gen_top_p", CFG.top_p)),
        max_tokens=int(st.session_state.get("gen_max_tokens", CFG.max_tokens)),
        streaming=bool(st.session_state.get("gen_streaming", CFG.streaming)),
        quality_escalation=CFG.quality_escalation,
    )


def with_quality_boost(params: PerfConfig) -> PerfConfig:
    if not CFG.quality_escalation:
        return params
    return replace(
        params,
        rag_k=max(params.rag_k, 8),
        use_multipass=True,
        use_reranker=True,
        max_tokens=max(params.max_tokens, 2000),
        temperature=max(params.temperature, 0.7),
        top_p=max(params.top_p, 0.95),
    )


def _contains_image_parts(message: Optional[Dict[str, Any]]) -> bool:
    if not message:
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for part in content:
        if isinstance(part, dict) and part.get("type") in {"input_image", "image_url"}:
            return True
    return False


def is_org_verify_stream_error(e: Exception) -> bool:
    try:
        err = getattr(e, "response", None) or getattr(e, "error", None) or {}
        msg = str(getattr(err, "message", "")) or str(e)
        code = getattr(err, "code", "") or ""
        param = getattr(err, "param", "") or ""
        return ("must be verified to stream this model" in msg.lower()) or (
            code == "unsupported_value" and param == "stream"
        )
    except Exception:
        return False


def _init_session_state() -> None:
    env_key = os.getenv("OPENAI_API_KEY")
    st.session_state.setdefault("api_key", env_key if env_key else None)
    st.session_state.setdefault("messages", [dict(GLOBAL_SYSTEM_MESSAGE)])
    st.session_state.setdefault("selected_model", CFG.default_model)
    st.session_state.setdefault("rag_index", None)
    st.session_state.setdefault("rag_texts", [])
    st.session_state.setdefault("rag_meta", [])
    st.session_state.setdefault("rag_docs", [])
    st.session_state.setdefault("rag_embedding_model", EMBEDDING_MODEL)
    st.session_state.setdefault("rag_diagnostics", None)
    st.session_state.setdefault("chat_attachments", [])
    st.session_state.setdefault("chat_images", [])
    st.session_state.setdefault("show_logs", False)
    st.session_state.setdefault("last_call_log", None)
    st.session_state.setdefault("_sidebar_feedback", None)
    st.session_state.setdefault("_chat_file_warning", [])
    st.session_state.setdefault("_chat_image_warning", [])
    st.session_state.setdefault("_pending_rag_reset", False)
    st.session_state.setdefault("_pending_rag_index", False)
    st.session_state.setdefault("_pending_chat_submit", None)
    st.session_state.setdefault(SIDEBAR_UPLOAD_KEY, [])
    st.session_state.setdefault(CHAT_FILE_UPLOAD_KEY, [])
    st.session_state.setdefault(CHAT_IMAGE_UPLOAD_KEY, [])


def _reset_chat() -> None:
    st.session_state.messages = [dict(GLOBAL_SYSTEM_MESSAGE)]


def _reset_rag_state() -> None:
    st.session_state.rag_index = None
    st.session_state.rag_texts = []
    st.session_state.rag_meta = []
    st.session_state.rag_docs = []
    st.session_state.rag_embedding_model = EMBEDDING_MODEL
    st.session_state.rag_diagnostics = None


def _remove_api_key() -> None:
    st.session_state.api_key = None
    _reset_chat()
    _reset_rag_state()
    st.rerun()


def _request_rag_reset() -> None:
    st.session_state["_pending_rag_reset"] = True
    st.rerun()


def _request_indexing() -> None:
    st.session_state["_pending_rag_index"] = True
    st.rerun()


def _process_pending_actions() -> None:
    feedback: Optional[tuple[str, str]] = None

    st.session_state["_sidebar_feedback"] = None

    if st.session_state.pop("_pending_rag_reset", False):
        _reset_rag_state()
        feedback = ("success", "Base documentaire réinitialisée.")

    if st.session_state.pop("_pending_rag_index", False):
        files = st.session_state.get(SIDEBAR_UPLOAD_KEY) or []
        if files:
            _handle_indexing(files)
        else:
            feedback = ("warning", "Aucun fichier sélectionné pour l'indexation.")

    if feedback is not None:
        st.session_state["_sidebar_feedback"] = feedback


def set_defaults_if_needed(*, force: bool = False) -> None:
    _ = force
    _init_perf_state()


def _rag_is_ready() -> bool:
    return bool(st.session_state.rag_index is not None and st.session_state.rag_texts)


class SessionVectorStore:
    """Lightweight adapter exposing a similarity_search interface for the RAG pipeline."""

    def __init__(
        self,
        client: OpenAI,
        index,
        texts: Sequence[str],
        metas: Sequence[Dict[str, Any]],
        embedding_model: str,
    ) -> None:
        self._client = client
        self._index = index
        self._texts = list(texts or [])
        self._metas = list(metas or [])
        self._embedding_model = embedding_model or EMBEDDING_MODEL

    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        if self._index is None or not self._texts:
            return []
        query_embedding = embed_texts(self._client, self._embedding_model, [query])
        if getattr(query_embedding, "size", 0) == 0:
            return []
        limit = min(k, len(self._texts))
        if limit <= 0:
            return []
        _distances, indices = self._index.search(query_embedding, limit)
        documents: List[Document] = []
        seen: set[int] = set()
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._texts) or idx in seen:
                continue
            seen.add(int(idx))
            documents.append(
                Document(
                    page_content=self._texts[idx],
                    metadata=dict(self._metas[idx] or {}),
                )
            )
            if len(documents) >= limit:
                break
        return documents

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = PERFORMANCE_DEFAULT_K,
        fetch_k: int = 24,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        _ = lambda_mult  # unused but kept for signature compatibility
        limit = max(fetch_k, k)
        docs = self.similarity_search(query, k=limit)
        return docs[:k]


def _format_sources_line(sources: Sequence[Dict[str, Any]]) -> str:
    if not sources:
        return ""
    badges = [format_source_badge(entry["meta"], entry["index"]) for entry in sources]
    return "Sources : " + ", ".join(badges)


def _build_source_entries(hits: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        _, meta, score = hit
        entries.append({"index": idx, "meta": meta, "score": score})
    return entries


def _session_retrieve(vectorstore: SessionVectorStore, query: str, *, k: Optional[int] = None) -> List[Document]:
    state = st.session_state
    top_k = int(k or state.get("rag_k", PERFORMANCE_DEFAULT_K))
    if state.get("use_mmr", True):
        fetch_k = max(int(state.get("mmr_fetch_k", 24)), top_k * 6)
        lambda_mult = float(state.get("mmr_lambda", 0.5))
        try:
            return vectorstore.max_marginal_relevance_search(
                query,
                k=top_k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
        except Exception:
            pass
    return vectorstore.similarity_search(query, k=top_k)


def _handle_indexing(uploaded_files: Sequence[Any]) -> None:
    if not uploaded_files:
        return

    if FAISS_IMPORT_ERROR is not None:
        st.error(f"faiss-cpu est requis pour activer le RAG : {FAISS_IMPORT_ERROR}")
        return

    if len(uploaded_files) > MAX_FILES:
        st.warning(f"Merci de limiter l'indexation à {MAX_FILES} fichiers simultanés.")
        return

    oversized: List[tuple[str, int]] = []
    valid_files = []
    for uploaded_file in uploaded_files:
        size = getattr(uploaded_file, "size", None)
        if size is None:
            try:
                size = len(uploaded_file.getvalue())
            except Exception:
                size = 0
        if not ALLOW_LARGE_FILES and size and size > MAX_FILE_BYTES:
            oversized.append((uploaded_file.name, size))
            continue
        valid_files.append(uploaded_file)

    if oversized:
        st.warning(
            "\n".join(
                [
                    f"Les fichiers suivants dépassent {DEFAULT_MAX_FILE_MB} Mo et ont été ignorés :",
                    *[f"• {name} ({format_bytes(size)})" for name, size in oversized],
                ]
            )
        )

    if not valid_files:
        st.info("Aucun fichier indexé.")
        return

    with st.status("Indexation en cours…", expanded=False) as status:
        status.update(label="📥 Lecture des fichiers…", state="running")
        chunk_texts: List[str] = []
        chunk_metas: List[Dict[str, Any]] = []
        doc_summaries: List[Dict[str, Any]] = []
        ingestion_warnings: List[str] = []

        for uploaded_file in valid_files:
            try:
                chunks, summary, warnings = load_file_to_chunks(
                    uploaded_file,
                    max_chars=CHUNK_MAX_CHARS,
                    overlap=CHUNK_OVERLAP,
                )
            except Exception as exc:  # noqa: BLE001 - surface in UI
                ingestion_warnings.append(f"{uploaded_file.name} : {exc}")
                continue

            if summary:
                doc_summaries.append(summary)
            if warnings:
                ingestion_warnings.extend(warnings)
            if not chunks:
                status.write(f"⚠️ {uploaded_file.name} — aucun texte exploitable.")
                continue

            status.write(f"✔️ {uploaded_file.name} — {len(chunks)} chunks")
            for chunk in chunks:
                chunk_texts.append(chunk.text)
                chunk_metas.append(chunk.meta)

        if not chunk_texts:
            status.update(label="⚠️ Aucun contenu indexable", state="error")
            for warn in ingestion_warnings:
                st.warning(warn)
            return

        status.update(label="🧠 Calcul des embeddings…", state="running")
        client = OpenAI(api_key=st.session_state.api_key)
        embedding_model = st.session_state.rag_embedding_model or EMBEDDING_MODEL
        embeddings = embed_texts(client, embedding_model, chunk_texts)

        if embeddings.size == 0:
            status.update(label="⚠️ Impossible de calculer les embeddings", state="error")
            return

        status.update(label="📚 Construction de l'index FAISS…", state="running")
        try:
            st.session_state.rag_index = add_embeddings_to_index(
                st.session_state.rag_index,
                embeddings,
            )
        except Exception as exc:  # noqa: BLE001 - surface in UI
            status.update(label="❌ Échec de la mise à jour de l'index", state="error")
            st.error(f"Impossible de mettre à jour l'index FAISS : {exc}")
            return

        st.session_state.rag_texts.extend(chunk_texts)
        st.session_state.rag_meta.extend(chunk_metas)
        st.session_state.rag_docs.extend(doc_summaries)
        st.session_state.rag_embedding_model = embedding_model

        status.update(label="✅ Indexation terminée", state="complete")

    st.success(f"{len(chunk_texts)} nouveaux chunks ont été ajoutés à la base documentaire.")
    for warn in ingestion_warnings:
        st.warning(warn)


def _add_chat_attachments() -> None:
    files = st.session_state.get(CHAT_FILE_UPLOAD_KEY) or []
    attachments = list(st.session_state.get("chat_attachments", []))
    existing = {(getattr(f, "name", None), getattr(f, "size", None)) for f in attachments}
    warnings: List[str] = []

    for upload in files:
        key = (getattr(upload, "name", None), getattr(upload, "size", None))
        if key in existing:
            continue
        if len(attachments) >= MAX_FILES:
            warnings.append(f"Maximum {MAX_FILES} fichiers par envoi.")
            break
        size = getattr(upload, "size", None)
        if size is None:
            try:
                size = len(upload.getvalue())
            except Exception:
                size = 0
        if not ALLOW_LARGE_FILES and size and size > MAX_FILE_BYTES:
            warnings.append(
                f"{getattr(upload, 'name', 'Fichier')} dépasse la limite de {DEFAULT_MAX_FILE_MB} Mo et a été ignoré."
            )
            continue
        attachments.append(upload)
        existing.add(key)

    st.session_state.chat_attachments = attachments
    st.session_state["_chat_file_warning"] = warnings
    st.session_state[CHAT_FILE_UPLOAD_KEY] = []
    st.rerun()


def _add_chat_images() -> None:
    files = st.session_state.get(CHAT_IMAGE_UPLOAD_KEY) or []
    images = list(st.session_state.get("chat_images", []))
    existing = {(getattr(f, "name", None), getattr(f, "size", None)) for f in images}
    warnings: List[str] = []

    for upload in files:
        key = (getattr(upload, "name", None), getattr(upload, "size", None))
        if key in existing:
            continue
        if len(images) >= MAX_IMAGE_ATTACHMENTS:
            warnings.append(f"Maximum {MAX_IMAGE_ATTACHMENTS} images par envoi.")
            break
        images.append(upload)
        existing.add(key)

    st.session_state.chat_images = images
    st.session_state["_chat_image_warning"] = warnings
    st.session_state[CHAT_IMAGE_UPLOAD_KEY] = []
    st.rerun()


def _remove_chat_attachment(index: int) -> None:
    attachments = list(st.session_state.get("chat_attachments", []))
    if 0 <= index < len(attachments):
        attachments.pop(index)
        st.session_state.chat_attachments = attachments
    st.rerun()


def _remove_chat_image(index: int) -> None:
    images = list(st.session_state.get("chat_images", []))
    if 0 <= index < len(images):
        images.pop(index)
        st.session_state.chat_images = images
    st.rerun()


def _render_key_gate() -> None:
    st.markdown(
        """
        <style>
            .login-wrapper {
                max-width: 420px;
                margin: 10vh auto;
                padding: 2.5rem;
                background: #ffffff;
                border-radius: 18px;
                border: 1px solid #e5e7eb;
                box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
            }
            .login-wrapper h1 {
                text-align: center;
                margin-bottom: 0.5rem;
            }
            .login-wrapper p {
                text-align: center;
                color: #6b7280;
                margin-bottom: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.form("api-key-form", clear_on_submit=False):
        st.markdown(
            """
            <div class='login-wrapper'>
                <h1>Bienvenue</h1>
                <p>Entrez votre clé API OpenAI pour démarrer la conversation.</p>
            """,
            unsafe_allow_html=True,
        )
        api_key_input = st.text_input(
            "Clé API OpenAI",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
        )
        submit = st.form_submit_button("Continuer", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        if not api_key_input:
            st.error("Merci de renseigner une clé API valide.")
        else:
            st.session_state.api_key = api_key_input.strip()
            _reset_chat()
            st.rerun()



def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Paramètres")
        try:
            default_index = AVAILABLE_MODELS.index(st.session_state.selected_model)
        except ValueError:
            default_index = 0
        st.selectbox("Modèle", AVAILABLE_MODELS, index=default_index, key="selected_model")

        st.button("Nouvelle conversation", on_click=_reset_chat)
        st.button("Changer de clé API", on_click=_remove_api_key)

        st.markdown("---")
        st.caption("Votre clé n'est jamais sauvegardée côté serveur.")
        st.caption("⚡ Mode performance activé (réglages masqués).")

        st.markdown("### Données")
        st.file_uploader(
            "📎 Importer des fichiers",
            type=[
                "csv",
                "tsv",
                "xlsx",
                "xls",
                "pdf",
                "docx",
                "txt",
                "md",
                "json",
            ],
            accept_multiple_files=True,
            key=SIDEBAR_UPLOAD_KEY,
        )
        help_hint = (
            f"Limite {DEFAULT_MAX_FILE_MB} Mo par fichier • CSV, TSV, XLSX, XLS, PDF, DOCX, TXT, MD, JSON/NDJSON"
        )
        if ALLOW_LARGE_FILES:
            help_hint += " — Les fichiers plus lourds seront traités par morceaux."
        st.caption(help_hint)

        uploaded_files = st.session_state.get(SIDEBAR_UPLOAD_KEY) or []
        st.button(
            "Indexer",
            use_container_width=True,
            disabled=not uploaded_files,
            on_click=_request_indexing,
        )
        st.button(
            "Réinitialiser base",
            use_container_width=True,
            on_click=_request_rag_reset,
        )

        feedback = st.session_state.get("_sidebar_feedback")
        if isinstance(feedback, tuple) and len(feedback) == 2:
            level, message = feedback
            if level == "success":
                st.success(message)
            elif level == "warning":
                st.warning(message)
            elif level == "error":
                st.error(message)
            else:
                st.info(message)

        st.markdown("---")
        doc_count = len(st.session_state.rag_docs)
        chunk_count = len(st.session_state.rag_texts)
        st.markdown(f"**Documents indexés :** {doc_count}")
        st.markdown(f"**Chunks :** {chunk_count}")
        st.markdown(f"**Modèle d'embeddings :** {st.session_state.rag_embedding_model or '-'}")

        if st.session_state.rag_docs:
            for doc in st.session_state.rag_docs:
                st.caption(
                    " • ".join(
                        [
                            doc.get("name", "Inconnu"),
                            human_readable_size(doc.get("size_bytes", 0)),
                            f"{doc.get('chunk_count', 0)} chunks",
                            f"~{doc.get('token_estimate', 0)} tokens",
                        ]
                    )
                )
        else:
            st.caption("Aucun document indexé.")

        st.markdown("---")
        with st.expander("Diagnostics", expanded=False):
            diag = st.session_state.get("rag_diagnostics") or {}
            if not diag:
                st.caption("Aucun diagnostic RAG pour le moment.")
            else:
                lines = [
                    f"- Modèle : {diag.get('model', '-')}",
                    f"- Mode : {diag.get('mode', '-')}",
                    f"- k : {diag.get('k', '-')}",
                    f"- Temps retrieval : {diag.get('retrieval_time_s', '-')} s",
                ]
                if diag.get("rerank_used") is not None:
                    lines.append(f"- Reranker : {'oui' if diag.get('rerank_used') else 'non'}")
                if diag.get("pass1_time_s") is not None:
                    lines.append(f"- Pass 1 : {diag['pass1_time_s']} s")
                if diag.get("pass2_time_s") is not None:
                    lines.append(f"- Pass 2 : {diag['pass2_time_s']} s")
                if diag.get("pass2_skipped"):
                    lines.append("- Pass 2 désactivé")
                if diag.get("history_chars"):
                    lines.append(f"- Résumé historique : {diag['history_chars']} caractères")
                pass1_tokens = (diag.get("pass1_usage") or {}).get("completion_tokens")
                pass2_tokens = (diag.get("pass2_usage") or {}).get("completion_tokens")
                if pass1_tokens:
                    lines.append(f"- Tokens pass 1 : {pass1_tokens}")
                if pass2_tokens:
                    lines.append(f"- Tokens pass 2 : {pass2_tokens}")
                st.markdown("\n".join(lines))

        st.markdown("---")
        st.checkbox(
            "Afficher les logs",
            key="show_logs",
        )
        if st.session_state.show_logs:
            st.markdown("### Logs")
            log = st.session_state.last_call_log
            if not log:
                st.caption("Aucun appel récent.")
            else:
                lines = [f"- **Modèle :** {log.get('model', '-')}"]
                if log.get("mode"):
                    lines.append(f"- **Mode :** {log.get('mode')}")
                lines.append(f"- **Type d’appel :** {log.get('call_type', '-')}")
                lines.append(f"- **Messages envoyés :** {log.get('messages_count', 0)}")
                tokens_before = log.get("prompt_tokens_before")
                tokens_after = log.get("prompt_tokens_after")
                if tokens_before and tokens_before != tokens_after:
                    lines.append(
                        f"- **Tokens invite (avant/après) :** {tokens_before} → {tokens_after}"
                    )
                elif tokens_after is not None:
                    lines.append(f"- **Tokens invite estimés :** {tokens_after}")
                lines.append(f"- **RAG :** {'oui' if log.get('rag_used') else 'non'}")
                if log.get("rag_hits"):
                    lines.append(f"  - {log['rag_hits']} extraits")
                if log.get("truncated_context"):
                    lines.append("  - Contexte RAG tronqué")
                if log.get("truncated_history"):
                    lines.append("  - Historique tronqué")
                image_count = log.get("image_count", 0)
                lines.append(f"- **Images :** {image_count}")
                images = log.get("images") or []
                for image in images:
                    name = image.get("name") or "Image"
                    size_bytes = image.get("size_bytes") or 0
                    lines.append(
                        f"  - {escape(name)} ({human_readable_size(size_bytes)})"
                    )
                attachments_count = log.get("attachments_count")
                if attachments_count is not None:
                    lines.append(f"- **Pièces jointes indexées :** {attachments_count}")
                usage_text = _format_usage(log.get("usage"))
                if usage_text:
                    lines.append(f"- {usage_text}")
                response_chars = log.get("response_chars")
                if response_chars is not None:
                    lines.append(
                        f"- **Longueur réponse :** {response_chars} caractères"
                    )
                if log.get("sources_count"):
                    lines.append(f"- **Sources citées :** {log['sources_count']}")
                if log.get("error"):
                    lines.append(f"- **Erreur :** {log['error']}")
                diag = log.get("pipeline_diagnostics") or {}
                if diag:
                    lines.append(
                        f"- **RAG (pipeline)** : retrieval {diag.get('retrieval_time_s', '-')} s, pass1 {diag.get('pass1_time_s', '-')} s"
                    )
                    if diag.get("pass2_time_s") is not None:
                        lines.append(f"  - Pass2 : {diag.get('pass2_time_s')} s")
                st.markdown("\n".join(lines))
                if log.get("error"):
                    st.caption(
                        "[Voir logs Streamlit Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/manage-your-app#view-your-app-logs)"
                    )


def _usage_to_dict(usage: Optional[object]) -> Optional[Dict[str, int]]:
    if usage is None:
        return None

    if isinstance(usage, dict):
        data = usage
    else:
        data = {key: getattr(usage, key, None) for key in ["total_tokens", "prompt_tokens", "completion_tokens"]}

    filtered = {key: value for key, value in data.items() if value is not None}
    return filtered or None


def _format_usage(usage: Optional[Dict[str, int]]) -> Optional[str]:
    if not usage:
        return None

    parts = []
    if "total_tokens" in usage:
        parts.append(f"Total : {usage['total_tokens']}")
    if "prompt_tokens" in usage:
        parts.append(f"Invite : {usage['prompt_tokens']}")
    if "completion_tokens" in usage:
        parts.append(f"Réponse : {usage['completion_tokens']}")

    if not parts:
        return None

    return "Usage tokens — " + ", ".join(parts)


def responses_stream(
    client: OpenAI,
    model: str,
    payload: Sequence[Dict[str, Any]],
    on_delta,
) -> Any:
    with client.responses.stream(
        model=model,
        input=list(payload),
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    on_delta(delta)
            elif event.type == "response.error":
                error_obj = getattr(event, "error", None)
                if isinstance(error_obj, Exception):
                    raise error_obj
                error_message = getattr(error_obj, "message", None) or "Erreur inconnue"
                exc = RuntimeError(error_message)
                if error_obj is not None:
                    setattr(exc, "response", error_obj)
                raise exc
        return stream.get_final_response()


def chat_stream(
    client: OpenAI,
    model: str,
    payload: Sequence[Dict[str, Any]],
    on_delta,
) -> Any:
    stream = client.chat.completions.create(
        model=model,
        messages=list(payload),
        stream=True,
        stream_options={"include_usage": True},
    )
    final_parts: List[str] = []
    usage_data = None
    for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        delta = getattr(choice.delta, "content", None) if choice else None
        if delta:
            final_parts.append(delta)
            on_delta(delta)
        if getattr(chunk, "usage", None):
            usage_data = chunk.usage
    return SimpleNamespace(output_text="".join(final_parts), usage=usage_data)


def do_call(
    client: OpenAI,
    model: str,
    payload_messages: Sequence[Dict[str, Any]],
    stream: bool = True,
    on_delta=None,
) -> Any:
    delta_callback = on_delta or (lambda *_args, **_kwargs: None)
    last_user = next(
        (message for message in reversed(payload_messages) if message.get("role") == "user"),
        None,
    )
    has_images = _contains_image_parts(last_user)

    if has_images and model not in VISION_MODELS:
        raise ValueError(
            "Ce modèle n’accepte pas d’images. Choisissez gpt-4o / gpt-4o-mini / gpt-5."
        )

    if has_images:
        payload = to_responses_input(list(payload_messages))
        if stream:
            return responses_stream(client, model, payload, delta_callback)
        return client.responses.create(model=model, input=payload)

    payload = to_chat_messages(list(payload_messages))
    if stream:
        return chat_stream(client, model, payload, delta_callback)

    completion = client.chat.completions.create(
        model=model,
        messages=payload,
        stream=False,
    )
    final_text_parts: List[str] = []
    if completion.choices:
        message = getattr(completion.choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            final_text_parts.append(content)
    final_text = "".join(final_text_parts)
    return SimpleNamespace(output_text=final_text, usage=getattr(completion, "usage", None))


def call_with_fallback(
    client: OpenAI,
    model: str,
    payload_messages: Sequence[Dict[str, Any]],
    use_stream: bool = True,
    on_delta=None,
) -> tuple[Any, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "model_used": model,
        "stream_was_used": use_stream,
        "fallback_applied": False,
        "stream_disabled": False,
        "model_changed": False,
    }
    try:
        result = do_call(client, model, payload_messages, stream=use_stream, on_delta=on_delta)
        return result, info
    except Exception as e:
        if is_org_verify_stream_error(e) and use_stream:
            result = do_call(client, model, payload_messages, stream=False, on_delta=on_delta)
            info["fallback_applied"] = True
            info["stream_was_used"] = False
            info["stream_disabled"] = True
            return result, info
        raise


def safe_llm_call(
    client: OpenAI,
    model: str,
    payload_messages: Sequence[Dict[str, Any]],
    stream: bool = True,
    on_delta=None,
) -> tuple[Any, Dict[str, Any]]:
    try:
        return call_with_fallback(
            client,
            model,
            payload_messages,
            use_stream=stream,
            on_delta=on_delta,
        )
    except Exception as e1:
        if not is_org_verify_stream_error(e1):
            raise
        for alt in PREFERRED_MODELS:
            if alt == model:
                continue
            try:
                result, info = call_with_fallback(
                    client,
                    alt,
                    payload_messages,
                    use_stream=stream,
                    on_delta=on_delta,
                )
                info["fallback_applied"] = True
                info["model_used"] = alt
                info["model_changed"] = True
                return result, info
            except Exception as e2:
                if not is_org_verify_stream_error(e2):
                    raise
        result, info = call_with_fallback(
            client,
            PREFERRED_MODELS[0],
            payload_messages,
            use_stream=False,
            on_delta=on_delta,
        )
        info["fallback_applied"] = True
        info["model_used"] = PREFERRED_MODELS[0]
        info["model_changed"] = PREFERRED_MODELS[0] != model
        info["stream_disabled"] = True
        info["stream_was_used"] = False
        return result, info


def call_llm(
    client: OpenAI,
    model: str,
    payload_messages: Sequence[Dict[str, Any]],
    on_delta,
    on_done,
    on_error,
) -> Optional[Dict[str, Any]]:
    try:
        result, info = safe_llm_call(
            client,
            model,
            payload_messages,
            stream=st.session_state.gen_streaming,
            on_delta=on_delta,
        )
    except Exception as exc:
        on_error(exc)
        return None

    try:
        on_done(result)
    except Exception as exc:
        on_error(exc)
        return None

    return info


CODE_BLOCK_PATTERN = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def _ensure_global_system_message() -> None:
    messages = st.session_state.messages
    if not isinstance(messages, list):
        st.session_state.messages = [dict(GLOBAL_SYSTEM_MESSAGE)]
        return
    if not any(
        msg.get("role") == "system" and msg.get("content") == GLOBAL_SYSTEM_PROMPT
        for msg in messages
    ):
        st.session_state.messages.insert(0, dict(GLOBAL_SYSTEM_MESSAGE))


def _next_render_counter() -> str:
    counter = st.session_state.get("_render_counter", 0)
    st.session_state["_render_counter"] = counter + 1
    return f"render-{counter}"


def render_answer_markdown_or_code(
    raw_text: str,
    *,
    preferred_lang: Optional[str] = None,
    dbml_mode: bool = False,
) -> None:
    key_root = _next_render_counter()
    target_lang = "dbml" if dbml_mode else preferred_lang
    lang, code = extract_code_block(raw_text, target_lang)

    if dbml_mode and code is None:
        code = try_extract_dbml_heuristic(raw_text)
        if code:
            lang = "dbml"

    if code:
        st.code(code, language=(lang or "text"))
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.download_button(
                "💾 Télécharger",
                code,
                file_name=f"modele.{(lang or 'txt')}",
                key=f"{key_root}-download",
            )
        with c2:
            st.button(
                "📋 Copier",
                key=f"{key_root}-copy",
                on_click=lambda text=code: st.session_state.__setitem__("__copy__", text),
            )
        with c3:
            if (lang or "").lower() == "dbml":
                url = "https://dbdiagram.io/d/new?code=" + _url.quote(code)
                st.link_button("↗️ Ouvrir sur dbdiagram.io", url)
    else:
        st.markdown(raw_text)


def _render_message_content(
    content: Any,
    *,
    role: str = "assistant",
    preferred_lang: Optional[str] = None,
    dbml_mode: bool = False,
) -> None:
    if not content:
        return

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                _render_message_content(
                    part.get("text", ""),
                    role=role,
                    preferred_lang=preferred_lang,
                    dbml_mode=dbml_mode,
                )
            elif part.get("type") == "input_image" and part.get("image_url"):
                st.image(part["image_url"], width=196)
        return

    if not isinstance(content, str):
        return

    if role == "assistant":
        render_answer_markdown_or_code(
            content,
            preferred_lang=preferred_lang,
            dbml_mode=dbml_mode,
        )
        return

    last_index = 0
    for match in CODE_BLOCK_PATTERN.finditer(content):
        text_segment = content[last_index : match.start()]
        if text_segment.strip():
            st.markdown(text_segment)
        elif text_segment:
            st.markdown(text_segment)

        language = match.group(1).strip() or None
        code = match.group(2)
        st.code(code.rstrip("\n"), language)
        last_index = match.end()

    remaining_text = content[last_index:]
    if remaining_text.strip():
        st.markdown(remaining_text)
    elif remaining_text:
        st.markdown(remaining_text)


def _render_chat_interface() -> None:
    _ensure_global_system_message()
    st.session_state["_render_counter"] = 0

    def _extract_text_from_response_output(output: Any) -> str:
        if not output:
            return ""
        pieces: List[str] = []
        for item in output:
            if isinstance(item, dict):
                content_items = item.get("content") or []
            else:
                content_items = getattr(item, "content", None) or []
            if not isinstance(content_items, list):
                content_items = [content_items]
            for content_item in content_items:
                text_value = None
                if isinstance(content_item, dict):
                    text_value = content_item.get("text") or content_item.get("value")
                else:
                    text_value = getattr(content_item, "text", None) or getattr(
                        content_item, "value", None
                    )
                if isinstance(text_value, dict):
                    text_value = text_value.get("value") or text_value.get("text")
                if isinstance(text_value, str):
                    pieces.append(text_value)
            if isinstance(item, dict):
                text_attr = item.get("text")
            else:
                text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str):
                pieces.append(text_attr)
        return "".join(pieces)

    _render_sidebar()

    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {padding-top: 2rem !important; padding-bottom: 6rem !important;}
        .stChatFloatingInputContainer {bottom: 1.5rem;}
        div[data-testid="stChatMessage"] {background: transparent;}
        div[data-testid="stChatMessageUser"] > div:nth-child(1) {
            background: #e7f5f0;
            color: #0f172a;
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stChatMessageAssistant"] > div:nth-child(1) {
            background: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 0.75rem 1rem;
        }
        .attachment-chip-wrapper {
            margin-top: 0.5rem;
        }
        .attachment-chip {
            display: inline-block;
            background: #e0f2fe;
            color: #0f172a;
            border-radius: 999px;
            padding: 0.15rem 0.6rem;
            margin-right: 0.35rem;
            margin-bottom: 0.25rem;
            font-size: 0.75rem;
            border: 1px solid #bae6fd;
        }
        .chat-header {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }
        .empty-state {
            margin-top: 15vh;
            text-align: center;
            color: #6b7280;
        }
        .empty-state h2 {
            color: #111827;
            font-size: 2.25rem;
            margin-bottom: 0.5rem;
        }
        .empty-state p {
            margin: 0.25rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='chat-header'>ChatGPT-like Chatbot</div>", unsafe_allow_html=True)

    if _rag_is_ready():
        st.markdown(f"🧠 RAG actif (k={st.session_state.rag_k})")

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        render_opts = message.get("render_options") or {}
        with st.chat_message(message["role"]):
            _render_message_content(
                message.get("content", ""),
                role=message.get("role", "assistant"),
                preferred_lang=render_opts.get("preferred_lang"),
                dbml_mode=render_opts.get("dbml_mode", False),
            )
            if message["role"] == "user" and message.get("attachments"):
                attachments = message.get("attachments") or []
                chips = " ".join(
                    [
                        f"<span class='attachment-chip'>{escape(a['name'])}</span>"
                        for a in attachments
                    ]
                )
                if chips:
                    st.markdown(
                        f"<div class='attachment-chip-wrapper'>{chips}</div>",
                        unsafe_allow_html=True,
                    )
            sources_line = _format_sources_line(message.get("sources") or [])
            if sources_line:
                st.caption(sources_line)
            usage_text = _format_usage(message.get("usage"))
            if usage_text:
                st.caption(usage_text)

    if not any(msg.get("role") != "system" for msg in st.session_state.messages):
        st.markdown(
            """
            <div class='empty-state'>
                <h2>Que voulez-vous savoir aujourd'hui ?</h2>
                <p>Choisissez un modèle dans la barre latérale et lancez la discussion.</p>
                <p>Déposez vos fichiers dans la barre latérale pour activer le RAG.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form("chat-composer", clear_on_submit=True):
        c1, c2, c3 = st.columns([0.08, 0.72, 0.20])
        with c1:
            with st.popover("📎", use_container_width=True):
                st.file_uploader(
                    "Importer des fichiers",
                    type=[
                        "csv",
                        "tsv",
                        "xlsx",
                        "xls",
                        "pdf",
                        "docx",
                        "txt",
                        "md",
                        "json",
                    ],
                    accept_multiple_files=True,
                    key=CHAT_FILE_UPLOAD_KEY,
                    on_change=_add_chat_attachments,
                )
                help_hint = (
                    f"Limite {DEFAULT_MAX_FILE_MB} Mo par fichier • CSV, TSV, XLSX, XLS, PDF, DOCX, TXT, MD, JSON/NDJSON"
                )
                if ALLOW_LARGE_FILES:
                    help_hint += " — Les fichiers plus lourds seront traités par morceaux."
                st.caption(help_hint)

                st.file_uploader(
                    "Images (PNG, JPG, WEBP, GIF)",
                    type=["png", "jpg", "jpeg", "webp", "gif"],
                    accept_multiple_files=True,
                    key=CHAT_IMAGE_UPLOAD_KEY,
                    on_change=_add_chat_images,
                )

        with c2:
            user_text = st.text_input(
                "Envoyer un message…",
                label_visibility="collapsed",
                key="chat_input",
            )

        with c3:
            send = st.form_submit_button("▶️ Envoyer", use_container_width=True)

    for warning in st.session_state.get("_chat_file_warning", []) or []:
        st.warning(warning)

    for warning in st.session_state.get("_chat_image_warning", []) or []:
        st.warning(warning)

    if st.session_state.chat_images:
        img_cols = st.columns(min(4, len(st.session_state.chat_images)))
        for i, f in enumerate(list(st.session_state.chat_images)):
            with img_cols[i % len(img_cols)]:
                st.image(f, caption=f.name, use_container_width=True)
                st.button(
                    f"❌ Retirer image {i}",
                    key=f"rm_img_{i}",
                    on_click=_remove_chat_image,
                    args=(i,),
                )

    if st.session_state.chat_attachments:
        chip_cols = st.columns(min(len(st.session_state.chat_attachments), 4))
        for i, f in enumerate(st.session_state.chat_attachments):
            with chip_cols[i % len(chip_cols)]:
                size = getattr(f, "size", None)
                if size is None:
                    try:
                        size = len(f.getvalue())
                    except Exception:
                        size = 0
                st.markdown(f"🧩 **{f.name}**  ({format_bytes(size)})")
                st.button(
                    f"❌ Retirer {i}",
                    key=f"rm_{i}",
                    on_click=_remove_chat_attachment,
                    args=(i,),
                )

    if send:
        st.session_state.quality_escalated = False
        text_value = user_text.strip() if user_text else ""
        dbml_requested = expects_dbml(text_value)
        attachments: List[Dict[str, Any]] = []
        oversized: List[tuple[str, int]] = []

        for upload in list(st.session_state.chat_attachments):
            size = getattr(upload, "size", None)
            if size is None:
                try:
                    size = len(upload.getvalue())
                except Exception:
                    size = 0
            if not ALLOW_LARGE_FILES and size and size > MAX_FILE_BYTES:
                oversized.append((upload.name, size))
                continue
            attachments.append({"name": upload.name, "data": upload.getvalue(), "size": size})

        if oversized:
            st.warning(
                "\n".join(
                    [
                        f"Les fichiers suivants dépassent {DEFAULT_MAX_FILE_MB} Mo et ont été ignorés :",
                        *[f"• {name} ({format_bytes(size)})" for name, size in oversized],
                    ]
                )
            )

        image_parts: List[Dict[str, Any]] = []
        image_stats: List[Dict[str, Any]] = []
        if st.session_state.chat_images:
            if len(st.session_state.chat_images) > MAX_IMAGE_ATTACHMENTS:
                st.session_state.chat_images = st.session_state.chat_images[:MAX_IMAGE_ATTACHMENTS]
            for f in list(st.session_state.chat_images):
                try:
                    image_parts.append(to_image_part(f))
                    size = getattr(f, "size", None)
                    if size is None:
                        try:
                            size = len(f.getvalue())
                        except Exception:
                            size = 0
                    image_stats.append(
                        {
                            "name": getattr(f, "name", "Image"),
                            "size_bytes": size or 0,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - display in UI
                    st.warning(f"Image ignorée ({f.name}) : {exc}")

        if not text_value and not attachments and not image_parts:
            st.info("Veuillez saisir un message ou ajouter des pièces jointes.")
            return

        selected_model = st.session_state.selected_model
        if image_parts and selected_model not in VISION_MODELS:
            st.warning("Ce modèle n’accepte pas d’images, choisissez gpt-4o/mini/gpt-5.")
            return

        if image_parts:
            user_content = []
            if text_value:
                user_content.append({"type": "text", "text": text_value})
            user_content.extend(image_parts)
        else:
            user_content = text_value or "(Pièces jointes uniquement)"

        user_payload: Dict[str, Any] = {"role": "user", "content": user_content}
        if attachments:
            user_payload["attachments"] = attachments
        st.session_state.messages.append(user_payload)
        st.session_state.chat_attachments = []
        st.session_state.chat_images = []

        if dbml_requested and not any(
            msg.get("role") == "system" and msg.get("content") == DBML_ENFORCEMENT_PROMPT
            for msg in st.session_state.messages
        ):
            st.session_state.messages.insert(
                0, {"role": "system", "content": DBML_ENFORCEMENT_PROMPT}
            )

        with st.chat_message("user"):
            _render_message_content(user_payload["content"], role="user")
            if attachments:
                chips = " ".join(
                    [f"<span class='attachment-chip'>{escape(a['name'])}</span>" for a in attachments]
                )
                if chips:
                    st.markdown(
                        f"<div class='attachment-chip-wrapper'>{chips}</div>",
                        unsafe_allow_html=True,
                    )

        indexing_stats: Optional[Dict[str, Any]] = None
        if attachments:
            try:
                with st.status("Indexation des pièces jointes…", expanded=False) as s:
                    configure_status_callbacks(
                        update=lambda label, state="running": s.update(label=label, state=state),
                        write=s.write,
                    )
                    try:
                        indexing_stats = index_files_from_chat(attachments)
                    finally:
                        clear_status_callbacks()
                    if (indexing_stats or {}).get("chunks"):
                        s.update(
                            label=f"✅ Indexation terminée ({indexing_stats['chunks']} chunks)",
                            state="complete",
                        )
                    else:
                        s.update(label="⚠️ Aucun contenu indexable", state="error")
            except Exception as error:  # noqa: BLE001 - surface in UI
                st.error(f"Indexation impossible : {error}")
                indexing_stats = None

        if indexing_stats and indexing_stats.get("warnings"):
            for warn in indexing_stats["warnings"]:
                st.warning(warn)

        client = OpenAI(api_key=st.session_state.api_key)
        use_rag = _rag_is_ready()
        if not use_rag:
            st.session_state.rag_diagnostics = None
        is_multimodal_request = bool(image_parts)
        query_for_rag = text_value or "Résume/analyse les documents joints."
        pipeline_result: Optional[Dict[str, Any]] = None
        pipeline_sources_info: List[Dict[str, Any]] = []

        vectorstore: Optional[SessionVectorStore] = None
        if use_rag:
            vectorstore = SessionVectorStore(
                client,
                st.session_state.rag_index,
                st.session_state.rag_texts,
                st.session_state.rag_meta,
                st.session_state.rag_embedding_model or EMBEDDING_MODEL,
            )

        if vectorstore is not None and not is_multimodal_request:
            base_cfg = effective_params_from_mode()
            active_cfg = base_cfg
            quality_applied = False

            def _run_with_cfg(cfg: PerfConfig) -> Dict[str, Any]:
                return run_rag_pipeline(
                    client,
                    selected_model,
                    vectorstore,
                    query_for_rag,
                    st.session_state.messages,
                    cfg=cfg,
                )

            try:
                if CFG.quality_escalation and need_quality_escalation(text_value or query_for_rag, use_rag):
                    boosted_cfg = with_quality_boost(base_cfg)
                    if boosted_cfg != base_cfg:
                        active_cfg = boosted_cfg
                        quality_applied = True

                pipeline_result = _run_with_cfg(active_cfg)
                st.session_state.rag_diagnostics = pipeline_result.get("diagnostics")

                final_answer_text = pipeline_result.get("answer", "") if pipeline_result else ""
                if (
                    pipeline_result
                    and output_is_poor(final_answer_text, expect_dbml=dbml_requested)
                    and not quality_applied
                    and CFG.quality_escalation
                ):
                    boosted_cfg = with_quality_boost(active_cfg)
                    if boosted_cfg != active_cfg:
                        active_cfg = boosted_cfg
                        quality_applied = True
                        pipeline_result = _run_with_cfg(active_cfg)
                        st.session_state.rag_diagnostics = pipeline_result.get("diagnostics")

                st.session_state.quality_escalated = quality_applied
            except Exception as error:  # noqa: BLE001 - fallback to classic path
                st.warning(f"Pipeline RAG indisponible : {error}")
                st.session_state.rag_diagnostics = None
                pipeline_result = None
                active_cfg = base_cfg

        if pipeline_result and pipeline_result.get("answer"):
            final_text = pipeline_result.get("answer", "")
            usage_data = pipeline_result.get("usage")
            docs = pipeline_result.get("documents") or []
            for idx, doc in enumerate(docs, start=1):
                meta = getattr(doc, "metadata", None) or {}
                pipeline_sources_info.append({"index": idx, "meta": meta})

            with st.chat_message("assistant"):
                preferred_lang = "dbml" if dbml_requested else None
                render_opts: Dict[str, Any] = {}
                if dbml_requested:
                    render_opts = {"preferred_lang": preferred_lang, "dbml_mode": True}
                _render_message_content(
                    final_text,
                    role="assistant",
                    preferred_lang=preferred_lang,
                    dbml_mode=dbml_requested,
                )
                if pipeline_sources_info:
                    st.markdown(
                        "Sources : "
                        + " • ".join(
                            [
                                format_source_badge(info["meta"], info["index"])
                                for info in pipeline_sources_info
                            ]
                        )
                    )
                usage_text = _format_usage(usage_data)
                if usage_text:
                    st.caption(usage_text)

            message_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": final_text,
                "usage": usage_data,
                "sources": pipeline_sources_info or None,
            }
            if render_opts:
                message_entry["render_options"] = render_opts
            st.session_state.messages.append(message_entry)

            call_context: Dict[str, Any] = {
                "model": selected_model,
                "call_type": "RAG Pipeline",
                "messages_count": len(st.session_state.messages),
                "prompt_tokens_before": None,
                "prompt_tokens_after": None,
                "truncated_context": False,
                "truncated_history": False,
                "image_count": len(image_stats),
                "images": image_stats,
                "attachments_count": len(attachments or []),
                "rag_used": bool(pipeline_sources_info),
                "rag_hits": len(pipeline_sources_info),
                "sources_count": len(pipeline_sources_info),
                "usage": usage_data,
                "response_chars": len(final_text or ""),
                "status": "success",
                "pipeline_diagnostics": pipeline_result.get("diagnostics"),
                "temperature": active_cfg.temperature,
                "top_p": active_cfg.top_p,
                "max_tokens": active_cfg.max_tokens,
                "rag_k": active_cfg.rag_k,
                "multipass": active_cfg.use_multipass,
                "rerank": active_cfg.use_reranker,
                "mode": "performance",
            }
            st.session_state.last_call_log = call_context
            return

        _ensure_global_system_message()

        messages_for_api = [
            {"role": msg.get("role"), "content": msg.get("content")}
            for msg in st.session_state.messages
        ]

        sources_info: List[Dict[str, Any]] = []

        hits: List[Any] = []
        if vectorstore is not None:
            try:
                hits = _session_retrieve(vectorstore, query_for_rag, k=st.session_state.rag_k)
            except Exception as error:  # noqa: BLE001 - surface gracefully
                st.warning(f"Recherche contextuelle indisponible : {error}")
                hits = []

        truncated_context_flag = False
        truncated_history_flag = False
        selected_model = st.session_state.selected_model

        if hits:
            context = format_context(hits)
            trimmed_context = truncate_context_text(
                context,
                selected_model,
                MAX_RAG_CONTEXT_TOKENS,
            )
            truncated_context_flag = trimmed_context != context
            sys_prefix = {
                "role": "system",
                "content": (
                    "Tu es un assistant. Utilise EXCLUSIVEMENT les extraits ci-dessous pour répondre. "
                    "Cites les sources entre crochets [n]. Si l’info manque, dis-le.\n\nExtraits :\n"
                    + trimmed_context
                ),
            }
            messages_for_api = [sys_prefix] + messages_for_api
            sources_info = _build_source_entries(hits)

        original_prompt_tokens = count_tokens_chat(messages_for_api, selected_model)
        messages_for_api = truncate_messages_to_budget(
            messages_for_api,
            selected_model,
            MAX_INPUT_TOKENS,
            RESERVE_OUTPUT_TOKENS,
        )
        prompt_tokens_after = count_tokens_chat(messages_for_api, selected_model)
        truncated_history_flag = prompt_tokens_after < original_prompt_tokens

        last_user_for_api = next(
            (msg for msg in reversed(messages_for_api) if msg.get("role") == "user"),
            None,
        )
        is_multimodal_request = _contains_image_parts(last_user_for_api)

        call_context: Dict[str, Any] = {
            "model": selected_model,
            "call_type": "Responses API" if is_multimodal_request else "Chat Completions",
            "messages_count": len(messages_for_api),
            "prompt_tokens_before": original_prompt_tokens,
            "prompt_tokens_after": prompt_tokens_after,
            "truncated_context": truncated_context_flag,
            "truncated_history": truncated_history_flag,
            "image_count": len(image_stats),
            "images": image_stats,
            "attachments_count": len(attachments or []),
            "rag_used": bool(hits),
            "rag_hits": len(hits),
            "sources_count": len(sources_info),
            "usage": None,
            "temperature": st.session_state.gen_temperature,
            "top_p": st.session_state.gen_top_p,
            "max_tokens": st.session_state.gen_max_tokens,
            "rag_k": st.session_state.rag_k,
            "multipass": st.session_state.use_multipass,
            "rerank": st.session_state.use_reranker,
            "streaming": st.session_state.gen_streaming,
            "mode": "performance",
            "pipeline_diagnostics": st.session_state.get("rag_diagnostics"),
        }

        with st.chat_message("assistant"):
            if truncated_context_flag or truncated_history_flag:
                st.caption("ℹ️ Contexte réduit pour respecter les limites du modèle.")
            status_box = st.empty()
            answer_box = st.empty()
            sources_placeholder = st.empty()
            usage_placeholder = st.empty()
            fallback_notice = st.empty()

            status_box.info("✍️ L’assistant est en train d’écrire…")

            first_token_displayed = False
            acc: List[str] = []
            usage_holder: Dict[str, Optional[Dict[str, int]]] = {"usage": None}
            should_rerun = {"value": False}

            def _to_text(delta: Any) -> str:
                if delta is None:
                    return ""
                if isinstance(delta, str):
                    return delta
                if isinstance(delta, dict):
                    text_value = delta.get("text") or delta.get("value")
                    if isinstance(text_value, str):
                        return text_value
                text_attr = getattr(delta, "text", None)
                if isinstance(text_attr, str):
                    return text_attr
                value_attr = getattr(delta, "value", None)
                if isinstance(value_attr, str):
                    return value_attr
                return str(delta)

            def on_delta(delta_text: Any) -> None:
                nonlocal first_token_displayed
                text = _to_text(delta_text)
                if not text:
                    return
                if not first_token_displayed:
                    status_box.empty()
                    answer_box.markdown("_Rédaction en cours…_")
                    first_token_displayed = True
                acc.append(text)

            def on_done(result: Any) -> None:
                status_box.empty()
                answer_box.empty()
                usage_holder["usage"] = _usage_to_dict(getattr(result, "usage", None))
                final_text = getattr(result, "output_text", None) or ""
                if not final_text:
                    final_text = "".join(acc)
                if not final_text and hasattr(result, "output"):
                    final_text = _extract_text_from_response_output(getattr(result, "output"))
                if not final_text:
                    final_text = ""

                render_opts: Dict[str, Any] = {}
                if final_text:
                    preferred_lang = "dbml" if dbml_requested else None
                    if dbml_requested:
                        render_opts = {"preferred_lang": preferred_lang, "dbml_mode": True}
                    _render_message_content(
                        final_text,
                        role="assistant",
                        preferred_lang=preferred_lang,
                        dbml_mode=dbml_requested,
                    )
                else:
                    st.info("Aucune réponse texte reçue.")

                if hits:
                    sources_placeholder.markdown(
                        "Sources : "
                        + " • ".join(
                            [
                                format_source_badge(hit[1], idx + 1)
                                for idx, hit in enumerate(hits)
                            ]
                        )
                    )
                else:
                    sources_line = _format_sources_line(sources_info)
                    if sources_line:
                        sources_placeholder.caption(sources_line)

                usage_text = _format_usage(usage_holder["usage"])
                if usage_text:
                    usage_placeholder.caption(usage_text)

                message_entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": final_text,
                    "usage": usage_holder["usage"],
                    "sources": sources_info if sources_info else None,
                }
                if render_opts:
                    message_entry["render_options"] = render_opts
                st.session_state.messages.append(message_entry)

                call_context["usage"] = usage_holder["usage"]
                call_context["response_chars"] = len(final_text)
                call_context["sources_count"] = len(sources_info)
                call_context["status"] = "success"
                st.session_state.last_call_log = call_context
                should_rerun["value"] = True

            def on_error(err: Any) -> None:
                status_box.empty()
                answer_box.empty()
                sources_placeholder.empty()
                usage_placeholder.empty()
                error_message = getattr(err, "message", None) or str(err)
                st.error(f"Erreur API : {error_message}")
                st.caption(
                    "[Voir logs Streamlit Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/manage-your-app#view-your-app-logs)"
                )
                call_context["error"] = error_message
                call_context["status"] = "error"
                call_context["response_chars"] = len("".join(acc))
                st.session_state.last_call_log = call_context

            outcome = call_llm(
                client,
                selected_model,
                messages_for_api,
                on_delta,
                on_done,
                on_error,
            )

            if outcome:
                effective_model = outcome.get("model_used", selected_model)
                call_context["effective_model"] = effective_model
                call_context["stream_was_used"] = outcome.get("stream_was_used", True)
                call_context["fallback_applied"] = outcome.get("fallback_applied", False)
                call_context["model_changed"] = outcome.get("model_changed", False)
                call_context["stream_disabled"] = outcome.get("stream_disabled", False)
                if outcome.get("fallback_applied"):
                    fallback_notice.info(
                        "ℹ️ Le modèle sélectionné requiert une organisation vérifiée pour le streaming. "
                        "Un fallback a été appliqué (streaming désactivé et/ou modèle alternatif)."
                    )
                else:
                    fallback_notice.empty()
            else:
                fallback_notice.empty()

        if should_rerun["value"]:
            st.rerun()


if __name__ == "__main__":
    _init_session_state()
    set_defaults_if_needed()
    _process_pending_actions()

    if not st.session_state.api_key:
        _render_key_gate()
    else:
        _render_chat_interface()
