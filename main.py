import os
import re
from html import escape
from typing import Any, Dict, List, Optional, Sequence
from types import SimpleNamespace

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from responses_schema import to_responses_input

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
    retrieve,
)
from token_utils import (
    count_tokens_chat,
    truncate_context_text,
    truncate_messages_to_budget,
)

load_dotenv()

st.set_page_config(page_title="ChatGPT-like Chatbot", layout="wide")

AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-5",
]


EMBEDDING_MODEL = "text-embedding-3-large"
VISION_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-5"}
MAX_INPUT_TOKENS = 300_000
RESERVE_OUTPUT_TOKENS = 1_000
MAX_RAG_CONTEXT_TOKENS = 30_000
RETRIEVAL_K = 4
MAX_FILES = 5
MAX_FILE_BYTES = DEFAULT_MAX_FILE_MB * 1024 * 1024
CHUNK_MAX_CHARS = 4000
CHUNK_OVERLAP = 400
MAX_IMAGE_ATTACHMENTS = 5


def _init_session_state() -> None:
    if "api_key" not in st.session_state:
        env_key = os.getenv("OPENAI_API_KEY")
        st.session_state.api_key = env_key if env_key else None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0]
    if "rag_index" not in st.session_state:
        st.session_state.rag_index = None
    if "rag_texts" not in st.session_state:
        st.session_state.rag_texts = []
    if "rag_meta" not in st.session_state:
        st.session_state.rag_meta = []
    if "rag_docs" not in st.session_state:
        st.session_state.rag_docs = []
    if "rag_embedding_model" not in st.session_state:
        st.session_state.rag_embedding_model = EMBEDDING_MODEL
    if "chat_attachments" not in st.session_state:
        st.session_state.chat_attachments = []
    if "chat_images" not in st.session_state:
        st.session_state.chat_images = []
    if "show_logs" not in st.session_state:
        st.session_state.show_logs = False
    if "last_call_log" not in st.session_state:
        st.session_state.last_call_log = None


def _reset_chat() -> None:
    st.session_state.messages = []


def _reset_rag_state() -> None:
    st.session_state.rag_index = None
    st.session_state.rag_texts = []
    st.session_state.rag_meta = []
    st.session_state.rag_docs = []
    st.session_state.rag_embedding_model = EMBEDDING_MODEL


def _remove_api_key() -> None:
    st.session_state.api_key = None
    _reset_chat()
    _reset_rag_state()
    st.rerun()


def _rag_is_ready() -> bool:
    return bool(st.session_state.rag_index is not None and st.session_state.rag_texts)


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



def _render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### Paramètres")
        try:
            default_index = AVAILABLE_MODELS.index(st.session_state.selected_model)
        except ValueError:
            default_index = 0
        model = st.selectbox("Modèle", AVAILABLE_MODELS, index=default_index)
        st.session_state.selected_model = model

        st.button("Nouvelle conversation", on_click=_reset_chat)
        st.button("Changer de clé API", on_click=_remove_api_key)

        st.markdown("---")
        st.caption("Votre clé n'est jamais sauvegardée côté serveur.")

        st.markdown("### Données")
        uploaded_files = st.file_uploader(
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
        )
        help_hint = (
            f"Limite {DEFAULT_MAX_FILE_MB} Mo par fichier • CSV, TSV, XLSX, XLS, PDF, DOCX, TXT, MD, JSON/NDJSON"
        )
        if ALLOW_LARGE_FILES:
            help_hint += " — Les fichiers plus lourds seront traités par morceaux."
        st.caption(help_hint)

        index_clicked = st.button(
            "Indexer",
            use_container_width=True,
            disabled=not uploaded_files,
        )
        reset_clicked = st.button(
            "Réinitialiser base",
            use_container_width=True,
        )

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
        show_logs = st.checkbox(
            "Afficher les logs",
            value=st.session_state.show_logs,
        )
        st.session_state.show_logs = show_logs
        if show_logs:
            st.markdown("### Logs")
            log = st.session_state.last_call_log
            if not log:
                st.caption("Aucun appel récent.")
            else:
                lines: List[str] = []
                lines.append(f"- **Modèle :** {log.get('model', '-')}")
                lines.append(f"- **Type d’appel :** {log.get('call_type', '-')}")
                lines.append(
                    f"- **Messages envoyés :** {log.get('messages_count', 0)}"
                )
                tokens_before = log.get("prompt_tokens_before")
                tokens_after = log.get("prompt_tokens_after")
                if tokens_before and tokens_before != tokens_after:
                    lines.append(
                        f"- **Tokens invite (avant/après) :** {tokens_before} → {tokens_after}"
                    )
                elif tokens_after is not None:
                    lines.append(f"- **Tokens invite estimés :** {tokens_after}")
                lines.append(
                    f"- **RAG :** {'oui' if log.get('rag_used') else 'non'}"
                )
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
                st.markdown("\n".join(lines))
                if log.get("error"):
                    st.caption(
                        "[Voir logs Streamlit Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/manage-your-app#view-your-app-logs)"
                    )

    return {
        "uploaded_files": uploaded_files,
        "index_clicked": index_clicked,
        "reset_clicked": reset_clicked,
    }


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


def stream_via_responses(
    client: OpenAI,
    model: str,
    payload_messages: Sequence[Dict[str, Any]],
    on_delta,
    on_done,
    on_error,
) -> None:
    try:
        payload = to_responses_input(list(payload_messages))
        with client.responses.stream(
            model=model,
            input=payload,
        ) as stream:
            for event in stream:
                try:
                    if event.type == "response.output_text.delta":
                        on_delta(getattr(event, "delta", ""))
                    elif event.type == "response.error":
                        on_error(getattr(event, "error", "Erreur inconnue"))
                        return
                except Exception as inner_exc:  # noqa: BLE001 - propagate to UI
                    on_error(inner_exc)
                    return
            try:
                result = stream.get_final_response()
            except Exception as final_exc:  # noqa: BLE001 - propagate to UI
                on_error(final_exc)
                return
            on_done(result)
    except Exception as exc:  # noqa: BLE001 - surface to UI
        on_error(exc)


def stream_via_chat_completions(
    client: OpenAI,
    model: str,
    payload_messages: Sequence[Dict[str, Any]],
    on_delta,
    on_done,
    on_error,
) -> None:
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=payload_messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        final_parts: List[str] = []
        usage_data = None
        for chunk in stream:
            try:
                choice = chunk.choices[0] if chunk.choices else None
                delta = getattr(choice.delta, "content", None) if choice else None
            except AttributeError:
                delta = None
            if delta:
                final_parts.append(delta)
                try:
                    on_delta(delta)
                except Exception as inner_exc:  # noqa: BLE001
                    on_error(inner_exc)
                    return
            if getattr(chunk, "usage", None):
                usage_data = chunk.usage
        result = SimpleNamespace(output_text="".join(final_parts), usage=usage_data)
        on_done(result)
    except Exception as exc:  # noqa: BLE001 - surface to UI
        on_error(exc)


def call_llm(
    client: OpenAI,
    model: str,
    payload_messages: Sequence[Dict[str, Any]],
    on_delta,
    on_done,
    on_error,
) -> None:
    last_user = next(
        (message for message in reversed(payload_messages) if message.get("role") == "user"),
        None,
    )
    is_multimodal = isinstance(last_user.get("content") if last_user else None, list)

    vision_capable = {"gpt-4o", "gpt-4o-mini", "gpt-5"}
    if is_multimodal and model not in vision_capable:
        on_error(
            ValueError(
                "Ce modèle n’accepte pas d’images. Choisissez gpt-4o / gpt-4o-mini / gpt-5."
            )
        )
        return

    if is_multimodal:
        stream_via_responses(client, model, payload_messages, on_delta, on_done, on_error)
    else:
        stream_via_chat_completions(client, model, payload_messages, on_delta, on_done, on_error)


CODE_BLOCK_PATTERN = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def _render_message_content(content: Any) -> None:
    if not content:
        return

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                _render_message_content(part.get("text", ""))
            elif part.get("type") == "input_image" and part.get("image_url"):
                st.image(part["image_url"], width=196)
        return

    if not isinstance(content, str):
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

    sidebar_state = _render_sidebar()

    if sidebar_state["reset_clicked"]:
        _reset_rag_state()
        st.sidebar.success("Base documentaire réinitialisée.")

    if sidebar_state["index_clicked"]:
        _handle_indexing(sidebar_state["uploaded_files"] or [])

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
        st.markdown(f"🧠 RAG actif (k={RETRIEVAL_K})")

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            _render_message_content(message.get("content", ""))
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

    if not st.session_state.messages:
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

    if "chat_attachments" not in st.session_state:
        st.session_state.chat_attachments = []
    if "chat_images" not in st.session_state:
        st.session_state.chat_images = []

    with st.form("chat-composer", clear_on_submit=True):
        c1, c2, c3 = st.columns([0.08, 0.72, 0.20])
        with c1:
            with st.popover("📎", use_container_width=True):
                files = st.file_uploader(
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
                )
                help_hint = (
                    f"Limite {DEFAULT_MAX_FILE_MB} Mo par fichier • CSV, TSV, XLSX, XLS, PDF, DOCX, TXT, MD, JSON/NDJSON"
                )
                if ALLOW_LARGE_FILES:
                    help_hint += " — Les fichiers plus lourds seront traités par morceaux."
                st.caption(help_hint)
                if files:
                    existing = {(f.name, f.size) for f in st.session_state.chat_attachments}
                    for f in files:
                        if (f.name, f.size) not in existing:
                            if len(st.session_state.chat_attachments) >= MAX_FILES:
                                st.warning(f"Maximum {MAX_FILES} fichiers par envoi.")
                                break
                            size = getattr(f, "size", None)
                            if size is None:
                                size = len(f.getvalue())
                            if not ALLOW_LARGE_FILES and size > MAX_FILE_BYTES:
                                st.warning(
                                    f"{f.name} dépasse la limite de {DEFAULT_MAX_FILE_MB} Mo et sera ignoré."
                                )
                                continue
                            st.session_state.chat_attachments.append(f)

                img_files = st.file_uploader(
                    "Images (PNG, JPG, WEBP, GIF)",
                    type=["png", "jpg", "jpeg", "webp", "gif"],
                    accept_multiple_files=True,
                    key="chat_img_uploader",
                )
                if img_files:
                    existing_images = {(f.name, getattr(f, "size", None)) for f in st.session_state.chat_images}
                    for f in img_files:
                        if len(st.session_state.chat_images) >= MAX_IMAGE_ATTACHMENTS:
                            st.warning(f"Maximum {MAX_IMAGE_ATTACHMENTS} images par envoi.")
                            break
                        key = (f.name, getattr(f, "size", None))
                        if key in existing_images:
                            continue
                        st.session_state.chat_images.append(f)
                        existing_images.add(key)

        with c2:
            user_text = st.text_input(
                "Envoyer un message…",
                label_visibility="collapsed",
                key="chat_input",
            )

        with c3:
            send = st.form_submit_button("▶️ Envoyer", use_container_width=True)

    if st.session_state.chat_images:
        img_cols = st.columns(min(4, len(st.session_state.chat_images)))
        for i, f in enumerate(list(st.session_state.chat_images)):
            with img_cols[i % len(img_cols)]:
                st.image(f, caption=f.name, use_container_width=True)
                if st.button(f"❌ Retirer image {i}", key=f"rm_img_{i}"):
                    st.session_state.chat_images.pop(i)
                    st.rerun()

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
                if st.button(f"❌ Retirer {i}", key=f"rm_{i}"):
                    st.session_state.chat_attachments.pop(i)
                    st.rerun()

    if send:
        text_value = user_text.strip() if user_text else ""
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
                user_content.append({"type": "input_text", "text": text_value})
            user_content.extend(image_parts)
        else:
            user_content = text_value or "(Pièces jointes uniquement)"

        user_payload: Dict[str, Any] = {"role": "user", "content": user_content}
        if attachments:
            user_payload["attachments"] = attachments
        st.session_state.messages.append(user_payload)
        st.session_state.chat_attachments = []
        st.session_state.chat_images = []

        with st.chat_message("user"):
            _render_message_content(user_payload["content"])
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

        messages_for_api = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
            if msg["role"] != "system"
        ]

        client = OpenAI(api_key=st.session_state.api_key)
        sources_info: List[Dict[str, Any]] = []

        use_rag = _rag_is_ready()
        hits: List[Any] = []
        if use_rag:
            try:
                query = text_value or "Résume/analyse les documents joints."
                hits = retrieve(
                    client,
                    query,
                    st.session_state.rag_index,
                    st.session_state.rag_texts,
                    st.session_state.rag_meta,
                    st.session_state.rag_embedding_model or EMBEDDING_MODEL,
                    k=RETRIEVAL_K,
                )
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
        is_multimodal_request = isinstance(
            last_user_for_api.get("content") if last_user_for_api else None,
            list,
        )

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
        }

        with st.chat_message("assistant"):
            if truncated_context_flag or truncated_history_flag:
                st.caption("ℹ️ Contexte réduit pour respecter les limites du modèle.")
            status_box = st.empty()
            answer_box = st.empty()
            sources_placeholder = st.empty()
            usage_placeholder = st.empty()

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
                    first_token_displayed = True
                acc.append(text)
                answer_box.markdown("".join(acc))

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

                if final_text:
                    _render_message_content(final_text)
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

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_text,
                        "usage": usage_holder["usage"],
                        "sources": sources_info if sources_info else None,
                    }
                )

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

            call_llm(
                client,
                selected_model,
                messages_for_api,
                on_delta,
                on_done,
                on_error,
            )

        if should_rerun["value"]:
            st.rerun()


if __name__ == "__main__":
    _init_session_state()

    if not st.session_state.api_key:
        _render_key_gate()
    else:
        _render_chat_interface()
