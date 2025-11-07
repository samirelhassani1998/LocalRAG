import os
import re
from html import escape
from typing import Any, Dict, List, Optional, Sequence

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

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
MAX_INPUT_TOKENS = 300_000
RESERVE_OUTPUT_TOKENS = 1_000
MAX_RAG_CONTEXT_TOKENS = 30_000
RETRIEVAL_K = 4
MAX_FILES = 5
MAX_FILE_BYTES = DEFAULT_MAX_FILE_MB * 1024 * 1024
CHUNK_MAX_CHARS = 4000
CHUNK_OVERLAP = 400


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

    return {
        "uploaded_files": uploaded_files,
        "index_clicked": index_clicked,
        "reset_clicked": reset_clicked,
    }


def _call_openai(client: OpenAI, messages: List[Dict[str, str]]):
    return client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
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


CODE_BLOCK_PATTERN = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def _render_message_content(content: str) -> None:
    if not content:
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
                    # ajoute sans dupliquer les mêmes noms+taille
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

        with c2:
            user_text = st.text_input(
                "Envoyer un message…",
                label_visibility="collapsed",
                key="chat_input",
            )

        with c3:
            send = st.form_submit_button("▶️ Envoyer", use_container_width=True)

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

        st.session_state.chat_attachments = []

        if oversized:
            st.warning(
                "\n".join(
                    [
                        f"Les fichiers suivants dépassent {DEFAULT_MAX_FILE_MB} Mo et ont été ignorés :",
                        *[f"• {name} ({format_bytes(size)})" for name, size in oversized],
                    ]
                )
            )

        if not text_value and not attachments:
            st.info("Veuillez saisir un message ou ajouter des pièces jointes.")
            return

        user_payload: Dict[str, Any] = {
            "role": "user",
            "content": text_value or "(Pièces jointes uniquement)",
        }
        if attachments:
            user_payload["attachments"] = attachments
        st.session_state.messages.append(user_payload)

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
        truncated_history_flag = (
            count_tokens_chat(messages_for_api, selected_model) < original_prompt_tokens
        )

        with st.chat_message("assistant"):
            if truncated_context_flag or truncated_history_flag:
                st.caption("ℹ️ Contexte réduit pour respecter les limites du modèle.")
            status_box = st.empty()
            answer_box = st.empty()
            sources_placeholder = st.empty()
            usage_placeholder = st.empty()

            status_box.info("✍️ L’assistant est en train d’écrire…")

            response_text = ""
            usage_data = None
            first_token_displayed = False

            try:
                for chunk in _call_openai(client, messages_for_api):
                    if chunk.choices and chunk.choices[0].delta:
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            if not first_token_displayed:
                                status_box.empty()
                                first_token_displayed = True
                            response_text += content
                            answer_box.markdown(response_text)
                    if getattr(chunk, "usage", None):
                        usage_data = _usage_to_dict(chunk.usage)
            except Exception as error:  # noqa: BLE001 - display gracefully in UI
                status_box.empty()
                st.error(f"Erreur lors de l'appel à l'API : {error}")
            else:
                status_box.empty()
                answer_box.empty()
                _render_message_content(response_text)
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
                usage_text = _format_usage(usage_data)
                if usage_text:
                    usage_placeholder.caption(usage_text)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "usage": usage_data,
                        "sources": sources_info if sources_info else None,
                    }
                )
        st.rerun()


if __name__ == "__main__":
    _init_session_state()

    if not st.session_state.api_key:
        _render_key_gate()
    else:
        _render_chat_interface()
