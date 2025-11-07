import os
import re
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

st.set_page_config(page_title="ChatGPT-like Chatbot", layout="wide")

AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-5",
]


def _init_session_state() -> None:
    if "api_key" not in st.session_state:
        env_key = os.getenv("OPENAI_API_KEY")
        st.session_state.api_key = env_key if env_key else None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0]


def _reset_chat() -> None:
    st.session_state.messages = []


def _remove_api_key() -> None:
    st.session_state.api_key = None
    _reset_chat()
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
        model = st.selectbox("Modèle", AVAILABLE_MODELS, index=default_index)
        st.session_state.selected_model = model

        st.button("Nouvelle conversation", on_click=_reset_chat)
        st.button("Changer de clé API", on_click=_remove_api_key)

        st.markdown("---")
        st.caption("Votre clé n'est jamais sauvegardée côté serveur.")


def _call_openai(messages: List[Dict[str, str]]):
    client = OpenAI(api_key=st.session_state.api_key)
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            _render_message_content(message.get("content", ""))
            usage_text = _format_usage(message.get("usage"))
            if usage_text:
                st.caption(usage_text)

    if not st.session_state.messages:
        st.markdown(
            """
            <div class='empty-state'>
                <h2>Que voulez-vous savoir aujourd'hui ?</h2>
                <p>Choisissez un modèle dans la barre latérale et lancez la discussion.</p>
                <p>Votre historique reste visible dans cette fenêtre, comme sur ChatGPT.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    prompt = st.chat_input("Envoyer un message")
    if prompt and prompt.strip():
        user_message = prompt.strip()
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            _render_message_content(user_message)

        messages_for_api = [
            {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            spinner_placeholder = st.empty()
            stream_placeholder = st.empty()
            usage_placeholder = st.empty()
            response_text = ""
            usage_data = None

            try:
                with spinner_placeholder.spinner("Assistant est en train d'écrire…"):
                    for chunk in _call_openai(messages_for_api):
                        if chunk.choices and chunk.choices[0].delta:
                            content = chunk.choices[0].delta.content or ""
                            if content:
                                response_text += content
                                stream_placeholder.markdown(response_text)
                        if getattr(chunk, "usage", None):
                            usage_data = _usage_to_dict(chunk.usage)
            except Exception as error:  # noqa: BLE001 - display gracefully in UI
                spinner_placeholder.empty()
                stream_placeholder.error(f"Erreur lors de l'appel à l'API : {error}")
            else:
                spinner_placeholder.empty()
                stream_placeholder.empty()
                _render_message_content(response_text)
                usage_text = _format_usage(usage_data)
                if usage_text:
                    usage_placeholder.caption(usage_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text, "usage": usage_data}
                )


if __name__ == "__main__":
    _init_session_state()

    if not st.session_state.api_key:
        _render_key_gate()
    else:
        _render_chat_interface()
