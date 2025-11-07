"""Streamlit app for a ChatGPT-powered assistant."""

from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st

try:  # Prefer the modern OpenAI client when available.
    from openai import OpenAI
    from openai import OpenAIError

    _import_error: Exception | None = None
    _legacy_openai = None
except ImportError as exc:  # pragma: no cover - fallback for older OpenAI clients.
    OpenAI = None
    OpenAIError = Exception
    _import_error = exc
    try:
        import openai as _legacy_openai  # type: ignore
    except ImportError as legacy_exc:  # pragma: no cover - if package missing entirely.
        _legacy_openai = None
        _import_error = legacy_exc


def _call_openai(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> str:
    """Send a conversation to OpenAI's chat completions API and return the answer."""
    if not api_key:
        raise ValueError("A valid OpenAI API key is required to call the API.")

    if OpenAI is not None:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    # Legacy client support (openai<1.0)
    if _legacy_openai is None:
        raise ValueError(
            "La dépendance 'openai' est introuvable. Ajoutez-la au fichier requirements.txt "
            "ou installez-la dans votre environnement."
        ) from _import_error

    _legacy_openai.api_key = api_key
    response = _legacy_openai.ChatCompletion.create(  # type: ignore[attr-defined]
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"]


def _dependencies_ready() -> bool:
    """Return True when at least one OpenAI client implementation is importable."""

    return OpenAI is not None or _legacy_openai is not None


def _render_dependency_warning() -> None:
    """Display a prominent warning in the UI when dependencies are missing."""

    if _dependencies_ready():
        return

    st.sidebar.error(
        "La bibliothèque `openai` est absente. Ajoutez-la au fichier ``requirements.txt`` "
        "ou installez-la dans votre environnement pour utiliser le chatbot."
    )
    st.stop()


def _initial_messages(system_prompt: str) -> List[Dict[str, str]]:
    """Return the initial conversation scaffold."""
    return [
        {
            "role": "system",
            "content": system_prompt.strip() or "You are a helpful assistant.",
        }
    ]


def _ensure_session_state(system_prompt: str) -> None:
    """Initialise Streamlit session state for the conversation."""
    if "messages" not in st.session_state:
        st.session_state.messages = _initial_messages(system_prompt)
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = system_prompt


def _reset_conversation(system_prompt: str) -> None:
    """Reset the conversation while keeping the latest system prompt."""
    st.session_state.system_prompt = system_prompt
    st.session_state.messages = _initial_messages(system_prompt)


def main() -> None:
    st.set_page_config(page_title="La Poste - ChatGPT", page_icon="📮", layout="wide")
    st.title("📮 Assistant conversationnel La Poste")

    default_system_prompt = "Tu es un assistant virtuel de La Poste. Réponds avec clarté et professionnalisme."

    _ensure_session_state(default_system_prompt)
    _render_dependency_warning()

    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Clé API OpenAI",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Saisissez votre clé API OpenAI personnelle.",
        )

        model = st.selectbox(
            "Modèle",
            options=[
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1-mini",
                "gpt-3.5-turbo",
            ],
            index=0,
            help="Choisissez le modèle ChatGPT à utiliser.",
        )

        temperature = st.slider(
            "Créativité (temperature)",
            min_value=0.0,
            max_value=1.5,
            value=0.7,
            step=0.05,
        )

        system_prompt = st.text_area(
            "Instruction système",
            value=st.session_state.get("system_prompt", default_system_prompt),
            help="Décrivez la personnalité ou la mission de l'assistant.",
        )

        if st.button("🧹 Réinitialiser la conversation"):
            _reset_conversation(system_prompt)
            st.success("Conversation réinitialisée")

    if system_prompt != st.session_state.get("system_prompt"):
        _reset_conversation(system_prompt)

    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Envoyez un message à l'assistant"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        if not api_key:
            st.error("Veuillez renseigner votre clé API OpenAI dans la barre latérale.")
            return

        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                try:
                    completion = _call_openai(
                        api_key=api_key,
                        model=model,
                        messages=st.session_state.messages,
                        temperature=temperature,
                    )
                except (OpenAIError, ValueError) as exc:  # type: ignore[arg-type]
                    st.error(f"Erreur lors de l'appel à l'API: {exc}")
                    return

                st.markdown(completion)

        st.session_state.messages.append({"role": "assistant", "content": completion})


if __name__ == "__main__":
    main()
