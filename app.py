"""Application Streamlit pour discuter avec un chatbot OpenAI."""

import streamlit as st
from openai import OpenAI


def init_session_state() -> None:
    """Initialise les éléments indispensables dans la session Streamlit."""

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("api_key", "")


def render_sidebar() -> None:
    """Affiche le panneau latéral pour renseigner la clé API et les actions."""

    with st.sidebar:
        st.header("Connexion à l'API")
        st.caption(
            "Obtenez votre clé sur https://platform.openai.com/ et gardez-la confidentielle."
        )

        with st.form("api_form", clear_on_submit=False):
            api_key_input = st.text_input(
                "Clé API OpenAI",
                type="password",
                help="Saisissez votre clé API personnelle avant de démarrer la discussion.",
                value=st.session_state.get("api_key", ""),
            )
            submitted = st.form_submit_button("Enregistrer la clé")

        if submitted:
            st.session_state.api_key = api_key_input.strip()
            if st.session_state.api_key:
                st.success("✅ Clé API enregistrée.")
            else:
                st.warning("⚠️ Aucune clé renseignée.")

        if st.button("🧹 Réinitialiser la conversation", use_container_width=True):
            st.session_state.messages = []

        if st.button("🚫 Oublier la clé", use_container_width=True):
            st.session_state.api_key = ""
            st.success("La clé API a été effacée de cette session.")


def display_conversation() -> None:
    """Affiche l'historique des échanges dans l'interface principale."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    st.set_page_config(page_title="Chatbot OpenAI", page_icon="🤖")
    st.title("Chatbot OpenAI")

    init_session_state()
    render_sidebar()

    if not st.session_state.api_key:
        st.info(
            "🔑 Veuillez renseigner votre clé API OpenAI dans la barre latérale pour démarrer la conversation."
        )
        st.stop()

    client = OpenAI(api_key=st.session_state.api_key)

    display_conversation()

    prompt = st.chat_input("Posez votre question...")

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.messages,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
    except Exception as error:  # pragma: no cover - dépend de l'API externe.
        st.error(f"❌ Erreur lors de l'appel à l'API : {error}")
        return

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)


if __name__ == "__main__":
    main()
