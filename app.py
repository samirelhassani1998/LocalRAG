from __future__ import annotations

import streamlit as st
from openai import OpenAI


DEFAULT_MODEL = "gpt-3.5-turbo"


def _bootstrap_state() -> None:
    """Initialise les valeurs en session (clé API, historique, modèle)."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "model" not in st.session_state:
        st.session_state.model = st.secrets.get("OPENAI_MODEL", DEFAULT_MODEL)

    if "api_key" in st.session_state and st.session_state.api_key:
        return

    secret_key = st.secrets.get("OPENAI_API_KEY", "")
    if secret_key:
        st.session_state.api_key = secret_key
        st.session_state.api_key_origin = "secret"
    else:
        st.session_state.api_key = ""
        st.session_state.api_key_origin = "manual"


st.set_page_config(page_title="Chatbot OpenAI", page_icon="🤖")

_bootstrap_state()

st.title("Chatbot OpenAI")
st.caption("Version en ligne : [laposte-rma9pkmqq66imsvz3tmgee.streamlit.app](https://laposte-rma9pkmqq66imsvz3tmgee.streamlit.app/)")

with st.sidebar:
    st.header("Connexion à l'API")
    st.caption("La clé peut être fournie manuellement ou via les *Streamlit Secrets*.")

    placeholder_value = "" if st.session_state.get("api_key_origin") == "secret" else st.session_state.get("api_key", "")
    api_key_input = st.text_input(
        "Clé API OpenAI",
        type="password",
        value=placeholder_value,
        help="Saisissez votre clé API personnelle. Elle n'est conservée que le temps de la session en cours.",
    )

    if api_key_input:
        st.session_state.api_key = api_key_input
        st.session_state.api_key_origin = "manual"

    if st.button("Effacer la clé"):
        st.session_state.api_key = ""
        st.session_state.api_key_origin = "manual"
        api_key_input = ""

    if st.session_state.get("api_key"):
        if st.session_state.get("api_key_origin") == "secret":
            st.success("Clé API chargée depuis la configuration sécurisée.")
        else:
            st.success("Clé API enregistrée pour la session en cours.")
    else:
        st.warning("Aucune clé active pour le moment.")

with st.sidebar.expander("Options avancées"):
    available_models = []
    for candidate in [st.session_state.model, DEFAULT_MODEL, "gpt-4o-mini"]:
        if candidate not in available_models:
            available_models.append(candidate)

    st.session_state.model = st.selectbox(
        "Modèle OpenAI",
        options=available_models,
        index=0,
        help="Sélectionnez le modèle de génération de texte. Le premier élément provient éventuellement des secrets.",
    )

if not st.session_state.get("api_key"):
    st.info("🔑 Veuillez renseigner votre clé API OpenAI dans la barre latérale pour démarrer la conversation.")
    st.stop()

client = OpenAI(api_key=st.session_state.api_key)

st.success("Clé API détectée. Vous pouvez discuter avec le chatbot !")

if st.button("🧹 Réinitialiser la conversation"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Posez votre question…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=st.session_state.messages,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
    except Exception as error:
        st.error(f"❌ Erreur lors de l'appel à l'API : {error}")
    else:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
