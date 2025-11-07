import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Chatbot OpenAI", page_icon="🤖")

st.title("Chatbot OpenAI")

with st.sidebar:
    st.header("Connexion à l'API")
    api_key = st.text_input("Clé API OpenAI", type="password", help="Saisissez votre clé API personnelle.")
    if st.button("Effacer la clé"):
        st.session_state.pop("api_key", None)
        api_key = ""

if api_key:
    st.session_state.api_key = api_key

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state or not st.session_state.api_key:
    st.info("🔑 Veuillez renseigner votre clé API OpenAI dans la barre latérale pour démarrer la conversation.")
    st.stop()

client = OpenAI(api_key=st.session_state.api_key)

st.success("Clé API détectée. Vous pouvez discuter avec le chatbot !")

if st.button("🧹 Réinitialiser la conversation"):
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Posez votre question...")

if prompt:
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
    except Exception as error:
        st.error(f"❌ Erreur lors de l'appel à l'API : {error}")
    else:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
