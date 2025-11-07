# Chatbot OpenAI avec Streamlit

Cette application Streamlit propose une interface de discussion alimentée par l'API d'OpenAI. L'utilisateur doit saisir sa clé API personnelle avant de pouvoir converser avec le chatbot.

## Installation

```bash
pip install streamlit openai
```

## Lancer l'application

```bash
streamlit run app.py
```

## Utilisation

1. Ouvrez l'interface web générée par Streamlit.
2. Renseignez votre clé API OpenAI dans la barre latérale.
3. Lancez la conversation depuis le champ de saisie situé en bas de la page.
4. Utilisez le bouton « 🧹 Réinitialiser la conversation » pour repartir de zéro.

> 💡 Votre clé n'est jamais stockée côté serveur : elle reste en mémoire dans votre session Streamlit uniquement le temps de votre navigation.
