# Chatbot OpenAI avec Streamlit

Cette application Streamlit propose une interface de discussion alimentée par l'API d'OpenAI. L'utilisateur doit saisir sa clé API personnelle avant de pouvoir converser avec le chatbot ou utiliser celle configurée par l'administrateur de l'application.

> 🌐 Version en ligne : https://laposte-rma9pkmqq66imsvz3tmgee.streamlit.app/

## Installation

```bash
pip install streamlit openai
```

## Lancer l'application en local

```bash
streamlit run app.py
```

## Déploiement sur Streamlit Cloud

1. Créez un fichier `.streamlit/secrets.toml` à la racine du dépôt avec la clé :

   ```toml
   OPENAI_API_KEY = "votre_cle_api"
   OPENAI_MODEL = "gpt-4o-mini"  # optionnel
   ```

2. Poussez le dépôt sur GitHub et connectez-vous à [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Déployez l'application en pointant vers `app.py`. Les secrets seront chargés automatiquement et la clé ne sera jamais affichée dans l'interface.

Une fois en ligne, l'application reste compatible avec une saisie manuelle de clé. Cela permet aux utilisateurs disposant de leur propre clé OpenAI de surcharger celle configurée côté serveur.

## Utilisation

1. Ouvrez l'interface web générée par Streamlit (locale ou déployée).
2. Renseignez votre clé API OpenAI dans la barre latérale ou utilisez celle fournie via les secrets si elle est disponible.
3. Lancez la conversation depuis le champ de saisie situé en bas de la page.
4. Utilisez le bouton « 🧹 Réinitialiser la conversation » pour repartir de zéro.

> 💡 Votre clé n'est jamais stockée côté serveur : elle reste en mémoire dans votre session Streamlit uniquement le temps de votre navigation.
