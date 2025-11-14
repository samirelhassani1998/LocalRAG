# ChatGPT-like Streamlit Chatbot

This project recreates a minimal ChatGPT-style user interface using Streamlit and the OpenAI API. Users can supply their own API key directly in the app, pick a model, and begin chatting immediately.
URL: https://laposte-57sgwe24hqzegfthseuprg.streamlit.app/

## Features

- **Inline API key capture**: Provide your OpenAI API key inside the app—no need for `st.secrets`.
- **Model selection**: Switch between available GPT-5 generation models (GPT-5.1, GPT-5.1 mini, GPT-5, GPT-5 mini) if enabled on your account.
- **Clean conversation view**: Messages are displayed in a vertically stacked chat log with a chat input area at the bottom of the screen.
- **Session persistence**: The API key and conversation history live in `st.session_state` during the browsing session.

## Getting Started

### Prerequisites

- Python 3.9+
- An OpenAI API key with access to the desired models

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run main.py
```

Open the provided local URL in your browser. The app will first ask for your API key. Once provided, it reveals the chat interface.

### Environment Variables (Optional)

If you do not want to type your API key every time, you can create a `.env` file at the project root and set `OPENAI_API_KEY=...`. The app will load it automatically on startup, but the key can still be changed from the UI at any time.

Additional knobs for advanced usage:

- `QUALITY_ESCALATION=0` — disables the automatic multi-pass / reranker escalation (enabled by default for maximum quality).

## Customization

- Update the list of models in `main.py` to match the ones available on your account.
- Adjust the theme colors in `.streamlit/config.toml` to tweak the look and feel.

## RAG & Upload

The chat interface now includes an optional retrieval-augmented generation (RAG) workflow:

- **Drag & drop documents** directly in the sidebar (`csv`, `xlsx`, `xls`, `pdf`, `docx`, `txt`, `md`). Up to five files can be indexed at a time and the per-file limit defaults to 20&nbsp;MB (configurable).
- **On-demand indexing** builds an in-memory FAISS index for the current session using OpenAI's `text-embedding-3-large` model. Files are read in memory only; nothing is persisted on disk and the API key never leaves the session.
- **Chunking & metadata**: each document is normalized, chunked (~4 000 chars with 400-char overlap), and enriched with metadata (source file, page/sheet/row range when applicable).
- **Contextual answers**: when the index is populated, every new user question retrieves the top-8 chunks (MMR + reranker) and injects them into the system prompt. Responses cite their sources and a badge indicates when RAG is active.
- **Reset anytime**: use the “Réinitialiser base” button to clear the FAISS index and associated documents from the session state.

By default the chatbot now runs in "mode qualité" with aggressive retrieval settings (`k=8`, MMR fetch 40, cross-encoder reranking, multi-pass generation and 2 000 output tokens). Those options are baked into the code for maximum robustness and no longer appear in the sidebar.

The sidebar summarises the indexed corpus (file sizes, estimated tokens, chunk counts, embedding model). If a PDF contains no extractable text (e.g. scanned documents), the app warns you and skips it.

## Limites et gros fichiers

- La limite d'upload est gouvernée par `DEFAULT_MAX_FILE_MB` (20&nbsp;Mo par défaut). Surcharger `MAX_FILE_MB` via `st.secrets` ou une variable d'environnement permet d'augmenter ou diminuer ce plafond.
- Quand `ALLOW_LARGE_FILES` vaut `true` (valeur par défaut), les documents au-delà de cette limite sont traités par morceaux plutôt qu'ignorés : CSV/TSV sont lus par blocs (`CSV_CHUNKSIZE_ROWS`), les classeurs Excel feuille par feuille (`EXCEL_MAX_SHEETS`), et les PDF page par page (`PDF_MAX_PAGES`). Les fichiers texte/DOCX suivent le flux habituel.
- `MAX_TOTAL_CHARS` borne le volume total de caractères ingérés pour éviter des coûts d'embeddings ou une consommation mémoire disproportionnée. Adaptez ce paramètre selon vos contraintes.
- Pour désactiver le traitement chunké et retrouver l'ancien comportement (fichiers volumineux ignorés), définissez `ALLOW_LARGE_FILES=false` dans l'environnement ou `st.secrets`.
- Si vous relevez `MAX_FILE_MB` au-delà de 200, pensez à synchroniser la configuration Streamlit (`.streamlit/config.toml`, clé `server.maxUploadSize`) afin que l'upload navigateur/serveur suive.
- Les fichiers massifs génèrent davantage de chunks et donc plus d'embeddings : surveillez vos coûts OpenAI, surtout avec `text-embedding-3-large`.

## License

This project is released under the [MIT License](LICENSE).
