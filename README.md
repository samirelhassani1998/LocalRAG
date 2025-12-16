# LocalRAG — Chatbot RAG Avancé avec Streamlit

Application de chat conversationnel **style ChatGPT** construite avec Streamlit et l'API OpenAI, enrichie d'un pipeline de **Retrieval-Augmented Generation (RAG)** performant. Indexez vos propres documents et obtenez des réponses contextualisées en quelques clics.

🔗 **Démo en ligne** : [laposte-57sgwe24hqzegfthseuprg.streamlit.app](https://laposte-57sgwe24hqzegfthseuprg.streamlit.app/)

---

## ✨ Fonctionnalités Clés

| Catégorie | Description |
|-----------|-------------|
| **Chat Intelligent** | Interface conversationnelle fluide avec streaming des réponses en temps réel |
| **RAG Avancé** | Indexation vectorielle FAISS, embeddings OpenAI `text-embedding-3-large`, reranking cross-encoder |
| **Multi-formats** | Support CSV, TSV, XLSX, XLS, PDF, DOCX, TXT, MD, JSON (y compris NDJSON) |
| **Vision** | Prise en charge des images (GPT-5.1, GPT-5) pour analyse visuelle |
| **Mode Qualité** | Multi-pass generation, MMR search (λ=0.35), top-k=8, reranking automatique |
| **Session Sécurisée** | Clé API saisie directement dans l'UI, données en mémoire uniquement |

---

## 🏗️ Architecture du Projet

```
LocalRAG/
├── main.py               # Application Streamlit principale (~2450 lignes)
├── rag_utils.py          # Ingestion de documents, chunking, embeddings
├── config.py             # Configuration RAG (PerfConfig dataclass)
├── adapters.py           # Conversion messages → schéma OpenAI Chat/Responses
├── token_utils.py        # Comptage et troncature de tokens
├── image_utils.py        # Traitement d'images pour vision
├── responses_schema.py   # Schémas de réponses structurées
├── rag/                  # Module RAG avancé
│   ├── pipeline.py       # Orchestration du pipeline RAG complet
│   ├── retriever.py      # Logique de récupération et reranking
│   ├── memory.py         # Résumé de l'historique de conversation
│   └── prompts.py        # Templates de prompts système
├── utils/                # Utilitaires
│   ├── rendering.py      # Rendu et formatage
│   └── text_normalize.py # Normalisation de texte
├── quality/              # Modules d'amélioration de qualité
├── tests/                # Tests unitaires
├── .streamlit/           # Configuration Streamlit
└── requirements.txt      # Dépendances Python
```

---

## 🚀 Installation & Démarrage

### Prérequis

- Python 3.9+
- Clé API OpenAI (avec accès aux modèles GPT)

### Installation

```bash
# Cloner le repository
git clone https://github.com/votre-username/LocalRAG.git
cd LocalRAG

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement

```bash
streamlit run main.py
```

L'application s'ouvre dans votre navigateur. Entrez votre clé API OpenAI pour commencer.

---

## 📄 Workflow RAG

1. **Upload de documents** — Glissez-déposez jusqu'à 5 fichiers (20 Mo max par défaut) dans la sidebar
2. **Indexation automatique** — Chunking intelligent (~4000 caractères, 400 overlap), création de l'index FAISS
3. **Recherche contextuelle** — MMR (Maximal Marginal Relevance) + reranking cross-encoder
4. **Génération multi-pass** — Première réponse puis amélioration automatique
5. **Réponses sourcées** — Chaque réponse cite ses sources avec numérotation

### Types de fichiers supportés

| Format | Extensions | Particularités |
|--------|------------|----------------|
| Texte | `.txt`, `.md` | Encodage auto-détecté |
| Tableur | `.csv`, `.tsv`, `.xlsx`, `.xls` | Parsing par feuilles/lignes |
| Document | `.pdf`, `.docx` | Extraction par pages |
| Données | `.json` | Support NDJSON et streaming |

---

## ⚙️ Configuration

### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `OPENAI_API_KEY` | — | Clé API OpenAI (optionnel si saisi dans l'UI) |
| `MAX_FILE_MB` | `20` | Taille maximale par fichier (Mo) |
| `ALLOW_LARGE_FILES` | `true` | Traitement chunké des gros fichiers |
| `MAX_TOTAL_CHARS` | — | Limite totale de caractères ingérés |
| `QUALITY_ESCALATION` | `1` | Active le mode qualité avancé (0 pour désactiver) |

### Configuration RAG (`config.py`)

```python
@dataclass(frozen=True)
class PerfConfig:
    default_model: str = "gpt-5.1"
    rag_k: int = 8                 # Nombre de chunks récupérés
    use_mmr: bool = True           # Maximal Marginal Relevance
    mmr_fetch_k: int = 40          # Taille du pool de candidats MMR
    mmr_lambda: float = 0.35       # Balance pertinence/diversité
    use_reranker: bool = True      # Cross-encoder reranking
    use_multipass: bool = True     # Génération en 2 passes
    temperature: float = 0.3
    max_tokens: int = 2000
```

---

## 🔧 Fonctionnalités Avancées

### Mode Vision

Les modèles GPT-5.1 et GPT-5 supportent l'analyse d'images. Uploadez des images dans le chat pour obtenir des descriptions, analyses ou réponses contextuelles.

### Gros Fichiers

- Fichiers > `MAX_FILE_MB` traités par morceaux (streaming)
- CSV/TSV : lecture par blocs
- Excel : feuille par feuille
- PDF : page par page

### Reranking Intelligent

1. **Cross-Encoder** (MS-MARCO MiniLM L-6) — scoring sémantique précis
2. **BM25 Fallback** — algorithme lexical si cross-encoder indisponible

---

## 📦 Dépendances Principales

- `streamlit` — Interface web
- `openai` — API OpenAI
- `faiss-cpu` — Indexation vectorielle
- `sentence-transformers` — Cross-encoder reranking
- `pypdf` — Extraction PDF
- `python-docx` — Extraction DOCX
- `pandas` / `openpyxl` — Traitement tableurs
- `tiktoken` — Comptage de tokens
- `rank-bm25` — Reranking BM25

---

## 🎨 Personnalisation

- **Modèles** : Modifiez `AVAILABLE_MODELS` dans `main.py`
- **Thème** : Ajustez `.streamlit/config.toml`
- **Prompts** : Éditez `rag/prompts.py` et `BASE_GLOBAL_SYSTEM_PROMPT`

---

## 📝 Licence

Ce projet est distribué sous licence [MIT](LICENSE).

---

## 🤝 Contribution

Les contributions sont bienvenues ! Ouvrez une issue ou soumettez une pull request.
