from dataclasses import dataclass


@dataclass(frozen=True)
class PerfConfig:
    """Container describing the Retrieval-Augmented Generation defaults."""

    # Modèle
    default_model: str = "gpt-5.1"

    # RAG
    rag_k: int = 8
    use_mmr: bool = True
    mmr_fetch_k: int = 40
    mmr_lambda: float = 0.35
    use_reranker: bool = True
    use_multipass: bool = True

    # Génération
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 2000
    streaming: bool = True

    # Quality escalation (activée par défaut)
    quality_escalation: bool = True
