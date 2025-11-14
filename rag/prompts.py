SYSTEM_BASE = """Tu es un assistant expert francophone.
Réponds de façon structurée, rigoureuse et sourcée.
N'invente jamais d'informations : si le contexte ne suffit pas, indique "Je ne sais pas" et précise ce qu'il manque."""

USER_TEMPLATE = """CONTEXTE ISSU DE VOS FICHIERS (top {k} passages):
-------------------------------
{context}

HISTORIQUE (RÉSUMÉ):
--------------------
{history}

QUESTION UTILISATEUR:
---------------------
{query}

DIRECTIVES:
1. Explique ta réponse étape par étape.
2. Cite systématiquement les sources sous la forme [n].
3. Termine par une section "Sources" listant les documents utilisés.
4. Si aucune source ne répond, écris "Je ne sais pas" et suggère des pistes.
"""

IMPROVE_TEMPLATE = """Améliore la réponse suivante:
- Conserve le format demandé (résumé, détails, sources).
- Vérifie et rappelle les sources [n].
- Clarifie les limites éventuelles.
Réponse initiale:
{draft}
"""
