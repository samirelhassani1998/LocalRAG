import re

DBML_HINTS = ("dbml", "dbdiagram", "modélisation", "modele", "modèle", "schéma", "schema", "diagramme")
CODE_HINTS = ("sql", "python", "pandas", "dbt", "spark", "graphviz", "dot", "ddl", "dml")
LONGFORM_HINTS = ("analyse", "étapes", "procedure", "plan", "architecture", "benchmark", "comparatif")


def need_quality_escalation(user_text: str, has_rag: bool) -> bool:
    t = user_text.lower()
    if any(k in t for k in DBML_HINTS):
        return True
    if any(k in t for k in CODE_HINTS):
        return True
    if has_rag and any(k in t for k in LONGFORM_HINTS):
        return True
    if re.search(r"\b(explique|d(é|e)taille|complet|exhaustif)\b", t):
        return True
    return False

def output_is_poor(text: str, expect_dbml: bool = False) -> bool:
    if not text or len(text.strip()) < 600:
        return True
    if expect_dbml and "```dbml" not in text:
        return True
    need_sections = ("###", "Résumé", "Détails")
    if sum(1 for s in need_sections if s.lower() in text.lower()) < 1:
        if "```" not in text:
            return True
    return False

def expects_dbml(user_text: str) -> bool:
    t = user_text.lower()
    return any(k in t for k in ("dbml", "dbdiagram"))
