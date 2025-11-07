import re
from typing import Tuple

FENCE_RE = re.compile(r"```([a-zA-Z0-9_-]+)?\s*([\s\S]*?)```", re.MULTILINE)


def extract_code_block(text: str, preferred_lang: str | None = None) -> Tuple[str | None, str | None]:
    """
    Retourne (language, code) si un bloc ```lang ... ``` est trouvé.
    Si preferred_lang est donné, on privilégie ce bloc-là.
    """
    fences = FENCE_RE.findall(text or "")
    if not fences:
        return None, None
    if preferred_lang:
        for lang, code in fences:
            if (lang or "").lower() == preferred_lang.lower():
                return lang or "", code.strip()
    lang, code = fences[0]
    return (lang or "").strip(), code.strip()


def try_extract_dbml_heuristic(text: str) -> str | None:
    """
    Si pas de bloc trouvé, tenter d’extraire du DBML en détectant des motifs usuels.
    """
    m = re.search(r"(Table\s+[A-Za-z_].+?)(?:\n{2,}|\Z)", text, re.DOTALL)
    return m.group(1).strip() if m else None
