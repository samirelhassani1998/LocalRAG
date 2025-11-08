"""Utilities to normalise textual data into safe Unicode strings."""

from __future__ import annotations

from typing import Iterable, Tuple

import unicodedata
from charset_normalizer import from_bytes


MAX_LEN_CHARS = 200_000  # coupe très long texte avant embeddings


def _detect_and_decode(data: bytes) -> str:
    """Detect encoding for bytes payloads and decode into text."""

    result = from_bytes(data).best()
    if result is not None:
        return str(result)  # -> str Unicode (UTF-8 en interne)
    # repli : décodage UTF-8 tolérant
    return data.decode("utf-8", errors="replace")


def ensure_text(obj: bytes | str) -> str:
    """Retourne un str Unicode sûr (UTF-8), normalisé NFC et sans caractères illégaux."""

    if isinstance(obj, bytes):
        text = _detect_and_decode(obj)
    else:
        text = obj
    # Normalisation Unicode et nettoyage basique
    text = unicodedata.normalize("NFC", text).replace("\x00", "")
    # Coupe les payloads démesurés (évite erreurs côté embeddings)
    if len(text) > MAX_LEN_CHARS:
        text = text[:MAX_LEN_CHARS]
    return text


def ensure_text_iter(chunks: Iterable[bytes | str]) -> Tuple[str, ...]:
    """Normalise une séquence de morceaux texte/binaire en str sûrs."""

    return tuple(ensure_text(c) for c in chunks if c is not None)


__all__ = ["ensure_text", "ensure_text_iter", "MAX_LEN_CHARS"]

