"""Utilities for document ingestion and retrieval-augmented generation."""
from __future__ import annotations

import io
import json
import math
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

import streamlit as st
from token_utils import count_tokens_text, truncate_context_text
from charset_normalizer import from_bytes
from utils.text_normalize import ensure_text, ensure_text_iter

try:  # pragma: no cover - handled gracefully at runtime
    import faiss
except ImportError as exc:  # pragma: no cover - surfaced in UI instead
    faiss = None
    FAISS_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when FAISS is available
    FAISS_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    import tiktoken
except ImportError:  # pragma: no cover - fall back to heuristic token estimate
    tiktoken = None

try:  # pragma: no cover - optional dependency
    from docx import Document
except ImportError as exc:  # pragma: no cover - surfaced during ingestion
    Document = None
    DOCX_IMPORT_ERROR = exc
else:  # pragma: no cover
    DOCX_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency for streaming JSON
    import ijson
except ImportError:  # pragma: no cover - fallback to in-memory parsing
    ijson = None

from pypdf import PdfReader


# Limite d'upload (Mo) — configurable
DEFAULT_MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", st.secrets.get("MAX_FILE_MB", 20)))
# Autoriser le traitement chunké > limite (au lieu d'ignorer)
ALLOW_LARGE_FILES = (
    os.getenv("ALLOW_LARGE_FILES", str(st.secrets.get("ALLOW_LARGE_FILES", "true"))).lower()
    == "true"
)

# Bornes de sécurité RAG (éviter coûts/ram démesurés)
MAX_TOTAL_CHARS = int(
    os.getenv("MAX_TOTAL_CHARS", st.secrets.get("MAX_TOTAL_CHARS", 2_000_000))
)
CSV_CHUNKSIZE_ROWS = int(
    os.getenv("CSV_CHUNKSIZE_ROWS", st.secrets.get("CSV_CHUNKSIZE_ROWS", 100_000))
)
EXCEL_MAX_SHEETS = int(
    os.getenv("EXCEL_MAX_SHEETS", st.secrets.get("EXCEL_MAX_SHEETS", 8))
)
PDF_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", st.secrets.get("PDF_MAX_PAGES", 1_000)))


TEXT_EXTENSIONS = {"txt", "md"}
CSV_EXTENSIONS = {"csv", "tsv"}
EXCEL_EXTENSIONS = {"xlsx", "xls"}
PDF_EXTENSIONS = {"pdf"}
DOCX_EXTENSIONS = {"docx"}
JSON_EXTENSIONS = {"json"}
SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | CSV_EXTENSIONS
    | EXCEL_EXTENSIONS
    | PDF_EXTENSIONS
    | DOCX_EXTENSIONS
    | JSON_EXTENSIONS
)


def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def read_csv_streamed(file_bytes: bytes, max_rows: int | None = None, sep: str = ",") -> str:
    """Lit un CSV en chunks pour limiter la RAM et retourne un gros texte tabulaire."""
    bio = BytesIO(file_bytes)
    rows = 0
    texts: List[str] = []
    char_total = 0
    encoding = _detect_encoding(file_bytes)
    try:
        reader = pd.read_csv(
            bio,
            chunksize=CSV_CHUNKSIZE_ROWS,
            sep=sep,
            encoding=encoding,
            engine="python",
            on_bad_lines="skip",
        )
    except UnicodeDecodeError:
        bio = BytesIO(file_bytes)
        reader = pd.read_csv(
            bio,
            chunksize=CSV_CHUNKSIZE_ROWS,
            sep=sep,
            encoding="utf-8",
            encoding_errors="ignore",
            engine="python",
            on_bad_lines="skip",
        )
    for chunk in reader:
        chunk_csv = ensure_text(chunk.to_csv(index=False))
        texts.append(chunk_csv)
        char_total += len(chunk_csv)
        rows += len(chunk)
        if max_rows and rows >= max_rows:
            break
        if char_total >= MAX_TOTAL_CHARS:
            break
    return ensure_text("\n".join(texts))


def read_excel_streamed(file_bytes: bytes) -> str:
    bio = BytesIO(file_bytes)
    sheets = pd.read_excel(bio, sheet_name=None)
    texts: List[str] = []
    char_total = 0
    for i, (name, df) in enumerate(sheets.items()):
        if i >= EXCEL_MAX_SHEETS:
            break
        sheet_name = ensure_text(name)
        text = ensure_text(df.to_csv(index=False))
        texts.append(ensure_text(f"### Sheet: {sheet_name}\n{text}"))
        char_total += len(texts[-1])
        if char_total >= MAX_TOTAL_CHARS:
            break
    return ensure_text("\n\n".join(texts))


def read_pdf_paged(file_bytes: bytes) -> List[Tuple[str, Dict[str, Any]]]:
    """Retourne [(texte_page, meta), ...] pour chaque page (tronqué si besoin)."""
    bio = BytesIO(file_bytes)
    reader = PdfReader(bio)
    pages: List[Tuple[str, Dict[str, Any]]] = []
    total = 0
    for i, page in enumerate(reader.pages):
        if i >= PDF_MAX_PAGES:
            break
        txt = ensure_text(page.extract_text() or "")
        if not txt.strip():
            continue  # page vide / scannée
        pages.append((txt, {"page": i + 1}))
        total += len(txt)
        if total >= MAX_TOTAL_CHARS:
            break
    return pages

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
EMBED_MAX_TOKENS_PER_REQUEST = 300_000
CHAT_MAX_FILES = 5
CHAT_MAX_FILE_SIZE = DEFAULT_MAX_FILE_MB * 1024 * 1024

_STATUS_UPDATE_CALLBACK: Optional[Callable[[str, str], None]] = None
_STATUS_WRITE_CALLBACK: Optional[Callable[[str], None]] = None


def should_stream_file(size_bytes: int) -> bool:
    return size_bytes > DEFAULT_MAX_FILE_MB * 1024 * 1024 and ALLOW_LARGE_FILES


def configure_status_callbacks(
    update: Optional[Callable[[str, str], None]] = None,
    write: Optional[Callable[[str], None]] = None,
) -> None:
    """Expose callbacks so Streamlit can reflect indexing progress."""

    global _STATUS_UPDATE_CALLBACK, _STATUS_WRITE_CALLBACK
    _STATUS_UPDATE_CALLBACK = update
    _STATUS_WRITE_CALLBACK = write


def clear_status_callbacks() -> None:
    """Reset progress callbacks after indexing."""

    configure_status_callbacks(None, None)


@dataclass
class DocumentChunk:
    text: str
    meta: Dict[str, Any]


class TokenBatch(list):
    """List-like container that tracks whether any item was truncated."""

    def __init__(self) -> None:
        super().__init__()
        self.truncated = False


def _clean_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 400) -> List[str]:
    """Chunk a long string into overlapping segments."""
    if not text:
        return []

    normalized = _clean_text(text)
    if len(normalized) < 50:
        return []

    chunks: List[str] = []
    start = 0
    step = max_chars - overlap
    if step <= 0:
        step = max_chars

    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        chunk = normalized[start:end].strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start += step

    return chunks


def _estimate_tokens(text: str) -> int:
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:  # pragma: no cover - fallback used if encoding fails
            pass
    return max(1, math.ceil(len(text) / 4))


def _detect_encoding(data: bytes) -> str:
    match = from_bytes(data).best()
    if match is None or not match.encoding:
        return "utf-8"
    return match.encoding


def _bytes_to_text(data: bytes) -> str:
    return ensure_text(data)


def _json_scalar_to_str(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _flatten_json_iter(value: Any, prefix: Optional[str] = None) -> Iterator[Tuple[str, str]]:
    if isinstance(value, dict):
        if not value:
            path = prefix or "value"
            yield path, "{}"
            return
        for key, child in value.items():
            key_str = str(key)
            new_prefix = f"{prefix}.{key_str}" if prefix else key_str
            yield from _flatten_json_iter(child, new_prefix)
        return
    if isinstance(value, list):
        if not value:
            path = prefix or "value"
            yield path, "[]"
            return
        for idx, child in enumerate(value):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _flatten_json_iter(child, new_prefix)
        return
    path = prefix or "value"
    yield path, _json_scalar_to_str(value)


def _json_structure_to_text(
    value: Any,
    prefix: Optional[str] = None,
    *,
    allow_tabular: bool = True,
) -> str:
    if isinstance(value, list):
        if allow_tabular and value and all(isinstance(item, dict) for item in value):
            try:
                table = pd.json_normalize(value)
            except Exception:
                table = None
            else:
                if not table.empty:
                    return table.to_csv(index=False)
        if not value:
            path = prefix or "value"
            return f"{path} = []"
        parts: List[str] = []
        for idx, item in enumerate(value):
            item_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            part = _json_structure_to_text(
                item,
                prefix=item_prefix,
                allow_tabular=allow_tabular,
            )
            if part:
                parts.append(part)
        return "\n\n".join(parts)
    if isinstance(value, dict):
        if not value:
            path = prefix or "value"
            return f"{path} = {{}}"
        lines = [
            f"{path} = {scalar}"
            for path, scalar in _flatten_json_iter(value, prefix)
        ]
        return "\n".join(lines)
    path = prefix or "value"
    return f"{path} = {_json_scalar_to_str(value)}"


def parse_json_bytes(data: bytes) -> str:
    """Parse JSON or NDJSON content fully in memory and return text."""

    decoded = ensure_text(data)
    try:
        parsed = json.loads(decoded)
    except json.JSONDecodeError as exc:
        records: List[Any] = []
        for raw_line in decoded.splitlines():
            line = ensure_text(raw_line).strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not records:
            raise ValueError("Contenu JSON invalide ou vide.") from exc
        text = _json_structure_to_text(records)
    else:
        text = _json_structure_to_text(parsed)

    text = ensure_text(text).strip()
    if len(text) > MAX_TOTAL_CHARS:
        return text[:MAX_TOTAL_CHARS]
    return text


def parse_json_streaming(data: bytes) -> Iterator[str]:
    """Stream JSON/NDJSON content into textual blocks limited by MAX_TOTAL_CHARS."""

    if ijson is None:
        yield parse_json_bytes(data)
        return

    total_chars = 0
    buffer: List[str] = []
    buffer_len = 0

    def emit_buffer() -> Optional[str]:
        nonlocal buffer, buffer_len, total_chars
        if not buffer:
            return None
        chunk = ensure_text("\n\n".join(buffer)).strip()
        buffer = []
        buffer_len = 0
        if not chunk:
            return None
        remaining = MAX_TOTAL_CHARS - total_chars
        if remaining <= 0:
            return None
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        total_chars += len(chunk)
        return chunk

    def append_text(text: str) -> Optional[str]:
        nonlocal buffer_len
        stripped = ensure_text(text).strip()
        if not stripped:
            return None
        if len(stripped) > MAX_TOTAL_CHARS:
            stripped = stripped[:MAX_TOTAL_CHARS]
        buffer.append(stripped)
        buffer_len += len(stripped)
        threshold = max(10_000, MAX_TOTAL_CHARS // 4)
        if buffer_len >= threshold or total_chars + buffer_len >= MAX_TOTAL_CHARS:
            return emit_buffer()
        return None

    bio = BytesIO(data)
    emitted = False
    try:
        for idx, item in enumerate(ijson.items(bio, "item"), start=1):
            if total_chars >= MAX_TOTAL_CHARS:
                break
            text = _json_structure_to_text(
                item,
                prefix=f"item[{idx}]",
                allow_tabular=False,
            )
            maybe_chunk = append_text(text)
            if maybe_chunk:
                emitted = True
                yield maybe_chunk
            if total_chars >= MAX_TOTAL_CHARS:
                break
        final_chunk = emit_buffer()
        if final_chunk:
            emitted = True
            yield final_chunk
        if emitted:
            return
    except ijson.JSONError:
        pass

    bio.seek(0)
    try:
        for line_no, raw_line in enumerate(bio, start=1):
            if total_chars >= MAX_TOTAL_CHARS:
                break
            line = ensure_text(raw_line).strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _json_structure_to_text(
                item,
                prefix=f"line[{line_no}]",
                allow_tabular=False,
            )
            maybe_chunk = append_text(text)
            if maybe_chunk:
                emitted = True
                yield maybe_chunk
            if total_chars >= MAX_TOTAL_CHARS:
                break
        final_chunk = emit_buffer()
        if final_chunk:
            emitted = True
            yield final_chunk
        if emitted:
            return
    except Exception:
        pass

    fallback = parse_json_bytes(data)
    if fallback:
        if len(fallback) > MAX_TOTAL_CHARS:
            fallback = fallback[:MAX_TOTAL_CHARS]
        yield fallback


def _ingest_text_document(
    name: str,
    data: bytes,
    file_type: str,
    *,
    max_chars: int,
    overlap: int,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    safe_name = ensure_text(name)
    text_content = _bytes_to_text(data)
    if len(text_content) > MAX_TOTAL_CHARS:
        text_content = text_content[:MAX_TOTAL_CHARS]
    chunks = [
        DocumentChunk(
            text=ensure_text(chunk),
            meta={"source": safe_name, "type": file_type},
        )
        for chunk in chunk_text(text_content, max_chars=max_chars, overlap=overlap)
    ]
    summary = {
        "name": safe_name,
        "type": file_type,
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, []


def _ingest_csv(
    name: str,
    data: bytes,
    *,
    max_chars: int,
    overlap: int,
    stream: bool = False,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    safe_name = ensure_text(name)
    warnings: List[str] = []
    lower_name = name.lower()
    sep = "\t" if lower_name.endswith(".tsv") else ","
    if stream:
        csv_text = read_csv_streamed(data, sep=sep)
    else:
        encoding = _detect_encoding(data)
        try:
            df = pd.read_csv(
                io.BytesIO(data),
                encoding=encoding,
                sep=sep,
                engine="python",
                on_bad_lines="skip",
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                io.BytesIO(data),
                encoding="utf-8",
                encoding_errors="ignore",
                sep=sep,
                engine="python",
                on_bad_lines="skip",
            )
        except Exception as exc:
            return [], {}, [f"Erreur lors de la lecture CSV {safe_name}: {exc}"]

        if df.empty:
            warnings.append(f"Le fichier CSV {safe_name} est vide.")
            summary = {
                "name": safe_name,
                "type": "csv",
                "size_bytes": len(data),
                "chunk_count": 0,
                "token_estimate": 0,
            }
            return [], summary, warnings

        csv_text = ensure_text(df.to_csv(index=False))
        if len(csv_text) > MAX_TOTAL_CHARS:
            csv_text = csv_text[:MAX_TOTAL_CHARS]

    csv_text = ensure_text(csv_text)

    chunks = [
        DocumentChunk(
            text=ensure_text(chunk),
            meta={"source": safe_name, "type": "csv"},
        )
        for chunk in chunk_text(csv_text, max_chars=max_chars, overlap=overlap)
    ]

    summary = {
        "name": safe_name,
        "type": "csv",
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, warnings


def _ingest_excel(
    name: str,
    data: bytes,
    *,
    max_chars: int,
    overlap: int,
    stream: bool = False,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    safe_name = ensure_text(name)
    warnings: List[str] = []
    if stream:
        excel_text = read_excel_streamed(data)
        chunks = [
            DocumentChunk(text=ensure_text(chunk), meta={"source": safe_name, "type": "excel"})
            for chunk in chunk_text(excel_text, max_chars=max_chars, overlap=overlap)
        ]
    else:
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        except Exception as exc:
            return [], {}, [f"Erreur lors de la lecture Excel {safe_name}: {exc}"]

        chunks = []
        for sheet_name, df in sheets.items():
            if df.empty:
                warnings.append(
                    f"La feuille '{ensure_text(sheet_name)}' dans {safe_name} est vide."
                )
                continue
            safe_sheet_name = ensure_text(sheet_name)
            text = ensure_text(df.to_csv(index=False))
            if len(text) > MAX_TOTAL_CHARS:
                text = text[:MAX_TOTAL_CHARS]
            for part in chunk_text(text, max_chars=max_chars, overlap=overlap):
                chunks.append(
                    DocumentChunk(
                        text=ensure_text(part),
                        meta={
                            "source": safe_name,
                            "type": "excel",
                            "sheet": safe_sheet_name,
                        },
                    )
                )

    summary = {
        "name": safe_name,
        "type": "excel",
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, warnings


def _ingest_pdf(
    name: str,
    data: bytes,
    *,
    max_chars: int,
    overlap: int,
    stream: bool = False,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    safe_name = ensure_text(name)
    chunks: List[DocumentChunk] = []
    warnings: List[str] = []

    if stream:
        for page_text, meta in read_pdf_paged(data):
            cleaned = _clean_text(page_text)
            if not cleaned:
                continue
            for chunk in chunk_text(cleaned, max_chars=max_chars, overlap=overlap):
                chunk_meta = {"source": safe_name, "type": "pdf", **meta}
                chunks.append(DocumentChunk(text=ensure_text(chunk), meta=chunk_meta))
    else:
        pdf_reader = PdfReader(io.BytesIO(data))
        total_chars = 0
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            if total_chars >= MAX_TOTAL_CHARS:
                break
            try:
                text = ensure_text(page.extract_text() or "")
            except Exception as exc:
                warnings.append(f"Extraction impossible page {page_number} ({safe_name}) : {exc}")
                continue
            cleaned = _clean_text(text)
            if not cleaned:
                warnings.append(
                    f"Page {page_number} du PDF {safe_name} semble ne contenir aucun texte."
                )
                continue
            remaining = MAX_TOTAL_CHARS - total_chars
            truncated = cleaned[:remaining]
            total_chars += len(truncated)
            for chunk in chunk_text(truncated, max_chars=max_chars, overlap=overlap):
                chunks.append(
                    DocumentChunk(
                        text=ensure_text(chunk),
                        meta={"source": safe_name, "type": "pdf", "page": page_number},
                    )
                )
            if total_chars >= MAX_TOTAL_CHARS:
                break

    summary = {
        "name": safe_name,
        "type": "pdf",
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, warnings


def _ingest_docx(
    name: str,
    data: bytes,
    *,
    max_chars: int,
    overlap: int,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    if Document is None:
        return [], {}, [f"Le support DOCX est indisponible : {DOCX_IMPORT_ERROR}"]

    safe_name = ensure_text(name)
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        return [], {}, [f"Erreur lors de la lecture DOCX {safe_name}: {exc}"]

    paragraphs = [ensure_text(p.text) for p in document.paragraphs if p.text.strip()]
    full_text = ensure_text("\n".join(paragraphs))
    if len(full_text) > MAX_TOTAL_CHARS:
        full_text = full_text[:MAX_TOTAL_CHARS]
    chunks = [
        DocumentChunk(text=ensure_text(chunk), meta={"source": safe_name, "type": "docx"})
        for chunk in chunk_text(full_text, max_chars=max_chars, overlap=overlap)
    ]
    summary = {
        "name": safe_name,
        "type": "docx",
        "size_bytes": len(data),
        "chunk_count": len(chunks),
        "token_estimate": sum(_estimate_tokens(chunk.text) for chunk in chunks),
    }
    return chunks, summary, []


def _ingest_json(
    name: str,
    data: bytes,
    *,
    size_bytes: int,
    max_chars: int,
    overlap: int,
) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    stream_mode = should_stream_file(size_bytes)
    safe_name = ensure_text(name)
    try:
        blocks: Iterable[str]
        if stream_mode:
            blocks = parse_json_streaming(data)
        else:
            blocks = [parse_json_bytes(data)]
    except ValueError as exc:
        return [], {}, [f"Erreur lors de la lecture JSON {safe_name}: {exc}"]

    chunks: List[DocumentChunk] = []
    token_estimate = 0
    for block_index, block in enumerate(blocks, start=1):
        if not block:
            continue
        for chunk in chunk_text(block, max_chars=max_chars, overlap=overlap):
            meta: Dict[str, Any] = {"source": safe_name, "type": "json"}
            if stream_mode:
                meta["segment"] = block_index
            chunks.append(DocumentChunk(text=ensure_text(chunk), meta=meta))
            token_estimate += _estimate_tokens(chunk)

    warnings: List[str] = []
    if not chunks:
        warnings.append(
            f"Le fichier JSON {safe_name} ne contient pas de contenu exploitable."
        )

    summary = {
        "name": safe_name,
        "type": "json",
        "size_bytes": size_bytes,
        "chunk_count": len(chunks),
        "token_estimate": token_estimate,
    }
    return chunks, summary, warnings


def load_file_to_chunks(uploaded_file, max_chars: int = 4000, overlap: int = 400) -> Tuple[List[DocumentChunk], Dict[str, Any], List[str]]:
    name = uploaded_file.name
    safe_name = ensure_text(name)
    extension = name.split(".")[-1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return [], {}, [f"Format non supporté : {safe_name}"]

    size = getattr(uploaded_file, "size", None)
    data = uploaded_file.getvalue()
    if size is None:
        size = len(data)

    if size > CHAT_MAX_FILE_SIZE and not ALLOW_LARGE_FILES:
        limit_mb = DEFAULT_MAX_FILE_MB
        return [], {}, [
            f"{safe_name} ({format_bytes(size)}) dépasse {limit_mb} Mo et a été ignoré."
        ]

    stream_mode = should_stream_file(size)

    if extension in TEXT_EXTENSIONS:
        return _ingest_text_document(name, data, extension, max_chars=max_chars, overlap=overlap)
    if extension in CSV_EXTENSIONS:
        return _ingest_csv(name, data, max_chars=max_chars, overlap=overlap, stream=stream_mode)
    if extension in EXCEL_EXTENSIONS:
        return _ingest_excel(name, data, max_chars=max_chars, overlap=overlap, stream=stream_mode)
    if extension in PDF_EXTENSIONS:
        return _ingest_pdf(name, data, max_chars=max_chars, overlap=overlap, stream=stream_mode)
    if extension in DOCX_EXTENSIONS:
        return _ingest_docx(name, data, max_chars=max_chars, overlap=overlap)
    if extension in JSON_EXTENSIONS:
        return _ingest_json(
            name,
            data,
            size_bytes=size,
            max_chars=max_chars,
            overlap=overlap,
        )

    return [], {}, [f"Extension inconnue : {safe_name}"]


class _InMemoryUploadedFile:
    """Minimal wrapper to reuse ingestion pipeline with in-memory bytes."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data or b""
        self.size = len(self._data)

    def getvalue(self) -> bytes:
        return self._data


def _normalise_chat_files(files: Sequence[Any]) -> List[Tuple[str, bytes, int]]:
    normalised: List[Tuple[str, bytes, int]] = []
    for file in files:
        if file is None:
            continue
        if isinstance(file, dict):
            name = str(file.get("name") or "document")
            data = file.get("data") or b""
            if not isinstance(data, (bytes, bytearray)):
                data = bytes(data)
            size = int(file.get("size") or len(data))
        else:
            name = getattr(file, "name", "document")
            try:
                data = file.getvalue()
            except Exception:  # noqa: BLE001 - best effort fallback
                data = b""
            size = getattr(file, "size", None)
            if size is None:
                size = len(data)
        normalised.append((name, bytes(data), int(size)))
    return normalised


def index_files_from_chat(files: Sequence[Any]) -> Dict[str, Any]:
    """Index chat attachments into the shared FAISS store."""

    if not files:
        return {"documents": 0, "chunks": 0, "warnings": []}

    if FAISS_IMPORT_ERROR is not None:
        raise RuntimeError(f"faiss-cpu est requis pour activer le RAG : {FAISS_IMPORT_ERROR}")

    api_key = st.session_state.get("api_key")
    if not api_key:
        raise RuntimeError("Aucune clé API disponible pour l'indexation.")

    normalised = _normalise_chat_files(files)
    if not normalised:
        return {"documents": 0, "chunks": 0, "warnings": []}

    warnings: List[str] = []
    chunk_texts: List[str] = []
    chunk_metas: List[Dict[str, Any]] = []
    doc_summaries: List[Dict[str, Any]] = []

    if len(normalised) > CHAT_MAX_FILES:
        warnings.append(
            f"Seuls les {CHAT_MAX_FILES} premiers fichiers ont été indexés sur {len(normalised)} fournis."
        )
        if _STATUS_WRITE_CALLBACK:
            _STATUS_WRITE_CALLBACK(
                f"ℹ️ Seuls les {CHAT_MAX_FILES} premiers fichiers seront traités."
            )
        normalised = normalised[:CHAT_MAX_FILES]

    if _STATUS_UPDATE_CALLBACK:
        _STATUS_UPDATE_CALLBACK("📥 Lecture des fichiers…", "running")

    for name, data, size in normalised:
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        safe_name = ensure_text(name)

        if extension == "json":
            stream_mode = should_stream_file(size)
            try:
                if stream_mode:
                    blocks: Iterable[str] = parse_json_streaming(data)
                else:
                    blocks = [parse_json_bytes(data)]
            except ValueError as exc:
                warn_msg = (
                    f"Impossible d'indexer {safe_name} ({format_bytes(size)}): {exc}"
                )
                st.warning(warn_msg)
                warnings.append(warn_msg)
                if _STATUS_WRITE_CALLBACK:
                    _STATUS_WRITE_CALLBACK(f"❌ {safe_name} : {exc}")
                continue

            chunk_count = 0
            token_estimate = 0
            for block_index, block in enumerate(blocks, start=1):
                if not block:
                    continue
                for chunk in chunk_text(block):
                    meta: Dict[str, Any] = {"source": safe_name, "type": "json"}
                    if stream_mode:
                        meta["segment"] = block_index
                    chunk_texts.append(ensure_text(chunk))
                    chunk_metas.append(meta)
                    chunk_count += 1
                    token_estimate += _estimate_tokens(chunk)

            if chunk_count == 0:
                msg = f"⚠️ {safe_name} — aucun contenu JSON exploitable."
                warnings.append(msg)
                if _STATUS_WRITE_CALLBACK:
                    _STATUS_WRITE_CALLBACK(msg)
                continue

            summary = {
                "name": safe_name,
                "type": "json",
                "size_bytes": size,
                "chunk_count": chunk_count,
                "token_estimate": token_estimate,
            }
            doc_summaries.append(summary)
            if _STATUS_WRITE_CALLBACK:
                _STATUS_WRITE_CALLBACK(f"✔️ {safe_name} — {chunk_count} chunks")
            continue

        in_memory = _InMemoryUploadedFile(name, data)
        try:
            chunks, summary, chunk_warnings = load_file_to_chunks(in_memory)
        except Exception as exc:  # noqa: BLE001 - feedback in UI
            warn_msg = f"Impossible d'indexer {safe_name} ({format_bytes(size)}): {exc}"
            st.warning(warn_msg)
            warnings.append(warn_msg)
            if _STATUS_WRITE_CALLBACK:
                _STATUS_WRITE_CALLBACK(f"❌ {safe_name} : {exc}")
            continue

        if summary:
            doc_summaries.append(summary)
        if chunk_warnings:
            warnings.extend(chunk_warnings)
            if _STATUS_WRITE_CALLBACK:
                for warn in chunk_warnings:
                    _STATUS_WRITE_CALLBACK(f"⚠️ {warn}")
        if not chunks:
            if not chunk_warnings:
                msg = f"⚠️ {safe_name} — aucun texte exploitable."
                warnings.append(msg)
                if _STATUS_WRITE_CALLBACK:
                    _STATUS_WRITE_CALLBACK(msg)
            continue

        for chunk in chunks:
            chunk_texts.append(ensure_text(chunk.text))
            chunk_metas.append(chunk.meta)
        if _STATUS_WRITE_CALLBACK:
            _STATUS_WRITE_CALLBACK(f"✔️ {safe_name} — {len(chunks)} chunks")

    if not chunk_texts:
        return {"documents": 0, "chunks": 0, "warnings": warnings}

    if _STATUS_UPDATE_CALLBACK:
        _STATUS_UPDATE_CALLBACK("🧠 Calcul des embeddings…", "running")

    client = OpenAI(api_key=api_key)
    embedding_model = st.session_state.get("rag_embedding_model") or DEFAULT_EMBEDDING_MODEL
    embeddings = embed_texts(client, embedding_model, chunk_texts)

    if embeddings.size == 0:
        warnings.append("Impossible de calculer les embeddings pour les pièces jointes.")
        if _STATUS_WRITE_CALLBACK:
            _STATUS_WRITE_CALLBACK("❌ Échec du calcul des embeddings.")
        return {"documents": len(doc_summaries), "chunks": 0, "warnings": warnings}

    if _STATUS_UPDATE_CALLBACK:
        _STATUS_UPDATE_CALLBACK("📚 Mise à jour de l'index FAISS…", "running")

    try:
        st.session_state.rag_index = add_embeddings_to_index(
            st.session_state.get("rag_index"),
            embeddings,
        )
    except Exception as exc:  # noqa: BLE001 - propagate as runtime error
        raise RuntimeError(f"Impossible de mettre à jour l'index FAISS : {exc}") from exc

    st.session_state.rag_texts.extend(chunk_texts)
    st.session_state.rag_meta.extend(chunk_metas)
    st.session_state.rag_docs.extend(doc_summaries)
    st.session_state.rag_embedding_model = embedding_model

    return {
        "documents": len(doc_summaries),
        "chunks": len(chunk_texts),
        "warnings": warnings,
    }


def batch_by_token_budget(texts: Sequence[str], model: str, max_tokens: int):
    """Yield batches of texts whose combined tokens stay under the budget."""

    if max_tokens <= 0:
        for text in texts:
            batch = TokenBatch()
            current = truncate_context_text(text or "", model, 0)
            if current != (text or ""):
                batch.truncated = True
            batch.append(current)
            yield batch
        return

    batch = TokenBatch()
    total = 0
    for text in texts:
        current = text or ""
        token_count = count_tokens_text(current, model)
        truncated = False
        if token_count > max_tokens:
            current = truncate_context_text(current, model, max_tokens)
            truncated = current != (text or "")
            token_count = count_tokens_text(current, model)
        if total + token_count > max_tokens and batch:
            yield batch
            batch = TokenBatch()
            total = 0
        batch.append(current)
        total += token_count
        if truncated:
            batch.truncated = True
    if batch:
        yield batch


def embed_texts(client: OpenAI, model: str, texts: Sequence[str]) -> np.ndarray:
    safe_texts = ensure_text_iter(texts)
    if not safe_texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors: List[List[float]] = []
    truncated_any = False

    for batch in batch_by_token_budget(safe_texts, model, EMBED_MAX_TOKENS_PER_REQUEST):
        truncated_any = truncated_any or getattr(batch, "truncated", False)
        safe_batch = ensure_text_iter(batch)
        if not safe_batch:
            continue
        response = client.embeddings.create(model=model, input=list(safe_batch))
        vectors.extend(item.embedding for item in response.data)

    if truncated_any:
        st.warning("Document volumineux compressé pour rester sous la limite de tokens.")

    if not vectors:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms
    return np.ascontiguousarray(normalized.astype(np.float32))


def ensure_faiss_index(dimension: int):
    if faiss is None:
        raise RuntimeError(f"faiss-cpu est requis : {FAISS_IMPORT_ERROR}")
    index = faiss.IndexFlatIP(dimension)
    return index


def add_embeddings_to_index(index, embeddings: np.ndarray):
    if embeddings.size == 0:
        return index
    if index is None:
        index = ensure_faiss_index(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(
    client: OpenAI,
    query: str,
    index,
    texts: Sequence[str],
    metas: Sequence[Dict[str, Any]],
    model: str,
    k: int = 4,
) -> List[Tuple[str, Dict[str, Any], float]]:
    if index is None or not texts:
        return []

    query_embedding = embed_texts(client, model, [query])
    if query_embedding.size == 0:
        return []

    query_array = np.ascontiguousarray(query_embedding)
    distances, indices = index.search(query_array, min(k, len(texts)))
    hits: List[Tuple[str, Dict[str, Any], float]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(texts):
            continue
        hits.append((texts[idx], metas[idx], float(score)))
    return hits


def format_context(hits: Iterable[Tuple[str, Dict[str, Any], float]]) -> str:
    sections: List[str] = []
    for pos, (chunk, meta, _) in enumerate(hits, start=1):
        descriptor_parts = [meta.get("source", "inconnu")]
        if meta.get("page") is not None:
            descriptor_parts.append(f"page {meta['page']}")
        if meta.get("sheet"):
            descriptor_parts.append(f"feuille {meta['sheet']}")
        if meta.get("row_range"):
            descriptor_parts.append(f"lignes {meta['row_range']}")
        descriptor = " ".join(descriptor_parts)
        sections.append(f"[{pos}] Source: {descriptor}\n{chunk}")
    return "\n\n".join(sections)


def format_source_badge(meta: Dict[str, Any], index: int) -> str:
    parts = [f"[{index}] {meta.get('source', 'inconnu')}"]
    if meta.get("page") is not None:
        parts.append(f"page {meta['page']}")
    if meta.get("sheet"):
        parts.append(f"feuille {meta['sheet']}")
    if meta.get("row_range"):
        parts.append(f"lignes {meta['row_range']}")
    return " ".join(parts)


def human_readable_size(num_bytes: int) -> str:
    if num_bytes == 0:
        return "0 o"
    units = ["o", "Ko", "Mo", "Go"]
    size = float(num_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    return f"{size:.2f} {units[unit_index]}"
