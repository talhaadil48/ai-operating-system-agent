"""Text extraction, chunking, and lightweight embedding helpers for RAG."""

from __future__ import annotations

import hashlib
import io
import math
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

_WORD_RE = re.compile(r"[a-z0-9]+")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".log",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".html",
    ".htm",
    ".py",
}


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(self.parts)


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_bytes(data: bytes, filename: str, content_type: str | None = None) -> str:
    suffix = Path(filename or "").suffix.lower()
    content_type = (content_type or "").lower()

    if suffix in _TEXT_EXTENSIONS or content_type.startswith("text/") or content_type in {
        "application/json",
        "application/xml",
        "application/csv",
        "text/csv",
    }:
        return data.decode("utf-8", errors="replace")

    if suffix == ".docx":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            document_xml = archive.read("word/document.xml")
        root = ET.fromstring(document_xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            parts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
            if parts:
                paragraphs.append("".join(parts))
        return "\n\n".join(paragraphs)

    if suffix == ".pdf":
        if PdfReader is None:
            raise ValueError("PDF upload requires the optional 'pypdf' package.")
        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for page in reader.pages:
            extracted = page.extract_text() or ""
            if extracted:
                pages.append(extracted)
        return "\n\n".join(pages)

    if suffix in {".html", ".htm"}:
        parser = _HTMLStripper()
        parser.feed(data.decode("utf-8", errors="replace"))
        return parser.text()

    return data.decode("utf-8", errors="replace")


def split_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    paragraphs = [paragraph.strip() for paragraph in _PARA_SPLIT_RE.split(cleaned) if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            flush()
            sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(paragraph) if sentence.strip()]
            sentence_buffer = ""
            for sentence in sentences:
                candidate = f"{sentence_buffer} {sentence}".strip() if sentence_buffer else sentence
                if len(candidate) <= max_chars:
                    sentence_buffer = candidate
                else:
                    if sentence_buffer:
                        chunks.append(sentence_buffer)
                    if len(sentence) > max_chars:
                        start = 0
                        while start < len(sentence):
                            end = min(len(sentence), start + max_chars)
                            piece = sentence[start:end].strip()
                            if piece:
                                chunks.append(piece)
                            if end >= len(sentence):
                                break
                            start = max(0, end - overlap)
                        sentence_buffer = ""
                    else:
                        sentence_buffer = sentence
            if sentence_buffer:
                chunks.append(sentence_buffer)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        flush()
        current = paragraph

    flush()

    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = []
        for index, chunk in enumerate(chunks):
            if index == 0:
                overlapped.append(chunk)
                continue
            prefix = chunks[index - 1][-overlap:]
            candidate = f"{prefix}\n{chunk}".strip()
            overlapped.append(candidate if len(candidate) <= max_chars else chunk)
        chunks = overlapped

    return chunks


def embed_text(text: str, dimension: int = 512) -> list[float]:
    tokens = _WORD_RE.findall(normalize_text(text).lower())
    if not tokens:
        return [0.0] * dimension

    vector = [0.0] * dimension
    for index, token in enumerate(tokens):
        weight = 1.0 + math.log1p(len(token))
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        slot = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[slot] += sign * weight

        if index + 1 < len(tokens):
            bigram = f"{token}_{tokens[index + 1]}"
            digest = hashlib.blake2b(bigram.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "big") % dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[slot] += sign * 0.75

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector