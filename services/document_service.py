"""Document ingestion helpers for the assistant retrieval index."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from config import get_settings
from utils.config_validator import ConfigValidator


@dataclass
class DocumentChunk:
    """Lightweight representation of a text chunk for retrieval indexing."""

    chunk_id: str
    text: str
    metadata: Dict[str, str]


class DocumentService:
    """Load and chunk Refrakt knowledge sources for retrieval."""

    def __init__(self, chunk_chars: int = 1200, overlap_chars: int = 200):
        self._settings = get_settings()
        self._chunk_chars = chunk_chars
        self._overlap_chars = overlap_chars

    def load_static_corpus(self) -> List[DocumentChunk]:
        """Gather document chunks from canonical Refrakt documentation."""
        chunks: List[DocumentChunk] = []

        chunks.extend(self._load_markdown_file(self._settings.PROMPT_TEMPLATE_PATH, label="prompt"))
        chunks.extend(self._load_config_examples())
        validator_chunk = self._build_validator_summary_chunk()
        if validator_chunk:
            chunks.append(validator_chunk)

        return [chunk for chunk in chunks if chunk and chunk.text.strip()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_markdown_file(self, path: Path, label: str) -> List[DocumentChunk]:
        if not path.exists():
            return []

        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []

        normalized = self._normalize_markdown(text)
        base_id = self._slugify(label)
        return self._chunk_text(normalized, source=str(path), base_id=base_id)

    def _load_config_examples(self) -> List[DocumentChunk]:
        configs_dir = (self._settings.PROJECT_ROOT / "configs").resolve()
        if not configs_dir.exists():
            return []

        chunks: List[DocumentChunk] = []
        for path in sorted(configs_dir.glob("*.y*l")):
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue

            text = f"Refrakt configuration template `{path.name}`:\n\n{raw}"
            base_id = self._slugify(f"config-{path.stem}")
            chunks.extend(self._chunk_text(text, source=str(path), base_id=base_id))
        return chunks

    def _build_validator_summary_chunk(self) -> Optional[DocumentChunk]:
        try:
            validator = ConfigValidator()
        except Exception:
            return None

        supported_models = ", ".join(validator.supported_model_names())
        unsupported_aliases = validator.unsupported_model_names()
        unsupported_snippet = ", ".join(unsupported_aliases[:10])

        text_lines = [
            "Refrakt Supported Models Overview",
            "",
            f"Supported model templates: {supported_models}.",
        ]
        if unsupported_snippet:
            text_lines.extend(
                [
                    "",
                    "Frequently requested but currently unsupported models include: "
                    f"{unsupported_snippet}.",
                ]
            )

        text = "\n".join(text_lines)
        return DocumentChunk(
            chunk_id=f"validator-summary-{self._hash_key(text)}",
            text=text,
            metadata={
                "source": "config_validator",
                "category": "models",
            },
        )

    def _chunk_text(self, text: str, *, source: str, base_id: str) -> List[DocumentChunk]:
        paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
        if not paragraphs:
            return []

        chunks: List[DocumentChunk] = []
        buffer: List[str] = []
        current_chars = 0

        for para in paragraphs:
            para_len = len(para)
            if current_chars + para_len > self._chunk_chars and buffer:
                chunk_text = "\n\n".join(buffer)
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{base_id}-{len(chunks)}",
                        text=chunk_text,
                        metadata={
                            "source": source,
                        },
                    )
                )
                # Start new buffer with overlap for continuity
                if self._overlap_chars > 0 and chunk_text:
                    overlap_text = chunk_text[-self._overlap_chars :]
                    buffer = [overlap_text, para]
                    current_chars = len(overlap_text) + para_len
                else:
                    buffer = [para]
                    current_chars = para_len
            else:
                buffer.append(para)
                current_chars += para_len

        if buffer:
            chunk_text = "\n\n".join(buffer)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{base_id}-{len(chunks)}",
                    text=chunk_text,
                    metadata={
                        "source": source,
                    },
                )
            )

        return chunks

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        # Remove fenced code block markers and collapse multiple hashes
        text = re.sub(r"`{3,}", "", text)
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
        text = text.replace("\r\n", "\n")
        return text

    @staticmethod
    def _slugify(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
        return cleaned or "doc"

    @staticmethod
    def _hash_key(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


