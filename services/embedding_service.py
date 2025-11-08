"""Embedding generation with lightweight on-disk caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
import numpy as np

from config import get_settings


class EmbeddingService:
    """Wrapper around OpenAI embeddings with persistent caching."""

    def __init__(self):
        settings = get_settings()
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.ASSISTANT_EMBEDDING_MODEL
        self._cache_path = settings.ASSISTANT_INDEX_PATH / "embedding_cache.json"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[float]] = self._load_cache()

    def embed(self, text: str) -> List[float]:
        """Return embedding vector for text, using cache when available."""
        key = self._hash_text(text)
        if key in self._cache:
            return np.asarray(self._cache[key], dtype=float)

        response = self._client.embeddings.create(
            model=self._model,
            input=[text],
        )
        vector = response.data[0].embedding
        self._cache[key] = list(vector)
        self._persist_cache()
        return np.asarray(vector, dtype=float)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_cache(self) -> Dict[str, List[float]]:
        if not self._cache_path.exists():
            return {}
        try:
            with self._cache_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return {k: list(map(float, v)) for k, v in data.items()}
        except (OSError, ValueError, TypeError):
            pass
        return {}

    def _persist_cache(self) -> None:
        try:
            with self._cache_path.open("w", encoding="utf-8") as handle:
                json.dump(self._cache, handle)
        except OSError:
            pass

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


