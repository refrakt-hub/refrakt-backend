"""Wrapper utilities around simple-vector-store for assistant retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from simple_vector_store import SimpleVectorStore


@dataclass
class VectorResult:
    text: str
    metadata: Dict[str, str]
    score: float


class VectorStoreManager:
    """Manage persistence and querying of the assistant vector store."""

    def __init__(self, storage_path: Path):
        self._storage_path = storage_path
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._store: Optional[SimpleVectorStore] = None
        self._loaded = False

    def is_ready(self) -> bool:
        return self._loaded and self._store is not None

    def load_or_initialize(self) -> None:
        """Load store from disk if present."""
        store = SimpleVectorStore()
        if self._storage_path.exists():
            try:
                store.load(str(self._storage_path))
                self._store = store
                self._loaded = True
                return
            except Exception:
                # fall back to rebuilding if load fails
                pass
        self._store = SimpleVectorStore()
        self._loaded = False

    def rebuild(self, chunks: List[Dict[str, str]], embed_fn) -> None:
        """Create a fresh index from provided chunks."""
        if not chunks:
            self._store = SimpleVectorStore()
            self._loaded = True
            self._persist()
            return

        vectors = []
        for chunk in chunks:
            vector = embed_fn(chunk["text"])
            vectors.append((chunk, vector))

        vector_dim = len(vectors[0][1])
        store = SimpleVectorStore(vector_dim=vector_dim)

        for chunk, vector in vectors:
            metadata = dict(chunk.get("metadata", {}))
            metadata["chunk_id"] = chunk["chunk_id"]
            store.add_item(vector=vector, text=chunk["text"], metadata=metadata)

        self._store = store
        self._loaded = True
        self._persist()

    def search(self, query_text: str, embed_fn, *, k: int = 4, min_score: float = 0.15) -> List[VectorResult]:
        if not self._store or not self._loaded:
            return []

        vector = embed_fn(query_text)
        try:
            matches = self._store.search_vector(vector, k=k)
        except Exception:
            return []

        results: List[VectorResult] = []
        for item_id, score in matches:
            if score < min_score:
                continue
            item = self._store.get_item(item_id)
            results.append(
                VectorResult(
                    text=item.get("text", ""),
                    metadata=item.get("metadata", {}) or {},
                    score=float(score),
                )
            )
        return results

    def _persist(self) -> None:
        if not self._store:
            return
        try:
            self._store.save(str(self._storage_path))
        except Exception:
            pass


