"""FAISS index decorator for protected, cosine-ready vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from privrag.core.guard import PrivRAGGuard


class FaissPrivGuardIndex:
    """Guard ``add`` and ``search`` calls on a FAISS-like index.

    Use with ``faiss.IndexFlatIP`` (or an inner-product index) because the guard
    emits unit-normalized vectors, making inner product equivalent to cosine.
    """

    def __init__(self, index: Any, guard: PrivRAGGuard) -> None:
        self.index = index
        self.guard = guard

    def add(self, embeddings: np.ndarray | list[list[float]]) -> None:
        self.index.add(self.guard.protect_batch(embeddings).astype(np.float32))

    def add_with_ids(self, embeddings: np.ndarray | list[list[float]], ids: np.ndarray | list[int]) -> None:
        if not hasattr(self.index, "add_with_ids"):
            raise TypeError("the wrapped FAISS index does not support add_with_ids")
        self.index.add_with_ids(self.guard.protect_batch(embeddings).astype(np.float32), np.asarray(ids))

    def search(self, query_embeddings: np.ndarray | list[list[float]], k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(query_embeddings, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries[None, :]
        return self.index.search(self.guard.protect_batch(queries).astype(np.float32), k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.index, name)
