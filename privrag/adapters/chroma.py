"""A small Chroma collection decorator that protects all supplied embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np

from privrag.core.guard import PrivRAGGuard


class ChromaPrivGuard:
    """Decorate a Chroma-like collection so vector writes and queries are guarded.

    The adapter accepts any object implementing Chroma's ``add``, ``upsert`` and
    ``query`` interface, which also makes it straightforward to test without a
    running Chroma service.  Pass explicit embeddings: allowing Chroma to embed
    documents internally would bypass the protection boundary.
    """

    def __init__(self, collection: Any, guard: PrivRAGGuard) -> None:
        self.collection = collection
        self.guard = guard

    @classmethod
    def from_client(cls, client: Any, name: str, guard: PrivRAGGuard, **kwargs: Any) -> "ChromaPrivGuard":
        """Create/get a collection through a Chroma client and wrap it."""
        collection = client.get_or_create_collection(name=name, **kwargs)
        return cls(collection, guard)

    def add(self, *, embeddings: Any | None = None, **kwargs: Any) -> Any:
        return self.collection.add(embeddings=self._protect_embeddings(embeddings), **kwargs)

    def upsert(self, *, embeddings: Any | None = None, **kwargs: Any) -> Any:
        return self.collection.upsert(embeddings=self._protect_embeddings(embeddings), **kwargs)

    def query(self, *, query_embeddings: Any | None = None, **kwargs: Any) -> Any:
        if query_embeddings is None:
            raise ValueError("query_embeddings are required so PrivRAG-Guard can protect the query")
        protected = self._protect_embeddings(query_embeddings)
        return self.collection.query(query_embeddings=protected, **kwargs)

    def get(self, **kwargs: Any) -> Any:
        """Proxy reads unchanged; stored embeddings remain protected."""
        return self.collection.get(**kwargs)

    def _protect_embeddings(self, embeddings: Any | None) -> list[list[float]]:
        if embeddings is None:
            raise ValueError(
                "explicit embeddings are required; document-side embedding inside Chroma would bypass protection"
            )
        matrix = np.asarray(embeddings, dtype=np.float64)
        was_single_vector = matrix.ndim == 1
        if was_single_vector:
            matrix = matrix[None, :]
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a vector or a two-dimensional batch")
        return self.guard.protect_batch(matrix).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.collection, name)
