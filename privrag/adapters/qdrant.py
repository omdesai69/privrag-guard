"""Optional Qdrant client boundary for protected vectors."""

from __future__ import annotations

from typing import Any, Sequence

from privrag.core.guard import PrivRAGGuard


class QdrantPrivGuard:
    """Protect vectors passed to a ``qdrant_client.QdrantClient`` instance."""

    def __init__(self, client: Any, guard: PrivRAGGuard) -> None:
        self.client = client
        self.guard = guard

    def upsert_vectors(
        self,
        collection_name: str,
        ids: Sequence[str | int],
        embeddings: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any] | None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create Qdrant PointStruct records with sanitized vectors."""
        try:
            from qdrant_client.models import PointStruct
        except ImportError as error:  # pragma: no cover - dependency optional
            raise ImportError("Install PrivRAG-Guard with the 'qdrant' extra to use this adapter") from error
        protected = self.guard.protect_batch(embeddings).tolist()
        if len(ids) != len(protected):
            raise ValueError("ids and embeddings must have equal lengths")
        records = payloads or [None] * len(ids)
        if len(records) != len(ids):
            raise ValueError("payloads and embeddings must have equal lengths")
        points = [PointStruct(id=item_id, vector=vector, payload=payload) for item_id, vector, payload in zip(ids, protected, records)]
        return self.client.upsert(collection_name=collection_name, points=points, **kwargs)

    def query(self, collection_name: str, query_embedding: Sequence[float], limit: int = 5, **kwargs: Any) -> Any:
        vector = self.guard.protect_vector(query_embedding).tolist()
        if hasattr(self.client, "query_points"):
            return self.client.query_points(collection_name=collection_name, query=vector, limit=limit, **kwargs)
        return self.client.search(collection_name=collection_name, query_vector=vector, limit=limit, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)
