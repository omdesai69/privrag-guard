"""LangChain Embeddings wrapper without a hard LangChain dependency."""

from __future__ import annotations

from typing import Any, Sequence

from privrag.core.guard import PrivRAGGuard


class LangChainPrivGuardEmbeddings:
    """Wrap any LangChain-compatible embeddings object at the provider boundary."""

    def __init__(self, embeddings: Any, guard: PrivRAGGuard) -> None:
        if not hasattr(embeddings, "embed_documents") or not hasattr(embeddings, "embed_query"):
            raise TypeError("embeddings must implement embed_documents and embed_query")
        self.embeddings = embeddings
        self.guard = guard

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raw = self.embeddings.embed_documents(list(texts))
        return self.guard.protect_batch(raw).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.guard.protect_vector(self.embeddings.embed_query(text)).tolist()

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if hasattr(self.embeddings, "aembed_documents"):
            raw = await self.embeddings.aembed_documents(list(texts))
            return self.guard.protect_batch(raw).tolist()
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        if hasattr(self.embeddings, "aembed_query"):
            raw = await self.embeddings.aembed_query(text)
            return self.guard.protect_vector(raw).tolist()
        return self.embed_query(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.embeddings, name)
