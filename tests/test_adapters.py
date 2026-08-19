"""Test all four adapter wrappers."""

import asyncio

import numpy as np

from privrag import PrivRAGGuard
from privrag.adapters import ChromaPrivGuard, LangChainPrivGuardEmbeddings


class FakeCollection:
    def __init__(self) -> None:
        self.writes: list[dict] = []
        self.queries: list[dict] = []

    def add(self, **kwargs):
        self.writes.append(kwargs)

    def upsert(self, **kwargs):
        self.writes.append(kwargs)

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return {"ids": [["a"]]}

    def get(self, **kwargs):
        return {"ids": ["a"]}


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(len(t)), 1.0, 0.5, -0.25] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.5, -0.25]

    async def aembed_query(self, text):
        return self.embed_query(text)


def test_chroma_adapter_sanitizes_writes_and_queries() -> None:
    collection = FakeCollection()
    adapter = ChromaPrivGuard(collection, PrivRAGGuard(random_state=5))
    adapter.add(ids=["a", "b"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    adapter.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
    stored = np.asarray(collection.writes[0]["embeddings"])
    queried = np.asarray(collection.queries[0]["query_embeddings"])
    assert np.allclose(np.linalg.norm(stored, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(queried, axis=1), 1.0)


def test_langchain_wrapper_protects_sync_and_async() -> None:
    wrapper = LangChainPrivGuardEmbeddings(FakeEmbeddings(), PrivRAGGuard(random_state=12))
    docs = np.asarray(wrapper.embed_documents(["one", "three"]))
    query = np.asarray(wrapper.embed_query("query"))
    async_query = np.asarray(asyncio.run(wrapper.aembed_query("query")))
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert np.isclose(np.linalg.norm(query), 1.0)
    assert np.isclose(np.linalg.norm(async_query), 1.0)


class FakeFaissIndex:
    def __init__(self) -> None:
        self.added = []
        self.searched = []

    def add(self, vectors):
        self.added.append(vectors)

    def add_with_ids(self, vectors, ids):
        self.added.append((vectors, ids))

    def search(self, vectors, k):
        self.searched.append((vectors, k))
        return np.ones((len(vectors), k)), np.zeros((len(vectors), k), dtype=int)


def test_faiss_adapter_protects_add_and_search() -> None:
    from privrag.adapters import FaissPrivGuardIndex
    raw_index = FakeFaissIndex()
    adapter = FaissPrivGuardIndex(raw_index, PrivRAGGuard(random_state=42))
    adapter.add([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    adapter.add_with_ids([[1.0, 2.0, 3.0]], [99])
    distances, indices = adapter.search([1.0, 2.0, 3.0], k=2)

    added_vecs = raw_index.added[0]
    assert np.allclose(np.linalg.norm(added_vecs, axis=1), 1.0)
    assert raw_index.added[1][1][0] == 99
    assert len(raw_index.searched) == 1


class FakeQdrantClient:
    def __init__(self) -> None:
        self.points_queried = []

    def query_points(self, collection_name, query, limit, **kwargs):
        self.points_queried.append((collection_name, query, limit))
        return ["result"]


def test_qdrant_adapter_query() -> None:
    from privrag.adapters import QdrantPrivGuard
    client = FakeQdrantClient()
    adapter = QdrantPrivGuard(client, PrivRAGGuard(random_state=42))
    res = adapter.query("test_collection", [1.0, 2.0, 3.0], limit=3)
    assert res == ["result"]
    assert len(client.points_queried) == 1
    queried_vec = client.points_queried[0][1]
    assert np.isclose(np.linalg.norm(queried_vec), 1.0)

