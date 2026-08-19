"""Semantic subspace projection utilities."""

from __future__ import annotations

import numpy as np


class SubspaceProjectionEngine:
    """Learns an optional semantic basis and keeps perturbations out of it.

    Before fitting, each vector's own direction is treated as critical.  After
    ``fit``, the learned principal semantic basis is also retained.  This makes
    the perturbation orthogonal to the signal directions used for cosine search.
    """

    def __init__(self, n_components: int | None = None) -> None:
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be positive when provided")
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray | list[list[float]]) -> "SubspaceProjectionEngine":
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            raise ValueError("fit requires a matrix with at least two embeddings")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings must contain only finite numbers")
        self.mean_ = matrix.mean(axis=0)
        centered = matrix - self.mean_
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        components = self.n_components or min(32, right_vectors.shape[0])
        self.components_ = right_vectors[: min(components, right_vectors.shape[0])].T
        return self

    @staticmethod
    def normalize(vector: np.ndarray | list[float]) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("vector must be non-empty and one-dimensional")
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            raise ValueError("zero vectors cannot be normalized")
        return array / norm

    def project_noise(self, noise: np.ndarray, semantic_vector: np.ndarray) -> np.ndarray:
        """Remove components aligned with a vector and learned semantic basis."""
        candidate = np.asarray(noise, dtype=np.float64).copy()
        anchor = self.normalize(semantic_vector)
        if candidate.shape != anchor.shape:
            raise ValueError("noise and semantic_vector must have the same shape")
        candidate -= anchor * float(np.dot(candidate, anchor))
        if self.components_ is not None:
            if self.components_.shape[0] != candidate.size:
                raise ValueError("fitted subspace dimension does not match embedding")
            candidate -= self.components_ @ (self.components_.T @ candidate)
            # Re-remove the vector direction because bases need not be orthogonal.
            candidate -= anchor * float(np.dot(candidate, anchor))
        return candidate

    def project_noise_batch(self, noise: np.ndarray, semantic_vectors: np.ndarray) -> np.ndarray:
        """Vectorized equivalent of :meth:`project_noise` for ingestion batches."""
        candidate = np.asarray(noise, dtype=np.float64).copy()
        anchors = np.asarray(semantic_vectors, dtype=np.float64)
        if candidate.ndim != 2 or anchors.ndim != 2 or candidate.shape != anchors.shape:
            raise ValueError("noise and semantic_vectors must be equally shaped matrices")
        norms = np.linalg.norm(anchors, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("semantic_vectors cannot contain zero rows")
        anchors = anchors / norms
        candidate -= np.sum(candidate * anchors, axis=1, keepdims=True) * anchors
        if self.components_ is not None:
            if self.components_.shape[0] != candidate.shape[1]:
                raise ValueError("fitted subspace dimension does not match embedding")
            candidate -= (candidate @ self.components_) @ self.components_.T
            candidate -= np.sum(candidate * anchors, axis=1, keepdims=True) * anchors
        return candidate

    def project_vector(self, vector: np.ndarray | list[float]) -> np.ndarray:
        """Project a vector into the learned semantic basis and normalize it."""
        value = np.asarray(vector, dtype=np.float64)
        if self.components_ is None:
            return self.normalize(value)
        if value.size != self.components_.shape[0]:
            raise ValueError("fitted subspace dimension does not match embedding")
        projected = self.components_ @ (self.components_.T @ value)
        return self.normalize(projected)
