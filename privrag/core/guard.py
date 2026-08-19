"""The high-level PrivRAG protection transform."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from privrag.core.dp_engine import DifferentialPrivacyEngine, Mechanism
from privrag.core.subspace_proj import SubspaceProjectionEngine


@dataclass(frozen=True)
class UtilityMetrics:
    mean_cosine_similarity: float
    min_cosine_similarity: float
    mean_cosine_loss: float
    recall_at_k: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "min_cosine_similarity": self.min_cosine_similarity,
            "mean_cosine_loss": self.mean_cosine_loss,
            "recall_at_k": self.recall_at_k,
        }


class PrivRAGGuard:
    """Protect embeddings before they are persisted or queried.

    The transform combines three operations: normalize, add bounded noise in a
    non-critical subspace, and apply a keyed orthogonal rotation.  Rotation keeps
    every cosine relationship unchanged while hiding the provider's coordinate
    system from an attacker without the guard key.  All embeddings in one index
    must use the same guard instance/configuration.

    ``noise_fraction`` is deliberately bounded to make RAG useful.  This is a
    practical defense layer, not a substitute for a formal end-to-end DP proof:
    truncating a textbook DP sample to meet a utility budget changes its formal
    distribution.  Use ``dp_engine.add_noise`` when strict raw DP calibration is
    required and evaluate recall for the intended corpus.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        mechanism: Mechanism = "gaussian",
        noise_fraction: float = 0.10,
        keyed_rotation: bool = True,
        random_state: int | None = None,
        passphrase: str | None = None,
        subspace_engine: SubspaceProjectionEngine | None = None,
    ) -> None:
        if not 0.0 <= noise_fraction < 1.0:
            raise ValueError("noise_fraction must be in [0, 1)")
        if passphrase is not None and not passphrase:
            raise ValueError("passphrase must not be empty")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.clip_norm = float(clip_norm)
        self.mechanism = mechanism
        self.noise_fraction = float(noise_fraction)
        self.keyed_rotation = keyed_rotation
        self.passphrase = passphrase
        self.dp_engine = DifferentialPrivacyEngine(epsilon, delta, clip_norm, mechanism, random_state)
        self.subspace_engine = subspace_engine or SubspaceProjectionEngine()
        self._rotation_seed = self._derive_rotation_seed(passphrase, random_state)
        self._rotation_rng = np.random.default_rng(self._rotation_seed)
        self._rotation: np.ndarray | None = None

    def fit_subspace(self, reference_embeddings: np.ndarray | list[list[float]]) -> "PrivRAGGuard":
        """Learn corpus-level semantic directions from representative vectors."""
        self.subspace_engine.fit(reference_embeddings)
        return self

    def protect_vector(self, embedding: np.ndarray | list[float]) -> np.ndarray:
        """Return a unit-length protected embedding suitable for cosine indexes."""
        return self._protect_matrix(np.asarray(embedding, dtype=np.float64)[None, :])[0]

    def protect_batch(self, embeddings: np.ndarray | list[list[float]]) -> np.ndarray:
        """Protect an ingestion batch using fully vectorized NumPy operations."""
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty two-dimensional matrix")
        return self._protect_matrix(matrix)

    def save_key(self, path: str | Path) -> None:
        """Persist the initialized rotation matrix for another trusted process.

        The file is raw key material. Store it only in a protected secret volume
        with restrictive permissions; use ``passphrase`` if deterministic key
        derivation is a better fit for your deployment.
        """
        if self._rotation is None:
            raise RuntimeError("No rotation matrix has been initialized yet; protect a vector first.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as key_file:
            np.save(key_file, self._rotation, allow_pickle=False)

    def load_key(self, path: str | Path) -> "PrivRAGGuard":
        """Load and validate a rotation matrix into this guard instance."""
        source = Path(path)
        with source.open("rb") as key_file:
            rotation = np.load(key_file, allow_pickle=False)
        self._set_rotation(rotation)
        return self

    @classmethod
    def from_key_file(cls, path: str | Path, **kwargs: object) -> "PrivRAGGuard":
        """Construct a guard and load an existing trusted rotation key."""
        guard = cls(**kwargs)
        return guard.load_key(path)

    def benchmark_utility(
        self,
        raw_vectors: np.ndarray | list[list[float]],
        protected_vectors: np.ndarray | list[list[float]],
        k: int = 5,
    ) -> dict[str, float]:
        """Compare self-alignment and nearest-neighbour overlap of two corpora."""
        raw = self._normalize_matrix(raw_vectors)
        protected = self._normalize_matrix(protected_vectors)
        if raw.shape != protected.shape:
            raise ValueError("raw_vectors and protected_vectors must have the same shape")
        n = raw.shape[0]
        if n < 2:
            raise ValueError("benchmark requires at least two vectors")
        k = max(1, min(k, n - 1))
        raw_scores = raw @ raw.T
        protected_scores = protected @ protected.T
        # A keyed rotation intentionally changes coordinate values.  Utility is
        # therefore measured by preservation of document-to-document cosine
        # relationships, which is what a vector index actually consumes.
        upper = np.triu_indices(n, k=1)
        relation_loss = np.abs(raw_scores[upper] - protected_scores[upper])
        relation_retention = 1.0 - relation_loss
        np.fill_diagonal(raw_scores, -np.inf)
        np.fill_diagonal(protected_scores, -np.inf)
        raw_neighbours = np.argpartition(raw_scores, -k, axis=1)[:, -k:]
        protected_neighbours = np.argpartition(protected_scores, -k, axis=1)[:, -k:]
        recall = float(np.mean([len(set(a).intersection(b)) / k for a, b in zip(raw_neighbours, protected_neighbours)]))
        return UtilityMetrics(
            mean_cosine_similarity=float(np.mean(relation_retention)),
            min_cosine_similarity=float(np.min(relation_retention)),
            mean_cosine_loss=float(np.mean(relation_loss)),
            recall_at_k=recall,
        ).as_dict()

    def _get_rotation(self, dimension: int) -> np.ndarray:
        if self._rotation is None:
            matrix = self._rotation_rng.normal(size=(dimension, dimension))
            rotation, diagonal = np.linalg.qr(matrix)
            diag_signs = np.sign(np.diag(diagonal))
            diag_signs[diag_signs == 0.0] = 1.0
            self._set_rotation(rotation * diag_signs)
        elif self._rotation.shape != (dimension, dimension):
            raise ValueError("one PrivRAGGuard instance can only protect one embedding dimension")
        return self._rotation

    def _protect_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("embeddings must be a non-empty two-dimensional matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("embeddings must contain only finite numbers")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("zero vectors cannot be protected")
        clipped = matrix * np.minimum(1.0, self.clip_norm / norms)
        semantic = clipped / np.linalg.norm(clipped, axis=1, keepdims=True)
        if self.noise_fraction:
            dp_noise = self.dp_engine.sample_noise_matrix(*semantic.shape, self.mechanism)
            orthogonal_noise = self.subspace_engine.project_noise_batch(dp_noise, semantic)
            noise_norms = np.linalg.norm(orthogonal_noise, axis=1, keepdims=True)
            target = min(0.45, self.noise_fraction / self.epsilon)
            active = noise_norms[:, 0] > 1e-12
            if np.any(active):
                semantic[active] += orthogonal_noise[active] * (target / noise_norms[active])
                semantic /= np.linalg.norm(semantic, axis=1, keepdims=True)
        if self.keyed_rotation:
            semantic = semantic @ self._get_rotation(semantic.shape[1])
        return semantic.astype(np.float32)

    def _set_rotation(self, rotation: np.ndarray) -> None:
        candidate = np.asarray(rotation, dtype=np.float64)
        if candidate.ndim != 2 or candidate.shape[0] == 0 or candidate.shape[0] != candidate.shape[1]:
            raise ValueError("rotation key must be a non-empty square matrix")
        if not np.isfinite(candidate).all() or not np.allclose(candidate.T @ candidate, np.eye(candidate.shape[0]), atol=1e-6):
            raise ValueError("rotation key must be an orthogonal matrix")
        self._rotation = candidate

    @staticmethod
    def _derive_rotation_seed(passphrase: str | None, random_state: int | None) -> int | None:
        if passphrase is None:
            return random_state
        key = hashlib.scrypt(
            passphrase.encode("utf-8"),
            salt=b"privrag-guard:rotation-key:v1",
            n=2**14,
            r=8,
            p=1,
            dklen=16,
        )
        return int.from_bytes(key, byteorder="big", signed=False)

    @staticmethod
    def _normalize_matrix(vectors: np.ndarray | list[list[float]]) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("vectors must be a non-empty two-dimensional matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("vectors cannot contain zero rows")
        return matrix / norms
