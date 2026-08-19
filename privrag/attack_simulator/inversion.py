"""A deliberately simplified embedding inversion demonstration.

This is a synthetic token-dictionary attack for education and regression tests,
not an implementation of an attack against a real embedding provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


@dataclass(frozen=True)
class ReconstructionResult:
    tokens: list[str]
    confidences: list[float]
    mean_confidence: float

    def as_dict(self) -> dict[str, object]:
        return {"tokens": self.tokens, "confidences": self.confidences, "mean_confidence": self.mean_confidence}


class EmbeddingInversionSimulator:
    """Toy attacker that ranks known token vectors by cosine similarity.

    A protected vector from a guard using keyed rotation is intentionally in a
    different coordinate system, so this unkeyed dictionary cannot match it.
    """

    def __init__(self, dimension: int = 128, random_state: int | None = 7) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension
        self.rng = np.random.default_rng(random_state)
        self._tokens: list[str] = []
        self._codebook: np.ndarray | None = None

    def fit(self, vocabulary: list[str]) -> "EmbeddingInversionSimulator":
        clean = list(dict.fromkeys(token.lower() for token in vocabulary if token.strip()))
        if not clean:
            raise ValueError("vocabulary must contain at least one token")
        codebook = self.rng.normal(size=(len(clean), self.dimension))
        codebook /= np.linalg.norm(codebook, axis=1, keepdims=True)
        self._tokens, self._codebook = clean, codebook
        return self

    def encode(self, text: str) -> np.ndarray:
        """Create a synthetic embedding by averaging known token code vectors."""
        self._require_fit()
        tokens = re.findall(r"[a-zA-Z0-9_'-]+", text.lower())
        positions = [self._tokens.index(token) for token in tokens if token in self._tokens]
        if not positions:
            raise ValueError("text contains no tokens available in the simulator vocabulary")
        vector = self._codebook[positions].mean(axis=0)
        return vector / np.linalg.norm(vector)

    def reconstruct(self, embedding: np.ndarray | list[float], top_k: int = 5) -> ReconstructionResult:
        self._require_fit()
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.shape != (self.dimension,):
            raise ValueError(f"embedding must have dimension {self.dimension}")
        vector /= np.linalg.norm(vector)
        scores = self._codebook @ vector
        top_k = min(max(1, top_k), len(self._tokens))
        indices = np.argsort(scores)[-top_k:][::-1]
        confidence = ((scores[indices] + 1.0) / 2.0).clip(0.0, 1.0)
        return ReconstructionResult(
            tokens=[self._tokens[i] for i in indices],
            confidences=confidence.round(3).tolist(),
            mean_confidence=float(confidence.mean()),
        )

    def compare(self, raw_embedding: np.ndarray, protected_embedding: np.ndarray, top_k: int = 5) -> dict[str, dict[str, object]]:
        return {
            "raw": self.reconstruct(raw_embedding, top_k).as_dict(),
            "protected": self.reconstruct(protected_embedding, top_k).as_dict(),
        }

    def _require_fit(self) -> None:
        if self._codebook is None:
            raise RuntimeError("call fit(vocabulary) before encoding or reconstructing")
