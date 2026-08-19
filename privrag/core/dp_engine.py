"""Differential-privacy noise primitives for embedding vectors."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

Mechanism = Literal["laplace", "gaussian"]


class DifferentialPrivacyEngine:
    """Create calibrated Laplace or Gaussian noise.

    The public ``sample_noise`` method exposes standard textbook calibration for a
    vector whose L2 sensitivity is at most ``2 * clip_norm``.  The guard uses a
    separately bounded *utility layer* on top of these samples; see ``guard.py``
    for the important security/utility trade-off.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        mechanism: Mechanism = "gaussian",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1")
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if mechanism not in ("laplace", "gaussian"):
            raise ValueError("mechanism must be 'laplace' or 'gaussian'")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.clip_norm = float(clip_norm)
        self.mechanism: Mechanism = mechanism
        self.rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    @property
    def sensitivity(self) -> float:
        """L2 sensitivity after clipping a single embedding."""
        return 2.0 * self.clip_norm

    @property
    def laplace_scale(self) -> float:
        """Per-coordinate Laplace scale for pure epsilon-DP calibration."""
        return self.sensitivity / self.epsilon

    @property
    def gaussian_sigma(self) -> float:
        """Classic (epsilon, delta)-DP Gaussian standard deviation."""
        return self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def clip(self, embedding: np.ndarray | list[float]) -> np.ndarray:
        """Return a copy whose L2 norm is bounded by ``clip_norm``."""
        vector = self._as_vector(embedding)
        norm = float(np.linalg.norm(vector))
        return vector if norm <= self.clip_norm or norm == 0.0 else vector * (self.clip_norm / norm)

    def sample_noise(self, dimension: int, mechanism: Mechanism | None = None) -> np.ndarray:
        """Sample calibrated, unbounded DP noise for ``dimension`` coordinates."""
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        chosen = mechanism or self.mechanism
        if chosen == "laplace":
            return self.rng.laplace(0.0, self.laplace_scale, size=dimension)
        if chosen == "gaussian":
            return self.rng.normal(0.0, self.gaussian_sigma, size=dimension)
        raise ValueError("mechanism must be 'laplace' or 'gaussian'")

    def sample_noise_matrix(
        self, rows: int, dimension: int, mechanism: Mechanism | None = None
    ) -> np.ndarray:
        """Sample calibrated noise for a batch without Python-level loops."""
        if rows <= 0 or dimension <= 0:
            raise ValueError("rows and dimension must be positive")
        chosen = mechanism or self.mechanism
        shape = (rows, dimension)
        if chosen == "laplace":
            return self.rng.laplace(0.0, self.laplace_scale, size=shape)
        if chosen == "gaussian":
            return self.rng.normal(0.0, self.gaussian_sigma, size=shape)
        raise ValueError("mechanism must be 'laplace' or 'gaussian'")

    def add_noise(
        self, embedding: np.ndarray | list[float], mechanism: Mechanism | None = None
    ) -> np.ndarray:
        """Clip an embedding and add a textbook-calibrated DP sample."""
        clipped = self.clip(embedding)
        return clipped + self.sample_noise(clipped.size, mechanism)

    @staticmethod
    def _as_vector(embedding: np.ndarray | list[float]) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float64)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding must be a non-empty one-dimensional vector")
        if not np.isfinite(vector).all():
            raise ValueError("embedding must contain only finite numbers")
        return vector.copy()
