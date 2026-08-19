import numpy as np
import pytest

from privrag.core.subspace_proj import SubspaceProjectionEngine


def test_noise_projection_is_orthogonal_to_the_semantic_direction() -> None:
    engine = SubspaceProjectionEngine()
    semantic = np.array([1.0, 2.0, -1.0, 0.5])
    noise = np.array([0.7, -0.3, 1.2, -1.1])
    projected = engine.project_noise(noise, semantic)
    assert np.dot(projected, engine.normalize(semantic)) == pytest.approx(0.0, abs=1e-12)


def test_fitted_subspace_and_batch_projection_have_expected_shapes() -> None:
    rng = np.random.default_rng(4)
    corpus = rng.normal(size=(30, 10))
    engine = SubspaceProjectionEngine(n_components=3).fit(corpus)
    projected = engine.project_vector(corpus[0])
    noise = engine.project_noise_batch(rng.normal(size=(4, 10)), corpus[:4])
    assert engine.components_.shape == (10, 3)
    assert np.linalg.norm(projected) == pytest.approx(1.0)
    assert noise.shape == (4, 10)
