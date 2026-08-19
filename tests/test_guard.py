import numpy as np
import pytest

from privrag import PrivRAGGuard


def test_protected_vectors_are_unit_length_and_preserve_pairwise_cosines() -> None:
    rng = np.random.default_rng(10)
    raw = rng.normal(size=(60, 128))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    guard = PrivRAGGuard(noise_fraction=0.08, random_state=3)
    protected = guard.protect_batch(raw)
    assert np.allclose(np.linalg.norm(protected, axis=1), 1.0, atol=1e-6)
    metrics = guard.benchmark_utility(raw, protected, k=5)
    assert metrics["mean_cosine_loss"] < 0.02
    assert metrics["mean_cosine_similarity"] > 0.98


def test_guard_rejects_zero_vectors() -> None:
    with pytest.raises(ValueError):
        PrivRAGGuard().protect_vector([0.0, 0.0])


def test_passphrase_and_saved_key_keep_processes_in_the_same_index_space(tmp_path) -> None:
    vector = np.array([0.4, -0.2, 0.1, 0.8])
    first = PrivRAGGuard(noise_fraction=0.0, passphrase="deployment-only-secret", random_state=1)
    second = PrivRAGGuard(noise_fraction=0.0, passphrase="deployment-only-secret", random_state=999)
    protected = first.protect_vector(vector)
    assert np.allclose(protected, second.protect_vector(vector))

    key_path = tmp_path / "rotation.key"
    first.save_key(key_path)
    restored = PrivRAGGuard.from_key_file(key_path, noise_fraction=0.0, random_state=55)
    assert np.allclose(protected, restored.protect_vector(vector))


def test_batch_path_matches_single_vector_path_for_a_seeded_guard() -> None:
    vectors = np.random.default_rng(8).normal(size=(12, 16))
    batch_guard = PrivRAGGuard(random_state=123)
    single_guard = PrivRAGGuard(random_state=123)
    batch = batch_guard.protect_batch(vectors)
    singles = np.vstack([single_guard.protect_vector(vector) for vector in vectors])
    assert np.allclose(batch, singles, atol=1e-6)
