from privrag import PrivRAGGuard
from privrag.attack_simulator import EmbeddingInversionSimulator


def test_synthetic_attack_confidence_drops_in_protected_coordinate_space() -> None:
    vocabulary = ["patient", "diabetes", "medication", "12345", "clinic", "report"]
    simulator = EmbeddingInversionSimulator(dimension=64, random_state=7).fit(vocabulary)
    raw = simulator.encode("patient diabetes medication 12345")
    protected = PrivRAGGuard(noise_fraction=0.05, random_state=21).protect_vector(raw)
    result = simulator.compare(raw, protected, top_k=4)
    assert "patient" in result["raw"]["tokens"]
    assert result["protected"]["mean_confidence"] < result["raw"]["mean_confidence"]
