import numpy as np
import pytest

from privrag.core.dp_engine import DifferentialPrivacyEngine


def test_clip_never_exceeds_configured_norm() -> None:
    engine = DifferentialPrivacyEngine(clip_norm=2.0, random_state=1)
    clipped = engine.clip([3.0, 4.0])
    assert np.linalg.norm(clipped) == pytest.approx(2.0)


def test_calibration_and_noise_shape_are_valid() -> None:
    engine = DifferentialPrivacyEngine(epsilon=2.0, delta=1e-6, clip_norm=1.5, random_state=2)
    assert engine.laplace_scale == pytest.approx(1.5)
    assert engine.gaussian_sigma > engine.laplace_scale
    noise = engine.sample_noise(64, "gaussian")
    assert noise.shape == (64,)
    assert np.isfinite(noise).all()


def test_invalid_privacy_parameters_fail_early() -> None:
    with pytest.raises(ValueError):
        DifferentialPrivacyEngine(epsilon=0)
    with pytest.raises(ValueError):
        DifferentialPrivacyEngine(delta=1)
