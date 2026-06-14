"""Calibration improves ECE on a deliberately miscalibrated fixture."""
from __future__ import annotations

import numpy as np

from praman.calibrate import Calibrator, brier, ece

RNG = np.random.default_rng(11)


def _miscalibrated(n=4000):
    """Logits that are overconfident: true p = sigmoid(z/3) but we present z."""
    z = RNG.normal(0, 4, size=n)
    true_p = 1 / (1 + np.exp(-z / 3.0))      # the real probability is softer
    y = (RNG.uniform(size=n) < true_p).astype(int)
    return z, y


def test_temperature_reduces_ece():
    z, y = _miscalibrated()
    cal = Calibrator(method="temperature").fit(z, y)
    assert cal.metrics["after"]["ece"] < cal.metrics["before"]["ece"]
    # the overconfidence should push T > 1 (softening)
    assert cal.temperature > 1.2


def test_isotonic_reduces_ece():
    z, y = _miscalibrated()
    cal = Calibrator(method="isotonic").fit(z, y)
    assert cal.metrics["after"]["ece"] <= cal.metrics["before"]["ece"] + 1e-6


def test_ece_brier_bounds():
    p = RNG.uniform(size=500); y = RNG.integers(0, 2, size=500)
    assert 0.0 <= ece(p, y) <= 1.0
    assert 0.0 <= brier(p, y) <= 1.0


def test_calibrator_roundtrip(tmp_path):
    z, y = _miscalibrated(1000)
    cal = Calibrator(method="temperature").fit(z, y)
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = Calibrator.load(path)
    assert abs(loaded.temperature - cal.temperature) < 1e-6
    np.testing.assert_allclose(loaded.transform(z[:20]), cal.transform(z[:20]), atol=1e-6)
