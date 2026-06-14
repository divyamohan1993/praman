"""CRC/RCPS correctness: monotonicity, no-feasible default, expected-risk control."""
from __future__ import annotations

import numpy as np

from praman.riskcontrol import (bootstrap_validate, crc_threshold, miss_rate,
                                rcps_threshold, validate_risk)

RNG = np.random.default_rng(7)


def _toy_pos(n=600):
    return RNG.beta(5, 2, size=n)  # ungrounded scores skew high


def test_crc_threshold_monotone_in_alpha():
    u = _toy_pos(800)
    ts = [crc_threshold(u, a) for a in (0.01, 0.05, 0.10, 0.20, 0.40)]
    assert all(ts[i] <= ts[i + 1] + 1e-9 for i in range(len(ts) - 1)), ts


def test_crc_no_feasible_returns_approve_nothing():
    u = _toy_pos(200)
    # alpha below the finite-sample resolution 1/(n+1) -> approve nothing (t=0)
    assert crc_threshold(u, 1e-9) == 0.0


def test_crc_controls_expected_fnr():
    """CRC is an in-expectation guarantee: mean realized FNR over splits <= alpha."""
    alpha = 0.1
    realized = []
    for _ in range(300):
        cal = RNG.beta(5, 2, size=400)
        t = crc_threshold(cal, alpha)
        test = RNG.beta(5, 2, size=1500)
        realized.append(float(np.mean(test < t)))
    assert np.mean(realized) <= alpha + 1e-3, np.mean(realized)


def test_rcps_high_probability_control():
    """RCPS (Hoeffding): fraction of splits exceeding alpha <= delta (+ MC slack)."""
    alpha, delta = 0.1, 0.05
    realized = []
    for _ in range(300):
        cal = RNG.beta(5, 2, size=500)
        t = rcps_threshold(cal, alpha, delta)
        test = RNG.beta(5, 2, size=1500)
        realized.append(float(np.mean(test < t)))
    assert np.mean(np.array(realized) > alpha) <= delta + 0.04


def test_miss_rate_monotone_in_t():
    u = _toy_pos(500)
    ts = np.linspace(0, 1, 50)
    ms = [miss_rate(u, t) for t in ts]
    assert all(ms[i] <= ms[i + 1] + 1e-9 for i in range(len(ms) - 1))


def test_validate_risk_schema():
    u_pos = _toy_pos(300)
    u_test = RNG.beta(3, 3, size=400)
    y_test = (u_test > 0.5).astype(int)
    r = validate_risk(u_pos, u_test, y_test, 0.1)
    assert {"threshold", "realized_fnr", "coverage", "contamination", "joint"} <= set(r)


def test_bootstrap_reports_exceed_fraction():
    u = RNG.beta(3, 3, size=1000)
    y = (u > 0.5).astype(int)
    out = bootstrap_validate(u, y, 0.1, n_boot=50)
    assert 0.0 <= out["frac_exceed_alpha"] <= 1.0
    assert "mean_realized_fnr" in out
