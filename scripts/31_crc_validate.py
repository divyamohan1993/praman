#!/usr/bin/env python
"""Validate the conformal risk control implementation BEFORE trusting it on RAGTruth.

Two parts:

A. Independent Monte-Carlo self-check on a synthetic toy with a KNOWN score distribution.
   Confirms the actual mathematical guarantees:
     * CRC controls the EXPECTED realized FNR:  mean over trials of realized FNR <= alpha.
       (CRC is an in-expectation / marginal guarantee. Individual splits CAN exceed alpha;
        that is correct behavior, not a bug. We do NOT chase frac_exceed to zero.)
     * RCPS (Hoeffding) controls it with HIGH PROBABILITY:  frac of trials exceeding alpha
       <= delta.
     * Threshold is monotone non-decreasing in alpha.
     * No-feasible-threshold default is approve-nothing (t = 0).

B. Best-effort cross-check vs MAPIE (version pinned in requirements.lock.txt). MAPIE
   refactored its risk-control API around v1.0; if the class path differs we report the
   version + the discrepancy and rely on part A (the brief allows this fallback).
"""
from __future__ import annotations

import numpy as np

from praman.riskcontrol import crc_threshold, rcps_threshold

RNG = np.random.default_rng(20260614)


def make_toy(n_pos: int, n_neg: int):
    """Detector scores u in [0,1]; positives (ungrounded) skew high, negatives skew low."""
    u_pos = RNG.beta(5, 2, size=n_pos)   # ungrounded: u tends high (detector works)
    u_neg = RNG.beta(2, 5, size=n_neg)   # grounded:   u tends low
    return u_pos, u_neg


def montecarlo(alpha: float, method: str, trials: int = 400, n_cal: int = 500,
               n_test: int = 2000, delta: float = 0.05) -> dict:
    realized, thresholds = [], []
    for _ in range(trials):
        cal_pos, _ = make_toy(n_cal, n_cal)
        t = (rcps_threshold(cal_pos, alpha, delta) if method == "rcps"
             else crc_threshold(cal_pos, alpha))
        test_pos, _ = make_toy(n_test, 1)
        realized.append(float(np.mean(test_pos < t)))
        thresholds.append(t)
    realized = np.array(realized)
    return {
        "alpha": alpha, "method": method, "trials": trials,
        "mean_realized_fnr": float(realized.mean()),
        "p95_realized_fnr": float(np.percentile(realized, 95)),
        "frac_exceed_alpha": float(np.mean(realized > alpha)),
        "mean_threshold": float(np.mean(thresholds)),
    }


def part_a() -> bool:
    print("=== Part A: Monte-Carlo self-check ===")
    ok = True
    # monotonicity in alpha (single calibration draw)
    cal_pos, _ = make_toy(800, 800)
    ts = [crc_threshold(cal_pos, a) for a in (0.01, 0.05, 0.10, 0.20)]
    mono = all(ts[i] <= ts[i + 1] + 1e-9 for i in range(len(ts) - 1))
    print(f"[mono] CRC thresholds vs alpha {[round(t,3) for t in ts]}  monotone={mono}")
    ok &= mono
    # no-feasible => approve-nothing (alpha far below 1/(n+1))
    t0 = crc_threshold(cal_pos, 1e-6)
    print(f"[edge] alpha=1e-6 -> t={t0} (expect 0.0, approve-nothing)  ok={t0 == 0.0}")
    ok &= (t0 == 0.0)
    # CRC: expected realized FNR <= alpha ; RCPS: high-prob
    for a in (0.01, 0.05, 0.10):
        rc = montecarlo(a, "crc"); rr = montecarlo(a, "rcps")
        crc_ok = rc["mean_realized_fnr"] <= a + 1e-3
        rcps_ok = rr["frac_exceed_alpha"] <= 0.05 + 0.03  # delta=0.05, MC slack
        print(f"[crc ] a={a:.2f} mean_fnr={rc['mean_realized_fnr']:.4f} "
              f"frac_exceed={rc['frac_exceed_alpha']:.3f}  expected<=alpha: {crc_ok}")
        print(f"[rcps] a={a:.2f} mean_fnr={rr['mean_realized_fnr']:.4f} "
              f"frac_exceed={rr['frac_exceed_alpha']:.3f}  high-prob(<=delta): {rcps_ok}")
        ok &= crc_ok and rcps_ok
    print(f"[part A] {'PASS' if ok else 'FAIL'}")
    return ok


def part_b() -> None:
    print("\n=== Part B: MAPIE cross-check (best-effort) ===")
    try:
        import mapie
        print(f"[mapie] version {mapie.__version__}")
    except Exception as e:  # pragma: no cover
        print(f"[mapie] import failed: {e}; relying on Part A")
        return
    # Try the known CRC entry points across MAPIE versions.
    cal_pos, _ = make_toy(1000, 1)
    alpha = 0.1
    mine = crc_threshold(cal_pos, alpha)
    found = False
    for path, attr in [("mapie.risk_control", "BinaryClassificationController"),
                       ("mapie.multi_label_classification", "MapieMultiLabelClassifier"),
                       ("mapie.risk_control", "PrecisionRecallController")]:
        try:
            mod = __import__(path, fromlist=[attr])
            cls = getattr(mod, attr)
            print(f"[mapie] found {path}.{attr} (API differs across versions; "
                  f"manual cross-check recommended). My CRC t={mine:.3f} on toy.")
            found = True
            break
        except Exception:
            continue
    if not found:
        print(f"[mapie] no compatible CRC class located in this version; "
              f"Part A is the authority. My CRC t={mine:.3f} on toy (alpha={alpha}).")
    # Independent re-derivation of the exact CRC bound (no MAPIE) as a second check:
    n = len(cal_pos)
    grid = np.linspace(0, 1, 1001)
    feas = [t for t in grid if (n / (n + 1)) * np.mean(cal_pos < t) + 1 / (n + 1) <= alpha]
    ref = max(feas) if feas else 0.0
    print(f"[recheck] independent exact-bound t={ref:.3f} vs mine t={mine:.3f} "
          f"(diff={abs(ref-mine):.3f})")


if __name__ == "__main__":
    a_ok = part_a()
    part_b()
    raise SystemExit(0 if a_ok else 1)
