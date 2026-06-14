"""Conformal risk control: provably bound the missed-hallucination rate.

THE quantity (DoD 12, brief 7.4): the missed-hallucination rate = the false-negative
rate of the hallucination detector = P(auto-approve | claim is ungrounded). We pick a
threshold t so this is <= alpha (CRC finite-sample, or RCPS at confidence 1-delta).

Conventions
-----------
* detector score  u = 1 - p_grounded   (higher u => more likely ungrounded)
* auto-approve a claim  iff  u < t      (equivalently p_grounded > 1 - t)
* miss(t) on the ungrounded calibration claims = mean(u_pos < t)  -- monotone NON-DECREASING in t
* we want the LARGEST t with the corrected miss bound <= alpha (max coverage at the guarantee)
* if no t is feasible, return t = 0.0 => approve NOTHING (safest), and the caller is warned

We ALSO report, honestly, two related-but-different quantities (brief 4 vs 7.4 differ):
  - contamination = P(ungrounded | approved)   (what the 4 prose describes)
  - joint         = P(ungrounded AND approved)
Only the FNR carries the headline finite-sample guarantee here; the others are diagnostics.

Cross-checked against MAPIE on a toy set (scripts/30_crc_validate.py) before trusting on RAGTruth.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_GRID = np.linspace(0.0, 1.0, 501)


# --------------------------------------------------------------------------- #
# Core threshold selection
# --------------------------------------------------------------------------- #
def miss_rate(u_pos: np.ndarray, t: float) -> float:
    """Empirical missed-hallucination rate at threshold t (auto-approve iff u < t)."""
    if len(u_pos) == 0:
        return 0.0
    return float(np.mean(u_pos < t))


def crc_threshold(u_pos: Sequence[float], alpha: float,
                  grid: np.ndarray = _GRID) -> float:
    """CRC: largest t whose finite-sample-corrected FNR bound <= alpha.

    Correction (Angelopoulos et al., monotone bounded loss):
        corrected(t) = n/(n+1) * miss_hat(t) + 1/(n+1) <= alpha
    n = number of ungrounded (positive) calibration claims. Returns 0.0 (approve-nothing)
    if nothing is feasible (e.g. alpha < 1/(n+1), below the calibration resolution).
    """
    u = np.asarray(u_pos, dtype=float)
    n = len(u)
    if n == 0:
        return 0.0
    best = 0.0
    feasible_found = False
    for t in np.sort(grid):  # ascending; corrected(t) is non-decreasing in t
        corrected = (n / (n + 1.0)) * miss_rate(u, t) + 1.0 / (n + 1.0)
        if corrected <= alpha:
            best = float(t)
            feasible_found = True
        elif feasible_found:
            break  # once we leave the feasible region we never re-enter (monotone)
    return best


def _ucb(miss_hat: float, n: int, delta: float, bound: str, var: float | None = None) -> float:
    """Upper confidence bound on the true FNR, for RCPS."""
    if n == 0:
        return 1.0
    if bound == "hoeffding":
        return miss_hat + math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    if bound == "bernstein":
        v = var if var is not None else miss_hat * (1.0 - miss_hat)
        return (miss_hat + math.sqrt(2.0 * v * math.log(2.0 / delta) / n)
                + 7.0 * math.log(2.0 / delta) / (3.0 * (n - 1) if n > 1 else 1))
    raise ValueError(f"unknown bound {bound}")


def rcps_threshold(u_pos: Sequence[float], alpha: float, delta: float = 0.05,
                   bound: str = "hoeffding", grid: np.ndarray = _GRID) -> float:
    """RCPS: largest t whose (1-delta) upper confidence bound on FNR <= alpha."""
    u = np.asarray(u_pos, dtype=float)
    n = len(u)
    if n == 0:
        return 0.0
    best = 0.0
    feasible_found = False
    for t in np.sort(grid):
        m = miss_rate(u, t)
        var = float(np.var(u < t)) if n > 1 else m * (1 - m)
        if _ucb(m, n, delta, bound, var) <= alpha:
            best = float(t)
            feasible_found = True
        elif feasible_found:
            break
    return best


def ltt_overflag_threshold(u_neg: Sequence[float], beta: float, delta: float = 0.05,
                           grid: np.ndarray = _GRID) -> float:
    """Secondary control (LTT, fixed-sequence): bound the OVER-flag rate.

    over-flag = flagging a GROUNDED claim = u_neg >= t. We want P(flag | grounded) <= beta
    with confidence 1-delta. Fixed-sequence test from high t (few flags) downward; stop at
    the first t whose Hoeffding UCB on the over-flag rate exceeds beta. Returns the smallest
    SAFE t (most flagging still within the precision budget).
    """
    u = np.asarray(u_neg, dtype=float)
    n = len(u)
    if n == 0:
        return 0.0
    safe = 1.0
    for t in np.sort(grid)[::-1]:  # descending: start where we flag almost nothing
        over = float(np.mean(u >= t))
        ucb = over + math.sqrt(math.log(1.0 / delta) / (2.0 * n))
        if ucb <= beta:
            safe = float(t)
        else:
            break
    return safe


# --------------------------------------------------------------------------- #
# Non-exchangeable (NN-reweighted) threshold for the OOD slice
# --------------------------------------------------------------------------- #
def nonexchangeable_threshold(u_pos: np.ndarray, feats_pos: np.ndarray,
                              feat_query: np.ndarray, alpha: float, k: int = 50,
                              grid: np.ndarray = _GRID) -> float:
    """CRC with calibration positives reweighted by similarity to the query point.

    A basic instantiation of non-exchangeable conformal (Barber et al.; Ulmer et al.):
    weight each calibration positive by 1/(1+dist) to the query, restrict to its k nearest,
    and run the CRC correction on that reweighted empirical FNR. Cheap (no extra model);
    features supplied by the caller (e.g. [u, len(claim), len(evidence)] or embeddings).
    """
    u = np.asarray(u_pos, dtype=float)
    if len(u) == 0:
        return 0.0
    d = np.linalg.norm(feats_pos - feat_query[None, :], axis=1)
    idx = np.argsort(d)[:k]
    w = 1.0 / (1.0 + d[idx])
    w = w / w.sum()
    uu = u[idx]
    n = len(uu)
    best = 0.0
    feasible = False
    for t in np.sort(grid):
        miss = float(np.sum(w * (uu < t)))
        corrected = (n / (n + 1.0)) * miss + 1.0 / (n + 1.0)
        if corrected <= alpha:
            best = float(t); feasible = True
        elif feasible:
            break
    return best


# --------------------------------------------------------------------------- #
# Controller: fit per-alpha (and per-group) thresholds, serialize, apply
# --------------------------------------------------------------------------- #
@dataclass
class RiskController:
    method: str = "crc"                       # crc | rcps
    delta: float = 0.05
    bound: str = "hoeffding"
    thresholds: dict[str, float] = field(default_factory=dict)        # alpha(str) -> t
    group_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)  # group -> alpha -> t
    meta: dict[str, Any] = field(default_factory=dict)

    def _select(self, u_pos: np.ndarray, alpha: float) -> float:
        if self.method == "rcps":
            return rcps_threshold(u_pos, alpha, self.delta, self.bound)
        return crc_threshold(u_pos, alpha)

    def fit(self, u: np.ndarray, ungrounded: np.ndarray, alphas: Sequence[float],
            groups: np.ndarray | None = None) -> "RiskController":
        """u: detector scores (1-p_grounded); ungrounded: 0/1 truth; fit thresholds per alpha."""
        u = np.asarray(u, dtype=float)
        y = np.asarray(ungrounded, dtype=int)
        u_pos = u[y == 1]
        for a in alphas:
            self.thresholds[f"{a:g}"] = self._select(u_pos, a)
        if groups is not None:
            groups = np.asarray(groups)
            for g in np.unique(groups):
                gp = u[(y == 1) & (groups == g)]
                self.group_thresholds[str(g)] = {f"{a:g}": self._select(gp, a) for a in alphas}
        self.meta["n_pos"] = int((y == 1).sum())
        self.meta["n_total"] = int(len(y))
        return self

    def threshold(self, alpha: float, group: str | None = None) -> float:
        key = f"{alpha:g}"
        if group is not None and group in self.group_thresholds and key in self.group_thresholds[group]:
            return self.group_thresholds[group][key]
        return self.thresholds.get(key, 0.0)

    def approve(self, p_grounded: float, alpha: float, group: str | None = None) -> bool:
        """Auto-approve iff detector score u < t (i.e. confident enough it's grounded)."""
        u = 1.0 - float(p_grounded)
        return u < self.threshold(alpha, group)

    # --- persistence ---
    def to_json(self) -> str:
        return json.dumps({
            "method": self.method, "delta": self.delta, "bound": self.bound,
            "thresholds": self.thresholds, "group_thresholds": self.group_thresholds,
            "meta": self.meta,
        }, indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RiskController":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(method=d["method"], delta=d.get("delta", 0.05), bound=d.get("bound", "hoeffding"),
                   thresholds=d.get("thresholds", {}), group_thresholds=d.get("group_thresholds", {}),
                   meta=d.get("meta", {}))


# --------------------------------------------------------------------------- #
# Validation: realized risk over repeated splits (the honesty protocol)
# --------------------------------------------------------------------------- #
def validate_risk(u_calib_pos: np.ndarray, u_test: np.ndarray, y_test: np.ndarray,
                  alpha: float, method: str = "crc", delta: float = 0.05,
                  bound: str = "hoeffding") -> dict[str, float]:
    """One trial: fit t on calib positives, measure realized FNR + coverage on test."""
    t = (rcps_threshold(u_calib_pos, alpha, delta, bound) if method == "rcps"
         else crc_threshold(u_calib_pos, alpha))
    y = np.asarray(y_test, dtype=int)
    u = np.asarray(u_test, dtype=float)
    approved = u < t
    pos = y == 1
    realized_fnr = float(np.mean(approved[pos])) if pos.any() else 0.0
    coverage = float(np.mean(approved))
    contamination = float(np.mean(y[approved] == 1)) if approved.any() else 0.0
    return {
        "threshold": t, "realized_fnr": realized_fnr, "coverage": coverage,
        "contamination": contamination, "joint": float(np.mean(approved & pos)),
    }


def bootstrap_validate(u: np.ndarray, y: np.ndarray, alpha: float, n_boot: int = 200,
                       calib_frac: float = 0.5, method: str = "crc", delta: float = 0.05,
                       bound: str = "hoeffding", seed: int = 1337) -> dict[str, Any]:
    """Repeat calib/test resamples; report mean realized FNR and the fraction exceeding alpha.

    The single most important honesty number: ``frac_exceed`` = how often the realized FNR
    on held-out test exceeds the target alpha. With correct CRC this should be small and
    NOT chased to zero (chasing it invalidates the guarantee). We do NOT tune on these.
    """
    rng = np.random.default_rng(seed)
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=int)
    n = len(u)
    fnrs, covs, conts, exceed = [], [], [], 0
    for _ in range(n_boot):
        perm = rng.permutation(n)
        k = int(calib_frac * n)
        ci, ti = perm[:k], perm[k:]
        u_calib_pos = u[ci][y[ci] == 1]
        r = validate_risk(u_calib_pos, u[ti], y[ti], alpha, method, delta, bound)
        fnrs.append(r["realized_fnr"]); covs.append(r["coverage"]); conts.append(r["contamination"])
        if r["realized_fnr"] > alpha:
            exceed += 1
    return {
        "alpha": alpha, "method": method, "n_boot": n_boot,
        "mean_realized_fnr": float(np.mean(fnrs)), "p95_realized_fnr": float(np.percentile(fnrs, 95)),
        "mean_coverage": float(np.mean(covs)), "mean_contamination": float(np.mean(conts)),
        "frac_exceed_alpha": exceed / n_boot,
    }
