"""Evaluation: detection quality, guarantee validation, OOD honesty, ablations.

The "positive" class for DETECTION is ungrounded (the thing we must catch). The detector
score is u = 1 - p_grounded. Higher u => more confident the claim is ungrounded.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .riskcontrol import bootstrap_validate, crc_threshold, rcps_threshold, validate_risk


def detection_metrics(u: Sequence[float], ungrounded: Sequence[int]) -> dict[str, float]:
    """AUROC / AUPRC / F1 / balanced-accuracy for hallucination detection."""
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                                 f1_score, roc_auc_score)
    u = np.asarray(u, dtype=float); y = np.asarray(ungrounded, dtype=int)
    out: dict[str, float] = {"n": int(len(y)), "n_ungrounded": int(y.sum()),
                             "base_rate": float(y.mean()) if len(y) else 0.0}
    if 0 < y.sum() < len(y):
        out["auroc"] = float(roc_auc_score(y, u))
        out["auprc"] = float(average_precision_score(y, u))
        pred = (u >= 0.5).astype(int)
        out["f1@0.5"] = float(f1_score(y, pred, zero_division=0))
        out["balanced_acc@0.5"] = float(balanced_accuracy_score(y, pred))
    return out


def guarantee_table(u_calib_pos: np.ndarray, u_test: np.ndarray, y_test: np.ndarray,
                    alphas: Sequence[float], method: str = "crc", delta: float = 0.05,
                    bound: str = "hoeffding") -> list[dict[str, Any]]:
    """Per-alpha: chosen threshold, realized FNR on test, coverage, contamination."""
    rows = []
    for a in alphas:
        r = validate_risk(u_calib_pos, u_test, y_test, a, method, delta, bound)
        rows.append({"alpha": a, **{k: round(v, 4) for k, v in r.items()}})
    return rows


def bootstrap_table(u: np.ndarray, y: np.ndarray, alphas: Sequence[float],
                    n_boot: int = 200, method: str = "crc", delta: float = 0.05,
                    bound: str = "hoeffding") -> list[dict[str, Any]]:
    """Per-alpha bootstrap: mean realized FNR + fraction of splits exceeding alpha."""
    return [bootstrap_validate(u, y, a, n_boot=n_boot, method=method, delta=delta, bound=bound)
            for a in alphas]


def ablation_thresholds(u_calib: np.ndarray, y_calib: np.ndarray, alpha: float
                        ) -> dict[str, float]:
    """Baselines vs CRC at one alpha: naive 0.5, the alpha-quantile, and CRC."""
    u_pos = u_calib[y_calib == 1]
    return {
        "naive_0.5": 0.5,
        "quantile": float(np.quantile(u_pos, alpha)) if len(u_pos) else 0.0,
        "crc": crc_threshold(u_pos, alpha),
        "rcps_hoeffding": rcps_threshold(u_pos, alpha),
    }


def apply_threshold(u_test: np.ndarray, y_test: np.ndarray, t: float) -> dict[str, float]:
    approved = np.asarray(u_test) < t
    y = np.asarray(y_test, dtype=int)
    pos = y == 1
    return {
        "threshold": float(t),
        "realized_fnr": float(np.mean(approved[pos])) if pos.any() else 0.0,
        "coverage": float(np.mean(approved)),
        "contamination": float(np.mean(y[approved])) if approved.any() else 0.0,
    }
