"""Calibration: turn the verifier's raw "supported" logit into a real probability.

We calibrate the single binary logit z (sigmoid(z) == raw P(supported)) against the
groundedness label. Two methods, both reported with ECE + Brier before/after:
  * temperature: p = sigmoid(z / T), T fit by NLL (1 parameter, robust, monotone).
  * isotonic:    p = isotonic_fit(sigmoid(z)) (non-parametric, more flexible, can overfit).

Calibration must be fit on data DISJOINT from the conformal calibration + test splits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins)."""
    probs = np.asarray(probs, dtype=float); labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e, N = 0.0, len(probs)
    if N == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if m.sum() > 0:
            e += (m.sum() / N) * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float); labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2)) if len(probs) else 0.0


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, list]:
    """Data for a reliability diagram: per-bin confidence vs empirical accuracy."""
    probs = np.asarray(probs, dtype=float); labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if m.sum() > 0:
            conf.append(float(probs[m].mean())); acc.append(float(labels[m].mean()))
            cnt.append(int(m.sum()))
        else:
            conf.append(float((lo + hi) / 2)); acc.append(float("nan")); cnt.append(0)
    return {"bin_conf": conf, "bin_acc": acc, "bin_count": cnt}


def fit_temperature(z: np.ndarray, labels: np.ndarray, iters: int = 200) -> float:
    """Fit T by minimizing BCE of sigmoid(z/T). Uses torch LBFGS; pure-numpy fallback."""
    z = np.asarray(z, dtype=float); labels = np.asarray(labels, dtype=float)
    try:
        import torch
        T = torch.ones(1, requires_grad=True)
        opt = torch.optim.LBFGS([T], lr=0.05, max_iter=iters)
        bce = torch.nn.BCEWithLogitsLoss()
        lg = torch.tensor(z, dtype=torch.float32); lb = torch.tensor(labels, dtype=torch.float32)

        def closure():
            opt.zero_grad()
            loss = bce(lg / T.clamp(min=1e-2), lb)
            loss.backward()
            return loss
        opt.step(closure)
        return float(T.clamp(min=1e-2).item())
    except Exception:
        # golden-section search on NLL over T in [0.05, 20]
        def nll(T: float) -> float:
            p = _sigmoid(z / max(T, 1e-2))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
        lo, hi = 0.05, 20.0
        gr = (math_sqrt5 := 5 ** 0.5 - 1) / 2
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        for _ in range(80):
            if nll(c) < nll(d):
                hi = d
            else:
                lo = c
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        return float((lo + hi) / 2)


@dataclass
class Calibrator:
    method: str = "temperature"          # temperature | isotonic | none
    temperature: float = 1.0
    iso_x: list[float] | None = None     # isotonic support (sorted raw p)
    iso_y: list[float] | None = None     # isotonic calibrated p
    metrics: dict[str, Any] = None       # filled by fit()

    def fit(self, z: np.ndarray, labels: np.ndarray) -> "Calibrator":
        z = np.asarray(z, dtype=float); labels = np.asarray(labels, dtype=float)
        raw_p = _sigmoid(z)
        before = {"ece": ece(raw_p, labels), "brier": brier(raw_p, labels)}
        if self.method == "temperature":
            self.temperature = fit_temperature(z, labels)
        elif self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(raw_p, labels)
            xs = np.linspace(0, 1, 201)
            self.iso_x = xs.tolist(); self.iso_y = ir.predict(xs).tolist()
        cal_p = self.transform(z)
        after = {"ece": ece(cal_p, labels), "brier": brier(cal_p, labels)}
        self.metrics = {"before": before, "after": after, "n": int(len(labels)),
                        "reliability_after": reliability_bins(cal_p, labels)}
        return self

    def transform(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if self.method == "temperature":
            return _sigmoid(z / max(self.temperature, 1e-2))
        if self.method == "isotonic" and self.iso_x is not None:
            return np.interp(_sigmoid(z), self.iso_x, self.iso_y)
        return _sigmoid(z)

    # --- persistence ---
    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "temperature": self.temperature,
                "iso_x": self.iso_x, "iso_y": self.iso_y, "metrics": self.metrics}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(method=d["method"], temperature=d.get("temperature", 1.0),
                   iso_x=d.get("iso_x"), iso_y=d.get("iso_y"), metrics=d.get("metrics"))
