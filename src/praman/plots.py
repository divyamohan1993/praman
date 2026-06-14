"""Plots: reliability diagram, risk-coverage curve, desired-vs-realized risk.

Matplotlib only (no seaborn). Headless-safe (Agg backend). Each function takes already
-computed arrays/metrics and writes a PNG; no model or data loading here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def reliability_diagram(metrics_after: dict[str, list], out_path: str | Path,
                        title: str = "Reliability (after calibration)") -> Path:
    conf = metrics_after["bin_conf"]; acc = metrics_after["bin_acc"]; cnt = metrics_after["bin_count"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    xs = [c for c, a in zip(conf, acc) if a == a]  # drop NaN bins
    ys = [a for a in acc if a == a]
    ax.plot(xs, ys, "o-", color="#1f77b4", label="model")
    ax.set_xlabel("confidence  P(grounded)"); ax.set_ylabel("empirical accuracy")
    ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left")
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def risk_coverage(curve: dict[str, list[float]], out_path: str | Path,
                  title: str = "Risk-coverage") -> Path:
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot(curve["coverage"], curve["selective_risk"], "-", color="#d62728")
    ax.set_xlabel("coverage (auto-approval rate)")
    ax.set_ylabel("selective risk (ungrounded among approved)")
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)


def desired_vs_realized(rows: Sequence[dict[str, Any]], out_path: str | Path,
                        title: str = "Desired vs realized missed-hallucination rate") -> Path:
    """rows: [{alpha, realized_fnr, coverage}, ...] from the guarantee table."""
    alphas = [r["alpha"] for r in rows]
    realized = [r["realized_fnr"] for r in rows]
    cover = [r.get("coverage") for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    amax = max(alphas) * 1.1
    ax.plot([0, amax], [0, amax], "--", color="gray", lw=1, label="realized = target")
    ax.scatter(alphas, realized, color="#2ca02c", zorder=3, label="realized FNR")
    for a, r in zip(alphas, realized):
        ax.annotate(f"a={a}", (a, r), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.fill_between([0, amax], [0, amax], amax, color="red", alpha=0.05, label="violates bound")
    ax.set_xlabel("target alpha"); ax.set_ylabel("realized missed-hallucination rate")
    ax.set_title(title); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(alphas, cover, "o:", color="#1f77b4", alpha=0.6)
    ax2.set_ylabel("coverage (auto-approval rate)", color="#1f77b4")
    fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return Path(out_path)
