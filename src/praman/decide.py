"""Decision + abstention: accept / escalate / reject, per claim and per output.

The CRC threshold ``t_approve`` defines the auto-approve region (u < t_approve) at the
chosen alpha. Outside it we either reject (confidently ungrounded, recoverable) or
escalate to a human (uncertain, or any non-approval on a high-severity class).

PRAMAN right-sizes review; on high-severity / irreversible classes it NEVER auto-acts,
it escalates. The deployer sets alpha and which classes are high-severity (configs/policy.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

ACCEPT, ESCALATE, REJECT = "accept", "escalate", "reject"


@dataclass
class ClaimDecision:
    decision: str
    p_grounded: float
    u: float
    t_approve: float
    t_reject: float


def decide_claim(p_grounded: float, t_approve: float, t_reject: float | None = None,
                 severity: str = "normal") -> ClaimDecision:
    """One claim -> accept | escalate | reject.

    * accept  : u < t_approve (CRC guarantees the missed-hallucination rate here <= alpha)
    * reject  : u >= t_reject (confidently ungrounded) AND severity != high
    * escalate: everything else, and ALL non-approvals on high-severity classes
    """
    u = 1.0 - float(p_grounded)
    if t_reject is None:
        t_reject = max(t_approve, 1.0 - t_approve)  # symmetric default band
    if u < t_approve:
        return ClaimDecision(ACCEPT, p_grounded, u, t_approve, t_reject)
    if severity == "high":
        return ClaimDecision(ESCALATE, p_grounded, u, t_approve, t_reject)
    if u >= t_reject:
        return ClaimDecision(REJECT, p_grounded, u, t_approve, t_reject)
    return ClaimDecision(ESCALATE, p_grounded, u, t_approve, t_reject)


def aggregate_output(decisions: Sequence[str], severity: str = "normal") -> str:
    """Aggregate per-claim decisions into one output decision.

    High-severity: any non-accept => escalate (force a human gate on the whole output).
    Normal: escalate dominates reject dominates accept.
    """
    ds = list(decisions)
    if not ds:
        return ACCEPT
    if severity == "high" and any(d != ACCEPT for d in ds):
        return ESCALATE
    if ESCALATE in ds:
        return ESCALATE
    if REJECT in ds:
        return REJECT
    return ACCEPT


# ------------------------------------------------------------------ #
# Risk-coverage / selective-prediction metrics (brief 8.4)
# ------------------------------------------------------------------ #
def risk_coverage_curve(u: Sequence[float], ungrounded: Sequence[int],
                        n_points: int = 101) -> dict[str, list[float]]:
    """Sweep the approve threshold; report coverage (auto-approval rate) vs selective
    risk (missed-hallucination rate among approved-and-ungrounded / approved)."""
    import numpy as np
    u = np.asarray(u, dtype=float); y = np.asarray(ungrounded, dtype=int)
    ts = np.linspace(0, 1, n_points)
    cov, risk = [], []
    for t in ts:
        approved = u < t
        c = float(np.mean(approved))
        r = float(np.mean(y[approved])) if approved.any() else 0.0  # contamination of approvals
        cov.append(c); risk.append(r)
    return {"threshold": ts.tolist(), "coverage": cov, "selective_risk": risk}


def aurc(curve: dict[str, list[float]]) -> float:
    """Area under the risk-coverage curve (lower is better). Trapezoid over coverage."""
    import numpy as np
    cov = np.asarray(curve["coverage"]); risk = np.asarray(curve["selective_risk"])
    order = np.argsort(cov)
    return float(np.trapezoid(risk[order], cov[order])) if len(cov) > 1 else 0.0
