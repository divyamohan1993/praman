"""PRAMAN: on-prem, CPU-only, Indic-first grounded-claim verifier with provable risk
control and a regulator-ready audit trail.

    from praman import Praman
    p = Praman.load("artifacts/praman-verifier")
    out = p.verify(output_text="...", evidence=["..."], alpha=0.05)
"""
from __future__ import annotations

from ._env import airgap, assert_cpu_only, configure_threads, set_offline
from .audit import audit, field_mapping, write_jsonl
from .calibrate import Calibrator, brier, ece, fit_temperature
from .decide import aggregate_output, aurc, decide_claim, risk_coverage_curve
from .decompose import decompose
from .pipeline import Praman
from .riskcontrol import (RiskController, bootstrap_validate, crc_threshold,
                          rcps_threshold, validate_risk)
from .verifier import Verifier, VerifierConfig

__version__ = "0.1.0"

__all__ = [
    "Praman", "Verifier", "VerifierConfig", "Calibrator", "RiskController",
    "decompose", "audit", "field_mapping", "write_jsonl",
    "crc_threshold", "rcps_threshold", "validate_risk", "bootstrap_validate",
    "decide_claim", "aggregate_output", "risk_coverage_curve", "aurc",
    "ece", "brier", "fit_temperature",
    "airgap", "set_offline", "configure_threads", "assert_cpu_only",
    "__version__",
]
