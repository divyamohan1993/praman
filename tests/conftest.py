"""Shared fixtures. Tests avoid the 440MB model by using a deterministic fake verifier,
so pytest is fast AND offline (which makes the air-gap test honest)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from praman.calibrate import Calibrator
from praman.pipeline import Praman
from praman.riskcontrol import RiskController
from praman.verifier import VerifierConfig


class FakeVerifier:
    """Pure-python stand-in: 'supported' if the claim's key token appears in evidence."""
    def __init__(self) -> None:
        self.cfg = VerifierConfig(hf_id="fake-verifier", kind="nli")
        self._id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        self._support_idx = 0
        self.backend = "torch"

    def score_multi(self, claim: str, passages):
        passages = list(passages) or [""]
        joined = " ".join(passages).lower()
        # crude grounding signal: token overlap fraction of the claim
        toks = [t for t in claim.lower().split() if len(t) > 3]
        if not toks:
            p = 0.5
        else:
            p = sum(t in joined for t in toks) / len(toks)
            p = min(0.98, max(0.02, p))
        z = math.log(p / (1 - p))
        return p, z, 0

    def score_pairs(self, claims, evidences):
        ps, zs = [], []
        for c, e in zip(claims, evidences):
            p, z, _ = self.score_multi(c, [e])
            ps.append(p); zs.append(z)
        return np.array(ps), np.array(zs)


@pytest.fixture
def fake_praman() -> Praman:
    cal = Calibrator(method="none")
    risk = RiskController(method="crc", thresholds={"0.01": 0.1, "0.05": 0.4, "0.1": 0.6})
    return Praman(FakeVerifier(), cal, risk,
                  policy={"default": {"alpha": 0.05, "class": "general", "severity": "normal"},
                          "classes": {"clinical": {"alpha": 0.01, "severity": "high"}}},
                  versions={"model": "fake", "calib": "none", "risk": "crc"})
