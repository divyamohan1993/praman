"""verify() schema + the air-gap proof.

The air-gap test runs verify() inside praman._env.airgap(), which monkeypatches sockets
to refuse any outbound connection (we never touch the box's real networking, which hosts
other services). It proves the inference path needs no network.
"""
from __future__ import annotations

import socket

import pytest

from praman._env import NetworkBlockedError, airgap


def test_airgap_blocks_outbound_sockets():
    with airgap():
        with pytest.raises((NetworkBlockedError, OSError)):
            socket.create_connection(("8.8.8.8", 53), timeout=1)


def test_verify_schema(fake_praman):
    out = fake_praman.verify(
        output_text="The agency approved the drug in 2021. It reduces risk by 40%.",
        evidence=["The agency approved the drug in 2021 after review."],
        alpha=0.05,
    )
    assert set(out) >= {"claims", "output_decision", "alpha", "audit"}
    assert len(out["claims"]) == 2
    for c in out["claims"]:
        assert set(c) >= {"text", "p_grounded", "evidence_span", "decision"}
        assert 0.0 <= c["p_grounded"] <= 1.0
        assert c["decision"] in {"accept", "escalate", "reject"}
    assert len(out["audit"]) == len(out["claims"])
    assert out["output_decision"] in {"accept", "escalate", "reject"}


def test_verify_runs_airgapped(fake_praman):
    with airgap():
        out = fake_praman.verify(
            output_text="The drug was approved in 2021.",
            evidence=["The agency approved the drug in 2021."],
            alpha=0.05,
        )
    assert out["claims"][0]["decision"] in {"accept", "escalate", "reject"}


def test_high_severity_policy_escalates(fake_praman):
    out = fake_praman.verify(
        output_text="A completely unsupported invented claim about quantum llamas.",
        evidence=["The agency approved the drug in 2021."],
        policy={"class": "clinical", "severity": "high"},
    )
    assert out["output_decision"] == "escalate"
