from __future__ import annotations

from praman.audit import audit, field_mapping
from praman.decide import ACCEPT, ESCALATE, REJECT, aggregate_output, decide_claim


def test_decide_accept_when_confident_grounded():
    d = decide_claim(p_grounded=0.95, t_approve=0.4)  # u=0.05 < 0.4
    assert d.decision == ACCEPT


def test_decide_reject_when_confident_ungrounded_normal():
    d = decide_claim(p_grounded=0.02, t_approve=0.4, t_reject=0.6, severity="normal")
    assert d.decision == REJECT  # u=0.98 >= 0.6


def test_decide_high_severity_never_auto_rejects():
    d = decide_claim(p_grounded=0.02, t_approve=0.4, t_reject=0.6, severity="high")
    assert d.decision == ESCALATE  # high severity: any non-approval escalates


def test_decide_escalate_in_uncertain_band():
    d = decide_claim(p_grounded=0.5, t_approve=0.2, t_reject=0.9, severity="normal")
    assert d.decision == ESCALATE


def test_aggregate_high_severity_escalates_on_any_flag():
    assert aggregate_output([ACCEPT, REJECT], severity="high") == ESCALATE
    assert aggregate_output([ACCEPT, ACCEPT], severity="high") == ACCEPT


def test_aggregate_normal_precedence():
    assert aggregate_output([ACCEPT, ESCALATE, REJECT]) == ESCALATE
    assert aggregate_output([ACCEPT, REJECT]) == REJECT
    assert aggregate_output([ACCEPT, ACCEPT]) == ACCEPT
    assert aggregate_output([]) == ACCEPT


def test_audit_record_schema_and_hash_determinism():
    r1 = audit("claim x", "evidence y", 0.3, "reject",
               {"alpha": 0.05, "class": "general", "method": "crc"},
               {"model": "m", "calib": "c", "risk": "r"}, ts=1.0)
    r2 = audit("claim x", "evidence y", 0.3, "reject",
               {"alpha": 0.05, "class": "general", "method": "crc"},
               {"model": "m", "calib": "c", "risk": "r"}, ts=2.0)
    required = {"ts", "claim", "evidence_span", "p_grounded", "decision", "policy",
                "model_version", "calib_version", "content_hash"}
    assert required <= set(r1)
    assert r1["content_hash"] == r2["content_hash"]            # hash independent of ts
    r3 = audit("claim X", "evidence y", 0.3, "reject", {}, {}, ts=1.0)
    assert r3["content_hash"] != r1["content_hash"]            # tamper changes hash


def test_field_mapping_covers_all_audit_fields():
    fm = field_mapping()
    for f in ("ts", "claim", "evidence_span", "p_grounded", "decision", "policy",
              "model_version", "calib_version", "content_hash"):
        assert f in fm
        assert {"eu_ai_act", "nist_rmf", "hipaa"} <= set(fm[f])
