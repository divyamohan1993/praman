"""Audit record: the regulator-ready, tamper-evident trail (brief 7.6).

One record per claim, exported as JSONL. Each field maps to an EU AI Act / NIST AI RMF /
HIPAA evidence requirement (see docs/regulator-field-mapping.md and ``field_mapping()``).
The content_hash makes records tamper-evident: any edit to claim/evidence breaks the hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import orjson


def audit(claim: str, evidence_span: str, p_grounded: float, decision: str,
          policy: dict[str, Any], versions: dict[str, str], ts: float | None = None) -> dict[str, Any]:
    """Build one audit record. ``ts`` is injected (no wall-clock side effects inside)."""
    return {
        "ts": ts,
        "claim": claim,
        "evidence_span": evidence_span,
        "p_grounded": round(float(p_grounded), 4),
        "decision": decision,                       # accept | reject | escalate
        "policy": policy,                           # {"alpha":..,"delta":..,"class":..,"method":"crc"}
        "model_version": versions.get("model"),
        "calib_version": versions.get("calib"),
        "risk_version": versions.get("risk"),
        "content_hash": hashlib.sha256((claim + "||" + evidence_span).encode("utf-8")).hexdigest(),
    }


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Append audit records as JSONL. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "ab") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")
            n += 1
    return n


def field_mapping() -> dict[str, dict[str, str]]:
    """Audit field -> the regulator requirement it satisfies. Evidence aid, not legal advice."""
    return {
        "ts": {
            "eu_ai_act": "Art 12 record-keeping / Art 50 timing of transparency disclosure",
            "nist_rmf": "MEASURE 2.x: time-stamped measurement of system behaviour",
            "hipaa": "Security Rule audit controls 164.312(b): when ePHI-touching output was assessed",
        },
        "claim": {
            "eu_ai_act": "Art 50: the AI-generated content subject to transparency",
            "nist_rmf": "MAP 1.x: the system output being characterized",
            "hipaa": "Risk analysis 164.308(a)(1)(ii)(A): the artifact whose risk is analyzed",
        },
        "evidence_span": {
            "eu_ai_act": "Art 50: basis for the inaccuracy/accuracy disclosure",
            "nist_rmf": "MEASURE 2.8: traceability of the grounding evidence",
            "hipaa": "Risk analysis: the source data the determination relied on",
        },
        "p_grounded": {
            "eu_ai_act": "Art 50: the detected likelihood the content is grounded/accurate",
            "nist_rmf": "MEASURE 2.x: quantified, calibrated risk metric",
            "hipaa": "Risk analysis: likelihood component of the risk determination",
        },
        "decision": {
            "eu_ai_act": "Art 14 human oversight: accept/escalate/reject routing",
            "nist_rmf": "MANAGE 1.x: the risk-response action taken",
            "hipaa": "Risk management 164.308(a)(1)(ii)(B): the mitigation applied",
        },
        "policy": {
            "eu_ai_act": "Art 9 risk-management system: the documented risk policy (alpha/delta/class)",
            "nist_rmf": "GOVERN 1.x: the codified risk tolerance and method",
            "hipaa": "164.308(a)(1): the documented risk-management policy in force",
        },
        "model_version": {
            "eu_ai_act": "Art 11 technical documentation: model provenance",
            "nist_rmf": "MAP 4.x: model identity for reproducibility",
            "hipaa": "164.312(b): which tool version produced the assessment",
        },
        "calib_version": {
            "eu_ai_act": "Art 11: calibration provenance behind the probability",
            "nist_rmf": "MEASURE 2.x: provenance of the calibration that makes the metric valid",
            "hipaa": "Risk analysis reproducibility",
        },
        "risk_version": {
            "eu_ai_act": "Art 9: provenance of the risk-control thresholds in force",
            "nist_rmf": "GOVERN/MANAGE: which control configuration was applied",
            "hipaa": "164.308(a)(1)(ii)(B): provenance of the safeguard configuration",
        },
        "content_hash": {
            "eu_ai_act": "Art 12 record-keeping: tamper-evidence over the logged record",
            "nist_rmf": "MEASURE 2.x: integrity of the measurement record",
            "hipaa": "164.312(c)(1) integrity controls + 164.312(b) audit integrity",
        },
    }
