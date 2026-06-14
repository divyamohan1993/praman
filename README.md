# PRAMAN

> **प्रमाण (pramāṇa)**: "proof, a valid means of knowledge."

AI keeps making things up, and that one fact is the single biggest reason it cannot be deployed where the stakes are real: roughly 62% of enterprise users name hallucination as the #1 barrier, even RAG-grounded legal tools still hallucinate on 17 to 33% of queries, and 76 to 90% of agent deployments never make it to production. The blocker is not intelligence. It is trust you can prove.

PRAMAN is the proof. Give it a generated output (or an agent's action and its stated justification) plus the evidence it should rest on. It breaks the output into atomic claims, scores each claim's support against the evidence with a small CPU model, calibrates those scores into real probabilities, and then uses **conformal risk control** to choose thresholds that **provably bound the rate of auto-approving an ungrounded claim**, the catastrophic "wrong-as-right" error, at a level you set with a stated confidence. Each claim comes back accept, escalate, or reject, with a **regulator-ready audit record**. The whole thing runs **on-prem, on the CPU, air-gapped**, and it is built **Indic-first**.

The honest one-liner: PRAMAN gives you **provably right-sized review and a defensible, documented audit trail**. It bounds a rate; it is not a per-item certificate. It guarantees faithfulness to your evidence, not truth in the world. It does not remove the human on catastrophic, irreversible decisions; there, it is defense-in-depth, triage, and audit. What it does remove is the human on the reversible, bounded-cost, high-volume majority, which is most of the volume and most of the cost. The catastrophic error is clamped, and the residual risk is known instead of unknown.

`CPU-only, on-prem, Apache-2.0`

## Install

```bash
pip install praman
```

## Quickstart

```python
from praman import Praman
p = Praman.load("dmj-one/praman-verifier")   # or local artifacts/, fully offline
out = p.verify(
    output_text="The drug was approved in 2019 and reduces risk by 40%.",
    evidence=["... the agency approved the drug in 2021 ...", "..."],
    alpha=0.05, policy={"class": "clinical", "severity": "high"},
)
# out == {
#   "claims": [
#     {"text": "The drug was approved in 2019", "p_grounded": 0.08,
#      "evidence_span": "approved the drug in 2021", "decision": "reject"},
#     {"text": "reduces risk by 40%", "p_grounded": 0.55, "decision": "escalate"},
#   ],
#   "output_decision": "escalate",       # aggregate: any reject/escalate on high-severity → escalate
#   "alpha": 0.05,
#   "audit": [ {...}, {...} ],           # one record per claim
# }
```

`p.verify(...)` runs entirely offline: no external network calls in the inference path.

## Learn more

- [Technical report](REPORT.md): market framing and the wedge, the method, the guarantee-validation and OOD-honesty results, the Indic result, and the honest limits, with citations.
- [Model card](docs/model-card.md): intended use, supported languages, evaluation tables, the verbatim honest-scope section, and the regulator field mapping.
- [Regulator field mapping](docs/regulator-field-mapping.md): each audit-record field mapped to EU AI Act Article 50, NIST AI RMF, and the HIPAA Security Rule AI risk-analysis update.

## License and ownership

Apache-2.0. Owner: Divya Mohan (dmj.one). Hugging Face model id (target): `dmj-one/praman-verifier`.
