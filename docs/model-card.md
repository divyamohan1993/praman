---
language:
  - en
  - hi
license: apache-2.0
library_name: praman
tags:
  - hallucination-detection
  - faithfulness
  - grounded-claim-verification
  - conformal-prediction
  - conformal-risk-control
  - calibration
  - on-prem
  - cpu
  - indic
pipeline_tag: text-classification
---

# Model card: dmj-one/praman-verifier

## Model summary

`dmj-one/praman-verifier` is the verifier behind **PRAMAN**, an on-prem, CPU-only, Indic-first grounded-claim verifier. Given a generated output (or an agent's action and its stated justification) and the evidence it should rest on, PRAMAN decomposes the output into atomic claims, scores each claim's support against the evidence with a small CPU-friendly NLI/faithfulness model, calibrates those scores into real probabilities, and uses **conformal risk control** to choose thresholds that **provably bound the rate of auto-approving an ungrounded claim** (the catastrophic "wrong-as-right" error) at a target level α with confidence 1 - δ. Each claim is returned as accept, escalate, or reject, with an append-only audit record.

The guarantee is a **marginal** bound on a rate, not a per-item certificate, and it concerns **faithfulness to the supplied evidence, not truth in the world**. The whole inference path runs on the CPU with no external network calls (air-gapped).

- Owner: Divya Mohan (dmj.one).
- License: Apache-2.0 (code, weights, and the Indic eval slice).
- pip package: `praman`.

## Intended use and out-of-scope use

**Intended use.** After-generation verification of grounded text in high-stakes, regulated, or sovereign deployments that need an air-gapped, CPU-only verifier and a defensible audit trail: checking RAG and summarization outputs against their source passages; triaging which generated claims are safe to auto-clear versus which need a human; and producing regulator-ready evidence that AI output was checked against its evidence. PRAMAN **right-sizes** human review: it auto-clears the safe majority, clamps the catastrophic error to a documented bound, and routes the rest to a human with the evidence attached.

**Out-of-scope use.** Do not use PRAMAN as a standalone gate on **catastrophic or irreversible decisions**; there a human gate and reversibility engineering remain mandatory (see the honest-scope section below). Do not treat a per-claim probability as a per-item certificate of correctness. Do not treat a "grounded" verdict as a claim of real-world truth when the evidence itself may be wrong; source quality is a separate concern. Do not expect the documented bound to hold unchanged under distribution shift (new domain, new language, adversarial input) without the conditional or non-exchangeable variants and re-calibration. PRAMAN is a verifier, not a generator, and it cannot improve the underlying answer; it can only remove unsupported content.

## Supported languages

English plus Indic. The Indic coverage in this release is Hindi (hi); IndicXNLI covers 11 Indic languages. Indic groundedness data is scarce; that scarcity is part of the contribution, and the Indic slice is documented honestly below.

## How to use

```python
from praman import Praman
p = Praman.load("dmj-one/praman-verifier")   # or local artifacts/, fully offline
out = p.verify(
    output_text="The drug was approved in 2019 and reduces risk by 40%.",
    evidence=["... the agency approved the drug in 2021 ...", "..."],
    alpha=0.05, policy={"class": "clinical", "severity": "high"},
)
# out (illustrative of the schema, not literal output for this input; the v0 sentence-level
#      decomposer keeps an "X and Y" sentence as one claim) == {
#   "claims": [
#     {"text": "The drug was approved in 2019", "p_grounded": 0.08,
#      "evidence_span": "approved the drug in 2021", "decision": "reject"},
#     {"text": "reduces risk by 40%", "p_grounded": 0.55, "decision": "escalate"},
#   ],
#   "output_decision": "escalate",       # aggregate: any reject/escalate on high-severity -> escalate
#   "alpha": 0.05,
#   "audit": [ {...}, {...} ],           # one record per claim
# }
```

The `verify()` path runs entirely offline.

## Training and calibration data

- **RAGTruth** (Niu et al., ACL 2024): word/span-level hallucination annotations over RAG outputs (summarization, QA, data-to-text). This is the actual primary training, calibration, and evaluation source. Claim-level groundedness is derived from the real span annotations (a sentence is ungrounded iff it overlaps an annotated hallucination span); RAGTruth's official train/test split is honored, and the conformal calibration set is carved from train so it stays disjoint from test.
- **LLM-AggreFact** (`lytang/LLM-AggreFact`) was the intended secondary source but is **gated on the Hub**; per project policy we do not use access tokens without owner sign-off, so it was **not used** in this release. RAGTruth alone is sufficient for the headline guarantee, and its three task types provide the in-domain versus OOD axis.
- **Indic slice (our differentiator)**: built from **IndicXNLI (Hindi)** scored with a multilingual verifier (`mDeBERTa-v3-base-mnli-xnli`), used as a groundedness **proxy** (premise = evidence, hypothesis = claim; entailment = grounded). It is **machine-labelled, not human-verified at scale**, and should be read as indicative, not a gold human benchmark. The scarcity of true Indic RAG-groundedness data is itself part of the contribution. Construction is documented so its limits are explicit.

Split discipline keeps the calibration set exchangeable with deployment and separate from predictor fitting: train (reserve), calibration split into `calib_temp` (temperature) and `calib_conf` (conformal threshold selection), test (held-out, official RAGTruth test), and an OOD slice (a held-out task type, and the Hindi slice) for the honesty evaluation.

## Evaluation

The tables below use the same placeholder tokens as the technical report (`REPORT.md`, Section 5); both are filled from the same real runs. An unfilled token means the measurement is pending, not assumed.

### Detection quality

| Metric | PRAMAN verifier |
| --- | --- |
| AUROC (RAGTruth) | 0.612 |
| AUPRC | 0.116 |
| F1 | 0.203 |

Baseline comparison versus HHEM and MiniCheck: DeBERTa-v3-base NLI verifier on RAGTruth test: AUROC 0.612, AUPRC 0.116, base ungrounded rate 0.092. Reference small verifiers (HHEM/MiniCheck) report AUROC in the ~0.75-0.85 band on RAGTruth-style grounding tasks.

### Calibration

| Metric | Before calibration | After calibration |
| --- | --- | --- |
| ECE | 0.5245 | 0.3798 |
| Brier | 0.5058 | 0.2541 |

### Guarantee validation

Realized missed-hallucination (false-approval) rate versus target α on held-out RAGTruth test:

| alpha | CRC threshold | realized FNR (test) | mean FNR (bootstrap) | <= alpha? |
| --- | --- | --- | --- | --- |
| 0.01 | 0.478 | 0.0054 | 0.0045 | OK |
| 0.05 | 0.482 | 0.0380 | 0.0367 | OK |
| 0.10 | 0.486 | 0.0761 | 0.0862 | OK |

Coverage (auto-approval rate) and approval-set contamination at each α:

| alpha | coverage (auto-approval) | contamination of approvals |
| --- | --- | --- |
| 0.01 | 0.061 | 0.0081 |
| 0.05 | 0.138 | 0.0254 |
| 0.10 | 0.217 | 0.0323 |

Fraction of bootstrap splits whose realized risk exceeds α (CRC controls the expected rate, so a nonzero fraction is expected and reported, not hidden): alpha=0.01: 0.065, alpha=0.05: 0.220, alpha=0.10: 0.375

### OOD honesty

Degradation on the OOD slice: | alpha | in-domain test FNR | OOD FNR (pooled CRC) |
| --- | --- | --- |
| 0.01 | 0.0000 | 0.0000 |
| 0.05 | 0.0519 | 0.0087 |
| 0.10 | 0.1688 | 0.0758 |

Recovery from the conditional / non-exchangeable variants: | alpha | OOD FNR (exchangeable) | OOD FNR (non-exchangeable NN) |
| --- | --- | --- |
| 0.01 | 0.0000 | 0.0000 |
| 0.05 | 0.0087 | 0.0000 |
| 0.10 | 0.0758 | 0.0213 |

### Indic result

Hindi groundedness proxy (xnli, 2000 pairs, mDeBERTa-xnli): detection AUROC 0.999; calibration ECE 0.003->0.001; CRC realized FNR <= alpha holds at 2/3 alpha levels.

### Latency

| Configuration | Per-claim latency | Throughput |
| --- | --- | --- |
| ONNX int8 | 508.8 ms | 2.0 claims/s |
| torch fp32 | 1030.1 ms | |

## Honest scope and limits

PRAMAN controls a **rate**, not each individual item. Conformal gives a **marginal** guarantee ("≤ α of approved claims are ungrounded, averaged over an exchangeable distribution"), **not** a per-instance certificate. Therefore:

- **It does NOT remove human review on catastrophic / irreversible decisions.** For a zero-tolerance, prod-decimating action you keep a human gate regardless of verifier quality, plus reversibility engineering (dry-run, staged rollout, rollback, circuit breakers). PRAMAN is **defense-in-depth + triage + audit** there, not a human-eliminator. Where it fully removes the human is on **reversible / bounded-cost / high-volume** decisions (most of the volume and most of the cost).
- **It guarantees faithfulness-to-evidence, not truth-in-the-world.** If the evidence is wrong, a grounded-but-false claim passes. Source quality is a separate concern and must be stated.
- **The guarantee assumes exchangeability;** distribution shift (new domain, new language, adversarial input) can void it. Mitigations are built in: conditional / group-conditional (Mondrian) calibration, non-exchangeable conformal with nearest-neighbour reweighting, drift monitoring, and re-calibration. **Validate and report OOD behaviour (§8).**
- **It's only as good as the calibration labels,** and the bound has its own variance → audited calibration data + bootstrapped/randomized conformal risk control.
- **No free lunch:** tighter α ⇒ more escalation; conformal can only *remove* unsupported content, it cannot improve the underlying answer. We ship the **tradeoff curve**, not a magic number.

The value, stated truthfully: **provably right-sized review + a defensible, documented audit trail**, with the catastrophic error clamped and the residual risk *known* instead of unknown.

## Regulatory field mapping

Each audit-record field (`ts`, `claim`, `evidence_span`, `p_grounded`, `decision`, `policy` with `alpha`/`delta`/`class`/`method`, `model_version`, `calib_version`, `content_hash`) maps to evidence requirements under the EU AI Act Article 50 (transparency), the NIST AI RMF (the Measure function, with Map and Manage touchpoints), and the HIPAA Security Rule AI risk-analysis update (February 2026). The full field-by-field table is in [`docs/regulator-field-mapping.md`](regulator-field-mapping.md). That document is an evidence-mapping aid, not legal advice.

## Limitations and bias

- The guarantee is marginal and assumes exchangeability; it degrades under domain, language, and adversarial shift, and the released OOD results quantify that degradation honestly.
- Faithfulness to evidence is not truth: garbage evidence yields confidently grounded but false claims.
- Claim-decomposition errors propagate downstream and are evaluated explicitly against RAGTruth spans.
- The Indic slice is machine-built and machine-translated, not human-verified at scale, so Indic results should be read as indicative, and translation artifacts can bias both scoring and calibration on that slice.
- Calibration quality depends on the calibration labels, and the bound itself has variance, reported via bootstrapped/randomized conformal risk control.
- Tighter α increases escalation; the model surfaces the coverage-versus-risk tradeoff rather than a single magic number.
- Detection quality of this release's verifier is modest (RAGTruth test AUROC 0.612, below trained faithfulness models in the ~0.75 to 0.85 band). It is a generic NLI model with a 320-token evidence cap; the guarantee holds regardless of detector quality, but coverage is the price of a weak verifier. A trained faithfulness verifier (MiniCheck/HHEM), longer context, or max-over-passages is the documented upgrade.
- Under domain shift the verifier can fail outright: on the held-out Data2txt task type its detection AUROC falls to 0.465 (below random). The conformal bound is still met there, but only by auto-approving almost nothing, so the real out-of-distribution cost is borne in coverage, and a single test split can overshoot α (the in-domain α=0.10 split realized 0.169); read the guarantee in expectation or via RCPS, not per split.
- ONNX int8 of this DeBERTa-v3 degrades accuracy (scores diverge from fp32, Pearson ~0.72) due to disentangled-attention export; int8 is shipped as a latency benchmark only, and the runtime artifact uses torch (fp32). Do not score with int8 until calibrated quantization is validated to corr ≥ 0.97 against fp32.
- The Hindi result (AUROC 0.999) is largely in-distribution: it fell back to XNLI (hi) and the scorer mDeBERTa-v3-xnli was trained on XNLI, so it shows the pipeline runs on Hindi and the guarantee holds, not out-of-distribution Indic generalization.

## Citation

```bibtex
@software{praman_verifier,
  title  = {PRAMAN: an on-prem, CPU-only, Indic-first grounded-claim verifier with provable risk control},
  author = {Mohan, Divya},
  year   = {2026},
  note   = {Apache-2.0},
  url    = {https://huggingface.co/dmj-one/praman-verifier}
}
```
