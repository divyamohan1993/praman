# Changelog

All notable changes to PRAMAN are documented here. Format follows Keep a Changelog;
this project uses calendar-dated entries during the initial build.

## [Unreleased]

### 2026-06-14

#### Added
- Core package `praman`: claim decomposition, NLI verifier (DeBERTa-v3, ONNX-int8 runtime
  path), temperature/isotonic calibration with ECE/Brier, conformal risk control
  (CRC + RCPS + LTT), accept/escalate/reject decision with abstention, append-only
  content-hashed audit records, and the frozen `Praman.verify()` public API.
- `data.py`: RAGTruth loader that derives claim-level groundedness from real span
  annotations (a sentence is ungrounded iff it overlaps an annotated hallucination span),
  honoring RAGTruth's official train/test split and carving the conformal calibration set
  from train (disjoint from test).
- Conformal risk control validated against an independent exact-bound re-derivation and
  cross-checked against MAPIE on a synthetic toy: CRC controls the expected realized FNR,
  RCPS controls it with high probability, thresholds monotone in alpha, no-feasible default
  is approve-nothing.
- Robustness: group-conditional (Mondrian) and non-exchangeable (nearest-neighbour
  reweighted) conformal variants; leave-one-domain-out OOD honesty evaluation.
- Indic slice via IndicXNLI (Hindi) scored with mDeBERTa-v3-xnli as a groundedness proxy.
- Packaging: ONNX int8 export + latency benchmark, FastAPI service (loopback only), CLI,
  reproduction scripts (`scripts/00_setup.sh` ... `90_report.py`), `Makefile`,
  `requirements.lock.txt`.
- Tests: data shapes, calibration improves ECE, CRC monotonicity + expected-risk control,
  verify() schema, in-process air-gap proof, audit-record schema + field mapping.
- Docs: `README.md`, `REPORT.md`, Hugging Face model card, and the EU AI Act / NIST AI RMF /
  HIPAA regulator field-mapping document. The honest-scope section is reproduced verbatim
  from the brief in both the report and the model card.

#### Changed
- Data source: pivoted from the gated `lytang/LLM-AggreFact` to the non-gated RAGTruth
  (the brief's stated primary), avoiding the need for an access token.
- Runtime/full-run inference uses ONNX int8 (AVX-512 VNNI) for speed on the CPU box; torch
  remains the training/eval path and the fallback when int8 op-gaps occur.

#### Fixed
- `pipeline.py` imported the `audit` submodule under a name shadowed by the re-exported
  `audit` function in `__init__`, causing an AttributeError in `verify()`; now imports the
  function directly. Caught by the verify() tests.
