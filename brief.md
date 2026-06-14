# PRAMAN — on-prem grounded-claim verifier with provable risk control + audit

> **प्रमाण (pramāṇa)**: "proof / valid means of knowledge." PRAMAN takes a generated output (or an agent's action + its stated justification) plus the evidence it should be based on, and returns, per claim: a calibrated grounded/ungrounded verdict, a **distribution-free guarantee** on the error that matters most (auto-approving an ungrounded claim), an accept / escalate / reject decision, and a **regulator-ready audit record** — all running **fully on-prem / CPU / air-gapped**, and built **Indic-first**.

- **Codename:** PRAMAN · **pip package:** `praman` · **HF model id (target):** `dmj-one/praman-verifier`
- **Owner:** Divya Mohan (dmj.one) · **License:** Apache-2.0 (code + weights + the Indic eval slice we build)
- **This file is the source of truth.** Read it fully before doing anything. `AGENTS.md` says *how* to work; this says *what* and *why*.

---

## 0. TL;DR for the coding agent

Build a **verifier**, not a generator. Given `(output_text, evidence)`, decompose the output into atomic claims, score each claim's support against the evidence with a **small discriminative model** (NLI / faithfulness model — CPU-friendly), **calibrate** the scores, then use **conformal risk control** to pick thresholds that **provably bound the rate of auto-approving an ungrounded claim** (the catastrophic "says-wrong-as-right" error). Anything the verifier isn't confident about is **escalated to a human**; everything is **logged as an audit record**. The whole thing runs on the CPU VM with **no external network calls in the inference path** (on-prem requirement).

The product value (read §2 and §4 carefully — this is the honest scope):
- It does **not** eliminate human review on catastrophic/irreversible decisions. It **right-sizes** review: auto-clear the safe majority, clamp the expensive error to a documented bound, triage the rest with evidence so each review takes seconds, and produce a defensible audit trail. That is what a regulated/sovereign buyer needs to go from "can't deploy" to "deployed and defensible."

**Definition of done** is §12. **Chase the targets in §8.4.** Prefer the simple, correct, reproducible thing; log decisions in `PROGRESS.md`.

---

## 1. What the market wants (the framing — backed by the demand data)

The market does **not** want to fire reviewers (that's unbuildable — see §4). It wants to **deploy AI in high-stakes / regulated / sovereign workflows with risk that is bounded, documented, and recoverable**, and to prove that to a regulator. Evidence we gathered:

- **Hallucination is the #1 named barrier to deployment** (~62% of enterprise users), the single biggest trust blocker; even RAG-grounded legal tools still hallucinate on 17–33% of queries — so *after-generation verification* is needed.
- Enterprises are **stuck**: ~76% throw human-in-the-loop review at hallucinations (doesn't scale); only ~20% have mature AI governance; 76–90% of agent deployments fail in production. The bottleneck is **operability/trust, not intelligence.**
- **Money follows the control layer:** capital flows to agent-execution infrastructure (runtimes, identity, observability, control, recovery) and to vertical agents in regulated domains.
- **Regulation now mandates it:** EU AI Act transparency (detect & disclose inaccurate outputs) by Aug 2026; HIPAA AI risk-analysis rule (Feb 2026); NIST AI RMF Measure function.
- **Bharat:** sovereign Indic models (Sarvam, BharatGen, Gnani) are being pushed into governance, health, agriculture, citizen services. The named bottleneck is **data/eval/trust**, and these deployments need **on-prem / data-sovereign** tooling in **Indian languages**.

**The wedge (why this exact shape, and why it isn't repetition):** the funded incumbents (Galileo, Maxim, Atlan, Vectara, Webcite) ship hallucination detection as **cloud SaaS, English-first, heuristic scores**. PRAMAN combines three things none of them do together:
1. **A provable, documented bound** (conformal risk control) instead of a vibe score — the rigor is the moat.
2. **On-prem / edge / air-gapped** operation — the thing regulated, defence, health, finance, and sovereign deployments legally require and cloud APIs can't offer. The CPU constraint is the feature.
3. **Indic-first** — where no guaranteed verifier exists and public-administration outputs must be defensible.

---

## 2. Contributions (what's publishable / useful)

- A CPU/on-prem grounded-claim verifier whose **false-approval rate is provably bounded** (RCPS/CRC/LTT), with empirical validation **including under distribution shift** (most papers only show the i.i.d. case).
- A **regulator-ready audit schema** mapping each verdict to EU AI Act / NIST RMF / HIPAA evidence fields.
- An **Indic groundedness eval slice** (a real, scarce resource) + verifier results on it.
- Honest characterization of **where the guarantee holds and where it breaks** (the coverage-vs-usefulness tradeoff, OOD degradation) — the thing the field underreports.

---

## 3. Method overview (pipeline)

```
output_text + evidence
   │
   ▼
[1 claim decomposition]  → atomic, independently-checkable claims
   │
   ▼
[2 evidence alignment]   → (claim, evidence-span) pairs  (retrieve if evidence not pre-aligned)
   │
   ▼
[3 verifier scoring]     → support score s_i ∈ [0,1] per claim   (small NLI/faithfulness model, CPU)
   │
   ▼
[4 calibration]          → calibrated P(grounded) per claim      (per-bucket temperature/isotonic)
   │
   ▼
[5 conformal risk control] → threshold(s) s.t. missed-hallucination rate ≤ α w.p. ≥ 1-δ
   │
   ▼
[6 decision + abstention] → accept / escalate(human) / reject, per claim and per output
   │
   ▼
[7 audit record]         → claim, evidence span, score, policy(α,δ), decision, versions, hash
```

---

## 4. Honest scope & limits (READ — this defines the product and keeps it from being a fable)

PRAMAN controls a **rate**, not each individual item. Conformal gives a **marginal** guarantee ("≤ α of approved claims are ungrounded, averaged over an exchangeable distribution"), **not** a per-instance certificate. Therefore:

- **It does NOT remove human review on catastrophic / irreversible decisions.** For a zero-tolerance, prod-decimating action you keep a human gate regardless of verifier quality, plus reversibility engineering (dry-run, staged rollout, rollback, circuit breakers). PRAMAN is **defense-in-depth + triage + audit** there, not a human-eliminator. Where it fully removes the human is on **reversible / bounded-cost / high-volume** decisions (most of the volume and most of the cost).
- **It guarantees faithfulness-to-evidence, not truth-in-the-world.** If the evidence is wrong, a grounded-but-false claim passes. Source quality is a separate concern and must be stated.
- **The guarantee assumes exchangeability;** distribution shift (new domain, new language, adversarial input) can void it. Mitigations are built in: conditional / group-conditional (Mondrian) calibration, non-exchangeable conformal with nearest-neighbour reweighting, drift monitoring, and re-calibration. **Validate and report OOD behaviour (§8).**
- **It's only as good as the calibration labels,** and the bound has its own variance → audited calibration data + bootstrapped/randomized conformal risk control.
- **No free lunch:** tighter α ⇒ more escalation; conformal can only *remove* unsupported content, it cannot improve the underlying answer. We ship the **tradeoff curve**, not a magic number.

The value, stated truthfully: **provably right-sized review + a defensible, documented audit trail**, with the catastrophic error clamped and the residual risk *known* instead of unknown.

---

## 5. Hardware & environment (Oracle OCI, on-prem posture)

**Target:** `VM.Standard.E5.Flex`, **6 OCPU (= 12 vCPU)** on **AMD EPYC "Genoa" (Zen 4)**. **CPU-only — no GPU.** Do not install CUDA wheels or call `torch.cuda`. **On-prem rule: the inference path makes NO external network calls.** (Network is allowed only for downloading models/datasets during build; the runtime `verify()` path must work air-gapped.)

- **Memory:** ≥ 48–64 GB recommended.
- **OS:** Ubuntu 24.04. **Python:** 3.12 (3.13 if deps resolve).
- **SIMD:** Genoa has AVX-512 + VNNI (int8) + BF16 (no AMX). Use **ONNX Runtime int8** for the verifier inference path (VNNI); benchmark BF16/IPEX but don't assume gains.

### 5.1 Setup (script as `scripts/00_setup.sh` and run it)
```bash
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv build-essential git
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU torch only
pip install -U \
  "transformers" "sentence-transformers" "datasets" "accelerate" \
  "scikit-learn" "numpy" "pandas" "scipy" \
  "onnx" "onnxruntime" "optimum[onnxruntime]" \
  "mapie" "netcal" \
  "matplotlib" "tqdm" "pyyaml" "fastapi" "uvicorn" "pydantic" "orjson"
pip freeze > requirements.lock.txt   # commit this
```

### 5.2 Thread tuning (at process start)
```bash
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 TOKENIZERS_PARALLELISM=false
```
```python
import torch; torch.set_num_threads(12); torch.set_num_interop_threads(2)
```

---

## 6. Data & benchmarks (open, CPU-evaluable, used to PROVE the guarantee)

Verify exact HF paths/configs at runtime with `ds.features`; adapt, don't hard-code.

- **RAGTruth** (Niu et al., ACL 2024) — word/span-level hallucination annotations over RAG outputs (summarization, QA, data-to-text). **Primary** training/calibration/eval source. (GitHub `ParticleMedia/RAGTruth`; mirrored on HF.)
- **LLM-AggreFact** (the MiniCheck benchmark; `lytang/LLM-AggreFact` on HF) — aggregated fact-checking-against-grounding-documents datasets. Standard for evaluating grounded verifiers.
- **Faithfulness/NLI sets:** AggreFact, SummaC, FaithBench (summarization consistency); ANLI, MNLI, FEVER (entailment) for pretraining/baselines.
- **AA-Omniscience** — abstention-aware benchmark (rewards "knowing limits"); use for the selective-prediction framing.
- **Indic slice (our differentiator):** build a small, human-checked Hindi/Indic groundedness set — e.g., translate a RAGTruth subset and verify a sample by hand, and/or use Indic NLI/FEVER-like resources. **High-quality Indic groundedness data is scarce — that scarcity is the contribution.** Keep it small but clean; document construction.

Split discipline: **train** (verifier head/fine-tune) / **calibration** (temperature + conformal — keep exchangeable with deployment) / **test** (held-out) / **OOD slice** (different domain or language, for the honesty eval). Conformal coverage assumes the calibration set wasn't used to fit the predictor; if strict, split calibration into `calib_temp` and `calib_conf`.

---

## 7. Method detail + reference code (the substance)

### 7.1 Claim decomposition
Break the output into atomic claims (FActScore-style). Start deterministic and cheap; upgrade only if it limits quality.
```python
import re
def decompose(text: str) -> list[str]:
    # v0: sentence/clause split. Upgrade later to a small instruct SLM or dependency-based splitter.
    parts = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]
```
Evaluate decomposition quality on RAGTruth (its span annotations let you check claim coverage). Optional decontextualization (resolve pronouns) improves verifiability.

### 7.2 Verifier scoring (CPU, discriminative — benchmark a few, pick by accuracy/latency)
Candidates, all CPU-runnable and small:
- **MiniCheck** (Tang et al., EMNLP 2024) — small fact-checkers reported to match GPT-4 on grounding-doc fact-checking at a fraction of the cost. Strong default.
- **Vectara HHEM (open)** — `vectara/hallucination_evaluation_model` faithfulness scorer.
- **DeBERTa-v3 NLI cross-encoder** (entailment of claim by evidence). Robust, easy to fine-tune.
- **AlignScore** — unified factual-consistency function.
- **Indic:** a multilingual NLI / MuRIL / IndicBERT cross-encoder for the Indic slice.
Interface (return P(supported) for a (claim, evidence) pair):
```python
class Verifier:
    def score(self, claim: str, evidence: str) -> float:  # P(claim supported by evidence) in [0,1]
        ...
# For multi-passage evidence, score against each passage and take max (claim is grounded if ANY passage supports it).
```
Export the chosen model to **ONNX int8** for the runtime path; keep a torch path for training. No external API in the prod path.

### 7.3 Calibration (per score-bucket, post-hoc)
Make the score a real probability; report ECE/Brier before/after.
```python
import numpy as np, torch
def fit_temperature(logits, labels, iters=200):     # logits, labels: 1-D arrays for the "supported" class
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=iters)
    bce = torch.nn.BCEWithLogitsLoss()
    lg = torch.tensor(logits, dtype=torch.float32); lb = torch.tensor(labels, dtype=torch.float32)
    def closure():
        opt.zero_grad(); loss = bce(lg / T.clamp(min=1e-2), lb); loss.backward(); return loss
    opt.step(closure); return float(T.clamp(min=1e-2))

def ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(probs)
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i+1])
        if m.sum() > 0: e += (m.sum()/N) * abs(probs[m].mean() - labels[m].mean())
    return e
```
Also try per-bucket isotonic (`sklearn.isotonic`). Cross-check ECE with `netcal`.

### 7.4 Conformal risk control — bound the catastrophic error
**Positive class = "ungrounded / hallucinated."** Detector score `u = 1 - P(grounded)`. A **missed hallucination** = auto-approving an ungrounded claim = the catastrophic "wrong-as-right." Control the **false-negative rate of the hallucination detector** (= the false-approval rate) at level α with confidence 1−δ, via CRC/RCPS recall control. Reference (validate against MAPIE):
```python
import numpy as np
def crc_threshold(u_pos, alpha, grid=np.linspace(0,1,201)):
    # u_pos = calibrated detector scores u = 1 - P(grounded) on calib claims that ARE ungrounded (positives)
    n = len(u_pos); chosen = 1.0  # threshold t: predict "ungrounded/flag" if u >= t  → lower t = flag more = fewer misses
    for t in np.sort(grid)[::-1]:        # decrease t (flag more, catch more) until guarantee holds
        miss = np.mean(u_pos < t)        # ungrounded claims we FAIL to flag (auto-approve) at threshold t
        if (n/(n+1))*miss + 1/(n+1) <= alpha:   # CRC finite-sample correction
            chosen = t
        else:
            break
    return chosen   # auto-approve a claim iff u < chosen (i.e., P(grounded) high enough)
```
- **Batteries-included:** `mapie.multi_label_classification.MapieMultiLabelClassifier` implements **RCPS** (Hoeffding/Bernstein/WSR bounds), **CRC** (recall), **LTT** (precision). **Pin the MAPIE version and verify the class path** (MAPIE refactored around v1.0); if the API differs, use the from-scratch CRC and cite MAPIE for cross-validation. Use **LTT** if you also want to bound the *over-flagging* (precision) — the cheap error — as a secondary control.
- **Robustness:** add a **conditional** variant (per domain / source-type / language → Mondrian conformal) and a **non-exchangeable** variant (nearest-neighbour reweighting) for the OOD slice. Use **bootstrapped/randomized CRC** to report the bound's variance.

### 7.5 Decision + abstention
```python
def decide(p_grounded, t_approve, tau_escalate, severity="normal"):
    u = 1.0 - p_grounded
    if u < t_approve:                       # confidently grounded → accept
        return "accept"
    if u >= (1.0 - t_approve) and severity != "high":  # confidently ungrounded, recoverable → reject/flag
        return "reject"
    return "escalate"                       # uncertain (or high-severity) → human
```
`tau_escalate`/severity policy is **set per decision class by the deployer** (the enterprise picks α and which classes are zero-tolerance). Report risk–coverage curve, AURC, selective accuracy@coverage, and the escalation rate at each α.

### 7.6 Audit record (the regulator-ready differentiator)
```python
import hashlib, time, orjson
def audit(claim, evidence_span, p_grounded, decision, policy, versions):
    rec = {
        "ts": time.time(),
        "claim": claim,
        "evidence_span": evidence_span,
        "p_grounded": round(float(p_grounded), 4),
        "decision": decision,                      # accept | reject | escalate
        "policy": policy,                          # {"alpha": .., "delta": .., "class": "..", "method": "crc"}
        "model_version": versions["model"],
        "calib_version": versions["calib"],        # which calibration set/run produced the thresholds
        "content_hash": hashlib.sha256((claim + "||" + evidence_span).encode()).hexdigest(),
    }
    return rec
```
Export JSONL. Provide a mapping doc from these fields to EU AI Act Art 50 / NIST RMF Measure / HIPAA evidence requirements.

---

## 8. Evaluation protocol

### 8.1 Verifier quality
AUROC / AUPRC and balanced-accuracy / F1 for hallucination detection on RAGTruth and LLM-AggreFact; compare to reported HHEM / MiniCheck numbers. Calibration: ECE, Brier, reliability diagrams.

### 8.2 Guarantee validation (the core result)
On held-out test, confirm realized **false-approval (missed-hallucination) rate ≤ α** at the chosen threshold, across α ∈ {0.01, 0.05, 0.10}. **Repeat on the OOD slice and report where it degrades** (the honest figure). Plot desired-vs-realized risk; coverage (auto-approval rate) vs α; risk–coverage / AURC.

### 8.3 Cost / latency / on-prem
Per-claim verify latency on CPU (ONNX int8) and throughput; confirm the `verify()` path runs with networking disabled (air-gap test).

### 8.4 Targets to chase (honest)
- Detection AUROC competitive with reported small-verifier baselines (HHEM/MiniCheck ballpark) on RAGTruth/LLM-AggreFact.
- **Realized missed-hallucination rate ≤ α on test** for the controllable settings, with the auto-approval (coverage) rate reported as the cost.
- **Calibration ECE substantially reduced** post-calibration.
- **OOD honesty:** quantify the bound's degradation under domain/language shift, and show the conditional/non-exchangeable variant recovering some of it.
- Verify latency target < ~50 ms/claim single-thread (faster with int8), batchable; full air-gapped run.
- A working **Indic eval** result.

### 8.5 Baselines & ablations
0.5 threshold vs calibrated vs CRC-selected; raw uncalibrated score; (optional, needs budget) LLM-judge; max-over-passages vs concat evidence; with/without decontextualization; i.i.d. vs conditional vs non-exchangeable conformal.

---

## 9. Deliverables

1. **`praman` package**: `verify(output, evidence, alpha=0.05, policy=...) -> {claims:[...], output_decision, audit}` (see §11), plus `Verifier`, `calibrate`, `fit_risk_control`, `export_onnx`, `save/load`.
2. **ONNX int8** verifier + benchmark; **air-gapped** runtime (no external calls).
3. **FastAPI service** (`/verify`) runnable on the VM.
4. **HF model card** `dmj-one/praman-verifier`: task, supported languages, detection + calibration + guarantee tables, the honest-scope section verbatim, the EU-AI-Act/NIST/HIPAA field mapping.
5. **Technical report** `REPORT.md`: market framing (§1), method, the guarantee-validation + OOD-honesty figures, Indic result, limits (§4), with citations (§16).
6. **Reproducible scripts** (`scripts/00_setup.sh` … `90_report.py`), `Makefile`/`tasks.py`, fixed seeds, `configs/*.yaml`, `requirements.lock.txt`.
7. **Tests** (`pytest`): data shapes; calibration improves ECE on a fixture; CRC threshold monotonicity; realized risk ≤ α on a synthetic fixture; `verify()` schema; air-gap (no network) smoke test; audit-record schema.

---

## 10. Suggested repo layout
```
praman/
├── PROJECT_BRIEF.md  AGENTS.md  PROGRESS.md  README.md  REPORT.md
├── requirements.lock.txt  pyproject.toml  Makefile
├── configs/  data.yaml verifier.yaml calib.yaml risk.yaml policy.yaml
├── src/praman/
│   ├── data.py          # RAGTruth / LLM-AggreFact / NLI / Indic loaders, splits
│   ├── decompose.py     # claim decomposition (+ optional decontextualization)
│   ├── verifier.py      # NLI/faithfulness scoring + ONNX int8 export
│   ├── calibrate.py     # temperature/isotonic + ECE/Brier
│   ├── riskcontrol.py   # CRC/RCPS/LTT (+ conditional, non-exchangeable, MAPIE adapter)
│   ├── decide.py        # accept/escalate/reject + risk-coverage
│   ├── audit.py         # audit record + regulator field mapping
│   ├── pipeline.py      # verify(): ties it together
│   ├── eval.py          # metrics, guarantee validation, OOD, bootstrap CIs
│   ├── plots.py         # reliability diagrams, risk-coverage, desired-vs-realized risk
│   └── service.py       # FastAPI
├── data/ artifacts/ runs/ embeddings/   # gitignored
└── tests/
```

---

## 11. Public API (freeze early)
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

---

## 12. Definition of Done (hard)
- [ ] `pip install -e .` in a fresh venv on the VM; `verify(...)` runs end-to-end on CPU **with networking disabled**.
- [ ] `pytest` passes, covering §9.7.
- [ ] Artifacts (verifier + calibration + conformal thresholds + policy) saved and reloadable offline; `verify()` matches §11 schema.
- [ ] **Guarantee demonstrated:** realized missed-hallucination rate ≤ α on RAGTruth test at α∈{0.01,0.05,0.10}, with coverage reported; **plus the OOD-slice degradation honestly reported** and a conditional/non-exchangeable variant evaluated.
- [ ] Detection-quality table vs HHEM/MiniCheck baselines; calibration ECE before/after.
- [ ] **Indic eval** result reported.
- [ ] ONNX int8 latency benchmarked; audit JSONL exports; EU-AI-Act/NIST/HIPAA field mapping written.
- [ ] HF card + `REPORT.md` + `README.md`, **including the §4 honest-scope section verbatim**.
- [ ] `requirements.lock.txt` committed; one-command reproduction from `scripts/`.
- [ ] `PROGRESS.md` has the decision log + final results summary.

---

## 13. Execution plan (phases — checkpoint after each)
- **Phase 0 — setup & data (≈1 hr).** Env (§5), verify CPU-only + air-gap, load RAGTruth/LLM-AggreFact, confirm schemas, write `data.py`, smoke test.
- **Phase 1 — verifier baseline.** Wire claim decomposition + one verifier model (e.g., DeBERTa-NLI or HHEM-open); get a detection AUROC on RAGTruth before optimizing.
- **Phase 2 — calibration.** Temperature + isotonic; ECE/Brier/reliability diagrams.
- **Phase 3 — risk control.** CRC false-approval bound (from-scratch + MAPIE adapter); add RCPS bounds + LTT precision; validate realized risk ≤ α on test.
- **Phase 4 — robustness & decisions.** Conditional + non-exchangeable variants; OOD-slice eval; decision/abstention policy; risk-coverage/AURC; bootstrap CIs; ablations (§8.5).
- **Phase 5 — Indic slice.** Build/clean the small Indic groundedness set; multilingual verifier; report.
- **Phase 6 — packaging.** `verify()`/CLI/FastAPI, ONNX int8 + air-gap test, audit + field mapping, tests.
- **Phase 7 — report & card.** `REPORT.md` + HF card with all figures, limits, citations. Run §12.
- **Phase 8 — stretch (only if time/budget).** (a) LLM-judge baseline (**API budget → ask owner**). (b) Action-gating mode (verify an agent's action vs declared policy) — your security angle. (c) A real on-prem demo on a sovereign/Indic RAG flow.

---

## 14. Self-research directives
- Before pinning a library, check its latest release/changelog; verify `mapie`, `transformers`, the verifier model card, and dataset paths **at runtime**, adapt, and log discrepancies in `PROGRESS.md`.
- Read the §16 primary sources for the conformal/factuality methods — match the math, don't reinvent it.
- Cross-check your CRC implementation against MAPIE on a toy set before trusting it on RAGTruth.
- Benchmark verifier-model and int8 choices; record numbers. Prefer the model with the best accuracy/latency that runs offline.

## 15. Risks & gotchas
- **On-prem rule:** any external call in the `verify()` path is a bug. Add a test that runs with networking disabled.
- **No GPU / CUDA wheel** = bug. Set threads (§5.2).
- **Calibration vs conformal split hygiene;** keep calib exchangeable with test; if strict, split calib.
- **Rare/!i.i.d. conditions:** report which domains/languages the bound is valid for; don't overclaim.
- **Claim decomposition errors** propagate; evaluate it explicitly (RAGTruth spans help).
- **Marginal ≠ per-instance:** never present the guarantee as a per-item certificate (see §4).
- **ONNX int8 op gaps:** fall back to fp32 ONNX/torch and note it.
- **Reproducibility:** seed numpy/torch/sklearn; log versions with every result.

## 16. References
- Conformal factuality / guaranteed verification: Mohri & Hashimoto, *Language Models with Conformal Factuality Guarantees* (ICML 2024, arXiv:2402.10978); Abbasi-Yadkori et al., *Mitigating LLM Hallucinations via Conformal Abstention* (arXiv:2405.01563); *Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement* (arXiv:2407.18370); Cherian, Gibbs, Candès, *LLM validity via enhanced conformal prediction* (2024); Liu & Wu, *Multi-group UQ for long-form generation* (2024); Detommaso et al. (2024, conditional calibration of factuality); Ulmer et al., *Non-exchangeable conformal language generation with NN* (2024); *Taming Variability: Randomized & Bootstrapped Conformal Risk Control for LLMs* (arXiv:2509.23007); *Is Conformal Factuality for RAG-based LLMs Robust?* (arXiv:2603.16817, the limits paper).
- Faithfulness / hallucination detection: RAGTruth (Niu et al., ACL 2024); MiniCheck + LLM-AggreFact (Tang et al., EMNLP 2024); AlignScore (Zha et al., ACL 2023); SelfCheckGPT (Manakul et al., 2023); FActScore (Min et al., EMNLP 2023); Vectara HHEM (open).
- Risk control: Bates et al., RCPS (arXiv:2101.02703); Angelopoulos et al., Conformal Risk Control (2022) & Learn-Then-Test. MAPIE docs: https://mapie.readthedocs.io.
- Small encoders / Indic: EmbeddingGemma-300M; Qwen3-Embedding-0.6B; MuRIL; IndicBERTv2; DeBERTa-v3 NLI.
- Market/regulation context (for the report): EU AI Act Art 50 (transparency, Aug 2026); HIPAA Security Rule AI update (Feb 2026); NIST AI RMF.

---

*Build the simplest correct verifier first, prove the bound on RAGTruth, then show honestly where it holds and breaks, add the Indic slice and the audit trail, package it to run offline. Log decisions in `PROGRESS.md`. Ship.*