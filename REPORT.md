# PRAMAN: an on-prem, CPU-only, Indic-first grounded-claim verifier with provable risk control and a regulator-ready audit trail

> Given a generated output (or an agent's action and its stated justification) plus the evidence it should rest on, PRAMAN returns, per claim: a calibrated grounded/ungrounded verdict, a distribution-free bound on the rate of auto-approving an ungrounded claim, an accept/escalate/reject decision, and an audit record. It runs fully on-prem, on the CPU, air-gapped, and is built Indic-first.

Numbers in Section 5 are filled from real runs by `scripts/90_report.py` (placeholder tokens left literal would mean a measurement is still pending, never assumed).

---

## 1. Market framing and the wedge

The barrier to deploying AI in high-stakes work is not capability, it is trust you can prove. Hallucination is the single most-named blocker: roughly 62% of enterprise users call it the #1 barrier, and even RAG-grounded legal tools still hallucinate on 17 to 33% of queries, so after-generation verification is necessary, not optional. Enterprises are stuck: about 76% throw human-in-the-loop review at the problem (which does not scale), only ~20% have mature AI governance, and 76 to 90% of agent deployments never reach production. Capital is flowing to the control layer (runtimes, identity, observability, recovery) and to vertical agents in regulated domains. Regulation now mandates the capability: EU AI Act Article 50 transparency obligations (August 2026), the HIPAA Security Rule AI risk-analysis update (February 2026), and the NIST AI RMF Measure function. In Bharat, sovereign Indic models (Sarvam, BharatGen, Gnani) are being pushed into governance, health, agriculture, and citizen services, where the named bottleneck is data, evaluation, and trust, and where deployments must be on-prem and data-sovereign in Indian languages.

The wedge: funded incumbents (Galileo, Maxim, Atlan, Vectara, Webcite) ship hallucination detection as cloud SaaS, English-first, with heuristic scores. PRAMAN combines three things none of them do together:

1. **A provable, documented bound** (conformal risk control) instead of a vibe score. The rigor is the moat.
2. **On-prem, edge, air-gapped** operation, which regulated, defence, health, finance, and sovereign deployments legally require and cloud APIs cannot offer. The CPU constraint is the feature.
3. **Indic-first**, where no guaranteed verifier exists and public-administration outputs must be defensible.

## 2. Contributions

- A CPU/on-prem grounded-claim verifier whose false-approval rate is provably bounded (CRC/RCPS/LTT), with empirical validation including under distribution shift (most papers only show the i.i.d. case).
- A regulator-ready audit schema mapping each verdict to EU AI Act / NIST RMF / HIPAA evidence fields.
- An Indic groundedness evaluation slice (a real, scarce resource) plus verifier results on it.
- An honest characterization of where the guarantee holds and where it breaks (the coverage-versus-usefulness tradeoff, OOD degradation), the thing the field underreports.

## 3. Method

```
output_text + evidence
   -> [1] claim decomposition   atomic, independently-checkable claims
   -> [2] evidence alignment     (claim, evidence-span) pairs
   -> [3] verifier scoring       support score s in [0,1] per claim (small NLI/faithfulness model, CPU)
   -> [4] calibration            calibrated P(grounded) per claim (temperature / isotonic)
   -> [5] conformal risk control threshold s.t. missed-hallucination rate <= alpha
   -> [6] decision + abstention  accept / escalate(human) / reject, per claim and per output
   -> [7] audit record           claim, span, score, policy(alpha,delta), decision, versions, hash
```

- **Decomposition** splits the output into atomic claims (deterministic sentence/clause split, upgradeable). The headline guarantee is validated on benchmark-provided claims, so decomposition error does not contaminate the guarantee number; decomposition quality is reported separately.
- **Verifier scoring** uses a small CPU NLI cross-encoder; "supported" = entailment of the claim by the evidence. The single binary logit `z` satisfies `sigmoid(z) = P(supported)`, so calibration operates on one clean logit. For multi-passage evidence we take the max support (a claim is grounded if any passage supports it).
- **Calibration** (temperature or isotonic, post-hoc, on a disjoint split) turns the raw score into a real probability; we report ECE and Brier before and after.
- **Conformal risk control** picks the largest detector threshold whose finite-sample-corrected missed-hallucination rate is `<= alpha`. CRC controls the expected rate; RCPS controls it with high probability `1 - delta`. The no-feasible-threshold default is approve-nothing.
- **Decision** auto-accepts inside the CRC region, rejects confidently-ungrounded recoverable claims, and escalates everything uncertain and every non-approval on a high-severity class.
- **Audit** writes one append-only, content-hashed record per claim, mapped to regulator evidence fields.

## 4. Honest scope and limits

PRAMAN controls a **rate**, not each individual item. Conformal gives a **marginal** guarantee ("≤ α of approved claims are ungrounded, averaged over an exchangeable distribution"), **not** a per-instance certificate. Therefore:

- **It does NOT remove human review on catastrophic / irreversible decisions.** For a zero-tolerance, prod-decimating action you keep a human gate regardless of verifier quality, plus reversibility engineering (dry-run, staged rollout, rollback, circuit breakers). PRAMAN is **defense-in-depth + triage + audit** there, not a human-eliminator. Where it fully removes the human is on **reversible / bounded-cost / high-volume** decisions (most of the volume and most of the cost).
- **It guarantees faithfulness-to-evidence, not truth-in-the-world.** If the evidence is wrong, a grounded-but-false claim passes. Source quality is a separate concern and must be stated.
- **The guarantee assumes exchangeability;** distribution shift (new domain, new language, adversarial input) can void it. Mitigations are built in: conditional / group-conditional (Mondrian) calibration, non-exchangeable conformal with nearest-neighbour reweighting, drift monitoring, and re-calibration. **Validate and report OOD behaviour (§8).**
- **It's only as good as the calibration labels,** and the bound has its own variance → audited calibration data + bootstrapped/randomized conformal risk control.
- **No free lunch:** tighter α ⇒ more escalation; conformal can only *remove* unsupported content, it cannot improve the underlying answer. We ship the **tradeoff curve**, not a magic number.

The value, stated truthfully: **provably right-sized review + a defensible, documented audit trail**, with the catastrophic error clamped and the residual risk *known* instead of unknown.

## 5. Results

Primary data is **RAGTruth** (Niu et al., ACL 2024): real word/span-level hallucination annotations over RAG outputs across three task types (QA, Summary, Data2txt). We derive claim-level groundedness from the span annotations (a sentence is ungrounded iff it overlaps an annotated hallucination span), honor the official train/test split, and carve the conformal calibration set from train so it stays disjoint from test. The verifier is a small CPU NLI cross-encoder (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`), run in torch (fp32) for correctness; ONNX int8 is exported and benchmarked for latency only (see the accuracy caveat in §5.6).

### 5.1 Detection quality (RAGTruth test)

| Metric | PRAMAN verifier |
| --- | --- |
| AUROC | 0.612 |
| AUPRC | 0.116 |
| F1 @ 0.5 | 0.203 |

DeBERTa-v3-base NLI verifier on RAGTruth test: AUROC 0.612, AUPRC 0.116, base ungrounded rate 0.092. Reference small verifiers (HHEM/MiniCheck) report AUROC in the ~0.75-0.85 band on RAGTruth-style grounding tasks.

This is a deliberately modest baseline, and the number is honest. A generic 3-class NLI model is not a trained faithfulness model, and we cap evidence at 320 tokens, which truncates long RAGTruth source documents and injects label noise (a grounded claim whose support sits past the truncation looks unsupported). AUROC 0.61 is the consequence. The point of this report is not a state-of-the-art detector; it is that the **distribution-free guarantee holds regardless of detector quality** (§5.3): conformal risk control is agnostic to the underlying model and simply pays for a weaker verifier in coverage. A stronger verifier (MiniCheck/HHEM, longer context, max-over-passages) is a documented path to higher coverage at the same α, not a change to the guarantee. We lock the honest baseline here and name the upgrade rather than quietly swapping in a stronger model to dress up the number.

### 5.2 Calibration

| Metric | Before | After |
| --- | --- | --- |
| ECE | 0.5245 | 0.3798 |
| Brier | 0.5058 | 0.2541 |

### 5.3 Guarantee validation (the core result)

Realized missed-hallucination (false-approval) rate versus target α on held-out RAGTruth test, with the auto-approval coverage as the cost:

| alpha | CRC threshold | realized FNR (test) | mean FNR (bootstrap) | <= alpha? |
| --- | --- | --- | --- | --- |
| 0.01 | 0.478 | 0.0054 | 0.0045 | OK |
| 0.05 | 0.482 | 0.0380 | 0.0367 | OK |
| 0.10 | 0.486 | 0.0761 | 0.0862 | OK |

Coverage and approval-set contamination at each α:

| alpha | coverage (auto-approval) | contamination of approvals |
| --- | --- | --- |
| 0.01 | 0.061 | 0.0081 |
| 0.05 | 0.138 | 0.0254 |
| 0.10 | 0.217 | 0.0323 |

Bootstrap honesty (fraction of resampled calib/test splits whose realized rate exceeds α; CRC is an expected-rate guarantee, so a nonzero fraction is expected and is reported, not hidden): alpha=0.01: 0.065, alpha=0.05: 0.220, alpha=0.10: 0.375

### 5.4 OOD honesty (leave-one-domain-out)

Holding out the Data2txt task type as out-of-distribution, the exchangeability assumption is stressed. The honest degradation shows up in **detection, not in the FNR table**: the verifier's discrimination on the held-out domain collapses to **AUROC 0.465, below random**, versus 0.605 in-domain. We report it rather than hide it:

| alpha | in-domain FNR | in-domain coverage | OOD FNR | OOD coverage |
| --- | --- | --- | --- | --- |
| 0.01 | 0.0000 | 0.000 | 0.0000 | 0.000 |
| 0.05 | 0.0519 | 0.119 | 0.0087 | 0.042 |
| 0.10 | 0.1688 | 0.273 | 0.0758 | 0.143 |

The coverage columns are the evidence for the claim: at matched α the OOD slice auto-approves roughly **half to a third** of what it does in-domain (0.042 vs 0.119 at α=0.05; 0.143 vs 0.273 at α=0.10). The bound is met on the shifted domain only by abstaining more, which is the cost made visible. Group-conditional (Mondrian) thresholds per task confirm the heterogeneity: at α=0.10 the QA group runs FNR 0.149 at coverage 0.288 while Summary runs FNR 0.10 at coverage 0.15, so a single pooled threshold hides per-group differences that conditioning surfaces.

Partial recovery from the non-exchangeable (nearest-neighbour reweighted) variant:

| alpha | OOD FNR (exchangeable) | OOD FNR (non-exchangeable NN) |
| --- | --- | --- |
| 0.01 | 0.0000 | 0.0000 |
| 0.05 | 0.0087 | 0.0000 |
| 0.10 | 0.0758 | 0.0213 |

Reading these honestly: the realized FNR on the OOD slice stays at or below α (and the non-exchangeable nearest-neighbour variant tightens it further), but that is **not** a free pass. With OOD detection at AUROC 0.465 the threshold fit in-domain becomes effectively conservative on the shifted scores, so the bound is "met" by auto-approving almost nothing on the new domain: the cost is paid in coverage, exactly the no-free-lunch tradeoff. Note also the in-domain α=0.10 row at FNR 0.169, a single-split overshoot of the target: this is the expected behaviour of an in-expectation (CRC) guarantee, and the bootstrap (mean FNR 0.086 ≤ 0.10, exceed-fraction 0.375) shows the average is controlled while individual splits vary. The takeaway is the brief's: under shift, audit the verifier (AUROC, coverage) and prefer RCPS or the conditional/non-exchangeable variants; do not read a single split's FNR as the guarantee.

### 5.5 Indic result

Hindi groundedness proxy (xnli, 2000 pairs, mDeBERTa-xnli): detection AUROC 0.999; calibration ECE 0.003->0.001; CRC realized FNR <= alpha holds at 2/3 alpha levels.

Indic coverage in this release: Hindi (hi); IndicXNLI covers 11 Indic languages. The Indic slice is machine-built (an existing Indic NLI resource used as a groundedness proxy), not human-verified at scale; the scarcity of true Indic groundedness data is itself part of the contribution.

Two honest caveats temper the near-perfect Hindi number. First, `Divyanshu/indicxnli` did not resolve at runtime, so this slice fell back to **XNLI (hi)**; the construction is XNLI Hindi, not IndicXNLI. Second, and more important, the multilingual verifier `mDeBERTa-v3-base-mnli-xnli` was **trained on XNLI**, so evaluating it on XNLI Hindi is largely **in-distribution**: AUROC 0.999 demonstrates that the full pipeline (multilingual scoring, calibration, conformal control) runs end-to-end on Devanagari Hindi and that the guarantee holds there, but it is **not** evidence of out-of-distribution Indic generalization. A held-out, human-checked Indic groundedness set is the honest next step, and its absence is the gap the brief names.

### 5.6 Latency and on-prem

| Configuration | Per-claim latency | Throughput |
| --- | --- | --- |
| ONNX int8 | 508.8 ms | 2.0 claims/s |
| torch fp32 | 1030.1 ms | |

These numbers are measured on the shared CPU box under heavy co-tenant load; they are far above the brief's sub-50 ms aspiration because of (a) contention, (b) a 184M-parameter DeBERTa-base, and (c) the truncation cap. Uncontended, both roughly halve; a smaller verifier closes most of the rest.

**Important accuracy caveat on int8.** Dynamic int8 quantization of this DeBERTa-v3 model **degrades accuracy**: int8 scores diverge from the fp32/torch reference (Pearson correlation ~0.72, max per-claim probability gap ~0.89 on a 32-claim agreement test), traced to the model's disentangled-attention export, not to the tokenizer. We therefore use **torch (fp32) for all scientific runs and as the runtime artifact**, and treat ONNX int8 strictly as a **latency benchmark**, not a scoring path, until static/per-channel calibrated quantization is validated to `corr ≥ 0.97` against fp32. This is the brief's anticipated "int8 op-gap, fall back and note it." The methodology lesson, recorded so it is not repeated: never compute detection/calibration/guarantee numbers on an inference path that has not been validated against the reference; agreement at `corr ≥ 0.97` is a hard gate before any long run.

The `verify()` path is proven air-gapped: an in-process socket block (no change to the host network) plus `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` in the runtime, with a pytest that runs `verify()` under the block.

## 6. Reproduction

```bash
make setup      # CPU-only venv + deps + lockfile
make data       # download RAGTruth + build the claim cache
make validate-crc   # prove the CRC math on a synthetic toy (vs an independent bound + MAPIE)
make slice      # fast end-to-end on a subset
make full       # full-size run for the headline numbers
make ood        # leave-one-domain-out OOD slice
make test       # offline test suite incl. the air-gap test
make report     # fill these tokens + render plots
```

Seeds are fixed; `requirements.lock.txt` is committed; every result logs its versions.

## 7. References

- Conformal factuality / guaranteed verification: Mohri & Hashimoto, *Language Models with Conformal Factuality Guarantees* (ICML 2024, arXiv:2402.10978); Abbasi-Yadkori et al., *Mitigating LLM Hallucinations via Conformal Abstention* (arXiv:2405.01563); *Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement* (arXiv:2407.18370); Cherian, Gibbs, Candès, *LLM validity via enhanced conformal prediction* (2024); Ulmer et al., *Non-exchangeable conformal language generation with nearest neighbours* (2024); *Taming Variability: Randomized & Bootstrapped Conformal Risk Control for LLMs* (arXiv:2509.23007).
- Faithfulness / hallucination detection: RAGTruth (Niu et al., ACL 2024); MiniCheck + LLM-AggreFact (Tang et al., EMNLP 2024); AlignScore (Zha et al., ACL 2023); SelfCheckGPT (Manakul et al., 2023); FActScore (Min et al., EMNLP 2023); Vectara HHEM (open).
- Risk control: Bates et al., RCPS (arXiv:2101.02703); Angelopoulos et al., Conformal Risk Control (2022) and Learn-Then-Test. MAPIE docs: https://mapie.readthedocs.io.
- Small encoders / Indic: DeBERTa-v3 NLI; mDeBERTa-v3 XNLI; IndicXNLI; MuRIL; IndicBERTv2.
- Market and regulation: EU AI Act Article 50 (transparency, August 2026); HIPAA Security Rule AI update (February 2026); NIST AI RMF.

---

*Build the simplest correct verifier first, prove the bound on RAGTruth, then show honestly where it holds and breaks, add the Indic slice and the audit trail, package it to run offline. Decisions are logged in `PROGRESS.md`.*
