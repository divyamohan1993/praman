# PROGRESS — PRAMAN build log

## ============ FINAL SUMMARY (2026-06-15) ============
**PRAMAN is built, validated, and reproducible. The distribution-free guarantee holds.**

Headline numbers (RAGTruth, real torch scores; verifier = DeBERTa-v3-base-mnli):
- **Guarantee (the core result): realized missed-hallucination rate ≤ α at every α.**
  α=0.01 → FNR 0.005 (coverage 6.2%); α=0.05 → 0.038 (13.8%); α=0.10 → 0.076 (21.7%).
  Bootstrap mean FNR ≤ α at all α; exceed-fraction 0.065 / 0.22 / 0.375 (expected for an
  in-expectation CRC guarantee, reported not hidden). CRC math independently validated
  (toy vs exact bound diff 0.001; cross-checked vs MAPIE 1.4.1).
- **Detection:** AUROC 0.612, AUPRC 0.116 (modest, honest generic-NLI baseline; below
  HHEM/MiniCheck ~0.75-0.85 — generic NLI + 320-tok evidence truncation. The guarantee is
  agnostic to detector quality; a weak verifier costs coverage, not validity).
- **Calibration:** ECE 0.524 → 0.380, Brier 0.506 → 0.254 (temperature).
- **OOD honesty (leave-one-domain-out, Data2txt held out):** detection collapses to AUROC
  0.465 (below random). FNR bound still met but only by near-zero auto-approval (cost = coverage);
  non-exchangeable NN variant tightens it. A single in-domain split overshot (α=0.10 → 0.169),
  illustrating expected-value (not per-split) control.
- **Indic (Hindi):** mDeBERTa-xnli AUROC 0.999, ECE 0.003 → 0.001, guarantee holds 2/3 α.
  Caveat: fell back to XNLI(hi) and mDeBERTa was trained on XNLI -> largely in-distribution;
  demonstrates the pipeline + guarantee on Devanagari Hindi, not OOD Indic generalization.
- **Latency / on-prem:** torch fp32 1030 ms/claim, ONNX int8 509 ms (2.0x) on the contended box.
  int8 ACCURACY-DEGRADED (scores diverge from fp32, corr ~0.72; DeBERTa-v3 disentangled-attention
  export) -> torch is the scientific + runtime path; int8 kept as latency benchmark only.
- **Tests:** 32 pytest pass (data shapes, calibration-improves-ECE, CRC monotonicity + expected-
  risk control, verify() schema, in-process air-gap, audit schema). `verify()` runs end-to-end
  offline on the assembled torch artifact (clinical example -> escalate; full §11 schema + audit).

Reproduce (on the box, `/home/ubuntu/praman`): `make setup && make data && make validate-crc`
then `bash scripts/99_overnight.sh` (full + OOD + Indic + artifact + report-fill + pytest), or
per-stage `make full / ood / report / test`. Artifacts: `artifacts/praman-verifier/` (torch,
offline), `runs/full/{metrics,latency,tokens}.json`, `runs/ood/robustness.json`,
`runs/indic/indic.json`, `runs/full/plots/`. Filled docs: REPORT.md + docs/model-card.md.

Honest limits (verbatim scope in REPORT §4 / model card): marginal bound, not a per-item
certificate; faithfulness-to-evidence, not truth; assumes exchangeability (OOD voids it);
modest verifier -> modest coverage; Indic result in-distribution; int8 degraded. No humans were
removed on catastrophic/irreversible decisions; PRAMAN right-sizes review + produces a defensible
audit trail.

Stretch (not done, named not faked): stronger verifier (MiniCheck/HHEM) for higher coverage;
seq-512 + max-over-passages ablation; a held-out human-checked Indic groundedness set; validated
static/per-channel int8. LLM-judge baseline + HF public push remain PARKED (cost / external).
## ===================================================

# PROGRESS — PRAMAN build log

Append-only cross-session memory (AGENTS.md). Newest entries at the bottom of each phase.
Times in UTC. Box = OCI `dmj-docker-TEMP` (VM.Standard.E5.Flex, 12 vCPU AMD Genoa, CPU-only),
project at `/home/ubuntu/praman`, venv at `.venv`. Co-hosted `udaan` Docker stack is NOT ours
and must not be touched (disk floor 4GB guards its postgres).

## BLOCKERS (top of file, cleared when resolved)
- **LLM-AggreFact is gated** on HF (needs auth token = a secret). Per brief we do NOT use
  secrets without asking. RESOLVED by pivoting to the brief's PRIMARY source RAGTruth
  (non-gated, GitHub). No blocker remains.
- HF model-card push (public) and any paid LLM-judge baseline remain PARKED (need owner go-ahead).

---

## Phase 0 — env + data wiring (2026-06-14)
- Box surveyed: 12 vCPU, 31 GB RAM, Python 3.12.3, AVX-512 + VNNI + BF16 (good for ONNX int8).
  Disk is the binding constraint: ~10 GB free on a single `/dev/sda1` shared with udaan's
  postgres. Every download is disk-guarded (`scripts/00_setup.sh`, floor 4 GB); heavy procs
  `nice`/`ionice`'d so udaan latency is untouched.
- `scripts/00_setup.sh`: venv + pip bootstrap (system python had no pip; ensurepip used),
  CPU-only torch (`2.12.0+cpu`, `cuda_available=False` verified), full dep stack, pip cache
  purged, `requirements.lock.txt` committed. No sudo/apt (keeps the shared box untouched).
- **Decision (data source):** brief's primary LLM-AggreFact is gated; pivoted to **RAGTruth**
  (GitHub ParticleMedia/RAGTruth, non-gated, ~36 MB). It is the brief's stated PRIMARY anyway,
  has real word/span hallucination annotations, an official train/test split (15090/2700
  responses), and 3 task types (QA/Summary/Data2txt) giving a natural leave-one-domain-out
  OOD axis. `src/praman/data.py` parses it: split each response into sentences with char
  offsets, label a sentence **ungrounded iff it overlaps any annotated hallucination span**;
  evidence = the source reference. Caches parsed claims to `data/ragtruth/claims_cache.jsonl`.
- Split summary (caps for thin slice): calib_temp 2400 (12.4% ungrounded), calib_conf 4000
  (11.0%), test 4000 (8.8%). Calibration carved from official TRAIN; test = official TEST
  (conformal hygiene: calib disjoint from test).

## Phase 1 — verifier (2026-06-14)
- **Decision (verifier):** ONE small CPU model, `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  (3-class NLI, ~440 MB). "supported" = entailment; the single binary logit
  `z = entail_logit - logsumexp(neutral, contra)` satisfies `sigmoid(z) = P(entailment)`, so
  calibration operates on one clean logit. Multi-passage handled by max-P (claim grounded if
  ANY passage supports it). Chosen for robustness + the clean (claim,evidence)->P interface +
  fitting the disk budget; a model swap is one config line (and we delete the old weights).
- Scoring split from analysis: `scripts/20_score.py` runs the model once and caches
  (u,p,z,y,grounded) per split to `runs/<id>/scores.npz` (+ `meta.json` for strings, so the
  npz loads without `allow_pickle`). `scripts/30_pipeline.py` does the cheap analysis on the
  cache so we never re-run the model while iterating.

## CRC validation (the headline math) — PASSED (2026-06-14)
- `scripts/31_crc_validate.py`, synthetic toy with known answer:
  - CRC thresholds monotone in alpha; no-feasible-alpha -> t=0 (approve-nothing). ✓
  - **CRC controls the EXPECTED realized FNR:** mean realized FNR over 400 trials was
    0.0099 / 0.0489 / 0.0980 for alpha 0.01 / 0.05 / 0.10 (all <= alpha). `frac_exceed_alpha`
    ~0.40 is CORRECT for an in-expectation guarantee and must NOT be chased to zero.
  - **RCPS (Hoeffding) gives high-probability control:** frac_exceed = 0.000 <= delta=0.05. ✓
  - MAPIE 1.4.1 present (`BinaryClassificationController`, API changed post-1.0); my CRC
    matched an independent exact-bound re-derivation to within 0.001 on the toy.
- Implication for the report: frame CRC as expected-risk control, RCPS as high-prob; report
  both. The brief's reference `crc_threshold` had a bug (broke immediately -> approve-all);
  ours returns the largest feasible t and defaults to approve-nothing.

## Shared-box contention (operational note, 2026-06-14)
- The box co-hosts an ACTIVE external job: `~/pqcsched/` running `./venv/bin/python
  scripts/calibrate.py 12 30 12` (post-quantum-crypto scheduler experiment, ~6-7 cores,
  separate venv). This is one of the "other things" not to disturb. Our scorer runs at
  `nice -n 15 ionice -c3`, so it politely yields; consequence is slow scoring (box
  oversubscribed). DO NOT raise our priority or kill theirs.
- Implication: prefer ONNX int8 (faster + lighter) for the full run; run heavy jobs in
  background overnight; keep thread pressure reasonable.

## Phase 5 decision — Indic slice (2026-06-14)
- No human annotator tonight -> use an existing non-gated Indic resource, not fabricated
  hand-verification. Chosen: **IndicXNLI (Hindi)** scored with **mDeBERTa-v3-base-mnli-xnli**
  as a groundedness PROXY (premise=evidence, hypothesis=claim; entailment=grounded). Runs the
  same calibrate->CRC->validate pipeline. Honest framing in the card: NLI-based proxy in Hindi,
  machine-labelled, not hand-verified RAG-groundedness; the scarcity of true Indic RAG-
  groundedness data is itself the contribution. `scripts/50_indic.py`.

## Speed reality + the ONNX pivot (2026-06-14)
- Measured torch CPU throughput on the box: **~606 ms/claim at seq256, ~1076 ms at seq512**
  (DeBERTa-v3-base, 184M params; oneDNN/mkl active). With the co-hosted pqcsched job
  contending, the first slice (3900 claims, seq512) ran >25 min and was abandoned.
- Pivot: (1) cap claim counts to what gives stable estimates (test ~3-9k, calib ~3-8k, not
  the full 130k); (2) use **ONNX int8 (VNNI)** for the runtime + full-run scoring (faster +
  the brief's required path); (3) seq length 256-384 for the heavy runs. `scripts/99_overnight.sh`
  exports ONNX first, then scores full + OOD via int8, then Indic + report-fill, all disk-guarded.
- GOTCHA fixed: do NOT set HF_HUB_OFFLINE for the SCORING/build step (model download is
  build-time; only the runtime verify() path is air-gapped). Offline flag broke model load.
- GOTCHA fixed: pipeline.py imported the audit MODULE as `_audit` but `__init__` rebinds the
  `audit` name to the function -> AttributeError. Now imports the function directly. Tests caught it.

## RESUME PLAN (if a fresh session picks this up)
Everything is authored in `d:\praman` (canonical). The box `/home/ubuntu/praman` has the venv +
RAGTruth cache + the verifier model cached. To finish from cold:
1. SSH: `ssh -i ~/.ssh/oci_dmj ubuntu@92.4.67.232` (box `dmj-docker-TEMP`, ap-mumbai-1).
2. Sync code up: `tar czf - src scripts configs | ssh ... 'cd ~/praman && tar xzf -'`.
3. Launch: `cd ~/praman && nohup nice -n 15 ionice -c3 bash scripts/99_overnight.sh > overnight.log 2>&1 &`
   (produces runs/full/{metrics,latency,tokens}.json, runs/ood/robustness.json, runs/indic/indic.json,
   artifacts/, plots, and fills REPORT.md + docs/model-card.md tokens).
4. Sync results back: pull runs/*/*.json + artifacts/*.json + the filled REPORT.md/model-card.md.
5. `make test` on the box (full pytest incl air-gap). Then write the §12 DoD checklist + final
   PROGRESS summary. Co-hosted pqcsched job + udaan Docker stack must stay untouched.

## Overnight run LAUNCHED (2026-06-14 ~23:03 UTC)
- `scripts/99_overnight.sh` running detached (`setsid`, survives SSH drops + session end) on the
  box, logging to `~/praman/overnight.log`. Caps: full 6000 train + 6000 test, OOD 5000/3000/4000,
  seq 320. Stages: ONNX int8 export+latency -> full score (ONNX int8 if export ok, else torch) +
  pipeline -> OOD robustness -> Indic -> artifact assemble -> report token-fill + plots -> pytest.
- Produces: runs/full/{metrics,latency,tokens}.json, runs/ood/robustness.json, runs/indic/indic.json,
  artifacts/{verifier_onnx,praman-verifier}/, plots, and fills REPORT.md + docs/model-card.md.
- To finalize after it completes: `bash scripts/sync_results.sh` (pulls the small json + filled
  docs back to d:\praman), then commit, write the DoD §12 checklist + final PROGRESS summary.
- Box contention: co-hosted `pqcsched` keeps load ~12-17; our job is nice'd and slow but steady.
  SSH drops under load are expected; the run is detached so they don't matter.

## Definition of Done (§12) — FINAL
- [x] `pip install -e .` in a fresh venv on the VM; package imports, CPU-only verified.
- [x] `verify()` end-to-end on CPU with networking disabled — scripts/80_demo.py loads the torch
      artifact inside praman.airgap() and returns the §11 schema (clinical example -> escalate).
- [x] `pytest` passes (32 tests: data shapes, calibration improves ECE, CRC monotonicity +
      expected-risk control, verify() schema, in-process air-gap, audit schema).
- [x] Artifacts saved + reloadable offline (artifacts/praman-verifier, backend=torch); verify()
      matches §11 schema.
- [x] Guarantee demonstrated: realized FNR ≤ α on RAGTruth test at α∈{0.01,0.05,0.10}
      (0.005/0.038/0.076) with coverage reported (6.2/13.8/21.7%); OOD degradation honestly
      reported (AUROC 0.465 on Data2txt); conditional (Mondrian) + non-exchangeable variants evaluated.
- [x] Detection-quality table vs HHEM/MiniCheck (AUROC 0.612 vs ~0.75-0.85, honestly framed);
      calibration ECE before/after (0.524 -> 0.380).
- [x] Indic eval reported (Hindi via XNLI + mDeBERTa, AUROC 0.999, with in-distribution caveat).
- [x] ONNX int8 latency benchmarked (509 ms vs 1030 ms torch); audit JSONL exports;
      EU-AI-Act/NIST/HIPAA field mapping (docs/regulator-field-mapping.md).
- [x] HF card + REPORT.md + README.md, including the §4 honest-scope section VERBATIM.
- [x] requirements.lock.txt committed; one-command reproduction via Makefile / 99_overnight.sh.
- [x] PROGRESS.md decision log + final results summary (top of this file).
NOTE: int8 scoring accuracy is degraded (corr 0.72) -> torch is the scientific/runtime path,
int8 is latency-only. This is the brief's anticipated op-gap, documented not hidden.

## Latency (DoD deliverable) measured (2026-06-14)
- ONNX int8 confirmed WORKING (the benchmark ran ONNX inference). On the contended box
  (load ~17, seq256, n=200): **int8 508.8 ms/claim, torch fp32 1030.1 ms/claim, 2.02x speedup.**
- Honest note for the report: 509 ms is far above the brief's <50 ms aspiration because of
  (a) heavy co-host contention, (b) a 184M-param DeBERTa-base, (c) seq256. Uncontended this
  roughly halves; a smaller verifier (DeBERTa-xsmall) or fewer threads in contention would
  close most of the rest. We report the real number and the cause, not the aspiration.
- Implication: full + OOD scoring ~3.5 h at this rate; the headline (full pipeline) lands
  first (~100 min into scoring), then OOD, then Indic, then report-fill. Graceful degradation
  if the night is short: the guarantee headline completes even if Indic/report get cut.

## CRITICAL CORRECTION — ONNX int8 unusable for scoring; torch is the scientific path (2026-06-15)
- The first "headline" was computed on ONNX int8 scores that were NEVER validated against
  torch. Diagnostic (`scripts/_probe_onnx_agree.py`): ONNX int8 vs torch on 32 claims gives
  max|diff|=0.885, mean=0.22, **corr=0.72** -> ONNX scores are CORRUPTED.
- Cause isolated: forcing the hf_id tokenizer changed nothing (corr still 0.72) -> NOT the
  tokenizer; it is the **int8 quantization/export of DeBERTa-v3** (disentangled attention is
  known-fragile for ONNX). The brief anticipated this ("int8 op gaps -> fall back, note it").
- Consequence: the first metrics (AUROC 0.61, ECE up, 2-19% coverage) are ARTIFACTS of
  corrupted scores, not real verifier quality. DISCARDED. Guarantee FNR<=alpha still held even
  on garbage (CRC is distribution-free) but the detection/coverage were meaningless.
- FIX (advisor-endorsed): score ALL scientific runs with **torch** (correct). Keep ONNX int8
  ONLY as the latency deliverable (509ms vs 1030ms torch, 2x) with an honest accuracy caveat.
  The offline artifact (70_build_artifact) now ships the TORCH model, backend=torch.
- METHODOLOGY LESSON (write into the report): never compute metrics on an inference path you
  have not validated against the reference. corr>=0.97 is now a hard gate before any long run.
- Re-running full + OOD + Indic on torch (load eased to ~8; ~2.3h). Caps cut: full 3000 train
  / 2000 test, OOD 2500/1500/2500, Indic 2000. This run's headline is the real one.

## Decisions still autonomous / pending
- Realized-risk-<=-alpha demonstration on the RAGTruth test slice: IN PROGRESS (thin slice).
- OOD (leave-one-domain-out Data2txt), conditional (Mondrian) + non-exchangeable variants,
  Indic slice, ONNX int8 export, full report: queued.
