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

## Decisions still autonomous / pending
- Realized-risk-<=-alpha demonstration on the RAGTruth test slice: IN PROGRESS (thin slice).
- OOD (leave-one-domain-out Data2txt), conditional (Mondrian) + non-exchangeable variants,
  Indic slice, ONNX int8 export, full report: queued.
