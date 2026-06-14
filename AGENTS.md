# AGENTS.md — operating manual for PRAMAN

**Read `PROJECT_BRIEF.md` in full before doing anything.** That is the spec. This file is *how to work*.

## What you are building
`praman`: an on-prem, CPU-only, Indic-first **grounded-claim verifier** that returns, per claim, a calibrated grounded/ungrounded verdict, a **distribution-free guarantee** on the rate of auto-approving an ungrounded claim, an accept/escalate/reject decision, and an **audit record**. Scope, method, data, and the honest limits are in `PROJECT_BRIEF.md` (read §4 carefully — the product *right-sizes* human review and documents risk; it does **not** eliminate humans on catastrophic/irreversible decisions, and you must never present the guarantee as a per-item certificate).

## Environment (non-negotiable)
- **CPU-only Oracle E5 VM (AMD Genoa, 6 OCPU / 12 vCPU). No GPU.** Never install CUDA wheels or call `.cuda()`. Verify `torch.cuda.is_available() is False` at startup.
- **On-prem rule:** the runtime `verify()` path makes **no external network calls**. Networking is allowed only to download models/datasets during build. Add a test that runs `verify()` with networking disabled.
- Work in `.venv` (Python 3.12). Set the thread env vars + `torch.set_num_threads(12)` at process start (brief §5.2). Commit `requirements.lock.txt`; verify current library/dataset APIs at runtime rather than trusting tutorials.

## How to work
1. **Work the phases in order** (brief §13). Get the simplest end-to-end verifier working and measured on RAGTruth before optimizing.
2. **Checkpoint after every phase:** commit code + artifacts and append a dated entry to `PROGRESS.md` (what you did, decisions + why, benchmark numbers, results, next step). `PROGRESS.md` is append-only and is your cross-session memory.
3. **Prove, don't assert.** The headline is the *guarantee validation* — realized missed-hallucination rate ≤ α on held-out data, plus the honest OOD-degradation figure. Cross-check your conformal code against MAPIE on a toy set first.
4. **Benchmark, don't guess** (verifier model, int8 vs fp32, threads). Numbers go in `PROGRESS.md`.
5. **When reality differs from the brief** (dataset fields, model card, library API), trust reality, adapt, and log it. The brief is a strong prior, not gospel.
6. **Run things on the VM**, read errors, fix, re-run. No untested hand-offs. `pytest` (incl. the air-gap and schema tests) gates "done."
7. **Never overclaim the guarantee.** Marginal ≠ per-instance; faithfulness ≠ truth; the bound assumes exchangeability. Report where it holds and where it breaks.

## Decide autonomously vs. ask Divya
**Decide and log:** verifier-model choice, decomposition strategy, hyperparameters, thresholds, conformal method, which conditions the bound is valid for, report contents.

**Stop and ask first (cost / secrets / external side effects):**
- **Spends money:** the optional LLM-judge baseline (Phase 8a) or any paid API; provisioning bigger/GPU instances.
- **Secrets:** an HF write token to push the model card, or any cloud key — use a `.env`, ask for values, never commit secrets.
- **Publishing externally:** pushing weights to the public HF hub — prepare it, get the go-ahead before making it public.
- **Scope changes** that blow the budget — note the trade-off and let Divya choose.
If blocked on one of these, keep progressing on everything that isn't blocked and flag the blocker at the top of `PROGRESS.md`.

## Done
When every box in `PROJECT_BRIEF.md` §12 is checked and committed, write a final summary at the top of `PROGRESS.md` (headline numbers — detection quality, realized risk vs α, coverage cost, OOD behaviour, Indic result, latency — plus how to reproduce and the honest limits) and stop. Surface remaining stretch ideas instead of gold-plating.