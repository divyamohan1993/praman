#!/usr/bin/env bash
# Chained overnight run on the shared OCI box. Polite (nice/ionice), disk-guarded, and
# resilient: each stage logs and a failure does not abort the rest. All heavy artifacts
# stay on the box; small numeric outputs (metrics/robustness/indic/latency/tokens json,
# calibration/riskcontrol) are what we sync back. Launch:
#   nohup nice -n 15 ionice -c3 bash scripts/99_overnight.sh > overnight.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HOME="$PWD/.cache/hf" TMPDIR="$PWD/.cache/tmp" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
MIN_FREE_GB=4

stamp() { date -u +%FT%TZ; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
guard() { local f; f=$(free_gb); if [ "${f:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "[$(stamp)] DISK FLOOR HIT (${f}GB < ${MIN_FREE_GB}GB) -> skipping remaining heavy stages"; return 1; fi
  echo "[$(stamp)] disk ${f}GB ok"; }
stage() { echo "=== [$(stamp)] STAGE: $* ==="; }

# Full-size caps (test pool ~13k claims; calib carved from train). Tuned for overnight.
FULL_TRAIN=${FULL_TRAIN:-10000}
FULL_TEST=${FULL_TEST:-9000}
OOD_TRAIN=${OOD_TRAIN:-8000}
OOD_TEST=${OOD_TEST:-4000}
OOD_OOD=${OOD_OOD:-5000}
MAXLEN=${MAXLEN:-384}   # shorter seq -> faster on CPU; modest accuracy cost (logged)

# Export ONNX int8 FIRST so scoring uses the fast VNNI path. If int8 export fails the
# backend silently stays torch via the BK fallback below.
stage "ONNX int8 export + latency benchmark"
guard && $PY -u scripts/60_export_onnx.py --run-id full
BK=torch
if [ -f artifacts/verifier_onnx/model_int8.onnx ] || [ -f artifacts/verifier_onnx/model.onnx ]; then
  BK=onnx; echo "[$(stamp)] scoring backend = ONNX (fast path)"
else
  echo "[$(stamp)] ONNX missing -> scoring backend = torch (slow fallback)"
fi

stage "full score (backend=$BK, maxlen=$MAXLEN)"
guard && $PY -u scripts/20_score.py --run-id full --backend $BK --max-length $MAXLEN \
  --max-train $FULL_TRAIN --max-test $FULL_TEST \
  && $PY -u scripts/30_pipeline.py --run-id full
echo "[$(stamp)] full pipeline rc=$?"

stage "OOD leave-one-domain-out (Data2txt held out)"
guard && $PY -u scripts/20_score.py --run-id ood --ood-task Data2txt --backend $BK --max-length $MAXLEN \
  --max-train $OOD_TRAIN --max-test $OOD_TEST --max-ood $OOD_OOD \
  && $PY -u scripts/40_robustness.py --run-id ood
echo "[$(stamp)] ood rc=$?"

stage "Indic slice (IndicXNLI Hindi + mDeBERTa-xnli)"
guard && $PY -u scripts/50_indic.py --run-id indic
echo "[$(stamp)] indic rc=$?"
# bring indic.json under runs/full so 90_report picks it up
cp -f runs/indic/indic.json runs/full/indic.json 2>/dev/null || true

stage "assemble offline artifact + fill report tokens + plots"
$PY -u scripts/70_build_artifact.py --run-id full || true
$PY -u scripts/90_report.py --run-id full || true

stage "full test suite"
$PY -m pytest -q 2>&1 | tail -15 || true

echo "=== [$(stamp)] OVERNIGHT DONE ==="
free_gb | xargs -I{} echo "free disk {}GB"
