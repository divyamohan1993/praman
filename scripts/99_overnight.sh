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

# SCORING BACKEND = torch. ONNX int8 of this DeBERTa-v3 degrades accuracy (scores diverge
# from torch, corr ~0.72; disentangled-attention export). int8 is kept ONLY for the latency
# benchmark (runs/full/latency.json, already measured), NOT for scoring. Caps cut to fit the
# contended box (~1s/claim torch): calib_conf >= ~1.5k keeps alpha=0.01 feasible.
BK=torch
MAXLEN=${MAXLEN:-320}
FULL_TRAIN=${FULL_TRAIN:-3000}    # -> calib_temp ~900, calib_conf ~1500 (~165 ungrounded positives)
FULL_TEST=${FULL_TEST:-2000}
OOD_TRAIN=${OOD_TRAIN:-2500}
OOD_TEST=${OOD_TEST:-1500}
OOD_OOD=${OOD_OOD:-2500}
INDIC_N=${INDIC_N:-2000}

# ONNX int8 export + latency: skip if already done (latency.json present from the prior run).
if [ -f runs/full/latency.json ]; then
  echo "[$(stamp)] latency.json present -> skip ONNX re-export (int8 latency already benchmarked)"
else
  stage "ONNX int8 export + latency benchmark"
  guard && $PY -u scripts/60_export_onnx.py --run-id full
fi

stage "full score (backend=$BK, maxlen=$MAXLEN) [correct numerics]"
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
guard && $PY -u scripts/50_indic.py --run-id indic --n $INDIC_N
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
