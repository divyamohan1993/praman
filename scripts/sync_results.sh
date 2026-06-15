#!/usr/bin/env bash
# Pull the small numeric run artifacts + filled docs back from the box into the repo.
# Heavy artifacts (scores.npz, ONNX weights, model caches) stay on the box.
# Usage (from repo root, on the dev machine):  bash scripts/sync_results.sh
set -uo pipefail
# Parameterized: set these in your shell, never commit real host/key values.
#   export SSHK=~/.ssh/your_key  HOST=user@your-vm-host  REMOTE=/home/you/praman
SSHK="${SSHK:?set SSHK to your SSH private-key path}"
HOST="${HOST:?set HOST to user@vm-host}"
SSHOPT="-i $SSHK -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
REMOTE="${REMOTE:-/home/ubuntu/praman}"

mkdir -p runs/full runs/ood runs/indic artifacts
# small json metrics + thresholds + plots
for f in runs/full/metrics.json runs/full/latency.json runs/full/tokens.json \
         runs/full/calibration.json runs/full/riskcontrol.json runs/full/indic.json \
         runs/ood/robustness.json runs/indic/indic.json; do
  scp $SSHOPT "$HOST:$REMOTE/$f" "$f" 2>/dev/null && echo "pulled $f" || echo "skip $f (absent)"
done
# plots
scp $SSHOPT -r "$HOST:$REMOTE/runs/full/plots" runs/full/ 2>/dev/null && echo "pulled plots" || true
# the filled docs (90_report.py edits them in place on the box)
for f in REPORT.md docs/model-card.md; do
  scp $SSHOPT "$HOST:$REMOTE/$f" "$f" 2>/dev/null && echo "pulled $f (filled)" || echo "skip $f"
done
# the manifest of the assembled offline artifact (weights stay on the box)
scp $SSHOPT "$HOST:$REMOTE/artifacts/praman-verifier/manifest.json" artifacts/ 2>/dev/null || true
echo "done."
