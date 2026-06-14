#!/usr/bin/env bash
# PRAMAN env bootstrap. CPU-only. Idempotent. No sudo/apt (tools assumed present
# on the shared OCI E5 box; we never touch the system or the co-hosted Docker stack).
set -euo pipefail

PROJ="${PRAMAN_HOME:-/home/ubuntu/praman}"
MIN_FREE_GB="${PRAMAN_MIN_FREE_GB:-4}"     # hard floor: postgres (udaan) lives on the same partition

free_gb() { df -BG --output=avail "$PROJ" 2>/dev/null | tail -1 | tr -dc '0-9'; }
guard_disk() {
  local f; f="$(free_gb)"
  if [ -z "$f" ] || [ "$f" -lt "$MIN_FREE_GB" ]; then
    echo "FATAL: free disk ${f:-?}GB < floor ${MIN_FREE_GB}GB at $PROJ. Aborting to protect co-hosted DB." >&2
    exit 17
  fi
  echo "[disk] ${f}GB free (floor ${MIN_FREE_GB}GB) OK"
}

echo "=== PRAMAN setup @ $(date -u +%FT%TZ) ==="
mkdir -p "$PROJ"
cd "$PROJ"
guard_disk

# keep all caches inside the project dir (same partition, but easy to measure + clean)
export PIP_CACHE_DIR="$PROJ/.cache/pip"
export HF_HOME="$PROJ/.cache/hf"
export TMPDIR="$PROJ/.cache/tmp"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

# --- venv ---
if [ ! -x "$PROJ/.venv/bin/python" ]; then
  echo "[venv] creating"
  python3 -m venv "$PROJ/.venv" || python3 -m venv --without-pip "$PROJ/.venv"
fi
VPY="$PROJ/.venv/bin/python"

# --- pip bootstrap (system python on this box has no pip module) ---
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
  echo "[pip] ensurepip"
  "$VPY" -m ensurepip --upgrade 2>/dev/null || {
    echo "[pip] ensurepip failed; fetching get-pip.py (build-time network OK)"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"
    "$VPY" "$TMPDIR/get-pip.py"
  }
fi
"$VPY" -m pip install -U pip wheel setuptools

guard_disk
echo "[torch] CPU-only wheel"
"$VPY" -m pip install --no-input torch --index-url https://download.pytorch.org/whl/cpu

guard_disk
echo "[deps] core stack (lean; sentence-transformers/model-specific added later)"
"$VPY" -m pip install --no-input -U \
  "transformers" "datasets" "accelerate" \
  "scikit-learn" "numpy" "pandas" "scipy" \
  "onnx" "onnxruntime" "optimum[onnxruntime]" \
  "mapie" "netcal" \
  "matplotlib" "tqdm" "pyyaml" "fastapi" "uvicorn" "pydantic" "orjson"

echo "[clean] purge pip cache to reclaim space"
"$VPY" -m pip cache purge || true
rm -rf "$TMPDIR"/* 2>/dev/null || true

echo "[freeze] requirements.lock.txt"
"$VPY" -m pip freeze > "$PROJ/requirements.lock.txt"

guard_disk
echo "[verify] CPU-only + threads"
"$VPY" - <<'PY'
import torch, os
assert torch.cuda.is_available() is False, "CUDA must be unavailable (CPU-only box)"
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("threads default", torch.get_num_threads())
PY

echo "=== setup done @ $(date -u +%FT%TZ) ==="
df -BG "$PROJ" | tail -1
