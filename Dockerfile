# PRAMAN container. CPU-only. The verifier model is baked at BUILD time (network allowed
# only here); the RUNTIME is air-gapped in-container (HF_HUB_OFFLINE), consistent with the
# on-prem thesis. Runs on Cloud Run / any container host: binds 0.0.0.0:$PORT.
#
# Build:  docker build -t praman .
# Run:    docker run --rm -p 8080:8080 -e PORT=8080 praman   # then GET /health, POST /verify
#
# NOTE on the product thesis: PRAMAN's value is on-prem / air-gapped operation. A managed cloud
# deploy (Cloud Run) is a convenient PUBLIC DEMO, not the product's deployment model; a
# regulated/sovereign buyer runs this image on THEIR own hardware. See README / REPORT §1.

# ---- builder: install deps + bake the offline artifact (downloads the model once) ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only torch from the dedicated index, then the rest of the stack.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -U pip wheel setuptools \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "transformers" "numpy" "scipy" "scikit-learn" "pyyaml" "orjson" \
       "onnx" "onnxruntime" "optimum[onnxruntime]" "fastapi" "uvicorn" "pydantic"
COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY runs/full/calibration.json runs/full/riskcontrol.json ./runs/full/
RUN pip install -e . --no-deps
# Build the offline artifact: downloads the HF verifier once, bundles calibration + risk + policy.
RUN python scripts/70_build_artifact.py --run-id full --out artifacts/praman-verifier

# ---- runtime: lean, non-root, air-gapped ----
FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PRAMAN_ARTIFACTS=/app/artifacts/praman-verifier \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 PRAMAN_THREADS=4 PORT=8080
RUN useradd -m -u 10001 praman
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/artifacts /app/artifacts
COPY --from=builder /app/src /app/src
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
WORKDIR /app
USER praman
EXPOSE 8080
# Cloud Run sets $PORT; bind 0.0.0.0 (Cloud Run terminates TLS at its edge).
CMD ["sh", "-c", "uvicorn praman.service:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
