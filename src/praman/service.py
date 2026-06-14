"""FastAPI service for PRAMAN. Deliverable (brief 9.3); written, not deployed.

Run locally (bind loopback only, no external exposure):
    uvicorn praman.service:app --host 127.0.0.1 --port 8973

Loads the artifact dir from PRAMAN_ARTIFACTS (default artifacts/praman-verifier), fully
offline. Endpoints: GET /health (shallow), GET /health/ready (deep, model loaded),
POST /verify. The verify path makes no external network calls.
"""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
except Exception:  # pragma: no cover - fastapi optional at import time
    FastAPI = None  # type: ignore

ARTIFACTS = os.environ.get("PRAMAN_ARTIFACTS", "artifacts/praman-verifier")

_state: dict[str, Any] = {"model": None, "error": None}


def _get_model():
    if _state["model"] is None and _state["error"] is None:
        try:
            from .pipeline import Praman
            _state["model"] = Praman.load(ARTIFACTS, offline=True)
        except Exception as e:  # surfaced via /health/ready
            _state["error"] = str(e)
    return _state["model"]


class VerifyRequest(BaseModel):
    output_text: str = Field(..., description="generated output to verify")
    evidence: list[str] | str = Field(..., description="evidence passage(s)")
    alpha: float | None = Field(None, ge=0.0, le=1.0)
    policy: dict[str, Any] | None = None


if FastAPI is not None:
    app = FastAPI(title="PRAMAN", version="0.1.0",
                  description="On-prem grounded-claim verifier with provable risk control.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        m = _get_model()
        if m is None:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not loaded")
        return {"status": "ready", "artifacts": ARTIFACTS}

    @app.post("/verify")
    def verify(req: VerifyRequest) -> dict[str, Any]:
        m = _get_model()
        if m is None:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not loaded")
        return m.verify(output_text=req.output_text, evidence=req.evidence,
                        alpha=req.alpha, policy=req.policy)
