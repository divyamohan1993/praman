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
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover - fastapi optional at import time
    FastAPI = None  # type: ignore

ARTIFACTS = os.environ.get("PRAMAN_ARTIFACTS", "artifacts/praman-verifier")
# Optional shared-secret gate: if set, /verify requires header X-API-Key. Unset = open (demo).
API_KEY = os.environ.get("PRAMAN_API_KEY", "").strip()
# Explicit CORS origins (never "*"). Comma-separated; default to the public demo host.
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "PRAMAN_CORS_ORIGINS", "https://praman.dmj.one,https://dmj.one").split(",") if o.strip()]
MAX_OUTPUT_CHARS = int(os.environ.get("PRAMAN_MAX_OUTPUT_CHARS", "20000"))
MAX_EVIDENCE_ITEMS = int(os.environ.get("PRAMAN_MAX_EVIDENCE_ITEMS", "32"))
MAX_EVIDENCE_CHARS = int(os.environ.get("PRAMAN_MAX_EVIDENCE_CHARS", "60000"))

_state: dict[str, Any] = {"model": None, "error": None}


def _require_key(x_api_key: str | None = Header(default=None)) -> None:
    """Deny-by-default when an API key is configured. No-op in open demo mode."""
    if API_KEY and (x_api_key or "") != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _get_model():
    if _state["model"] is None and _state["error"] is None:
        try:
            from .pipeline import Praman
            _state["model"] = Praman.load(ARTIFACTS, offline=True)
        except Exception as e:  # surfaced via /health/ready
            _state["error"] = str(e)
    return _state["model"]


class VerifyRequest(BaseModel):
    output_text: str = Field(..., min_length=1, max_length=MAX_OUTPUT_CHARS,
                             description="generated output to verify")
    evidence: list[str] | str = Field(..., description="evidence passage(s)")
    alpha: float | None = Field(None, ge=0.0, le=1.0)
    policy: dict[str, Any] | None = None

    def evidence_list(self) -> list[str]:
        ev = [self.evidence] if isinstance(self.evidence, str) else list(self.evidence)
        if len(ev) > MAX_EVIDENCE_ITEMS:
            raise HTTPException(status_code=413, detail=f"too many evidence items (>{MAX_EVIDENCE_ITEMS})")
        if sum(len(e) for e in ev) > MAX_EVIDENCE_CHARS:
            raise HTTPException(status_code=413, detail="evidence too large")
        return ev


_SECURITY_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


if FastAPI is not None:
    app = FastAPI(title="PRAMAN", version="0.1.0",
                  description="On-prem grounded-claim verifier with provable risk control.")
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-API-Key"], allow_credentials=False)

    @app.middleware("http")
    async def _headers(request: "Request", call_next):
        resp = await call_next(request)
        for k, v in _SECURITY_HEADERS.items():
            resp.headers[k] = v
        return resp

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        m = _get_model()
        if m is None:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not loaded")
        return {"status": "ready", "artifacts": ARTIFACTS}

    @app.post("/verify", dependencies=[Depends(_require_key)])
    def verify(req: VerifyRequest) -> dict[str, Any]:
        m = _get_model()
        if m is None:
            raise HTTPException(status_code=503, detail=_state["error"] or "model not loaded")
        evidence = req.evidence_list()  # enforces item + size caps (413 on abuse)
        return m.verify(output_text=req.output_text, evidence=evidence,
                        alpha=req.alpha, policy=req.policy)
