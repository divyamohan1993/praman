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
    from fastapi.responses import HTMLResponse, JSONResponse
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
    # No scripts (script-src falls back to default-src 'none'); inline styles only for the
    # landing page; everything else locked down.
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
                               "base-uri 'none'; frame-ancestors 'none'",
}

_LANDING = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>PRAMAN</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#0a0e14;color:#cdd6e4;font:15px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;display:flex;min-height:100vh;align-items:center;justify-content:center;padding:2rem}
.card{max-width:760px;width:100%}
h1{font-size:2.6rem;margin:0;letter-spacing:.04em;color:#e8eef5}
.dev{color:#5b6b7f;font-weight:400;font-size:1.4rem}
.live{display:inline-block;margin:.4rem 0 1.2rem;font-size:.8rem;color:#3fb950}
.live b{display:inline-block;width:.55rem;height:.55rem;background:#3fb950;border-radius:50%;margin-right:.4rem;box-shadow:0 0 8px #3fb950}
p{color:#9aa7b8}.lead{color:#cdd6e4;font-size:1.05rem}
pre{background:#11161f;border:1px solid #1e2733;border-radius:10px;padding:1rem;overflow:auto;color:#a9c7e8;font-size:13px}
a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
.row{display:flex;gap:1.4rem;flex-wrap:wrap;margin-top:1.2rem;font-size:.92rem}
.note{margin-top:1.6rem;font-size:.82rem;color:#5b6b7f;border-top:1px solid #1e2733;padding-top:1rem}
</style></head><body><div class=card>
<h1>PRAMAN <span class=dev>प्रमाण</span></h1>
<div class=live><b></b>live</div>
<p class=lead>On-prem, CPU-only, Indic-first grounded-claim verifier. Give it a generated output plus its evidence; it returns, per claim, a calibrated grounded/ungrounded verdict, a <b>distribution-free bound</b> on the rate of auto-approving an ungrounded claim, an accept / escalate / reject decision, and an audit record.</p>
<pre>curl -X POST https://praman.dmj.one/verify \\
  -H 'Content-Type: application/json' \\
  -d '{"output_text":"The drug was approved in 2019.",
       "evidence":["The agency approved the drug in 2021."],
       "alpha":0.05}'</pre>
<div class=row>
<a href="/docs">&#8594; Interactive API (/docs)</a>
<a href="/health">&#8594; Health</a>
<a href="https://github.com/divyamohan1993/praman">&#8594; Source + report</a>
</div>
<p class=note>It bounds a <b>rate</b>, not each item: a marginal guarantee, not a per-item certificate. It checks faithfulness to the supplied evidence, not truth in the world. It does not remove the human on catastrophic or irreversible decisions. This managed-cloud endpoint is a demo; the product runs air-gapped on the operator's own hardware.</p>
</div></body></html>"""


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

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return _LANDING

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
