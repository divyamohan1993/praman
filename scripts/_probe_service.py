"""Prove the FastAPI app serves /health + /verify in-process (TestClient, no uvicorn/network)."""
import os, json
os.environ.setdefault("HF_HOME", os.getcwd() + "/.cache/hf")
os.environ.setdefault("PRAMAN_ARTIFACTS", os.getcwd() + "/artifacts/praman-verifier")
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from fastapi.testclient import TestClient
from praman.service import app

c = TestClient(app)
print("GET /health ->", c.get("/health").status_code, c.get("/health").json())
r = c.get("/health/ready")
print("GET /health/ready ->", r.status_code, r.json())
body = {"output_text": "The drug was approved in 2019.",
        "evidence": ["A regulatory review concluded the agency approved the drug in 2021."],
        "alpha": 0.05, "policy": {"class": "clinical", "severity": "high"}}
v = c.post("/verify", json=body)
print("POST /verify ->", v.status_code)
out = v.json()
print(json.dumps({"output_decision": out.get("output_decision"),
                  "claims": [{"text": cl["text"][:40], "p_grounded": cl["p_grounded"],
                              "decision": cl["decision"]} for cl in out.get("claims", [])],
                  "alpha": out.get("alpha")}, indent=2))
