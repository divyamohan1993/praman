"""``Praman.verify()`` — ties the pipeline together. The public, frozen API (brief 11).

    from praman import Praman
    p = Praman.load("artifacts/praman-verifier")   # local, fully offline
    out = p.verify(output_text=..., evidence=[...], alpha=0.05,
                   policy={"class": "clinical", "severity": "high"})

Air-gapped: load reads a LOCAL artifact dir; verify() makes no network calls. Prove it
with praman._env.airgap() (the pytest air-gap test runs verify() inside that context).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import yaml

from .audit import audit as build_audit
from ._env import configure_threads, set_offline
from .calibrate import Calibrator
from .decide import aggregate_output, decide_claim
from .decompose import decompose
from .riskcontrol import RiskController
from .verifier import Verifier, VerifierConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_EVIDENCE_SNIPPET = 400  # chars of the supporting passage stored in the audit record


def _load_policy() -> dict[str, Any]:
    with open(CONFIG_DIR / "policy.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Praman:
    def __init__(self, verifier: Verifier, calibrator: Calibrator, risk: RiskController,
                 policy: dict[str, Any] | None = None, versions: dict[str, str] | None = None):
        self.verifier = verifier
        self.calibrator = calibrator
        self.risk = risk
        self.policy_cfg = policy or _load_policy()
        self.versions = versions or {}
        self._conditional_groups = bool(risk.group_thresholds)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path, offline: bool = True, backend: str = "auto") -> "Praman":
        """Load a saved artifact directory, fully offline by default."""
        if offline:
            set_offline()
        configure_threads()
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        vcfg = VerifierConfig(**manifest["verifier_config"])
        model_dir = path / "verifier"
        if backend == "auto":
            has_onnx = any((model_dir / c).exists() for c in ("model_int8.onnx", "model.onnx"))
            backend = "onnx" if has_onnx else "torch"
        verifier = Verifier(vcfg, backend=backend, model_dir=model_dir)
        calibrator = Calibrator.load(path / "calibration.json")
        risk = RiskController.load(path / "riskcontrol.json")
        policy = None
        if (path / "policy.yaml").exists():
            policy = yaml.safe_load((path / "policy.yaml").read_text(encoding="utf-8"))
        return cls(verifier, calibrator, risk, policy, manifest.get("versions", {}))

    def save(self, path: str | Path, verifier_src: str | Path | None = None) -> None:
        """Persist calib + risk + policy + manifest. Verifier weights copied from verifier_src."""
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        self.calibrator.save(path / "calibration.json")
        self.risk.save(path / "riskcontrol.json")
        (path / "policy.yaml").write_text(yaml.safe_dump(self.policy_cfg), encoding="utf-8")
        manifest = {
            "verifier_config": self.verifier.cfg.__dict__,
            "versions": self.versions,
            "id2label": self.verifier._id2label,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if verifier_src is not None:
            import shutil
            dst = path / "verifier"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(verifier_src, dst)

    # ------------------------------------------------------------------ #
    def _resolve_policy(self, policy: dict[str, Any] | None, alpha: float | None
                        ) -> tuple[float, str, str]:
        cls_name = (policy or {}).get("class", self.policy_cfg["default"]["class"])
        classes = self.policy_cfg.get("classes", {})
        base = classes.get(cls_name, self.policy_cfg["default"])
        a = alpha if alpha is not None else (policy or {}).get("alpha", base.get("alpha", 0.05))
        severity = (policy or {}).get("severity", base.get("severity", "normal"))
        return float(a), cls_name, severity

    def verify(self, output_text: str, evidence: str | Sequence[str],
               alpha: float | None = None, policy: dict[str, Any] | None = None,
               ts: float | None = None) -> dict[str, Any]:
        """Verify output_text against evidence. Returns the brief-11 schema dict."""
        a, cls_name, severity = self._resolve_policy(policy, alpha)
        passages = [evidence] if isinstance(evidence, str) else list(evidence)
        group = cls_name if (self._conditional_groups and cls_name in
                             {g for g in self.risk.group_thresholds}) else None
        t_approve = self.risk.threshold(a, group)

        claims = decompose(output_text)
        versions = {"model": self.versions.get("model", self.verifier.cfg.hf_id),
                    "calib": self.versions.get("calib", self.calibrator.method),
                    "risk": self.versions.get("risk", self.risk.method)}
        policy_rec = {"alpha": a, "delta": self.risk.delta, "class": cls_name, "method": self.risk.method}

        claim_out, audit_out, decisions = [], [], []
        now = ts if ts is not None else time.time()
        for c in claims:
            p_raw, z, best = self.verifier.score_multi(c, passages)
            p_grounded = float(self.calibrator.transform([z])[0])
            span = passages[best][:_EVIDENCE_SNIPPET] if passages else ""
            d = decide_claim(p_grounded, t_approve, severity=severity).decision
            decisions.append(d)
            claim_out.append({"text": c, "p_grounded": round(p_grounded, 4),
                              "evidence_span": span, "decision": d})
            audit_out.append(build_audit(c, span, p_grounded, d, policy_rec, versions, ts=now))

        return {
            "claims": claim_out,
            "output_decision": aggregate_output(decisions, severity=severity),
            "alpha": a,
            "policy": policy_rec,
            "audit": audit_out,
        }
