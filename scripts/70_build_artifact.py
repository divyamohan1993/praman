#!/usr/bin/env python
"""Assemble the loadable, offline PRAMAN artifact dir from a finished run.

artifacts/praman-verifier/
  verifier/            <- ONNX int8 (preferred) or torch weights + tokenizer + meta
  calibration.json     <- from runs/<id>
  riskcontrol.json     <- from runs/<id>
  policy.yaml          <- configs/policy.yaml
  manifest.json        <- versions + verifier_config + id2label

`Praman.load("artifacts/praman-verifier")` then runs fully offline.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full")
    ap.add_argument("--out", default="artifacts/praman-verifier")
    ap.add_argument("--verifier-src", default="artifacts/verifier_onnx",
                    help="ONNX dir (preferred) or an HF snapshot dir")
    args = ap.parse_args()

    run = ROOT / "runs" / args.run_id
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    from praman.verifier import VerifierConfig
    vcfg = VerifierConfig.from_yaml()

    # verifier weights
    src = ROOT / args.verifier_src
    dst = out / "verifier"
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        meta_path = src / "verifier_meta.yaml"
        id2label = yaml.safe_load(meta_path.read_text(encoding="utf-8"))["id2label"] if meta_path.exists() else {}
    else:
        print(f"[warn] verifier src {src} missing; manifest will reference hf_id {vcfg.hf_id}")
        id2label = {}

    for name in ("calibration.json", "riskcontrol.json"):
        s = run / name
        if s.exists():
            shutil.copy(s, out / name)
        else:
            raise FileNotFoundError(f"missing {s}; run scripts/30_pipeline.py --run-id {args.run_id} first")

    shutil.copy(ROOT / "configs" / "policy.yaml", out / "policy.yaml")

    manifest = {
        "name": "praman-verifier",
        "version": "0.1.0",
        "verifier_config": vcfg.__dict__,
        "id2label": {str(k): v for k, v in id2label.items()},
        "versions": {"model": vcfg.hf_id, "calib": "temperature", "risk": "crc"},
        "backend": "onnx" if (dst / "model_int8.onnx").exists() or (dst / "model.onnx").exists() else "torch",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] artifact at {out}  (backend={manifest['backend']})")
    print(f"       files: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
