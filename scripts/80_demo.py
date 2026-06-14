#!/usr/bin/env python
"""End-to-end demo of the frozen public API (brief 11), fully offline.

    python scripts/80_demo.py --artifacts artifacts/praman-verifier

Mirrors the brief's worked example (a drug approved in 2019 vs evidence saying 2021) and
prints the verify() result + writes an audit JSONL. Used as the DoD "verify() runs end-to-end
on CPU with networking disabled" check (wrapped in airgap()).
"""
from __future__ import annotations

import argparse

import orjson

from praman import Praman, airgap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/praman-verifier")
    ap.add_argument("--audit-out", default="runs/demo_audit.jsonl")
    args = ap.parse_args()

    # prove the inference path needs no network: load + verify entirely inside airgap()
    with airgap():
        p = Praman.load(args.artifacts, offline=True)
        out = p.verify(
            output_text="The drug was approved in 2019 and reduces risk by 40%.",
            evidence=["A regulatory review concluded the agency approved the drug in 2021.",
                      "The trial reported a relative risk reduction, magnitude not stated."],
            alpha=0.05,
            policy={"class": "clinical", "severity": "high"},
        )
    print(orjson.dumps(out, option=orjson.OPT_INDENT_2).decode())

    from praman.audit import write_jsonl
    n = write_jsonl(out["audit"], args.audit_out)
    print(f"\n[audit] wrote {n} records -> {args.audit_out}")
    print(f"[decision] output_decision = {out['output_decision']} (high-severity clinical class)")


if __name__ == "__main__":
    main()
