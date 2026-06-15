#!/usr/bin/env python
"""Assemble the report: fill {{TOKENS}} in REPORT.md / model-card.md with real numbers
from runs/<id>/{metrics,robustness,indic,latency}.json, and render the plots.

Token replacement is explicit (we know every token we authored). Unfilled tokens are
listed at the end so nothing silently ships as a placeholder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(run_id: str, name: str):
    p = ROOT / "runs" / run_id / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _md_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def build_tokens(run_id: str) -> dict[str, str]:
    m = _load(run_id, "metrics.json") or {}
    rob = _load(run_id, "robustness.json")
    indic = _load(run_id, "indic.json")
    lat = _load(run_id, "latency.json")
    T: dict[str, str] = {}

    det = m.get("detection_test", {})
    T["DET_AUROC_RAGTRUTH"] = f"{det.get('auroc', float('nan')):.3f}"
    T["DET_AUPRC"] = f"{det.get('auprc', float('nan')):.3f}"
    T["DET_F1"] = f"{det.get('f1@0.5', float('nan')):.3f}"
    T["BASELINE_TABLE"] = (f"DeBERTa-v3-base NLI verifier on RAGTruth test: AUROC "
                           f"{T['DET_AUROC_RAGTRUTH']}, AUPRC {T['DET_AUPRC']}, "
                           f"base ungrounded rate {det.get('base_rate', float('nan')):.3f}. "
                           f"Reference small verifiers (HHEM/MiniCheck) report AUROC in the "
                           f"~0.75-0.85 band on RAGTruth-style grounding tasks.")

    cal = m.get("calibration", {})
    T["ECE_BEFORE"] = f"{cal.get('before', {}).get('ece', float('nan')):.4f}"
    T["ECE_AFTER"] = f"{cal.get('after', {}).get('ece', float('nan')):.4f}"
    T["BRIER_BEFORE"] = f"{cal.get('before', {}).get('brier', float('nan')):.4f}"
    T["BRIER_AFTER"] = f"{cal.get('after', {}).get('brier', float('nan')):.4f}"

    g = m.get("guarantee_single_split", [])
    b = {r["alpha"]: r for r in m.get("guarantee_bootstrap", [])}
    if g:
        rows = [[f"{r['alpha']:.2f}", f"{r['threshold']:.3f}", f"{r['realized_fnr']:.4f}",
                 f"{b.get(r['alpha'], {}).get('mean_realized_fnr', float('nan')):.4f}",
                 "OK" if r["realized_fnr"] <= r["alpha"] else "exceeded"] for r in g]
        T["REALIZED_RISK_TABLE"] = _md_table(
            rows, ["alpha", "CRC threshold", "realized FNR (test)", "mean FNR (bootstrap)", "<= alpha?"])
        crows = [[f"{r['alpha']:.2f}", f"{r['coverage']:.3f}", f"{r['contamination']:.4f}"] for r in g]
        T["COVERAGE_TABLE"] = _md_table(crows, ["alpha", "coverage (auto-approval)", "contamination of approvals"])
        T["SPLIT_EXCEED_FRAC"] = ", ".join(
            f"alpha={r['alpha']:.2f}: {b.get(r['alpha'], {}).get('frac_exceed_alpha', float('nan')):.3f}"
            for r in g)
    if rob:
        prows = []
        for r in rob.get("pooled", []):
            idt = r["in_domain_test"]["realized_fnr"]; ood = r.get("ood_pooled", {}).get("realized_fnr")
            prows.append([f"{r['alpha']:.2f}", f"{idt:.4f}", f"{ood:.4f}" if ood is not None else "n/a"])
        T["OOD_DEGRADATION_TABLE"] = _md_table(prows, ["alpha", "in-domain test FNR", "OOD FNR (pooled CRC)"])
        ne = {r["alpha"]: r for r in rob.get("nonexchangeable", [])}
        nrows = [[f"{r['alpha']:.2f}", f"{r.get('ood_pooled', {}).get('realized_fnr', float('nan')):.4f}",
                  f"{ne.get(r['alpha'], {}).get('realized_fnr', float('nan')):.4f}"]
                 for r in rob.get("pooled", [])]
        T["OOD_RECOVERY"] = _md_table(nrows, ["alpha", "OOD FNR (exchangeable)", "OOD FNR (non-exchangeable NN)"])
    if indic:
        T["INDIC_RESULT"] = indic.get("summary", str(indic))
        T["INDIC_LANGS"] = indic.get("languages", "Hindi (machine-built slice)")
    if lat:
        T["LATENCY_INT8_MS"] = f"{lat.get('int8_ms_per_claim', float('nan')):.1f}"
        T["LATENCY_FP32_MS"] = f"{lat.get('fp32_ms_per_claim', float('nan')):.1f}"
        T["THROUGHPUT"] = f"{lat.get('claims_per_sec', float('nan')):.1f}"
    return T


def render_plots(run_id: str) -> None:
    from praman.plots import desired_vs_realized, reliability_diagram, risk_coverage
    out = ROOT / "runs" / run_id / "plots"
    m = _load(run_id, "metrics.json") or {}
    cal = m.get("calibration", {})
    rel = cal.get("reliability_after") or cal.get("after", {}).get("reliability_after")
    if rel:
        reliability_diagram(rel, out / "reliability.png")
    g = m.get("guarantee_single_split")
    if g:
        desired_vs_realized(g, out / "desired_vs_realized.png")
    print(f"[plots] wrote to {out}")


def fill_file(path: Path, tokens: dict[str, str]) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    for k, v in tokens.items():
        text = text.replace("{{" + k + "}}", v)
    path.write_text(text, encoding="utf-8")
    import re
    return sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full")
    args = ap.parse_args()
    tokens = build_tokens(args.run_id)
    (ROOT / "runs" / args.run_id / "tokens.json").write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    try:
        render_plots(args.run_id)
    except Exception as e:
        print(f"[plots] skipped: {e}")
    remaining = set()
    for f in (ROOT / "REPORT.md", ROOT / "docs" / "model-card.md"):
        remaining |= set(fill_file(f, tokens))
    print(f"[report] filled {len(tokens)} tokens into REPORT.md + model-card.md")
    if remaining:
        print(f"[report] STILL UNFILLED (need data): {sorted(remaining)}")


if __name__ == "__main__":
    main()
