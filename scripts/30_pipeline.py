#!/usr/bin/env python
"""Analysis on cached scores: calibrate -> conformal risk control -> validate the guarantee.

Reads runs/<id>/scores.npz (from 20_score.py), fits temperature calibration on calib_temp,
fits CRC thresholds on calib_conf, validates realized FNR <= alpha on the held-out test
split (single split + bootstrap), computes detection + calibration metrics, and writes
artifacts (calibration.json, riskcontrol.json) + metrics.json. Cheap; iterate freely.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from praman.calibrate import Calibrator
from praman.eval import (ablation_thresholds, apply_threshold, bootstrap_table,
                         detection_metrics, guarantee_table)
from praman.riskcontrol import RiskController

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [0.01, 0.05, 0.10]


def load_scores(run_id: str) -> dict[str, np.ndarray]:
    # numeric-only npz written by 20_score.py; no pickle needed.
    d = np.load(ROOT / "runs" / run_id / "scores.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def get(split: str, key: str, S: dict) -> np.ndarray:
    return S[f"{split}__{key}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="slice")
    ap.add_argument("--calib-method", default="temperature")
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    S = load_scores(args.run_id)
    out_dir = ROOT / "runs" / args.run_id
    model_id = json.loads((out_dir / "meta.json").read_text(encoding="utf-8")).get("model", "?")

    # --- calibration: fit on calib_temp (disjoint from calib_conf + test) ---
    z_t = get("calib_temp", "z", S); g_t = get("calib_temp", "grounded", S)
    cal = Calibrator(method=args.calib_method).fit(z_t, g_t)
    print(f"[calib] T={cal.temperature:.3f}  ECE {cal.metrics['before']['ece']:.4f} -> "
          f"{cal.metrics['after']['ece']:.4f}  Brier {cal.metrics['before']['brier']:.4f} -> "
          f"{cal.metrics['after']['brier']:.4f}")

    # calibrated detector score u = 1 - p_grounded on each split
    def u_of(split: str) -> np.ndarray:
        return 1.0 - cal.transform(get(split, "z", S))

    u_conf = u_of("calib_conf"); y_conf = get("calib_conf", "y", S).astype(int)
    u_test = u_of("test"); y_test = get("test", "y", S).astype(int)

    # --- detection quality on held-out test ---
    det = detection_metrics(u_test, y_test)
    print(f"[detect] test AUROC={det.get('auroc'):.4f} AUPRC={det.get('auprc'):.4f} "
          f"F1@.5={det.get('f1@0.5'):.4f} base_rate={det.get('base_rate'):.4f}")

    # --- conformal risk control: fit thresholds on calib_conf positives ---
    risk = RiskController(method="crc").fit(u_conf, y_conf, ALPHAS)
    u_conf_pos = u_conf[y_conf == 1]

    # --- guarantee validation on test (single split) ---
    gtab = guarantee_table(u_conf_pos, u_test, y_test, ALPHAS, method="crc")
    print("[guarantee] alpha -> threshold | realized_fnr | coverage | contamination")
    for r in gtab:
        flag = "OK" if r["realized_fnr"] <= r["alpha"] else "EXCEED"
        print(f"  a={r['alpha']:.2f}  t={r['threshold']:.3f}  fnr={r['realized_fnr']:.4f}  "
              f"cov={r['coverage']:.3f}  cont={r['contamination']:.4f}  [{flag}]")

    # --- bootstrap honesty: combine held-out conf+test as the resample pool ---
    u_pool = np.concatenate([u_conf, u_test]); y_pool = np.concatenate([y_conf, y_test])
    btab = bootstrap_table(u_pool, y_pool, ALPHAS, n_boot=args.n_boot, method="crc")
    print("[bootstrap] alpha -> mean_realized_fnr | frac_exceed_alpha | mean_coverage")
    for r in btab:
        print(f"  a={r['alpha']:.2f}  mean_fnr={r['mean_realized_fnr']:.4f}  "
              f"frac_exceed={r['frac_exceed_alpha']:.3f}  cov={r['mean_coverage']:.3f}")

    # --- ablations at alpha=0.05 ---
    abl = ablation_thresholds(u_conf, y_conf, 0.05)
    abl_eval = {name: apply_threshold(u_test, y_test, t) for name, t in abl.items()}

    # --- persist artifacts + metrics ---
    cal.save(out_dir / "calibration.json")
    risk.save(out_dir / "riskcontrol.json")
    metrics = {
        "model": model_id,
        "n": {k: int(get(k, "y", S).shape[0]) for k in ("calib_temp", "calib_conf", "test")
              if f"{k}__y" in S},
        "calibration": cal.metrics,
        "detection_test": det,
        "guarantee_single_split": gtab,
        "guarantee_bootstrap": btab,
        "ablation_alpha0.05": abl_eval,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_dir/'metrics.json'}, calibration.json, riskcontrol.json")


if __name__ == "__main__":
    main()
