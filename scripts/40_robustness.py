#!/usr/bin/env python
"""Robustness + OOD honesty: where the guarantee holds and where it breaks.

Consumes an OOD-configured scoring run (runs/<id>/scores.npz from
`20_score.py --ood-task Data2txt ...`), which carries in-domain calib_temp/calib_conf/test
(QA+Summary) and an held-out `ood` split (Data2txt). Reports, per alpha:

  * pooled (exchangeable) CRC: realized FNR on in-domain test (should hold ~<=alpha) vs on
    the OOD slice (the HONEST degradation figure, typically > alpha);
  * conditional / Mondrian per-task thresholds (in-domain per-group validity);
  * non-exchangeable nearest-neighbour reweighting on the OOD slice (partial recovery).

The point is honesty: we publish the degradation, not hide it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from praman.calibrate import Calibrator
from praman.riskcontrol import RiskController, crc_threshold, nonexchangeable_threshold

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [0.01, 0.05, 0.10]


def _load(run_id: str):
    d = np.load(ROOT / "runs" / run_id / "scores.npz", allow_pickle=False)
    meta = json.loads((ROOT / "runs" / run_id / "meta.json").read_text(encoding="utf-8"))
    return {k: d[k] for k in d.files}, meta


def _feats(S, split):
    """Standardizable feature vector for NN reweighting: [u, log claim_len, log doc_len]."""
    u = S[f"{split}__u"].astype(float)
    cl = np.log1p(S[f"{split}__claim_len"].astype(float))
    dl = np.log1p(S[f"{split}__doc_len"].astype(float))
    return np.stack([u, cl, dl], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="ood")
    args = ap.parse_args()
    S, meta = _load(args.run_id)

    cal = Calibrator(method="temperature").fit(S["calib_temp__z"], S["calib_temp__grounded"])
    def u_of(split): return 1.0 - cal.transform(S[f"{split}__z"])

    u_conf, y_conf = u_of("calib_conf"), S["calib_conf__y"].astype(int)
    u_test, y_test = u_of("test"), S["test__y"].astype(int)
    has_ood = "ood__u" in S
    if has_ood:
        u_ood, y_ood = u_of("ood"), S["ood__y"].astype(int)

    u_conf_pos = u_conf[y_conf == 1]
    report = {"run_id": args.run_id, "ood_task": meta.get("datasets", {}).get("ood", ["?"])[:1],
              "pooled": [], "conditional": [], "nonexchangeable": []}

    # standardize features on calib for NN
    if has_ood:
        f_conf = _feats(S, "calib_conf"); mu = f_conf.mean(0); sd = f_conf.std(0) + 1e-9
        f_conf_pos = ((f_conf - mu) / sd)[y_conf == 1]
        f_ood = (_feats(S, "ood") - mu) / sd

    def realized(u, y, t):
        approved = u < t; pos = y == 1
        return {"realized_fnr": float(np.mean(approved[pos])) if pos.any() else 0.0,
                "coverage": float(np.mean(approved))}

    print("alpha | in-domain test FNR | OOD FNR (pooled) | OOD FNR (non-exch)")
    for a in ALPHAS:
        t = crc_threshold(u_conf_pos, a)
        row = {"alpha": a, "threshold": round(t, 4), "in_domain_test": realized(u_test, y_test, t)}
        if has_ood:
            row["ood_pooled"] = realized(u_ood, y_ood, t)
        report["pooled"].append(row)

        # --- non-exchangeable NN-reweighted thresholds on OOD ---
        ne = None
        if has_ood:
            # subsample OOD for the O(n_ood * n_calib) per-point computation
            idx = np.arange(len(u_ood))
            if len(idx) > 1500:
                idx = np.random.default_rng(0).choice(idx, 1500, replace=False)
            approved, npos, nmiss = 0, 0, 0
            for i in idx:
                t_i = nonexchangeable_threshold(u_conf_pos, f_conf_pos, f_ood[i], a, k=50)
                appr = u_ood[i] < t_i
                approved += int(appr)
                if y_ood[i] == 1:
                    npos += 1; nmiss += int(appr)
            ne = {"realized_fnr": (nmiss / npos) if npos else 0.0, "coverage": approved / len(idx)}
            report["nonexchangeable"].append({"alpha": a, **{k: round(v, 4) for k, v in ne.items()}})

        idt = row["in_domain_test"]["realized_fnr"]
        ood_p = row.get("ood_pooled", {}).get("realized_fnr")
        ood_ne = ne["realized_fnr"] if ne else None
        print(f"  {a:.2f} | {idt:.4f} | {ood_p if ood_p is None else round(ood_p,4)} | "
              f"{ood_ne if ood_ne is None else round(ood_ne,4)}")

    # --- conditional / Mondrian per task on in-domain test ---
    ds_conf = np.array(meta["datasets"]["calib_conf"])
    ds_test = np.array(meta["datasets"]["test"])
    risk = RiskController(method="crc").fit(u_conf, y_conf, ALPHAS, groups=ds_conf)
    for a in ALPHAS:
        per_group = {}
        for g in np.unique(ds_test):
            m = ds_test == g
            t_g = risk.threshold(a, str(g))
            per_group[str(g)] = realized(u_test[m], y_test[m], t_g)
        report["conditional"].append({"alpha": a, "per_task": per_group})

    out = ROOT / "runs" / args.run_id / "robustness.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
