#!/usr/bin/env python
"""Score RAGTruth claims with the verifier and CACHE the scores (the expensive step).

Runs the NLI verifier over each split's (claim, evidence) pairs once, saving per-split
arrays (u = 1 - p_supported, z = supported logit, y = ungrounded label, dataset) to an
.npz. Downstream calibration / conformal / eval re-use the cache, so we never re-run the
model while iterating. CPU-only; threads set by praman._env.

Usage:
  python scripts/20_score.py --run-id slice --max-train 3000 --max-test 1500
  python scripts/20_score.py --run-id full  --max-train 100000 --max-test 100000
  python scripts/20_score.py --run-id ood   --ood-task Data2txt --max-ood 4000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from praman._env import assert_cpu_only, configure_threads
from praman.data import load_records, make_splits, records_to_arrays
from praman.eval import detection_metrics
from praman.verifier import Verifier, VerifierConfig

ROOT = Path(__file__).resolve().parents[1]


def score_split(verifier: Verifier, records) -> dict[str, np.ndarray]:
    arr = records_to_arrays(records)
    t0 = time.time()
    p, z = verifier.score_pairs(arr["claim"], arr["doc"])
    dt = time.time() - t0
    n = max(1, len(p))
    print(f"  scored {len(p)} pairs in {dt:.1f}s ({1000*dt/n:.1f} ms/claim)")
    claim_len = np.array([len(c) for c in arr["claim"]], dtype=np.int32)
    doc_len = np.array([len(d) for d in arr["doc"]], dtype=np.int32)
    return {
        "u": (1.0 - p).astype(np.float32),
        "p": p.astype(np.float32),
        "z": z.astype(np.float32),
        "y": np.asarray(arr["ungrounded"], dtype=np.int8),
        "grounded": np.asarray(arr["grounded"], dtype=np.int8),
        "claim_len": claim_len,
        "doc_len": doc_len,
    }, list(arr["dataset"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="slice")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--max-ood", type=int, default=None)
    ap.add_argument("--ood-task", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    ap.add_argument("--onnx-dir", default="artifacts/verifier_onnx")
    ap.add_argument("--max-length", type=int, default=None, help="override seq length (speed)")
    args = ap.parse_args()

    assert_cpu_only()
    print(f"[env] threads={configure_threads()}")
    recs, report = load_records()
    print(f"[data] {report['n_records']} claims; ungrounded by task {report['ungrounded_by_task']}")

    cfg = None
    if any(v is not None for v in (args.max_train, args.max_test, args.max_ood)):
        from praman.data import _load_cfg
        cfg = _load_cfg("data.yaml")
        lim = cfg.setdefault("limits", {})
        if args.max_train is not None: lim["max_train"] = args.max_train
        if args.max_test is not None: lim["max_test"] = args.max_test
        if args.max_ood is not None: lim["max_ood"] = args.max_ood

    splits = make_splits(recs, cfg=cfg, ood_task=args.ood_task)
    print(f"[splits] { {k: v['n'] for k, v in splits.summary().items()} }")

    vcfg = VerifierConfig.from_yaml()
    if args.model:
        vcfg.hf_id = args.model
    if args.max_length:
        vcfg.max_length = args.max_length
    model_dir = None
    if args.backend == "onnx":
        model_dir = ROOT / args.onnx_dir
    print(f"[verifier] loading {vcfg.hf_id} backend={args.backend} maxlen={vcfg.max_length} (CPU)")
    verifier = Verifier(vcfg, backend=args.backend, model_dir=model_dir)
    print(f"[verifier] id2label={verifier._id2label} support_idx={verifier._support_idx}")

    out_dir = ROOT / "runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    save: dict[str, np.ndarray] = {}
    meta: dict = {"model": vcfg.hf_id, "datasets": {}, "n": {}}
    for name in ("calib_temp", "calib_conf", "test", "ood"):
        split_recs = getattr(splits, name)
        if not split_recs:
            continue
        print(f"[score] {name}: {len(split_recs)} claims")
        s, ds = score_split(verifier, split_recs)
        for k, v in s.items():
            save[f"{name}__{k}"] = v
        meta["datasets"][name] = ds
        meta["n"][name] = len(ds)
        if name in ("test", "ood"):
            dm = detection_metrics(s["u"], s["y"])
            print(f"  [detection] {name}: AUROC={dm.get('auroc')} AUPRC={dm.get('auprc')} "
                  f"base_rate={dm.get('base_rate')}")

    # numeric-only npz (no object arrays -> loads without allow_pickle); strings in meta.json
    np.savez(out_dir / "scores.npz", **save)
    import json
    (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[done] saved {out_dir/'scores.npz'} + meta.json")


if __name__ == "__main__":
    main()
