#!/usr/bin/env python
"""Export the verifier to ONNX, dynamic-quantize to int8 (AVX-512 VNNI on Genoa), and
benchmark fp32-torch vs fp32-onnx vs int8-onnx per-claim latency. CPU-only.

DoD: ONNX int8 latency benchmarked; fall back to fp32 ONNX/torch on op gaps (logged).
Writes artifacts/verifier_onnx/{model.onnx, model_int8.onnx, tokenizer..., verifier_meta.yaml}
and runs/<id>/latency.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from praman._env import assert_cpu_only, configure_threads
from praman.verifier import Verifier, VerifierConfig

ROOT = Path(__file__).resolve().parents[1]


def export(onnx_dir: Path, hf_id: str) -> dict:
    from transformers import AutoTokenizer
    onnx_dir.mkdir(parents=True, exist_ok=True)
    info = {"int8": False, "fp32": False}
    from optimum.onnxruntime import ORTModelForSequenceClassification
    print(f"[onnx] exporting {hf_id} -> fp32 ONNX")
    model = ORTModelForSequenceClassification.from_pretrained(hf_id, export=True)
    model.save_pretrained(onnx_dir)
    AutoTokenizer.from_pretrained(hf_id).save_pretrained(onnx_dir)
    info["fp32"] = True
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    (onnx_dir / "verifier_meta.yaml").write_text(yaml.safe_dump({"id2label": id2label, "hf_id": hf_id}),
                                                 encoding="utf-8")
    # dynamic int8 quantization (VNNI)
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        quantizer = ORTQuantizer.from_pretrained(onnx_dir)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        quantizer.quantize(save_dir=onnx_dir, quantization_config=qconfig)
        # optimum writes model_quantized.onnx; standardize the name our loader expects
        q = next((p for p in onnx_dir.glob("*quantized*.onnx")), None)
        if q:
            shutil.move(str(q), str(onnx_dir / "model_int8.onnx"))
            info["int8"] = True
            print("[onnx] int8 quantization OK -> model_int8.onnx")
    except Exception as e:  # op gaps etc. -> keep fp32, log it
        print(f"[onnx] int8 quantization failed ({e}); falling back to fp32 ONNX")
    return info


def bench(verifier: Verifier, claims, docs, repeats: int = 3) -> float:
    # warmup
    verifier.score_pairs(claims[:8], docs[:8])
    best = 1e9
    for _ in range(repeats):
        t0 = time.time()
        verifier.score_pairs(claims, docs)
        best = min(best, time.time() - t0)
    return 1000.0 * best / max(1, len(claims))  # ms per claim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="full")
    ap.add_argument("--n-bench", type=int, default=200)
    args = ap.parse_args()
    assert_cpu_only(); configure_threads()
    vcfg = VerifierConfig.from_yaml()
    onnx_dir = ROOT / "artifacts" / "verifier_onnx"
    info = export(onnx_dir, vcfg.hf_id)

    # benchmark sample from RAGTruth test
    from praman.data import load_records, make_splits, records_to_arrays
    recs, _ = load_records()
    sp = make_splits(recs)
    arr = records_to_arrays(sp.test[: args.n_bench] or sp.calib_conf[: args.n_bench])
    claims, docs = arr["claim"], arr["doc"]

    lat = {"model": vcfg.hf_id, "n_bench": len(claims), "export": info}
    print("[bench] torch fp32 ...")
    lat["fp32_ms_per_claim"] = bench(Verifier(vcfg, backend="torch"), claims, docs)
    if info.get("int8"):
        print("[bench] onnx int8 ...")
        lat["int8_ms_per_claim"] = bench(Verifier(vcfg, backend="onnx", model_dir=onnx_dir), claims, docs)
        lat["claims_per_sec"] = 1000.0 / lat["int8_ms_per_claim"]
        lat["speedup_vs_torch"] = lat["fp32_ms_per_claim"] / lat["int8_ms_per_claim"]
    else:
        lat["int8_ms_per_claim"] = None
        lat["claims_per_sec"] = 1000.0 / lat["fp32_ms_per_claim"]

    out = ROOT / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "latency.json").write_text(json.dumps(lat, indent=2), encoding="utf-8")
    print(f"[done] {json.dumps(lat, indent=2)}")


if __name__ == "__main__":
    main()
