#!/usr/bin/env python
"""Indic groundedness slice (the differentiator). Honest construction.

High-quality Indic RAG-groundedness data is scarce (that scarcity is the brief's noted
contribution). With no human annotator available tonight, we use an EXISTING non-gated
Indic resource rather than fabricate hand-verification: IndicXNLI (Hindi), an NLI set, as
a groundedness PROXY (premise = evidence, hypothesis = claim; entailment => grounded,
neutral/contradiction => ungrounded). Scored with a multilingual verifier
(mDeBERTa-v3-base-mnli-xnli). We run the SAME calibrate -> CRC -> validate pipeline and
report detection + the realized-risk guarantee on Hindi. Limits are documented in the card.

Fallbacks if IndicXNLI is unavailable: XNLI (hi). Disk-guarded; one extra ~560MB model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from praman._env import assert_cpu_only, configure_threads
from praman.calibrate import Calibrator
from praman.eval import detection_metrics, guarantee_table
from praman.riskcontrol import RiskController
from praman.verifier import Verifier, VerifierConfig

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = [0.01, 0.05, 0.10]
MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


def load_indic(lang: str = "hi", n: int = 3000):
    """Return (premise, hypothesis, grounded) from IndicXNLI/XNLI Hindi."""
    from datasets import load_dataset
    last = None
    for spec in [("Divyanshu/indicxnli", lang), ("xnli", lang)]:
        try:
            ds = load_dataset(spec[0], spec[1], split="validation")
            prem = ds["premise"]; hyp = ds["hypothesis"]; lab = ds["label"]
            # XNLI label: 0=entailment, 1=neutral, 2=contradiction ; grounded = entailment
            grounded = [1 if int(l) == 0 else 0 for l in lab]
            idx = list(range(len(prem)))[:n]
            return ([prem[i] for i in idx], [hyp[i] for i in idx],
                    [grounded[i] for i in idx], spec[0])
        except Exception as e:  # noqa
            last = e
            continue
    raise RuntimeError(f"could not load Indic NLI data: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="indic")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    assert_cpu_only(); print(f"[env] threads={configure_threads()}")

    prem, hyp, grounded, src = load_indic(args.lang, args.n)
    grounded = np.asarray(grounded, dtype=int)
    print(f"[indic] {len(prem)} {args.lang} pairs from {src}; grounded rate {grounded.mean():.3f}")

    vcfg = VerifierConfig(hf_id=MODEL, kind="nli")
    print(f"[indic] scoring with {MODEL}")
    verifier = Verifier(vcfg, backend="torch")
    p, z = verifier.score_pairs(hyp, prem)  # premise=evidence, hypothesis=claim
    u = 1.0 - p; y = 1 - grounded  # ungrounded = positive

    # split: calib_temp / calib_conf / test
    rng = np.random.default_rng(1337)
    perm = rng.permutation(len(u))
    a, b = int(0.3 * len(u)), int(0.6 * len(u))
    ct, cc, te = perm[:a], perm[a:b], perm[b:]

    cal = Calibrator(method="temperature").fit(z[ct], grounded[ct])
    uc = 1.0 - cal.transform(z)
    det = detection_metrics(uc[te], y[te])
    risk = RiskController(method="crc").fit(uc[cc], y[cc], ALPHAS)
    gtab = guarantee_table(uc[cc][y[cc] == 1], uc[te], y[te], ALPHAS)

    print(f"[indic] detection AUROC={det.get('auroc')}  ECE {cal.metrics['before']['ece']:.4f}"
          f"->{cal.metrics['after']['ece']:.4f}")
    for r in gtab:
        flag = "OK" if r["realized_fnr"] <= r["alpha"] else "EXCEED"
        print(f"  a={r['alpha']:.2f} t={r['threshold']:.3f} fnr={r['realized_fnr']:.4f} "
              f"cov={r['coverage']:.3f} [{flag}]")

    summary = (f"Hindi groundedness proxy ({src}, {len(prem)} pairs, mDeBERTa-xnli): "
               f"detection AUROC {det.get('auroc'):.3f}; calibration ECE "
               f"{cal.metrics['before']['ece']:.3f}->{cal.metrics['after']['ece']:.3f}; "
               f"CRC realized FNR <= alpha holds at "
               f"{sum(r['realized_fnr'] <= r['alpha'] for r in gtab)}/{len(gtab)} alpha levels.")
    out = ROOT / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "indic.json").write_text(json.dumps({
        "languages": f"Hindi ({args.lang}); IndicXNLI covers 11 Indic languages",
        "source": src, "n": len(prem), "detection": det,
        "calibration": cal.metrics, "guarantee": gtab, "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"[done] {summary}")


if __name__ == "__main__":
    main()
