"""Diagnose: do ONNX-int8 scores agree with torch scores? If not, the ONNX tokenizer
(which throws a fix_mistral_regex warning) is mis-tokenizing and the ONNX headline is an
artifact. Compares p_supported on the same claims."""
import os
os.environ.setdefault("HF_HOME", os.getcwd() + "/.cache/hf")
import numpy as np
from praman.verifier import Verifier, VerifierConfig
from praman.data import load_records, make_splits, records_to_arrays

recs, _ = load_records(); sp = make_splits(recs)
arr = records_to_arrays(sp.test[:32])
c, d = arr["claim"], arr["doc"]

cfg = VerifierConfig.from_yaml(); cfg.max_length = 320
vt = Verifier(cfg, backend="torch")
pt, zt = vt.score_pairs(c, d)
vo = Verifier(cfg, backend="onnx", model_dir=os.getcwd() + "/artifacts/verifier_onnx")
po, zo = vo.score_pairs(c, d)
diff = np.abs(pt - po)
print(f"[resaved-tokenizer] max|diff|={diff.max():.4f} mean={diff.mean():.4f} corr={np.corrcoef(pt,po)[0,1]:.4f}")

# decisive test: force the ONNX verifier to use the hf_id tokenizer (isolates tokenizer cause)
vo.tokenizer = vt.tokenizer
po2, zo2 = vo.score_pairs(c, d)
diff2 = np.abs(pt - po2)
corr2 = np.corrcoef(pt, po2)[0, 1]
print(f"[hf_id-tokenizer]   max|diff|={diff2.max():.4f} mean={diff2.mean():.4f} corr={corr2:.4f}")
print(f"torch p[:6]={np.round(pt[:6],3)}")
print(f"onnx2 p[:6]={np.round(po2[:6],3)}")
if corr2 >= 0.97:
    print("VERDICT: TOKENIZER WAS THE BUG -> load tokenizer from hf_id; ONNX int8 usable (fast+correct)")
else:
    print("VERDICT: quant/export issue -> fall back to torch for scientific runs")
