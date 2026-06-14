"""Throughput probe: torch ms/claim at seq 256 vs 512. Diagnoses the slow scoring."""
import os, time
os.environ.setdefault("HF_HOME", os.getcwd() + "/.cache/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from praman.verifier import Verifier, VerifierConfig
from praman.data import load_records, make_splits, records_to_arrays

recs, _ = load_records()
sp = make_splits(recs)
arr = records_to_arrays(sp.test[:16])
import torch
print("torch threads", torch.get_num_threads(), "mkl", torch.backends.mkl.is_available(),
      "mkldnn", torch.backends.mkldnn.is_available())
for ml in (256, 512):
    c = VerifierConfig.from_yaml(); c.max_length = ml
    v = Verifier(c, backend="torch")
    v.score_pairs(arr["claim"][:4], arr["doc"][:4])  # warmup
    t = time.time(); v.score_pairs(arr["claim"], arr["doc"]); dt = time.time() - t
    print(f"seq{ml}: {1000*dt/16:.0f} ms/claim ({dt:.1f}s for 16)")
