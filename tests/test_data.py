"""Data parsing: span annotations -> sentence-level groundedness labels."""
from __future__ import annotations

import json
from pathlib import Path

from praman.data import Record, make_splits, parse_ragtruth, records_to_arrays


def _write_fixture(d: Path):
    src = {"source_id": "1", "task_type": "QA",
           "source_info": "The agency approved the drug in 2021.", "prompt": "p"}
    (d / "source_info.jsonl").write_text(json.dumps(src) + "\n", encoding="utf-8")
    resp_text = "The drug was approved in 2021. The drug cures everything instantly."
    # hallucination span covers the 2nd sentence ("cures everything instantly")
    start = resp_text.index("The drug cures")
    resp = {"source_id": "1", "response": resp_text, "split": "test",
            "labels": [{"start": start, "end": len(resp_text), "text": "...",
                        "label_type": "Baseless Info"}]}
    (d / "response.jsonl").write_text(json.dumps(resp) + "\n", encoding="utf-8")


def test_parse_labels_from_spans(tmp_path):
    _write_fixture(tmp_path)
    recs = parse_ragtruth(tmp_path)
    assert len(recs) == 2
    by_claim = {r.claim: r for r in recs}
    grounded = next(r for r in recs if "approved in 2021" in r.claim)
    hallu = next(r for r in recs if "cures everything" in r.claim)
    assert grounded.grounded == 1
    assert hallu.grounded == 0
    assert hallu.ungrounded == 1
    assert all(r.dataset == "QA" and r.split == "test" for r in recs)


def test_records_to_arrays_shapes():
    recs = [Record("doc", "claim a", 1, "QA", "train"),
            Record("doc", "claim b", 0, "QA", "train")]
    arr = records_to_arrays(recs)
    assert arr["grounded"] == [1, 0]
    assert arr["ungrounded"] == [0, 1]
    assert len(arr["claim"]) == len(arr["doc"]) == 2


def test_make_splits_disjoint_calib_from_test(tmp_path):
    # build synthetic records with official train/test, ensure calib carved from train only
    recs = [Record("d", f"c{i}", i % 2, "QA", "train") for i in range(100)]
    recs += [Record("d", f"t{i}", i % 2, "QA", "test") for i in range(40)]
    sp = make_splits(recs)
    train_claims = {r.claim for r in sp.train + sp.calib_temp + sp.calib_conf}
    test_claims = {r.claim for r in sp.test}
    assert train_claims.isdisjoint(test_claims)
    assert len(sp.test) == 40
