"""Data loading + split discipline for PRAMAN.

PRIMARY source: RAGTruth (Niu et al., ACL 2024), the brief's primary. Word/span-level
hallucination annotations over RAG outputs across three task types (QA, Summary,
Data2txt). Non-gated; downloaded from GitHub at build time (scripts/10_data.py).

We derive CLAIM-level groundedness labels from the real span annotations:
  * split each model response into sentences (with char offsets),
  * a sentence is UNGROUNDED (grounded=0) iff it overlaps any annotated hallucination span,
  * evidence (doc) = the source reference the response was supposed to be grounded in.

This yields, per record:  {"doc": str, "claim": str, "grounded": 0/1, "dataset": task_type,
"split": train|test}. RAGTruth ships an official train/test split (we honor it for the
held-out guarantee). The three task types give a natural leave-one-domain-out OOD axis.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ragtruth"
RAGTRUTH_BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _load_cfg(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Record:
    doc: str
    claim: str
    grounded: int           # 1 = supported by doc, 0 = ungrounded/hallucinated
    dataset: str            # RAGTruth task_type: QA | Summary | Data2txt
    split: str = "train"    # official RAGTruth split: train | test

    @property
    def ungrounded(self) -> int:
        return 1 - self.grounded


@dataclass
class Splits:
    train: list[Record] = field(default_factory=list)        # verifier-fit reserve (unused for pretrained)
    calib_temp: list[Record] = field(default_factory=list)   # temperature/isotonic
    calib_conf: list[Record] = field(default_factory=list)   # conformal threshold selection
    test: list[Record] = field(default_factory=list)         # held-out guarantee validation
    ood: list[Record] = field(default_factory=list)          # leave-one-domain-out OOD slice

    def summary(self) -> dict[str, Any]:
        def stat(rs: list[Record]) -> dict[str, Any]:
            n = len(rs); pos = sum(r.ungrounded for r in rs)
            return {"n": n, "ungrounded": pos,
                    "ungrounded_rate": round(pos / n, 4) if n else None,
                    "datasets": sorted({r.dataset for r in rs})}
        return {k: stat(getattr(self, k)) for k in
                ("train", "calib_temp", "calib_conf", "test", "ood")}


# --------------------------------------------------------------------------- #
# RAGTruth parsing
# --------------------------------------------------------------------------- #
def _split_offsets(text: str) -> list[tuple[str, int, int]]:
    """Sentence segments with [start, end) char offsets into ``text``."""
    spans, start = [], 0
    for m in _BOUNDARY.finditer(text):
        end = m.start()
        seg = text[start:end]
        if seg.strip():
            spans.append((seg.strip(), start, end))
        start = m.end()
    if start < len(text):
        seg = text[start:]
        if seg.strip():
            spans.append((seg.strip(), start, len(text)))
    return spans


def _evidence_of(src: dict[str, Any]) -> str:
    """Extract the grounding reference text from a RAGTruth source_info row."""
    si = src.get("source_info")
    if isinstance(si, str):
        return si
    if isinstance(si, list):
        return "\n".join(str(x) for x in si)
    if isinstance(si, dict):
        for k in ("passages", "reference", "context", "document", "text"):
            if k in si:
                v = si[k]
                return "\n".join(map(str, v)) if isinstance(v, list) else str(v)
        return json.dumps(si, ensure_ascii=False)
    # fall back to the prompt (contains the grounding content the model saw)
    return str(src.get("prompt", ""))


def parse_ragtruth(data_dir: Path = DATA_DIR, min_chars: int = 8) -> list[Record]:
    """Join responses to sources, derive sentence-level groundedness from spans."""
    sources = {}
    with open(data_dir / "source_info.jsonl", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            sources[str(s["source_id"])] = s
    records: list[Record] = []
    with open(data_dir / "response.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            src = sources.get(str(r["source_id"]))
            if src is None:
                continue
            evidence = _evidence_of(src)
            task = str(src.get("task_type", "unknown"))
            split = str(r.get("split", "train"))
            resp = r.get("response", "") or ""
            spans = [(int(l["start"]), int(l["end"])) for l in (r.get("labels") or [])
                     if "start" in l and "end" in l]
            for sent, s0, s1 in _split_offsets(resp):
                if len(sent) < min_chars:
                    continue
                grounded = 0 if any(s0 < le and ls < s1 for (ls, le) in spans) else 1
                records.append(Record(doc=evidence, claim=sent, grounded=grounded,
                                      dataset=task, split=split))
    return records


def _cache_path(data_dir: Path) -> Path:
    return data_dir / "claims_cache.jsonl"


def load_records(cfg: dict[str, Any] | None = None, verbose: bool = True,
                 use_cache: bool = True) -> tuple[list[Record], dict[str, Any]]:
    """Load normalized claim records (parse once, cache to jsonl)."""
    data_dir = DATA_DIR
    cache = _cache_path(data_dir)
    if use_cache and cache.exists():
        records = [Record(**json.loads(l)) for l in open(cache, encoding="utf-8")]
    else:
        records = parse_ragtruth(data_dir)
        with open(cache, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    from collections import Counter
    report = {
        "source": "RAGTruth (GitHub ParticleMedia/RAGTruth)",
        "n_records": len(records),
        "by_task": dict(Counter(r.dataset for r in records)),
        "by_split": dict(Counter(r.split for r in records)),
        "overall_ungrounded_rate": round(sum(r.ungrounded for r in records) / len(records), 4),
        "ungrounded_by_task": {t: round(
            sum(r.ungrounded for r in records if r.dataset == t) /
            max(1, sum(1 for r in records if r.dataset == t)), 4)
            for t in sorted({r.dataset for r in records})},
    }
    if verbose:
        print(f"[data] {report['n_records']} claim records from RAGTruth")
        print(f"[data] by task: {report['by_task']}")
        print(f"[data] by split: {report['by_split']}")
        print(f"[data] ungrounded rate overall {report['overall_ungrounded_rate']}, "
              f"by task {report['ungrounded_by_task']}")
    return records, report


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_splits(records: list[Record], cfg: dict[str, Any] | None = None,
                ood_task: str | None = None) -> Splits:
    """Honor RAGTruth's official train/test. Calibration carved from TRAIN (disjoint from
    test => conformal hygiene). If ``ood_task`` given, that task type is held out as OOD and
    the others form in-domain (leave-one-domain-out)."""
    dcfg = cfg or _load_cfg("data.yaml")
    seed = int(dcfg.get("seed", 1337))
    lim = dcfg.get("limits", {})
    rng = random.Random(seed)

    recs = list(records)
    ood: list[Record] = []
    if ood_task is not None:
        ood = [r for r in recs if r.dataset.lower() == ood_task.lower()]
        recs = [r for r in recs if r.dataset.lower() != ood_task.lower()]

    train_pool = [r for r in recs if r.split == "train"]
    test_pool = [r for r in recs if r.split == "test"]
    rng.shuffle(train_pool); rng.shuffle(test_pool); rng.shuffle(ood)

    # caps for thin-slice speed; raised for the final full run via configs/data.yaml
    train_pool = train_pool[: int(lim.get("max_train", len(train_pool)))]
    test_pool = test_pool[: int(lim.get("max_test", len(test_pool)))]
    ood = ood[: int(lim.get("max_ood", len(ood)))]

    s = dcfg["splits"]
    n = len(train_pool)
    # within TRAIN: a small verifier-fit reserve, then temperature calib, then conformal calib
    i_fit = int(s.get("train", 0.25) * n)
    i_temp = i_fit + int(s.get("calib_temp", 0.30) * n)
    return Splits(
        train=train_pool[:i_fit],
        calib_temp=train_pool[i_fit:i_temp],
        calib_conf=train_pool[i_temp:],
        test=test_pool,
        ood=ood,
    )


def records_to_arrays(records: Iterable[Record]) -> dict[str, list]:
    rs = list(records)
    return {
        "claim": [r.claim for r in rs],
        "doc": [r.doc for r in rs],
        "grounded": [r.grounded for r in rs],
        "ungrounded": [r.ungrounded for r in rs],
        "dataset": [r.dataset for r in rs],
    }


if __name__ == "__main__":  # python -m praman.data
    recs, rep = load_records()
    sp = make_splits(recs)
    print(json.dumps({"schema": rep, "splits": sp.summary()}, indent=2)[:2500])
