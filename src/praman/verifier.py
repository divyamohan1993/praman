"""Verifier: score whether evidence supports a claim. CPU, discriminative, small.

Returns, per (claim, evidence) pair:
  * p_supported in [0,1]  -- the model's probability the claim is supported, and
  * supported_logit       -- a single binary logit z with sigmoid(z) == p_supported,
                             which is what calibration (temperature scaling) operates on.

Backends:
  * torch (default): AutoModelForSequenceClassification, for training/eval.
  * onnx  (runtime): ONNX Runtime int8 (VNNI on Genoa), fp32 fallback if op gaps.

For an NLI model, "supported" = entailment, and
    z = entail_logit - logsumexp(other_class_logits)  =>  sigmoid(z) == softmax(entail).
For a binary faithfulness model, z = pos_logit - neg_logit (or logit(p) for a 1-logit head).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from ._env import configure_threads

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


@dataclass
class VerifierConfig:
    hf_id: str
    kind: str = "nli"                 # nli | faithfulness_binary
    entailment_label: str = "entailment"
    max_length: int = 512
    multi_passage: str = "max"
    batch_size: int = 16
    evidence_char_cap: int = 4000

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "VerifierConfig":
        path = Path(path) if path else CONFIG_DIR / "verifier.yaml"
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        m, s = y["model"], y.get("scoring", {})
        return cls(
            hf_id=m["hf_id"], kind=m.get("kind", "nli"),
            entailment_label=m.get("entailment_label", "entailment"),
            max_length=int(m.get("max_length", 512)),
            multi_passage=s.get("multi_passage", "max"),
            batch_size=int(s.get("batch_size", 16)),
            evidence_char_cap=int(s.get("evidence_char_cap", 4000)),
        )


class Verifier:
    def __init__(self, cfg: VerifierConfig, backend: str = "torch",
                 model_dir: str | Path | None = None):
        configure_threads()
        self.cfg = cfg
        self.backend = backend
        self._support_idx: int | None = None
        self._other_idx: list[int] = []
        self._id2label: dict[int, str] = {}
        if backend == "torch":
            self._init_torch(model_dir or cfg.hf_id)
        elif backend == "onnx":
            self._init_onnx(model_dir)
        else:
            raise ValueError(f"unknown backend {backend}")

    # ---- backends ----------------------------------------------------------
    def _resolve_indices(self, id2label: dict[int, str]) -> None:
        self._id2label = {int(k): str(v) for k, v in id2label.items()}
        n = len(self._id2label)
        if self.cfg.kind == "nli":
            target = self.cfg.entailment_label.lower()
            idx = next((i for i, lab in self._id2label.items() if lab.lower() == target), None)
            if idx is None:  # fuzzy fallback
                idx = next((i for i, lab in self._id2label.items() if "entail" in lab.lower()), 0)
            self._support_idx = idx
            self._other_idx = [i for i in self._id2label if i != idx]
        else:  # faithfulness_binary
            idx = next((i for i, lab in self._id2label.items()
                        if lab.lower() in {"1", "supported", "consistent", "factual", "positive", "label_1"}), None)
            if idx is None:
                idx = max(self._id2label) if n >= 2 else 0
            self._support_idx = idx
            self._other_idx = [i for i in self._id2label if i != idx]

    def _init_torch(self, model_dir: str | Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        id2label = getattr(self.model.config, "id2label", None) or {0: "0", 1: "1"}
        self._resolve_indices(id2label)

    def _init_onnx(self, model_dir: str | Path | None) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        # prefer int8, fall back to fp32
        candidates = ["model_int8.onnx", "model.onnx"]
        onnx_path = next((model_dir / c for c in candidates if (model_dir / c).exists()), None)
        if onnx_path is None:
            raise FileNotFoundError(f"no .onnx found in {model_dir}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = configure_threads()
        so.inter_op_num_threads = 2
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so,
                                            providers=["CPUExecutionProvider"])
        meta = yaml.safe_load((model_dir / "verifier_meta.yaml").read_text(encoding="utf-8"))
        self._resolve_indices({int(k): v for k, v in meta["id2label"].items()})
        self._onnx_path = onnx_path

    # ---- scoring -----------------------------------------------------------
    def _logits_to_p_z(self, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, C) logits -> (p_supported, supported_binary_logit) each (N,)."""
        sup = logits[:, self._support_idx]
        if self._other_idx:
            other = _logsumexp(logits[:, self._other_idx], axis=-1)
        else:  # single-logit head: treat raw logit as the binary logit
            other = np.zeros_like(sup)
        z = sup - other
        p = 1.0 / (1.0 + np.exp(-z))
        return p, z

    def _forward(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        enc = self.tokenizer(premises, hypotheses, truncation=True,
                             max_length=self.cfg.max_length, padding=True,
                             return_tensors="np" if self.backend == "onnx" else "pt")
        if self.backend == "torch":
            with self._torch.no_grad():
                out = self.model(**enc)
            return out.logits.cpu().numpy()
        feed = {k: v for k, v in enc.items() if k in {i.name for i in self.session.get_inputs()}}
        logits = self.session.run(None, feed)[0]
        return np.asarray(logits)

    def score_pairs(self, claims: Sequence[str], evidences: Sequence[str]
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Score aligned (claim, single-evidence) pairs. Returns (p_supported, z)."""
        assert len(claims) == len(evidences)
        cap = self.cfg.evidence_char_cap
        prem = [(e or "")[:cap] for e in evidences]  # premise = evidence
        hyp = [c or "" for c in claims]              # hypothesis = claim
        ps, zs = [], []
        bs = self.cfg.batch_size
        for i in range(0, len(hyp), bs):
            logits = self._forward(prem[i:i + bs], hyp[i:i + bs])
            p, z = self._logits_to_p_z(logits)
            ps.append(p); zs.append(z)
        if not ps:
            return np.array([]), np.array([])
        return np.concatenate(ps), np.concatenate(zs)

    def score_multi(self, claim: str, passages: Sequence[str]) -> tuple[float, float, int]:
        """Claim vs many passages. Grounded if ANY supports -> take max P. Returns (p, z, idx)."""
        if isinstance(passages, str):
            passages = [passages]
        passages = list(passages) or [""]
        p, z = self.score_pairs([claim] * len(passages), passages)
        best = int(np.argmax(p))
        return float(p[best]), float(z[best]), best

    def score(self, claim: str, evidence: str | Sequence[str]) -> float:
        """Public convenience: P(claim supported by evidence)."""
        passages = [evidence] if isinstance(evidence, str) else list(evidence)
        return self.score_multi(claim, passages)[0]
