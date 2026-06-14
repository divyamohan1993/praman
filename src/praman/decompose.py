"""Claim decomposition: output_text -> atomic, independently-checkable claims.

v0 is deterministic and cheap (sentence/clause split), per brief 7.1. It is used on
the runtime verify() path. The headline guarantee is validated on benchmark-provided
claims (LLM-AggreFact), so decomposition errors do not contaminate the guarantee number;
decomposition quality is reported separately. Upgrade path: a small instruct SLM or a
dependency-based splitter, only if it measurably limits quality.
"""
from __future__ import annotations

import re

# Common abbreviations whose trailing period must NOT end a sentence.
_ABBR = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "no", "fig", "al", "inc", "ltd", "co", "u.s", "u.k", "approx", "vol", "p", "pp",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WS = re.compile(r"\s+")


def _looks_like_abbrev(token: str) -> bool:
    t = token.lower().rstrip(".")
    return t in _ABBR or (len(t) == 1 and t.isalpha())


def decompose(text: str, min_chars: int = 2) -> list[str]:
    """Split text into atomic claims. Conservative: avoids splitting on abbreviations."""
    text = (text or "").strip()
    if not text:
        return []
    rough = _SENT_SPLIT.split(text)
    claims: list[str] = []
    buf = ""
    for part in rough:
        part = part.strip()
        if not part:
            continue
        candidate = (buf + " " + part).strip() if buf else part
        last_token = candidate.split()[-1] if candidate.split() else ""
        if _looks_like_abbrev(last_token):
            buf = candidate  # keep accumulating; the period was an abbreviation
        else:
            claims.append(_WS.sub(" ", candidate))
            buf = ""
    if buf:
        claims.append(_WS.sub(" ", buf))
    return [c for c in claims if len(c) >= min_chars]


# --- optional decontextualization (resolve a leading pronoun to the prior subject) ---
_LEAD_PRONOUN = re.compile(r"^(it|he|she|they|this|that|these|those|the\s+\w+)\b", re.I)


def decontextualize(claims: list[str]) -> list[str]:
    """Light heuristic: prepend the previous claim's subject to a pronoun-led claim.

    Cheap and reversible; improves verifiability of claims like "It was approved in 2019."
    Disabled by default in the pipeline; offered as an ablation (brief 8.5).
    """
    out: list[str] = []
    prev_subject = ""
    for c in claims:
        m = _LEAD_PRONOUN.match(c)
        if m and prev_subject and m.group(1).lower() in {"it", "he", "she", "they", "this", "that"}:
            out.append(f"{prev_subject} {c}")
        else:
            out.append(c)
        # crude subject = first 4 words (noun phrase heuristic)
        toks = c.split()
        prev_subject = " ".join(toks[:4]) if toks else prev_subject
    return out


if __name__ == "__main__":
    demo = "The drug was approved in 2019. It reduces risk by 40%. Dr. Rao led the study."
    print(decompose(demo))
