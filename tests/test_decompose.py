from __future__ import annotations

from praman.data import _evidence_of, _split_offsets
from praman.decompose import decompose, decontextualize


def test_decompose_basic_sentences():
    out = decompose("The drug was approved in 2019. It reduces risk by 40%.")
    assert out == ["The drug was approved in 2019.", "It reduces risk by 40%."]


def test_decompose_keeps_abbreviation():
    out = decompose("Dr. Rao led the study. Results were strong.")
    assert out[0].startswith("Dr. Rao led the study")
    assert len(out) == 2


def test_decompose_empty():
    assert decompose("") == []
    assert decompose("   \n  ") == []


def test_decontextualize_resolves_pronoun():
    claims = ["The drug was approved.", "It reduces risk by 40%."]
    out = decontextualize(claims)
    assert "drug" in out[1].lower()


def test_split_offsets_cover_text():
    text = "First claim here. Second one follows.\nThird on a new line."
    spans = _split_offsets(text)
    assert len(spans) == 3
    for seg, s, e in spans:
        assert text[s:e].strip().startswith(seg[:10].strip()[:5]) or seg in text[s:e + 2]
        assert 0 <= s < e <= len(text)


def test_evidence_of_handles_types():
    assert _evidence_of({"source_info": "plain text"}) == "plain text"
    assert "a" in _evidence_of({"source_info": ["a", "b"]})
    assert _evidence_of({"source_info": {"passages": ["x", "y"]}}).find("x") >= 0
