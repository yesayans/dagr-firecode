"""LLM citation validation drops unknown review ids."""

from __future__ import annotations

from src.llm_extractor import validate_citations


def test_drops_hallucinated_ids():
    provided = ["abc", "def", "ghi"]
    cited = ["abc", "hallucinated-99", "def", "also-fake"]
    assert validate_citations(cited, provided) == ["abc", "def"]


def test_empty_when_all_unknown():
    assert validate_citations(["x", "y"], ["a", "b"]) == []
