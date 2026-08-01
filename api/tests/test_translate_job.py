"""Job analysis translation helpers."""

from src.translate_job import _build_source_blob, translate_job_analysis
import asyncio


def test_build_source_blob_includes_review_snippets():
    job = {
        "summary": "Found 1 gap",
        "gaps": [
            {
                "id": "g1",
                "need": "Offline sync",
                "one_sentence_summary": "Users want offline.",
                "latent_reasoning": "because reviews say so",
                "confidence_rationale": "deterministic",
                "metrics": {
                    "surface_complaints": ["sync fails"],
                    "workarounds": ["use website"],
                },
                "evidence": [
                    {
                        "evidence_id": "r1",
                        "source_type": "review",
                        "title": "Review r1",
                        "snippet": "I wish offline worked",
                    },
                    {
                        "evidence_id": "i1",
                        "source_type": "github_issue",
                        "title": "Issue",
                        "snippet": "ignore me",
                    },
                ],
            }
        ],
    }
    blob = _build_source_blob(job)
    assert blob["summary"] == "Found 1 gap"
    assert blob["gaps"][0]["gap_id"] == "g1"
    assert blob["gaps"][0]["surface_complaints"] == ["sync fails"]
    assert len(blob["gaps"][0]["evidence"]) == 1
    assert blob["gaps"][0]["evidence"][0]["evidence_id"] == "r1"


def test_translate_en_is_identity_without_llm():
    job = {
        "status": "completed",
        "summary": "Hello",
        "gaps": [
            {
                "id": "g1",
                "need": "Need",
                "one_sentence_summary": "Sum",
                "latent_reasoning": "Why",
                "confidence_rationale": "How",
                "metrics": {},
                "evidence": [],
            }
        ],
    }

    class S:
        llm_enabled = False
        openrouter_api_key = None
        openrouter_model = "x"
        llm_base_url = "https://example.com"

    out = asyncio.run(translate_job_analysis(job, "en", settings=S()))  # type: ignore[arg-type]
    assert out["locale"] == "en"
    assert out["summary"] == "Hello"
    assert out["gaps"][0]["need"] == "Need"
    assert out["model"] is None
