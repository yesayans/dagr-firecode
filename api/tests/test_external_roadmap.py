"""External roadmap (URLs + pasted text) for closed-source apps."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.resolver import RoadmapResolver, _paste_text_to_items, _split_urls


def test_split_urls():
    assert _split_urls(["https://a.com/r , https://b.com/c\nhttps://a.com/r"]) == [
        "https://a.com/r",
        "https://b.com/c",
    ]


def test_paste_text_to_items_lines():
    df = _paste_text_to_items("Offline sync\nShared workspaces\nExport to PDF")
    assert len(df) == 3
    assert df.iloc[0]["source"] == "web"
    assert df.iloc[0]["kind"] == "planned"
    assert "Offline sync" in df.iloc[0]["text"]


def test_resolve_external_paste_only():
    r = RoadmapResolver()
    result = r.resolve(
        "Acme Notes",
        "custom.acme.notes",
        refresh=True,
        external_roadmap_text="Offline sync for teams\nDark mode everywhere\n",
    )
    assert result.roadmap_source == "web"
    assert len(result.roadmap_items) == 2
    assert result.github_repo is None


def test_resolve_external_urls_mocked():
    r = RoadmapResolver()
    pages = pd.DataFrame(
        [
            {
                "source_url": "https://example.com/roadmap",
                "title": "Acme Notes Roadmap",
                "text": "We are working on offline sync and shared folders for teams.",
            }
        ]
    )
    r.web = MagicMock()
    r.web.fetch_roadmap_pages.return_value = (pages, [])
    result = r.resolve(
        "Acme Notes",
        "custom.acme.notes",
        refresh=True,
        external_roadmap_urls=["https://example.com/roadmap"],
    )
    assert result.roadmap_source == "web"
    assert len(result.roadmap_items) >= 1
    assert result.web_urls == ["https://example.com/roadmap"]
