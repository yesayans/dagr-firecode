"""GitHub issue text cleaning for embedding."""

from src.data_ingestion import (
    compose_issue_text,
    is_version_milestone_title,
    strip_github_boilerplate,
)


def test_strip_checklist_and_environment():
    body = """
### Checklist
- [x] I have used the search function to see if someone else has already submitted the same feature request.
- [ ] I have read the documentation

### Steps to reproduce
1. Open app

### Environment
Android 12

Real problem description here about downloads failing.
"""
    cleaned = strip_github_boilerplate(body)
    assert "search function" not in cleaned.lower()
    assert "downloads failing" in cleaned
    assert "Checklist" not in cleaned
    assert "Environment" not in cleaned


def test_compose_weights_title():
    text = compose_issue_text("Sleep timer broken", "### Checklist\n- [x] I have used the search function\n\nTimer ignores bluetooth", title_weight=3)
    assert text.startswith("Sleep timer broken")
    assert text.count("Sleep timer broken") >= 3
    assert "search function" not in text.lower()
    assert "Timer ignores bluetooth" in text


def test_version_milestone_titles():
    assert is_version_milestone_title("1.5.2")
    assert is_version_milestone_title("v3.0.0")
    assert not is_version_milestone_title("Improve Sleep Timer UX")
    assert not is_version_milestone_title("1.5 rewrite")
