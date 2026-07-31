"""GitHub issues + milestones as a product roadmap source."""

from __future__ import annotations

import os
from typing import Any

import requests

from silent_stakeholder import EvidenceItem, ProductContext


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "SilentStakeholder/1.0"
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def fetch_github_roadmap(
    ctx: ProductContext,
    repo: str,
    *,
    max_issues: int = 80,
    session: requests.Session | None = None,
) -> ProductContext:
    """Fill planned_items / evidence from GitHub milestones + open issues."""
    session = session or _session()
    ctx.github_repo = repo
    ctx.meta["gh_url"] = f"https://github.com/{repo}"

    r = session.get(f"https://api.github.com/repos/{repo}", timeout=25)
    if r.status_code != 200:
        ctx.notes += f" GitHub repo fetch failed ({r.status_code})."
        return ctx

    data = r.json()
    ctx.meta["gh_stars"] = data.get("stargazers_count")
    ctx.meta["gh_open_issues"] = data.get("open_issues_count")
    if not ctx.display_name or ctx.display_name == ctx.product_id:
        ctx.display_name = data.get("name") or ctx.display_name

    milestones = session.get(
        f"https://api.github.com/repos/{repo}/milestones",
        params={"state": "all", "per_page": 50},
        timeout=25,
    )
    ms = milestones.json() if milestones.status_code == 200 else []
    if isinstance(ms, list):
        ctx.meta["gh_milestones_total"] = len(ms)
        ctx.meta["gh_milestones_open"] = sum(1 for m in ms if m.get("state") == "open")
        for m in ms:
            title = m.get("title") or ""
            if not title:
                continue
            label = f"[milestone:{m.get('state')}] {title}"
            ctx.planned_items.append(label)
            ctx.evidence.append(
                EvidenceItem(
                    id=f"gh-milestone-{m.get('number')}",
                    source="github_issue",
                    title=title,
                    url=m.get("html_url") or "",
                    snippet=(m.get("description") or "")[:300],
                    kind="planned",
                )
            )

    issues = session.get(
        f"https://api.github.com/repos/{repo}/issues",
        params={
            "state": "open",
            "per_page": min(max_issues, 100),
            "sort": "comments",
            "direction": "desc",
        },
        timeout=25,
    )
    issue_rows = issues.json() if issues.status_code == 200 else []
    if isinstance(issue_rows, list):
        for issue in issue_rows:
            if "pull_request" in issue:
                continue
            title = issue.get("title") or ""
            labels = [lb.get("name", "") for lb in (issue.get("labels") or [])]
            kind = "planned"
            if any("enhancement" in x.lower() or "feature" in x.lower() for x in labels):
                kind = "planned"
            ctx.planned_items.append(title)
            ctx.evidence.append(
                EvidenceItem(
                    id=f"gh-issue-{issue.get('number')}",
                    source="github_issue",
                    title=title,
                    url=issue.get("html_url") or "",
                    snippet=", ".join(labels),
                    kind=kind,
                )
            )

    if ctx.roadmap_source == "none":
        ctx.roadmap_source = "github"
    elif ctx.roadmap_source == "web":
        ctx.roadmap_source = "hybrid"
    return ctx


def guess_repo(package_or_name: str, known: dict[str, str] | None = None) -> str | None:
    known = known or {}
    if package_or_name in known:
        return known[package_or_name]
    low = package_or_name.lower()
    heuristics = [
        ("antennapod", "AntennaPod/AntennaPod"),
        ("anki", "ankidroid/Anki-Android"),
        ("wordpress", "wordpress-mobile/WordPress-Android"),
        ("osmand", "osmandapp/OsmAnd"),
        ("termux", "termux/termux-app"),
        ("uhabits", "iSoron/uhabits"),
        ("newpipe", "TeamNewPipe/NewPipe"),
        ("wikipedia", "wikimedia/apps-android-wikipedia"),
        ("duckduckgo", "duckduckgo/Android"),
        ("k9", "thunderbird/thunderbird-android"),
    ]
    for key, repo in heuristics:
        if key in low:
            return repo
    return None
