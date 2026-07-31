"""Resolve product roadmap from GitHub and/or the public web."""

from __future__ import annotations

from typing import Any

from silent_stakeholder import ProductContext
from silent_stakeholder.roadmap_github import fetch_github_roadmap, guess_repo
from silent_stakeholder.roadmap_web import fetch_web_roadmap


def resolve_product_context(
    *,
    product_id: str,
    display_name: str | None = None,
    package_name: str | None = None,
    github_repo: str | None = None,
    known_github: dict[str, str] | None = None,
    reviews: int = 0,
    avg_stars: float = 0.0,
    dataset: str = "",
    prefer_github: bool = True,
    use_web_fallback: bool = True,
    force_web: bool = False,
    max_issues: int = 60,
) -> ProductContext:
    """
    Generic entrypoint for ANY app.

    1. Try GitHub issues/milestones when a repo is known/guessable.
    2. If no GitHub roadmap (or force_web), search the internet for
       current features, public plans, and interview/promise signals.
    3. If web search discovers a GitHub repo, try GitHub again.
    """
    pkg = package_name or product_id
    name = display_name or pkg
    ctx = ProductContext(
        product_id=product_id,
        display_name=name,
        package_name=pkg,
        dataset=dataset,
        reviews=reviews,
        avg_stars=avg_stars,
    )

    repo = github_repo or guess_repo(pkg, known_github) or guess_repo(name, known_github)

    github_ok = False
    if prefer_github and repo:
        try:
            ctx = fetch_github_roadmap(ctx, repo, max_issues=max_issues)
            github_ok = bool(ctx.planned_items or ctx.evidence)
        except Exception as e:
            ctx.notes += f" GitHub resolve error: {e}."

    need_web = force_web or (use_web_fallback and not github_ok)
    if need_web:
        try:
            ctx = fetch_web_roadmap(ctx, app_name=name)
        except Exception as e:
            ctx.notes += f" Web resolve error: {e}."

    # Web may have discovered a repo — retry GitHub once
    if (
        use_web_fallback
        and ctx.github_repo
        and ctx.roadmap_source in ("web", "none")
        and not github_ok
    ):
        try:
            before = len(ctx.evidence)
            ctx = fetch_github_roadmap(ctx, ctx.github_repo, max_issues=max_issues)
            if len(ctx.evidence) > before or ctx.planned_items:
                github_ok = True
        except Exception as e:
            ctx.notes += f" GitHub retry error: {e}."

    if ctx.roadmap_source == "none":
        ctx.notes += (
            " No roadmap source available yet — supply --github or rely on richer web results."
        )

    return ctx


def context_summary(ctx: ProductContext) -> dict[str, Any]:
    return {
        "product_id": ctx.product_id,
        "display_name": ctx.display_name,
        "roadmap_source": ctx.roadmap_source,
        "github_repo": ctx.github_repo,
        "current_features_n": len(ctx.current_features),
        "planned_items_n": len(ctx.planned_items),
        "promised_unshipped_n": len(ctx.promised_unshipped),
        "evidence_n": len(ctx.evidence),
        "notes": ctx.notes.strip(),
    }
