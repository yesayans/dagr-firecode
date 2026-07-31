"""Shared schemas for Silent Stakeholder product context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RoadmapSource = Literal["github", "web", "hybrid", "none"]


@dataclass
class EvidenceItem:
    id: str
    source: str  # github_issue | web_page | interview | changelog | store | other
    title: str
    url: str = ""
    snippet: str = ""
    kind: str = ""  # current_feature | planned | promised_unshipped | interview_signal


@dataclass
class ProductContext:
    """Generic product identity + what the team is building / has shipped."""

    product_id: str
    display_name: str
    package_name: str = ""
    dataset: str = ""
    reviews: int = 0
    avg_stars: float = 0.0

    roadmap_source: RoadmapSource = "none"
    github_repo: str | None = None

    current_features: list[str] = field(default_factory=list)
    planned_items: list[str] = field(default_factory=list)
    promised_unshipped: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)

    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_row(self) -> dict[str, Any]:
        """CSV-friendly flat view for discovery tables."""
        return {
            "product_id": self.product_id,
            "app_name": self.display_name,
            "package_name": self.package_name or self.product_id,
            "reviews": self.reviews,
            "avg_stars": self.avg_stars,
            "dataset": self.dataset,
            "roadmap_source": self.roadmap_source,
            "github_repo": self.github_repo or "",
            "likely_applicable": self.roadmap_source != "none",
            "current_features": " | ".join(self.current_features[:12]),
            "planned_items": " | ".join(self.planned_items[:12]),
            "promised_unshipped": " | ".join(self.promised_unshipped[:12]),
            "evidence_count": len(self.evidence),
            "evidence_urls": " | ".join(e.url for e in self.evidence if e.url)[:2000],
            "notes": self.notes,
            "gh_stars": self.meta.get("gh_stars", ""),
            "gh_open_issues": self.meta.get("gh_open_issues", ""),
            "gh_milestones_open": self.meta.get("gh_milestones_open", ""),
            "gh_milestones_total": self.meta.get("gh_milestones_total", ""),
            "gh_url": self.meta.get("gh_url", ""),
        }
