"""Dataset validation with per-column diagnostics.

The upload page needs to tell a PM *which column* is wrong and *how many rows*
are affected. Returning a structured report rather than raising gives both the
batch script and the UI something useful to render.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field

APP_REQUIRED_COLUMNS = ("app_id", "app_name")
REVIEW_REQUIRED_COLUMNS = ("app_id", "review_text")

#: Tolerated header spellings, so a re-scrape with different column names does
#: not require a code change. Maps canonical name -> accepted aliases.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "app_id": ("app_id", "appid", "id", "package_name", "app_package"),
    "app_name": ("app_name", "name", "title", "app"),
    "description": ("description", "desc", "summary"),
    "score": ("score", "rating", "app_score", "average_rating"),
    "ratings_count": ("ratings_count", "ratings", "n_ratings", "reviews_count"),
    "downloads": ("downloads", "installs", "downloads_raw", "min_installs"),
    "categories": ("categories", "category", "genres", "genre"),
    "section": ("section", "collection", "chart"),
    "review_text": ("review_text", "content", "text", "review", "body"),
    "review_score": ("review_score", "rating", "stars", "score"),
    "review_date": ("review_date", "at", "date", "created_at", "timestamp"),
    "helpful_count": ("helpful_count", "thumbs_up_count", "helpful", "likes"),
}


class Severity(str, Enum):
    ERROR = "error"  # the pipeline cannot proceed
    WARNING = "warning"  # degraded but usable


class ValidationIssue(BaseModel):
    column: str
    severity: Severity
    message: str
    n_rows_affected: int = 0


class ColumnDiagnostic(BaseModel):
    column: str
    present: bool
    resolved_from: str | None = None  # the alias actually found in the file
    n_null: int = 0
    n_rows: int = 0
    sample_values: list[str] = Field(default_factory=list)

    @property
    def null_share(self) -> float:
        return self.n_null / self.n_rows if self.n_rows else 0.0


class ValidationReport(BaseModel):
    dataset: str
    n_rows: int = 0
    columns: list[ColumnDiagnostic] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity is Severity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    def raise_for_errors(self) -> None:
        if self.ok:
            return
        detail = "; ".join(f"{i.column}: {i.message}" for i in self.errors)
        raise DatasetValidationError(f"{self.dataset} failed validation - {detail}")

    def summary_line(self) -> str:
        n_err = len(self.errors)
        n_warn = len(self.issues) - n_err
        return f"{self.dataset}: {self.n_rows:,} rows, {n_err} error(s), {n_warn} warning(s)"


class DatasetValidationError(RuntimeError):
    """Raised when a dataset is unusable. Carries the human-readable reason."""


def resolve_columns(frame: pd.DataFrame, wanted: tuple[str, ...]) -> dict[str, str]:
    """Map canonical column names onto whatever the file actually calls them."""
    lowered = {str(c).strip().lower(): str(c) for c in frame.columns}
    resolved: dict[str, str] = {}
    for canonical in wanted:
        for alias in COLUMN_ALIASES.get(canonical, (canonical,)):
            if alias in lowered:
                resolved[canonical] = lowered[alias]
                break
    return resolved


def _diagnose(
    frame: pd.DataFrame, canonical: str, resolved: dict[str, str]
) -> ColumnDiagnostic:
    actual = resolved.get(canonical)
    if actual is None:
        return ColumnDiagnostic(column=canonical, present=False, n_rows=len(frame))
    series = frame[actual]
    sample = [str(v) for v in series.dropna().head(3).tolist()]
    return ColumnDiagnostic(
        column=canonical,
        present=True,
        resolved_from=actual if actual != canonical else None,
        n_null=int(series.isna().sum()),
        n_rows=len(frame),
        sample_values=[s[:80] for s in sample],
    )


def validate_apps(frame: pd.DataFrame) -> ValidationReport:
    wanted = (
        "app_id",
        "app_name",
        "description",
        "score",
        "ratings_count",
        "downloads",
        "categories",
        "section",
    )
    resolved = resolve_columns(frame, wanted)
    report = ValidationReport(
        dataset="apps_info",
        n_rows=len(frame),
        columns=[_diagnose(frame, c, resolved) for c in wanted],
    )

    for required in APP_REQUIRED_COLUMNS:
        if required not in resolved:
            report.issues.append(
                ValidationIssue(
                    column=required,
                    severity=Severity.ERROR,
                    message=f"required column missing (accepted names: "
                    f"{', '.join(COLUMN_ALIASES[required])})",
                )
            )

    if "app_id" in resolved:
        ids = frame[resolved["app_id"]].astype(str)
        n_dupe = int(ids.duplicated().sum())
        if n_dupe:
            report.issues.append(
                ValidationIssue(
                    column="app_id",
                    severity=Severity.WARNING,
                    message="duplicate app ids; the last row of each id wins",
                    n_rows_affected=n_dupe,
                )
            )
    for optional in ("categories", "score"):
        diag = next((c for c in report.columns if c.column == optional), None)
        if diag and diag.present and diag.null_share > 0.5:
            report.issues.append(
                ValidationIssue(
                    column=optional,
                    severity=Severity.WARNING,
                    message=f"{diag.null_share:.0%} of rows are null",
                    n_rows_affected=diag.n_null,
                )
            )
    return report


def validate_reviews(frame: pd.DataFrame) -> ValidationReport:
    wanted = ("app_id", "review_text", "review_score", "review_date", "helpful_count")
    resolved = resolve_columns(frame, wanted)
    report = ValidationReport(
        dataset="apps_reviews",
        n_rows=len(frame),
        columns=[_diagnose(frame, c, resolved) for c in wanted],
    )

    for required in REVIEW_REQUIRED_COLUMNS:
        if required not in resolved:
            report.issues.append(
                ValidationIssue(
                    column=required,
                    severity=Severity.ERROR,
                    message=f"required column missing (accepted names: "
                    f"{', '.join(COLUMN_ALIASES[required])})",
                )
            )

    if "review_text" in resolved:
        text = frame[resolved["review_text"]].astype("string")
        n_blank = int((text.fillna("").str.strip() == "").sum())
        if n_blank:
            report.issues.append(
                ValidationIssue(
                    column="review_text",
                    severity=Severity.WARNING,
                    message="blank review text; these rows are dropped",
                    n_rows_affected=n_blank,
                )
            )
    if "review_date" in resolved:
        parsed = pd.to_datetime(frame[resolved["review_date"]], errors="coerce")
        n_bad = int(parsed.isna().sum())
        if n_bad:
            report.issues.append(
                ValidationIssue(
                    column="review_date",
                    severity=Severity.WARNING,
                    message="unparseable dates; excluded from trends",
                    n_rows_affected=n_bad,
                )
            )
    return report
