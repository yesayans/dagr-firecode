"""Fetch real sealuzh review snapshots into data/reviews/ (not fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.data_ingestion import ReviewScraper  # noqa: E402

APPS = [
    ("de.danoeh.antennapod", "AntennaPod"),
    ("com.ichi2.anki", "AnkiDroid"),
    ("org.isoron.uhabits", "Loop Habit Tracker"),
    ("org.wordpress.android", "WordPress"),
]


def main() -> None:
    scraper = ReviewScraper()
    summary = []
    for pkg, name in APPS:
        # Skip AntennaPod re-fetch if already populated unless --force
        force = "--force" in sys.argv or pkg != "de.danoeh.antennapod"
        if pkg == "de.danoeh.antennapod" and scraper.cache_path(pkg).exists() and "--force" not in sys.argv:
            result = scraper.fetch_reviews(pkg, max_reviews=5000, force_refresh=False)
            summary.append(
                {
                    "app": name,
                    "package": pkg,
                    "rows": len(result.df),
                    "provenance": result.provenance,
                    "action": "kept-existing",
                }
            )
            continue
        print(f"Fetching {name} ({pkg})…")
        result = scraper.fetch_reviews(pkg, max_reviews=5000, force_refresh=True)
        summary.append(
            {
                "app": name,
                "package": pkg,
                "rows": len(result.df),
                "provenance": result.provenance,
                "degraded": result.degraded,
                "action": "fetched",
            }
        )
        print(summary[-1])
    print({"summary": summary})


if __name__ == "__main__":
    main()
