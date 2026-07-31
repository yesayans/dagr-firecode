"""Run AntennaPod analysis against live GitHub and print verdict distribution."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from src.config import get_settings  # noqa: E402
from src.pipeline import AnalysisPipeline, config_hash  # noqa: E402
from src.store import LocalJsonStore, load_catalog  # noqa: E402


def main() -> None:
    settings = get_settings()
    settings.resolve_github_credentials()
    print(
        {
            "threshold": settings.match_threshold_tfidf,
            "margin": settings.match_margin_tfidf,
            "backend": settings.embedding_backend,
            "token_source": settings.github_token_source,
        }
    )
    store = LocalJsonStore(settings=settings)
    catalog = load_catalog(settings)
    store.seed_apps_from_catalog(catalog)
    app = store.get_app_by_package("de.danoeh.antennapod")
    if not app:
        raise SystemExit("AntennaPod not in catalog")
    # Prefer known repo
    app = {**app, "github_repo": app.get("github_repo") or "AntennaPod/AntennaPod"}
    job = store.create_job(
        {
            "app_id": app["id"],
            "config_hash": config_hash(app["id"], 500, settings),
            "status": "queued",
            "stage": "queued",
        }
    )
    pipe = AnalysisPipeline(store, settings)
    # Force fresh GitHub resolve path
    pipe.resolver  # noqa: B018
    result = pipe.run(job["id"], app, max_reviews=500)
    gaps = result.get("gaps") or []
    verdicts = Counter(g.get("verdict") for g in gaps)
    validated = sum(1 for g in gaps if (g.get("metrics") or {}).get("validated_by_later_roadmap"))
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "roadmap_source": result.get("roadmap_source"),
                "stats": result.get("stats"),
                "n_gaps": len(gaps),
                "verdicts": dict(verdicts),
                "validated_by_later_roadmap_count": validated,
            },
            indent=2,
            default=str,
        )
    )
    print("\n=== top gaps ===")
    for i, g in enumerate(gaps[:12], 1):
        m = g.get("metrics") or {}
        later = m.get("later_addressed_by")
        print(
            f"\n{i}. [{g.get('verdict')}] sim={m.get('best_similarity')} "
            f"margin={m.get('best_similarity_margin')} "
            f"need_source={g.get('need_source')}"
        )
        print(f"   need: {(g.get('need') or '')[:160]}")
        print(f"   matched: {m.get('matched_item_title')}")
        print(f"   state={m.get('matched_item_state')} age_days={m.get('matched_item_age_days')}")
        if later:
            print(
                f"   later_addressed: sim={later.get('similarity')} "
                f"{(later.get('title') or '')[:100]}"
            )
        print(f"   validated_by_later_roadmap={m.get('validated_by_later_roadmap')}")


if __name__ == "__main__":
    main()
