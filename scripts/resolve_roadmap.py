"""
Resolve roadmap / feature context for ANY product.

Usage:
  python scripts/resolve_roadmap.py --name "AntennaPod" --package de.danoeh.antennapod
  python scripts/resolve_roadmap.py --name "DoorDash" --force-web
  python scripts/resolve_roadmap.py --name "Loop Habit Tracker" --package org.isoron.uhabits --github iSoron/uhabits
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from silent_stakeholder.resolve import context_summary, resolve_product_context


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resolve current features + roadmap for any app (GitHub and/or web)."
    )
    ap.add_argument("--name", required=True, help="Human product name")
    ap.add_argument("--package", default="", help="Android package / store id")
    ap.add_argument("--github", default="", help="owner/repo if known")
    ap.add_argument("--force-web", action="store_true", help="Always search the web")
    ap.add_argument("--no-web", action="store_true", help="Disable web fallback")
    ap.add_argument(
        "--out",
        default="",
        help="Output JSON path (default: data/roadmaps/<id>.json)",
    )
    args = ap.parse_args()

    product_id = args.package or args.name.lower().replace(" ", "-")
    ctx = resolve_product_context(
        product_id=product_id,
        display_name=args.name,
        package_name=args.package or product_id,
        github_repo=args.github or None,
        use_web_fallback=not args.no_web,
        force_web=args.force_web,
    )

    out = Path(args.out) if args.out else ROOT / "data" / "roadmaps" / f"{product_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ctx.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    summary = context_summary(ctx)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved full context: {out}")
    if ctx.current_features:
        print("\nCurrent features (sample):")
        for x in ctx.current_features[:5]:
            print(f"  - {x[:160]}")
    if ctx.planned_items:
        print("\nPlanned / tracked:")
        for x in ctx.planned_items[:5]:
            print(f"  - {x[:160]}")
    if ctx.promised_unshipped:
        print("\nPromised / interview signals (may be unshipped):")
        for x in ctx.promised_unshipped[:5]:
            print(f"  - {x[:160]}")


if __name__ == "__main__":
    main()
