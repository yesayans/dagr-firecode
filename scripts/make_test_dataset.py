#!/usr/bin/env python3
"""Generate small CSV fixtures for exercising the Upload Dataset page.

    python scripts/make_test_dataset.py

Writes three pairs into `data/fixtures/`:

* **valid** - the happy path. Three apps, themed review text, enough volume for
  density clustering to find real themes.
* **aliased** - identical rows with `appId` / `content` / `thumbsUpCount` style
  headers, to prove the column-alias resolver works without a code change.
* **broken** - a missing required column plus unparseable dates and blank text,
  to prove the per-column diagnostics name the problem.

Sizing is dictated by the app, not chosen freely: the reviews slider starts at
200 and steps by 200, so each app needs at least 600 reviews for the control to
land on clean values; and `min_cluster_size` floors at 15, so a theme needs
enough segments to survive clustering.

Deterministic - a fixed seed means regenerating produces identical ids, so a
re-upload reuses the cached run instead of looking like new data.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aipm.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("make_test_dataset")

SEED = 20260801
REVIEWS_PER_APP = 600
END_DATE = date(2026, 7, 1)


# ---------------------------------------------------------------------------
# Themes
#
# Each theme carries a complaint template set plus, deliberately, phrases where
# users describe a *workaround*. Those are the strongest hidden-need signal in
# the real corpus, so the fixtures have to contain them or the upload path would
# exercise the pipeline without exercising what it is for.
# ---------------------------------------------------------------------------

Theme = tuple[str, int, tuple[str, ...]]


NIMBUS_THEMES: tuple[Theme, ...] = (
    ("sync", 1, (
        "My notes stopped syncing between my laptop and my phone three days ago.",
        "Edits I make on the desktop never show up on mobile, so I email myself the text instead.",
        "Sync silently fails and I only notice when a note is missing at work.",
        "Two devices, two different versions of the same note, no warning which is newer.",
        "I have taken to copying important notes into a plain text file as a backup.",
        "The sync spinner runs forever and then just stops without an error.",
        "Notes I wrote on the train were gone by the time I opened the app at home.",
        "Had to log out and back in again to force a sync, which lost my drafts.",
    )),
    ("data loss", 1, (
        "Lost an hour of writing when the app crashed without saving.",
        "A long note reverted to an old version and there is no history to restore from.",
        "The app closed itself and my unsaved draft was simply gone.",
        "I now write everything in another app first and paste it in, which defeats the point.",
        "No autosave means one crash costs me a whole meeting's notes.",
        "Recovered nothing after the update wiped my most recent entries.",
    )),
    ("search", 2, (
        "Search does not find text that I can see with my own eyes in a note.",
        "Searching only matches titles, never the body, so I scroll manually instead.",
        "I know the word is in there but search returns nothing.",
        "Tags are useless because search ignores them completely.",
        "Finding an old note takes longer than rewriting it.",
    )),
    ("pricing", 2, (
        "I was charged for a yearly plan straight after the trial with no reminder.",
        "The upgrade screen does not say what the free tier actually loses.",
        "Cancelled the subscription and was billed again the following month.",
        "Paying for a plan and still hitting a device limit is not acceptable.",
        "Refund request has gone unanswered for two weeks.",
    )),
    ("praise", 5, (
        "Clean interface and it stays out of the way when I am writing.",
        "Best note app I have used, genuinely fast to open.",
        "Love the markdown support and the keyboard shortcuts.",
        "Simple, quick and does exactly what I need day to day.",
    )),
)

FITTRACK_THEMES: tuple[Theme, ...] = (
    ("gps", 1, (
        "GPS drifts badly so a five kilometre run is logged as seven.",
        "The route map shows me running through buildings and across a river.",
        "Distance is wrong every single time, so I now carry a separate watch.",
        "Pace readings jump around wildly and ruin the whole activity.",
        "It takes four minutes to get a GPS lock, by which point I have started.",
        "I have started logging runs manually afterwards because the tracking cannot be trusted.",
    )),
    ("battery", 1, (
        "Tracking a one hour walk drains forty percent of my battery.",
        "Phone gets hot and the battery is dead before I get home.",
        "Background tracking eats power even when I am not exercising.",
        "I keep a power bank in my pocket now just to finish a long ride.",
    )),
    ("watch sync", 2, (
        "Workouts on my watch never make it into the phone app.",
        "The watch and the app disagree about how many steps I took today.",
        "Pairing drops constantly and I have to re-pair the watch every week.",
        "Heart rate data is missing from half of my recorded sessions.",
        "I export from the watch app and import here manually, which is ridiculous.",
    )),
    ("export", 2, (
        "There is no way to get my own data out in a usable format.",
        "Export produces a file that no other fitness app will read.",
        "Years of training history locked in with no export button anywhere.",
        "I screenshot my stats each month because there is no proper export.",
    )),
    ("praise", 5, (
        "The training plans are genuinely well designed and easy to follow.",
        "Great app for keeping a streak going, the reminders work well.",
        "Interface is clear and logging a session takes seconds.",
        "Been using it for two years and the history view is excellent.",
    )),
)

PAYFLOW_THEMES: tuple[Theme, ...] = (
    ("failed transfer", 1, (
        "Transfer failed but the money left my account and has not come back.",
        "Sent a payment that shows as pending five days later with no explanation.",
        "The app said the transfer succeeded, the recipient never received anything.",
        "I use my bank's own app now because at least the transfers complete.",
        "Money disappeared for six days before reappearing with no message.",
    )),
    ("verification", 1, (
        "Stuck in an endless identity verification loop that never completes.",
        "Uploaded my documents four times and each time it asks again.",
        "Account locked for a security check with no way to speak to anyone.",
        "Verification rejected my passport twice with no reason given.",
        "I gave up and opened an account with a competitor instead.",
    )),
    ("support", 2, (
        "Support has not replied to my ticket in three weeks.",
        "The chat bot loops back to the same articles and there is no human option.",
        "No phone number anywhere for an issue involving actual money.",
        "Had to complain publicly before anyone responded to me.",
    )),
    ("fees", 2, (
        "The exchange rate at confirmation is worse than the one advertised.",
        "Fees are not shown until after the transfer is authorised.",
        "Charged a fee for a transfer that then failed, and the fee was not refunded.",
        "I compare against another provider before every transfer now.",
    )),
    ("praise", 5, (
        "When it works the transfer arrives within minutes, which is impressive.",
        "Clean app and the recipient management is well done.",
        "Much cheaper than my bank for international payments.",
        "Setting up a new recipient is quick and painless.",
    )),
)

APPS = (
    {
        "app_id": "nimbus",
        "app_name": "Nimbus Notes",
        "description": (
            "Fast, private note taking that syncs across every device. Markdown "
            "support, offline editing and full-text search."
        ),
        "score": 4.3,
        "ratings_count": 48210,
        "downloads": 5000000,
        "categories": "Productivity",
        "themes": NIMBUS_THEMES,
    },
    {
        "app_id": "fittrack",
        "app_name": "FitTrack Pro",
        "description": (
            "Run, ride and walk tracking with training plans, heart-rate zones "
            "and a workout history you can actually read."
        ),
        "score": 4.1,
        "ratings_count": 91004,
        "downloads": 10000000,
        "categories": "#4 top free health & fitness, Health & Fitness",
        "themes": FITTRACK_THEMES,
    },
    {
        "app_id": "payflow",
        "app_name": "PayFlow Wallet",
        "description": (
            "Send money abroad in minutes. Live exchange rates, recipient "
            "management and transfers to 90 countries."
        ),
        "score": 3.8,
        "ratings_count": 27655,
        "downloads": 1000000,
        "categories": "Finance",
        "themes": PAYFLOW_THEMES,
    },
)

#: Trailing filler, applied sparingly.
#:
#: These are deliberately *rare* and worded as short continuations. An earlier
#: version appended one to every review from a pool of ten standalone sentences;
#: segmentation split them off as their own clauses, and because they repeated
#: verbatim they formed the largest, most cohesive clusters in every app -
#: themes made entirely of "anyone else seeing this?" with no product signal.
#: Filler belongs in the corpus, but it must not out-mass the actual complaints.
QUALIFIERS = (
    " Please fix this.",
    " Very frustrating.",
    " This has been going on for months now.",
    " Otherwise a decent app.",
    " Genuinely considering switching.",
    " Happens on both my devices.",
    " Rating will go up when it is fixed.",
    " Nobody seems to be looking at it.",
)
QUALIFIER_PROBABILITY = 0.25


#: Connectives used to join two complaints into one review. Real reviews carry
#: several grievances at once, and the segmenter splits on exactly these - a
#: corpus of single-sentence reviews would never exercise that.
JOINERS = (" But ", " Also ", " On top of that ", " And then ")


def build_reviews(rng: random.Random) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for app in APPS:
        themes: tuple[Theme, ...] = app["themes"]  # type: ignore[assignment]
        complaint_themes = [t for t in themes if t[0] != "praise"]
        per_theme = REVIEWS_PER_APP // len(themes)

        for name, score, templates in themes:
            for index in range(per_theme):
                text = rng.choice(templates)

                # ~35% of complaints mention a second, different problem. This
                # produces multi-clause reviews so segmentation has real work to
                # do, and keeps near-duplicate rates realistic rather than
                # letting eight templates repeat verbatim 75 times each.
                if name != "praise" and rng.random() < 0.35:
                    other_name, _, other_templates = rng.choice(complaint_themes)
                    if other_name != name:
                        text += rng.choice(JOINERS) + rng.choice(other_templates).lower()

                if rng.random() < QUALIFIER_PROBABILITY:
                    text += rng.choice(QUALIFIERS)

                # Ratings wobble by one so the star distribution is not a
                # giveaway of the theme, and covers all five levels.
                rating = min(5, max(1, score + rng.choice((-1, 0, 0, 0, 1))))
                offset = rng.randint(0, 540)
                rows.append(
                    {
                        "app_id": app["app_id"],
                        "review_text": text,
                        "review_score": rating,
                        "review_date": (END_DATE - timedelta(days=offset)).isoformat(),
                        "helpful_count": max(0, int(rng.lognormvariate(1.0, 1.3)) - 1),
                        "_theme": name,
                        "_index": index,
                    }
                )
    rng.shuffle(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %-40s %5d rows", path.relative_to(ROOT), len(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data" / "fixtures",
        help="output directory (default: data/fixtures)",
    )
    args = parser.parse_args(argv)
    setup_logging()

    rng = random.Random(SEED)
    reviews = build_reviews(rng)
    apps = [{k: v for k, v in app.items() if k != "themes"} for app in APPS]
    out = args.out

    # --- 1. the happy path -------------------------------------------------
    write_csv(
        out / "valid_apps.csv", apps,
        ["app_id", "app_name", "description", "score", "ratings_count",
         "downloads", "categories"],
    )
    write_csv(
        out / "valid_reviews.csv", reviews,
        ["app_id", "review_text", "review_score", "review_date", "helpful_count"],
    )

    # --- 2. alias headers --------------------------------------------------
    # Same data under the header names a different scraper would emit. Proves
    # `resolve_columns` without needing a second code path.
    write_csv(
        out / "aliased_apps.csv",
        [
            {"appId": a["app_id"], "title": a["app_name"], "summary": a["description"],
             "rating": a["score"], "installs": a["downloads"], "genre": a["categories"]}
            for a in apps
        ],
        ["appId", "title", "summary", "rating", "installs", "genre"],
    )
    write_csv(
        out / "aliased_reviews.csv",
        [
            {"appId": r["app_id"], "content": r["review_text"], "stars": r["review_score"],
             "at": r["review_date"], "thumbsUpCount": r["helpful_count"]}
            for r in reviews
        ],
        ["appId", "content", "stars", "at", "thumbsUpCount"],
    )

    # --- 3. deliberately broken -------------------------------------------
    # `app_name` missing entirely -> a hard error naming the column.
    write_csv(
        out / "broken_apps.csv",
        [{"app_id": a["app_id"], "score": a["score"]} for a in apps],
        ["app_id", "score"],
    )
    # Blank text and unparseable dates -> warnings with row counts, not errors.
    broken_reviews = []
    for position, row in enumerate(reviews[:300]):
        broken_reviews.append(
            {
                "app_id": row["app_id"],
                "review_text": "" if position % 10 == 0 else row["review_text"],
                "review_score": row["review_score"],
                "review_date": "not a date" if position % 7 == 0 else row["review_date"],
                "helpful_count": row["helpful_count"],
            }
        )
    write_csv(
        out / "broken_reviews.csv", broken_reviews,
        ["app_id", "review_text", "review_score", "review_date", "helpful_count"],
    )

    print(f"\nFixtures written to {out.relative_to(ROOT)}/\n")
    print("  Upload Dataset page — try these pairs:")
    print("    happy path      valid_apps.csv    + valid_reviews.csv")
    print("    alias resolver  aliased_apps.csv  + aliased_reviews.csv")
    print("    diagnostics     broken_apps.csv   + broken_reviews.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
