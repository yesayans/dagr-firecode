# Handoff prompt for Cowork / Claude Code

Paste this as your first message in a Cowork session, after pointing it at
`~/Documents/ai-pm-assistant`.

---

I'm building a hackathon project: an AI Product Manager Assistant. It's a
Streamlit app that analyses Google Play Store reviews to discover *hidden user
needs*, not just sentiment.

The full architecture is in `ARCHITECTURE.md` in this folder. Read it first and
follow it — the design decisions in it are deliberate.

Current state (Phase 0, partially done):
- Directory tree is scaffolded
- `src/aipm/schemas.py` — complete, this is the contract every module codes against
- `src/aipm/config.py` — complete
- `src/aipm/utils/` — hashing, logging, timing — complete
- Everything else is empty directories

What I need next, in this order:
1. Finish Phase 0: storage repository ABC + SQLite implementation, Streamlit
   shell with all 6 pages rendering from fixture data, pyproject.toml, .env.example
2. Generate realistic fixture data so the UI is clickable before any real
   pipeline exists
3. Phase 1: ingest + preprocess + stats/trends working on real CSVs

Rules:
- No `st.` calls anywhere under `src/` — the UI layer is `app/` only
- The LLM never produces a number. Confidence, reach, impact and priority are
  computed in Python from the data.
- Every need must cite review IDs, and citations get validated before display
- Work in small steps and show me diffs before large rewrites

Datasets are `apps_info.csv` and `apps_reviews.csv` — I'll drop them in `data/raw/`.
