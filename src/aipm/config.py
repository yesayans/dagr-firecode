"""Central configuration. Everything tunable lives here, nothing is hardcoded elsewhere.

Two independent AI backends:

* **LLM** - any OpenAI-compatible endpoint (`LLM_BASE_URL`, `LLM_API_KEY`,
  `LLM_MODEL`). We never assume server-side `json_schema` enforcement; structured
  output is specified in the prompt and validated with pydantic.
* **Embeddings** - selected by `EMBED_BACKEND` (``api`` | ``local`` | ``fixture``).
  ``EMBED_MODEL`` and ``EMBED_DIM`` are deliberately ``None`` by default so each
  backend resolves its own identity instead of inheriting a hardcoded guess.

Secrets belong in ``.env`` (gitignored), never in this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

EmbedBackend = Literal["api", "local", "fixture"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM (OpenAI-compatible router) -----------------------------------
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.2
    llm_max_retries: int = 3
    llm_timeout_s: int = 60
    llm_max_output_tokens: int = 2048
    #: Sent instead of the OpenAI SDK's own User-Agent. Some routers sit behind a
    #: WAF that 403s anything advertising itself as an SDK client (autorouter.io
    #: blocks ``OpenAI/Python*``), so this is not cosmetic. Set empty to keep the
    #: SDK default.
    http_user_agent: str = "aipm/0.1.0"

    # --- Embeddings -------------------------------------------------------
    embed_backend: EmbedBackend = "local"
    #: ``None`` means "let the active backend decide". Never hardcode a model or
    #: dimension here: a value set at this level silently overrides every backend.
    embed_model: str | None = None
    embed_dim: int | None = None
    embed_batch_size: int = 256
    #: Only used by the ``api`` backend; falls back to the LLM credentials.
    embed_base_url: str = ""
    embed_api_key: str = ""

    # --- Storage ----------------------------------------------------------
    storage_backend: str = "sqlite"  # "sqlite" | "postgres"
    sqlite_path: Path = ROOT / "data" / "aipm.db"
    postgres_dsn: str = ""

    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    runs_dir: Path = ROOT / "data" / "runs"
    fixtures_dir: Path = ROOT / "data" / "fixtures"
    embedding_cache_dir: Path = ROOT / "data" / "cache" / "embeddings"

    apps_csv_name: str = "apps_info.csv"
    reviews_csv_name: str = "apps_reviews.csv"

    # --- Pipeline thresholds ---------------------------------------------
    min_segment_tokens: int = 4
    near_dup_threshold: float = 0.95
    min_cluster_size_floor: int = 15
    min_cluster_size_ratio: float = 0.01
    min_clusters_before_fallback: int = 3
    citation_relevance_threshold: float = 0.30
    n_representatives: int = 10
    max_clusters_to_label: int = 12

    # --- Confidence weights (must sum to 1.0) -----------------------------
    w_support: float = 0.25
    w_cohesion: float = 0.20
    w_separation: float = 0.10
    w_temporal: float = 0.15
    w_diversity: float = 0.15
    w_grounding: float = 0.15

    # --- Demo precompute --------------------------------------------------
    demo_n_apps: int = 8
    demo_min_apps: int = 5
    demo_max_apps: int = 10
    demo_max_reviews_per_app: int = 4000

    # --- UI ---------------------------------------------------------------
    demo_mode: bool = Field(
        default=True,
        description="Read only precomputed runs. Never calls the LLM. Use for demos.",
    )

    def confidence_weights(self) -> dict[str, float]:
        return {
            "support": self.w_support,
            "cohesion": self.w_cohesion,
            "separation": self.w_separation,
            "temporal": self.w_temporal,
            "diversity": self.w_diversity,
            "grounding": self.w_grounding,
        }

    # --- Derived paths ----------------------------------------------------
    @property
    def apps_csv(self) -> Path:
        return self.raw_dir / self.apps_csv_name

    @property
    def reviews_csv(self) -> Path:
        return self.raw_dir / self.reviews_csv_name

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.raw_dir,
            self.runs_dir,
            self.fixtures_dir,
            self.embedding_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Credential resolution -------------------------------------------
    def resolved_embed_base_url(self) -> str:
        return self.embed_base_url or self.llm_base_url

    def resolved_embed_api_key(self) -> str:
        return self.embed_api_key or self.llm_api_key

    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url)


settings = Settings()
