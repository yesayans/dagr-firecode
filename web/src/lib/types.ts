export type RoadmapSource = "github" | "web" | "hybrid" | "none";
export type Verdict = "IGNORED" | "UNDER-PRIORITIZED" | "MISUNDERSTOOD" | "UNVERIFIED";
export type JobStatus = "queued" | "running" | "completed" | "failed";
export type Stage =
  | "queued"
  | "resolving_roadmap"
  | "fetching_reviews"
  | "embedding"
  | "clustering"
  | "matching"
  | "extracting"
  | "persisting"
  | "done"
  | "failed";

export interface App {
  id: string;
  package_name: string;
  display_name: string;
  review_count: number;
  avg_stars: number | null;
  github_repo: string | null;
  roadmap_source: RoadmapSource;
  roadmap_item_count: number;
  sample_review: string | null;
}

export interface EvidenceItem {
  evidence_id: string;
  source_type:
    | "review"
    | "github_issue"
    | "github_milestone"
    | "web_page"
    | "interview"
    | "other";
  title: string | null;
  snippet: string | null;
  url: string | null;
  payload: Record<string, unknown>;
}

export interface LaterAddressedBy {
  title: string;
  url: string | null;
  state: string | null;
  date: string | null;
  similarity: number;
}

export interface GapMetrics {
  cluster_size: number;
  total_reviews: number;
  cluster_share: number;
  best_similarity: number | null;
  matched_item_title: string | null;
  matched_item_url: string | null;
  matched_item_state: string | null;
  matched_item_age_days: number | null;
  mean_rating: number;
  rating_spread: number;
  cohesion: number;
  components: {
    volume: number;
    novelty: number;
    consistency: number;
    severity: number;
    spread: number;
  };
  weights: {
    volume: number;
    novelty: number;
    consistency: number;
    severity: number;
    spread: number;
  };
  deterministic_confidence: number;
  llm_confidence: number | null;
  keywords: string[];
  review_window_start: string;
  review_window_end: string;
  reference_date: string;
  later_addressed_by: LaterAddressedBy | null;
  validated_by_later_roadmap: boolean;
}

export interface Gap {
  id: string;
  rank: number;
  need: string;
  one_sentence_summary: string;
  verdict: Verdict;
  confidence: number;
  confidence_rationale: string;
  latent_reasoning: string;
  metrics: GapMetrics;
  evidence: EvidenceItem[];
}

export interface JobCharts {
  period: "year" | "month" | string;
  reviews_by_period: { period: string; count: number }[];
  rating_histogram: { stars: number; count: number }[];
  need_bearing: { need_bearing: number; other: number };
}

export interface Job {
  id: string;
  app: App;
  status: JobStatus;
  stage: Stage;
  progress: number;
  error: string | null;
  roadmap_source: RoadmapSource;
  summary: string | null;
  stats: {
    total_reviews: number;
    clusters: number;
    roadmap_items: number;
    llm_used: boolean;
    embedding_backend: string;
    elapsed_s: number;
    degraded: string[];
    review_provenance: "hf" | "parquet_cache" | "fixture" | "csv_upload";
    reviews_total?: number;
    reviews_need_bearing?: number;
    review_window_start: string;
    review_window_end: string;
    reference_date: string;
    charts?: JobCharts;
  };
  gaps: Gap[];
  created_at: string;
  completed_at: string | null;
}

export interface HealthResponse {
  ok: boolean;
  store: string;
  llm_enabled: boolean;
  llm_model: string;
  github_token: boolean;
  embedding_backend: string;
  match_threshold: number;
}

export interface AnalyzeResponse {
  job_id: string;
  status: JobStatus;
}

export interface ResolveAppRequest {
  app_name: string;
  package_name: string;
  github_repo: string | null;
  refresh: boolean;
}

export interface AnalyzeRequest {
  app_id: string;
  max_reviews: number;
  force: boolean;
}

export interface ChatCitation {
  gap_rank: number | null;
  evidence_id: string | null;
  quote: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  citations?: ChatCitation[];
}

export interface ChatRequest {
  message: string;
  history: { role: "user" | "assistant"; content: string }[];
}

export interface ChatResponse {
  answer: string;
  citations: ChatCitation[];
  model: string;
}

export const STAGE_SEQUENCE: Stage[] = [
  "queued",
  "resolving_roadmap",
  "fetching_reviews",
  "embedding",
  "clustering",
  "matching",
  "extracting",
  "persisting",
  "done",
];

export const COMPONENT_KEYS = [
  "volume",
  "novelty",
  "consistency",
  "severity",
  "spread",
] as const;

export type ComponentKey = (typeof COMPONENT_KEYS)[number];
