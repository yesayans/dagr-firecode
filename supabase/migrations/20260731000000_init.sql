# Dagr — Silent Stakeholder
# Apps catalog, analysis jobs, ranked unmet-need gaps

create extension if not exists "pgcrypto";

-- Catalog of analyzable products (seeded from HF/Kaggle review datasets)
create table if not exists public.apps (
  id uuid primary key default gen_random_uuid(),
  package_name text not null unique,
  display_name text not null,
  dataset text not null default 'sealuzh/app_reviews',
  review_count int not null default 0,
  avg_stars numeric(3,2),
  github_repo text,
  roadmap_source text check (roadmap_source in ('github','web','hybrid','none')) default 'none',
  sample_review text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists apps_display_name_idx on public.apps using gin (to_tsvector('english', display_name));
create index if not exists apps_package_name_idx on public.apps (package_name);
create index if not exists apps_review_count_idx on public.apps (review_count desc);

-- Analysis runs
create table if not exists public.analysis_jobs (
  id uuid primary key default gen_random_uuid(),
  app_id uuid not null references public.apps(id) on delete cascade,
  status text not null default 'queued'
    check (status in ('queued','running','completed','failed')),
  error text,
  roadmap_snapshot jsonb not null default '{}'::jsonb,
  summary text,
  created_at timestamptz not null default now(),
  started_at timestamptz,
  completed_at timestamptz
);

create index if not exists analysis_jobs_app_id_idx on public.analysis_jobs (app_id);
create index if not exists analysis_jobs_status_idx on public.analysis_jobs (status);

-- Ranked gaps (top 3–5 unmet needs)
create table if not exists public.gaps (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.analysis_jobs(id) on delete cascade,
  rank int not null,
  need text not null,
  confidence numeric(5,2) not null check (confidence >= 0 and confidence <= 100),
  confidence_rationale text not null default '',
  verdict text not null check (verdict in ('IGNORED','UNDER-PRIORITIZED','MISUNDERSTOOD')),
  latent_reasoning text not null default '',
  created_at timestamptz not null default now(),
  unique (job_id, rank)
);

create index if not exists gaps_job_id_idx on public.gaps (job_id);

-- Evidence traces linked to gaps (review IDs, issue #, web URLs)
create table if not exists public.gap_evidence (
  id uuid primary key default gen_random_uuid(),
  gap_id uuid not null references public.gaps(id) on delete cascade,
  evidence_id text not null,
  source_type text not null
    check (source_type in ('review','github_issue','github_milestone','web_page','interview','other')),
  title text,
  snippet text,
  url text,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists gap_evidence_gap_id_idx on public.gap_evidence (gap_id);

-- Optional: store sampled reviews for a job (for evidence replay)
create table if not exists public.job_reviews (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.analysis_jobs(id) on delete cascade,
  external_id text not null,
  star int,
  review_text text not null,
  review_date text,
  created_at timestamptz not null default now()
);

create index if not exists job_reviews_job_id_idx on public.job_reviews (job_id);

-- Public read for demo; writes go through service role / API
alter table public.apps enable row level security;
alter table public.analysis_jobs enable row level security;
alter table public.gaps enable row level security;
alter table public.gap_evidence enable row level security;
alter table public.job_reviews enable row level security;

create policy "apps are publicly readable"
  on public.apps for select using (true);

create policy "jobs are publicly readable"
  on public.analysis_jobs for select using (true);

create policy "gaps are publicly readable"
  on public.gaps for select using (true);

create policy "evidence is publicly readable"
  on public.gap_evidence for select using (true);

create policy "job reviews are publicly readable"
  on public.job_reviews for select using (true);
