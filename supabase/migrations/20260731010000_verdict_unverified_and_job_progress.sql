-- Allow UNVERIFIED verdicts, persist confidence metrics, track job stage/progress.
-- Idempotent: safe to re-run.

alter table public.gaps drop constraint if exists gaps_verdict_check;
alter table public.gaps
  add constraint gaps_verdict_check
  check (verdict in ('IGNORED', 'UNDER-PRIORITIZED', 'MISUNDERSTOOD', 'UNVERIFIED'));

alter table public.gaps
  add column if not exists metrics jsonb not null default '{}'::jsonb;

alter table public.gaps
  add column if not exists one_sentence_summary text not null default '';

alter table public.analysis_jobs
  add column if not exists stage text;

alter table public.analysis_jobs
  add column if not exists progress int not null default 0;

alter table public.analysis_jobs
  add column if not exists stats jsonb not null default '{}'::jsonb;

alter table public.analysis_jobs
  add column if not exists config_hash text;

create index if not exists analysis_jobs_config_hash_idx
  on public.analysis_jobs (app_id, config_hash)
  where status = 'completed';
