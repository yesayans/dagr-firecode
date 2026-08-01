import type { Gap, Job, TranslateResponse } from "./types";

export function applyJobTranslation(
  job: Job,
  translation: TranslateResponse | null,
): Job {
  if (!translation) return job;
  const byId = new Map(translation.gaps.map((g) => [g.gap_id, g]));
  const gaps: Gap[] = job.gaps.map((gap) => {
    const t = byId.get(gap.id);
    if (!t) return gap;
    const evidence = gap.evidence.map((ev) => {
      const te = t.evidence.find((x) => x.evidence_id === ev.evidence_id);
      if (!te) return ev;
      return {
        ...ev,
        title: te.title || ev.title,
        snippet: te.snippet || ev.snippet,
      };
    });
    return {
      ...gap,
      need: t.need || gap.need,
      one_sentence_summary: t.one_sentence_summary || gap.one_sentence_summary,
      latent_reasoning: t.latent_reasoning || gap.latent_reasoning,
      confidence_rationale: t.confidence_rationale || gap.confidence_rationale,
      metrics: {
        ...gap.metrics,
        surface_complaints:
          t.surface_complaints.length > 0
            ? t.surface_complaints
            : gap.metrics.surface_complaints,
        workarounds:
          t.workarounds.length > 0 ? t.workarounds : gap.metrics.workarounds,
      },
      evidence,
    };
  });
  return {
    ...job,
    summary: translation.summary || job.summary,
    gaps,
  };
}
