import type {
  App,
  ComponentKey,
  EvidenceItem,
  Gap,
  GapMetrics,
  Job,
  LaterAddressedBy,
  RoadmapSource,
  Verdict,
} from "./types";
import { COMPONENT_KEYS } from "./types";

const REVIEW_WINDOW_START = "2016-04-01T00:00:00Z";
const REVIEW_WINDOW_END = "2016-09-30T23:59:59Z";

const WEIGHTS_ROADMAP = {
  volume: 0.3,
  novelty: 0.25,
  consistency: 0.2,
  severity: 0.15,
  spread: 0.1,
} as const;

const WEIGHTS_NONE = {
  volume: 0.35,
  novelty: 0.0,
  consistency: 0.3,
  severity: 0.2,
  spread: 0.15,
} as const;

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function buildMetrics(input: {
  mode: RoadmapSource;
  cluster_size: number;
  total_reviews: number;
  max_cluster_size: number;
  best_similarity: number | null;
  mean_rating: number;
  rating_spread: number;
  cohesion: number;
  matched_item_title: string | null;
  matched_item_url: string | null;
  matched_item_state: string | null;
  matched_item_age_days: number | null;
  llm_confidence: number | null;
  keywords: string[];
  validated_by_later_roadmap?: boolean;
  later_addressed_by?: LaterAddressedBy | null;
  review_window_start?: string;
  review_window_end?: string;
}): GapMetrics {
  const weights = input.mode === "none" ? WEIGHTS_NONE : WEIGHTS_ROADMAP;
  const volume =
    Math.log1p(input.cluster_size) / Math.log1p(input.max_cluster_size);
  const novelty =
    input.mode === "none" ? 1.0 : 1 - (input.best_similarity ?? 0);
  const consistency = input.cohesion;
  const severity = (5 - input.mean_rating) / 4;
  const spread = input.rating_spread;

  const components = {
    volume: round2(volume),
    novelty: round2(novelty),
    consistency: round2(consistency),
    severity: round2(severity),
    spread: round2(spread),
  };

  const deterministic = round2(
    100 *
      COMPONENT_KEYS.reduce(
        (sum, key: ComponentKey) => sum + weights[key] * components[key],
        0,
      ),
  );

  const review_window_start =
    input.review_window_start ?? REVIEW_WINDOW_START;
  const review_window_end = input.review_window_end ?? REVIEW_WINDOW_END;
  const validated = input.validated_by_later_roadmap ?? false;
  const later =
    input.later_addressed_by !== undefined
      ? input.later_addressed_by
      : null;

  return {
    cluster_size: input.cluster_size,
    total_reviews: input.total_reviews,
    cluster_share: round2(input.cluster_size / input.total_reviews),
    best_similarity: input.mode === "none" ? null : input.best_similarity,
    matched_item_title: input.matched_item_title,
    matched_item_url: input.matched_item_url,
    matched_item_state: input.matched_item_state,
    matched_item_age_days: input.matched_item_age_days,
    mean_rating: input.mean_rating,
    rating_spread: input.rating_spread,
    cohesion: input.cohesion,
    components,
    weights: { ...weights },
    deterministic_confidence: deterministic,
    llm_confidence: input.llm_confidence,
    keywords: input.keywords,
    review_window_start,
    review_window_end,
    reference_date: review_window_end,
    later_addressed_by: later,
    validated_by_later_roadmap: validated,
  };
}

function finalConfidence(metrics: GapMetrics): {
  confidence: number;
  rationale: string;
} {
  if (metrics.llm_confidence === null) {
    return {
      confidence: metrics.deterministic_confidence,
      rationale: `Deterministic only: round(100 × Σ wᵢcᵢ) = ${metrics.deterministic_confidence}`,
    };
  }
  const blended = round2(
    0.6 * metrics.deterministic_confidence + 0.4 * metrics.llm_confidence,
  );
  return {
    confidence: blended,
    rationale: `Blend 0.6×deterministic (${metrics.deterministic_confidence}) + 0.4×LLM (${metrics.llm_confidence}) = ${blended}`,
  };
}

function reviewEvidence(
  id: string,
  stars: number,
  text: string,
  metrics: GapMetrics,
): EvidenceItem {
  return {
    evidence_id: id,
    source_type: "review",
    title: `${stars}★ review`,
    snippet: text,
    url: null,
    payload: {
      review_id: id,
      stars,
      cluster_size: metrics.cluster_size,
      total_reviews: metrics.total_reviews,
      best_similarity: metrics.best_similarity,
      mean_rating: metrics.mean_rating,
      rating_spread: metrics.rating_spread,
      cohesion: metrics.cohesion,
      components: metrics.components,
      weights: metrics.weights,
    },
  };
}

export const MOCK_APPS: App[] = [
  {
    id: "app-antennapod",
    package_name: "de.danoeh.antennapod",
    display_name: "AntennaPod",
    review_count: 18420,
    avg_stars: 4.4,
    github_repo: "AntennaPod/AntennaPod",
    roadmap_source: "github",
    roadmap_item_count: 142,
    sample_review:
      "Love the open-source podcast app, but sleep timer keeps resetting when I switch episodes.",
  },
  {
    id: "app-signal",
    package_name: "org.thoughtcrime.securesms",
    display_name: "Signal",
    review_count: 412000,
    avg_stars: 4.6,
    github_repo: null,
    roadmap_source: "web",
    roadmap_item_count: 28,
    sample_review:
      "Private messaging is great but group voice notes still fail on older Android devices.",
  },
  {
    id: "app-instagram",
    package_name: "com.instagram.android",
    display_name: "Instagram",
    review_count: 125000000,
    avg_stars: 3.9,
    github_repo: null,
    roadmap_source: "none",
    roadmap_item_count: 0,
    sample_review:
      "Chronological feed option disappeared again — Reels push is exhausting.",
  },
  {
    id: "app-newpipe",
    package_name: "org.schabi.newpipe",
    display_name: "NewPipe",
    review_count: 9200,
    avg_stars: 4.5,
    github_repo: "TeamNewPipe/NewPipe",
    roadmap_source: "hybrid",
    roadmap_item_count: 96,
    sample_review:
      "Background playback is essential; sponsor-block integration still missing for live streams.",
  },
];

function makeGithubGaps(): Gap[] {
  const total = 2000;
  const maxCluster = 186;

  const specs: Array<{
    rank: number;
    need: string;
    summary: string;
    verdict: Verdict;
    cluster_size: number;
    best_similarity: number;
    mean_rating: number;
    rating_spread: number;
    cohesion: number;
    matched_title: string;
    matched_url: string;
    matched_state: string;
    age_days: number;
    llm: number | null;
    keywords: string[];
    reasoning: string;
    reviews: Array<{ id: string; stars: number; text: string }>;
    extra: EvidenceItem[];
    validated_by_later_roadmap: boolean;
    later_addressed_by: LaterAddressedBy | null;
  }> = [
    {
      rank: 1,
      need: "Reliable sleep timer across episode switches",
      summary:
        "Users lose the sleep timer when changing episodes or skipping chapters mid-session.",
      verdict: "IGNORED",
      cluster_size: 186,
      best_similarity: 0.21,
      mean_rating: 2.1,
      rating_spread: 0.8,
      cohesion: 0.84,
      matched_title: "Improve notification actions",
      matched_url: "https://github.com/AntennaPod/AntennaPod/issues/6120",
      matched_state: "open",
      age_days: 120,
      llm: 91,
      keywords: ["sleep timer", "episode switch", "reset"],
      reasoning:
        "A large, cohesive cluster complains about timer reset; nearest roadmap item is only loosely related to notifications.",
      reviews: [
        {
          id: "rev-ap-001",
          stars: 2,
          text: "Sleep timer resets every time I switch to the next episode. Makes bedtime listening useless.",
        },
        {
          id: "rev-ap-002",
          stars: 1,
          text: "Please fix the sleep timer — it forgets my 30-minute setting when I skip chapters.",
        },
      ],
      extra: [
        {
          evidence_id: "gh-issue-6120",
          source_type: "github_issue",
          title: "Improve notification actions",
          snippet: "Open issue about notification controls; no mention of sleep timer persistence.",
          url: "https://github.com/AntennaPod/AntennaPod/issues/6120",
          payload: { state: "open", number: 6120 },
        },
      ],
      // Surfaced from 2016 reviews with no contemporaneous match; team shipped it in 2019.
      validated_by_later_roadmap: true,
      later_addressed_by: {
        title: "Keep sleep timer across episode changes",
        url: "https://github.com/AntennaPod/AntennaPod/issues/3142",
        state: "closed",
        date: "2019-03-14T00:00:00Z",
        similarity: 0.78,
      },
    },
    {
      rank: 2,
      need: "Offline download queue that survives app kills",
      summary:
        "Download queues vanish after force-stop or OS memory reclaim on mid-range phones.",
      verdict: "UNDER-PRIORITIZED",
      cluster_size: 142,
      best_similarity: 0.62,
      mean_rating: 2.4,
      rating_spread: 0.6,
      cohesion: 0.79,
      matched_title: "Persist download queue across restarts",
      matched_url: "https://github.com/AntennaPod/AntennaPod/issues/4891",
      matched_state: "open",
      age_days: 410,
      llm: 84,
      keywords: ["downloads", "queue", "offline", "persist"],
      reasoning:
        "Matched open issue is stale (>365 days) with no milestone — classic under-prioritization signal.",
      reviews: [
        {
          id: "rev-ap-010",
          stars: 2,
          text: "Queued 40 episodes for a flight, killed the app once, queue gone. Offline travel is broken.",
        },
        {
          id: "rev-ap-011",
          stars: 3,
          text: "Downloads restart from zero after Android clears memory. Need durable queue.",
        },
      ],
      extra: [
        {
          evidence_id: "gh-issue-4891",
          source_type: "github_issue",
          title: "Persist download queue across restarts",
          snippet: "Open for 410 days, no milestone assigned.",
          url: "https://github.com/AntennaPod/AntennaPod/issues/4891",
          payload: { state: "open", number: 4891, age_days: 410 },
        },
        {
          evidence_id: "gh-ms-3.6",
          source_type: "github_milestone",
          title: "Milestone 3.6 — Downloads",
          snippet: "Milestone closed without including queue persistence.",
          url: "https://github.com/AntennaPod/AntennaPod/milestone/36",
          payload: { state: "closed" },
        },
      ],
      validated_by_later_roadmap: false,
      later_addressed_by: null,
    },
    {
      rank: 3,
      need: "Car Bluetooth resume without skipping ahead",
      summary:
        "Auto-resume after car Bluetooth reconnect jumps several minutes ahead of last position.",
      verdict: "MISUNDERSTOOD",
      cluster_size: 98,
      best_similarity: 0.71,
      mean_rating: 2.0,
      rating_spread: 0.4,
      cohesion: 0.88,
      matched_title: "Fix Bluetooth playback resume",
      matched_url: "https://github.com/AntennaPod/AntennaPod/issues/5502",
      matched_state: "closed",
      age_days: 45,
      llm: 88,
      keywords: ["bluetooth", "car", "resume", "skip"],
      reasoning:
        "Issue marked closed/shipped, yet recent reviews still report the skip-ahead bug after reconnect.",
      reviews: [
        {
          id: "rev-ap-020",
          stars: 1,
          text: "Still skips 2–3 minutes when my car reconnects. Claimed fixed in last release — not for me.",
        },
        {
          id: "rev-ap-021",
          stars: 2,
          text: "Bluetooth resume is worse after the 'fix'. Loses my place every commute.",
        },
      ],
      extra: [
        {
          evidence_id: "gh-issue-5502",
          source_type: "github_issue",
          title: "Fix Bluetooth playback resume",
          snippet: "Closed as completed 45 days ago; users still report skip-ahead.",
          url: "https://github.com/AntennaPod/AntennaPod/issues/5502",
          payload: { state: "closed", number: 5502 },
        },
      ],
      validated_by_later_roadmap: false,
      later_addressed_by: null,
    },
    {
      rank: 4,
      need: "Transcript search across subscribed shows",
      summary:
        "Power users want full-text search over episode transcripts, not just titles and descriptions.",
      verdict: "IGNORED",
      cluster_size: 74,
      best_similarity: 0.18,
      mean_rating: 3.0,
      rating_spread: 0.6,
      cohesion: 0.72,
      matched_title: "Improve episode search UI",
      matched_url: "https://github.com/AntennaPod/AntennaPod/issues/6011",
      matched_state: "open",
      age_days: 88,
      llm: null,
      keywords: ["transcript", "search", "chapters"],
      reasoning:
        "Roadmap search work targets titles/descriptions; transcript search demand is unmet.",
      reviews: [
        {
          id: "rev-ap-030",
          stars: 3,
          text: "Wish I could search inside transcripts — titles never match the moment I remember.",
        },
        {
          id: "rev-ap-031",
          stars: 2,
          text: "Transcript search would make this the best research podcast app. Nothing on the roadmap.",
        },
      ],
      extra: [
        {
          evidence_id: "gh-issue-6011",
          source_type: "github_issue",
          title: "Improve episode search UI",
          snippet: "UI polish for title/description search only.",
          url: "https://github.com/AntennaPod/AntennaPod/issues/6011",
          payload: { state: "open", number: 6011 },
        },
      ],
      // Never addressed after the 2016 review window — decade of standing.
      validated_by_later_roadmap: false,
      later_addressed_by: null,
    },
  ];

  return specs.map((s) => {
    const metrics = buildMetrics({
      mode: "github",
      cluster_size: s.cluster_size,
      total_reviews: total,
      max_cluster_size: maxCluster,
      best_similarity: s.best_similarity,
      mean_rating: s.mean_rating,
      rating_spread: s.rating_spread,
      cohesion: s.cohesion,
      matched_item_title: s.matched_title,
      matched_item_url: s.matched_url,
      matched_item_state: s.matched_state,
      matched_item_age_days: s.age_days,
      llm_confidence: s.llm,
      keywords: s.keywords,
      validated_by_later_roadmap: s.validated_by_later_roadmap,
      later_addressed_by: s.later_addressed_by,
    });
    const { confidence, rationale } = finalConfidence(metrics);
    return {
      id: `gap-gh-${s.rank}`,
      rank: s.rank,
      need: s.need,
      one_sentence_summary: s.summary,
      verdict: s.verdict,
      confidence,
      confidence_rationale: rationale,
      latent_reasoning: s.reasoning,
      metrics,
      evidence: [
        ...s.reviews.map((r) =>
          reviewEvidence(r.id, r.stars, r.text, metrics),
        ),
        ...s.extra,
      ],
    };
  });
}

function makeWebGaps(): Gap[] {
  const total = 2000;
  const maxCluster = 210;

  const specs: Array<{
    rank: number;
    need: string;
    summary: string;
    verdict: Verdict;
    cluster_size: number;
    best_similarity: number;
    mean_rating: number;
    rating_spread: number;
    cohesion: number;
    matched_title: string;
    matched_url: string;
    matched_state: string;
    age_days: number;
    llm: number;
    keywords: string[];
    reasoning: string;
    reviews: Array<{ id: string; stars: number; text: string }>;
  }> = [
    {
      rank: 1,
      need: "Stable group voice notes on older Android",
      summary:
        "Group voice notes fail to send or play on Android 10–11 devices after recent updates.",
      verdict: "IGNORED",
      cluster_size: 210,
      best_similarity: 0.19,
      mean_rating: 1.8,
      rating_spread: 0.6,
      cohesion: 0.86,
      matched_title: "Stories on desktop",
      matched_url: "https://signal.org/blog/signal-stories/",
      matched_state: "shipped",
      age_days: 30,
      llm: 89,
      keywords: ["voice notes", "groups", "android"],
      reasoning:
        "Changelog emphasises Stories; voice-note reliability on older Android is absent.",
      reviews: [
        {
          id: "rev-sig-001",
          stars: 1,
          text: "Group voice notes never send on my Android 10. Private chat voice works fine.",
        },
        {
          id: "rev-sig-002",
          stars: 2,
          text: "Can't play group voice messages after the last update. Pixel 3a.",
        },
      ],
    },
    {
      rank: 2,
      need: "Username discovery without sharing phone number",
      summary:
        "Users want to add contacts via username alone; phone-number requirement remains a blocker.",
      verdict: "UNDER-PRIORITIZED",
      cluster_size: 168,
      best_similarity: 0.58,
      mean_rating: 2.6,
      rating_spread: 0.8,
      cohesion: 0.81,
      matched_title: "Usernames — gradual rollout",
      matched_url: "https://signal.org/blog/phone-number-privacy-usernames/",
      matched_state: "in_progress",
      age_days: 390,
      llm: 82,
      keywords: ["username", "privacy", "phone number"],
      reasoning:
        "Blog post acknowledges usernames but rollout notes are stale relative to review volume.",
      reviews: [
        {
          id: "rev-sig-010",
          stars: 2,
          text: "Still can't find friends by username without giving my phone number. Privacy promise incomplete.",
        },
        {
          id: "rev-sig-011",
          stars: 3,
          text: "Username feature teased for ages. Reviews keep asking — when is it universal?",
        },
      ],
    },
    {
      rank: 3,
      need: "Reliable link previews that respect disappearing messages",
      summary:
        "Link previews leak preview text into notifications even when disappearing messages are on.",
      verdict: "MISUNDERSTOOD",
      cluster_size: 91,
      best_similarity: 0.66,
      mean_rating: 2.2,
      rating_spread: 0.4,
      cohesion: 0.77,
      matched_title: "Notification privacy improvements",
      matched_url: "https://signal.org/blog/hide-message-content/",
      matched_state: "shipped",
      age_days: 60,
      llm: 80,
      keywords: ["link preview", "notifications", "disappearing"],
      reasoning:
        "Changelog claims notification privacy shipped; users still see link-preview leaks.",
      reviews: [
        {
          id: "rev-sig-020",
          stars: 2,
          text: "Disappearing messages on, but link previews still show in the lock-screen notification.",
        },
        {
          id: "rev-sig-021",
          stars: 1,
          text: "Said they fixed notification content. Link titles still leak. Not fixed.",
        },
      ],
    },
    {
      rank: 4,
      need: "Export chat archive without desktop tether",
      summary:
        "Mobile-only users cannot export full chat history without pairing a desktop client.",
      verdict: "IGNORED",
      cluster_size: 64,
      best_similarity: 0.22,
      mean_rating: 2.9,
      rating_spread: 0.6,
      cohesion: 0.7,
      matched_title: "Backup improvements",
      matched_url: "https://support.signal.org/hc/en-us/articles/360007059752",
      matched_state: "documented",
      age_days: 200,
      llm: 74,
      keywords: ["export", "archive", "backup", "mobile"],
      reasoning:
        "Support docs cover encrypted backups, not portable chat export from mobile alone.",
      reviews: [
        {
          id: "rev-sig-030",
          stars: 3,
          text: "Need to export chats as PDF/text from the phone. Desktop-only export is a non-starter.",
        },
        {
          id: "rev-sig-031",
          stars: 2,
          text: "No laptop — can't archive years of chats. Mobile export please.",
        },
      ],
    },
  ];

  return specs.map((s) => {
    const metrics = buildMetrics({
      mode: "web",
      cluster_size: s.cluster_size,
      total_reviews: total,
      max_cluster_size: maxCluster,
      best_similarity: s.best_similarity,
      mean_rating: s.mean_rating,
      rating_spread: s.rating_spread,
      cohesion: s.cohesion,
      matched_item_title: s.matched_title,
      matched_item_url: s.matched_url,
      matched_item_state: s.matched_state,
      matched_item_age_days: s.age_days,
      llm_confidence: s.llm,
      keywords: s.keywords,
    });
    const { confidence, rationale } = finalConfidence(metrics);
    const webPage: EvidenceItem = {
      evidence_id: `web-${s.rank}`,
      source_type: "web_page",
      title: s.matched_title,
      snippet: `Roadmap/changelog entry used for similarity matching (s=${s.best_similarity}).`,
      url: s.matched_url,
      payload: { state: s.matched_state },
    };
    return {
      id: `gap-web-${s.rank}`,
      rank: s.rank,
      need: s.need,
      one_sentence_summary: s.summary,
      verdict: s.verdict,
      confidence,
      confidence_rationale: rationale,
      latent_reasoning: s.reasoning,
      metrics,
      evidence: [
        ...s.reviews.map((r) =>
          reviewEvidence(r.id, r.stars, r.text, metrics),
        ),
        webPage,
      ],
    };
  });
}

function makeNoneGaps(): Gap[] {
  const total = 2000;
  const maxCluster = 320;

  const specs: Array<{
    rank: number;
    need: string;
    summary: string;
    cluster_size: number;
    mean_rating: number;
    rating_spread: number;
    cohesion: number;
    llm: number;
    keywords: string[];
    reasoning: string;
    reviews: Array<{ id: string; stars: number; text: string }>;
  }> = [
    {
      rank: 1,
      need: "Optional chronological main feed",
      summary:
        "Users repeatedly ask to restore a pure chronological feed instead of algorithmic Reels push.",
      cluster_size: 320,
      mean_rating: 1.6,
      rating_spread: 0.8,
      cohesion: 0.9,
      llm: 93,
      keywords: ["chronological", "feed", "algorithm", "reels"],
      reasoning:
        "Highest-volume cluster with severe ratings; no public roadmap available to verify coverage.",
      reviews: [
        {
          id: "rev-ig-001",
          stars: 1,
          text: "Bring back chronological feed. Reels drowning out friends is why I opened one star.",
        },
        {
          id: "rev-ig-002",
          stars: 2,
          text: "I just want posts from people I follow, in order. Algorithm makes the app unusable.",
        },
      ],
    },
    {
      rank: 2,
      need: "Granular mute for suggested posts",
      summary:
        "Mute controls do not stop suggested posts and ads from reappearing in the same session.",
      cluster_size: 244,
      mean_rating: 2.0,
      rating_spread: 0.6,
      cohesion: 0.83,
      llm: 86,
      keywords: ["mute", "suggested", "ads"],
      reasoning:
        "Strong consistency around mute failing for suggested content; surfaced from reviews only.",
      reviews: [
        {
          id: "rev-ig-010",
          stars: 2,
          text: "I mute suggested posts and they come right back. Mute does nothing.",
        },
        {
          id: "rev-ig-011",
          stars: 1,
          text: "Can't escape suggested content. Need a real off switch, not a placebo mute.",
        },
      ],
    },
    {
      rank: 3,
      need: "Longer draft retention for Reels and carousels",
      summary:
        "Drafts disappear after app updates or overnight, losing hours of creator work.",
      cluster_size: 155,
      mean_rating: 2.3,
      rating_spread: 0.6,
      cohesion: 0.78,
      llm: 81,
      keywords: ["drafts", "reels", "carousel", "lost"],
      reasoning:
        "Creators report draft loss independently of feed complaints — distinct latent need.",
      reviews: [
        {
          id: "rev-ig-020",
          stars: 2,
          text: "Spent three hours on a Reel draft. Update overnight wiped it. Unforgivable.",
        },
        {
          id: "rev-ig-021",
          stars: 3,
          text: "Carousel drafts vanish if I leave the app. Save drafts to cloud please.",
        },
      ],
    },
    {
      rank: 4,
      need: "Cross-profile inbox for creator + personal accounts",
      summary:
        "Switching profiles drops unread DMs and notification badges, causing missed replies.",
      cluster_size: 118,
      mean_rating: 2.5,
      rating_spread: 0.4,
      cohesion: 0.74,
      llm: 77,
      keywords: ["multi-account", "inbox", "notifications"],
      reasoning:
        "Multi-account users form a coherent cluster about inbox blindness across profiles.",
      reviews: [
        {
          id: "rev-ig-030",
          stars: 2,
          text: "Switch to creator account and personal DMs go silent. Missed three client messages.",
        },
        {
          id: "rev-ig-031",
          stars: 3,
          text: "Need a unified inbox badge across profiles. Constantly missing replies.",
        },
      ],
    },
  ];

  return specs.map((s) => {
    const metrics = buildMetrics({
      mode: "none",
      cluster_size: s.cluster_size,
      total_reviews: total,
      max_cluster_size: maxCluster,
      best_similarity: null,
      mean_rating: s.mean_rating,
      rating_spread: s.rating_spread,
      cohesion: s.cohesion,
      matched_item_title: null,
      matched_item_url: null,
      matched_item_state: null,
      matched_item_age_days: null,
      llm_confidence: s.llm,
      keywords: s.keywords,
    });
    const { confidence, rationale } = finalConfidence(metrics);
    return {
      id: `gap-none-${s.rank}`,
      rank: s.rank,
      need: s.need,
      one_sentence_summary: s.summary,
      verdict: "UNVERIFIED" as const,
      confidence,
      confidence_rationale: rationale,
      latent_reasoning: s.reasoning,
      metrics,
      evidence: s.reviews.map((r) =>
        reviewEvidence(r.id, r.stars, r.text, metrics),
      ),
    };
  });
}

function makeHybridGaps(): Gap[] {
  const total = 1800;
  const maxCluster = 130;
  const metrics = buildMetrics({
    mode: "hybrid",
    cluster_size: 130,
    total_reviews: total,
    max_cluster_size: maxCluster,
    best_similarity: 0.24,
    mean_rating: 2.2,
    rating_spread: 0.6,
    cohesion: 0.85,
    matched_item_title: "SponsorBlock for VOD",
    matched_item_url: "https://github.com/TeamNewPipe/NewPipe/issues/9100",
    matched_item_state: "open",
    matched_item_age_days: 95,
    llm_confidence: 87,
    keywords: ["sponsorblock", "live", "streams"],
  });
  const { confidence, rationale } = finalConfidence(metrics);

  const gaps: Gap[] = [
    {
      id: "gap-hy-1",
      rank: 1,
      need: "SponsorBlock on live streams",
      one_sentence_summary:
        "Users want SponsorBlock segments during live and premiere streams, not only VOD.",
      verdict: "IGNORED",
      confidence,
      confidence_rationale: rationale,
      latent_reasoning:
        "GitHub issue covers VOD; web changelog never mentions live SponsorBlock.",
      metrics,
      evidence: [
        reviewEvidence(
          "rev-np-001",
          2,
          "SponsorBlock works for uploads but live streams still blast mid-roll ads. Please extend it.",
          metrics,
        ),
        {
          evidence_id: "gh-np-9100",
          source_type: "github_issue",
          title: "SponsorBlock for VOD",
          snippet: "Scoped to VOD only; live streams out of scope.",
          url: "https://github.com/TeamNewPipe/NewPipe/issues/9100",
          payload: { state: "open" },
        },
        {
          evidence_id: "web-np-changelog",
          source_type: "web_page",
          title: "NewPipe release notes",
          snippet: "Recent releases list player fixes; no live SponsorBlock.",
          url: "https://newpipe.net/",
          payload: {},
        },
      ],
    },
  ];

  // Pad to 4 gaps for hybrid fixture completeness
  const padSpecs = [
    {
      rank: 2,
      need: "Tablet-optimized playlist editor",
      summary: "Large-screen layout still uses phone list density for playlist reordering.",
      verdict: "UNDER-PRIORITIZED" as Verdict,
      cluster_size: 88,
      sim: 0.55,
      rating: 2.8,
      spread: 0.4,
      cohesion: 0.76,
      title: "Tablet UI improvements",
      url: "https://github.com/TeamNewPipe/NewPipe/issues/7201",
      state: "open",
      age: 400,
      llm: 78,
    },
    {
      rank: 3,
      need: "Remember playback speed per channel",
      summary: "Global speed resets when switching channels; users want per-channel defaults.",
      verdict: "MISUNDERSTOOD" as Verdict,
      cluster_size: 72,
      sim: 0.69,
      rating: 2.4,
      spread: 0.6,
      cohesion: 0.8,
      title: "Per-stream playback speed",
      url: "https://github.com/TeamNewPipe/NewPipe/issues/8012",
      state: "closed",
      age: 20,
      llm: 83,
    },
    {
      rank: 4,
      need: "Export subscriptions as OPML from settings",
      summary: "Subscription export is buried; users fail to migrate devices without it.",
      verdict: "IGNORED" as Verdict,
      cluster_size: 51,
      sim: 0.2,
      rating: 3.1,
      spread: 0.4,
      cohesion: 0.71,
      title: "Settings reorganisation",
      url: "https://newpipe.net/blog/",
      state: "documented",
      age: 150,
      llm: 72,
    },
  ];

  for (const p of padSpecs) {
    const m = buildMetrics({
      mode: "hybrid",
      cluster_size: p.cluster_size,
      total_reviews: total,
      max_cluster_size: maxCluster,
      best_similarity: p.sim,
      mean_rating: p.rating,
      rating_spread: p.spread,
      cohesion: p.cohesion,
      matched_item_title: p.title,
      matched_item_url: p.url,
      matched_item_state: p.state,
      matched_item_age_days: p.age,
      llm_confidence: p.llm,
      keywords: ["hybrid"],
    });
    const fin = finalConfidence(m);
    gaps.push({
      id: `gap-hy-${p.rank}`,
      rank: p.rank,
      need: p.need,
      one_sentence_summary: p.summary,
      verdict: p.verdict,
      confidence: fin.confidence,
      confidence_rationale: fin.rationale,
      latent_reasoning: `Hybrid evidence from GitHub + web changelog for "${p.title}".`,
      metrics: m,
      evidence: [
        reviewEvidence(
          `rev-np-0${p.rank}0`,
          Math.round(p.rating),
          `Users keep reporting: ${p.need.toLowerCase()}.`,
          m,
        ),
        {
          evidence_id: `hy-link-${p.rank}`,
          source_type: p.url.includes("github") ? "github_issue" : "web_page",
          title: p.title,
          snippet: `Matched item state=${p.state}, age=${p.age}d.`,
          url: p.url,
          payload: { state: p.state },
        },
      ],
    });
  }

  return gaps;
}

function completedJob(
  id: string,
  app: App,
  gaps: Gap[],
  summary: string,
  roadmapItems: number,
  opts?: { llm_used?: boolean; degraded?: string[] },
): Job {
  return {
    id,
    app,
    status: "completed",
    stage: "done",
    progress: 100,
    error: null,
    roadmap_source: app.roadmap_source,
    summary,
    stats: {
      total_reviews: 2000,
      clusters: 48,
      roadmap_items: roadmapItems,
      llm_used: opts?.llm_used ?? true,
      embedding_backend: "minilm",
      elapsed_s: 42.6,
      degraded: opts?.degraded ?? [],
      review_provenance: "fixture",
      reviews_total: 2000,
      reviews_need_bearing: 860,
      review_window_start: REVIEW_WINDOW_START,
      review_window_end: REVIEW_WINDOW_END,
      reference_date: REVIEW_WINDOW_END,
      charts: {
        period: "year",
        reviews_by_period: [
          { period: "2012", count: 90 },
          { period: "2013", count: 180 },
          { period: "2014", count: 420 },
          { period: "2015", count: 610 },
          { period: "2016", count: 790 },
        ],
        rating_histogram: [
          { stars: 1, count: 220 },
          { stars: 2, count: 310 },
          { stars: 3, count: 480 },
          { stars: 4, count: 520 },
          { stars: 5, count: 470 },
        ],
        need_bearing: { need_bearing: 860, other: 1140 },
      },
    },
    gaps,
    created_at: "2026-07-31T10:00:00Z",
    completed_at: "2026-07-31T10:00:43Z",
  };
}

const antennapod = MOCK_APPS[0];
const signal = MOCK_APPS[1];
const instagram = MOCK_APPS[2];
const newpipe = MOCK_APPS[3];

export const MOCK_JOBS: Record<string, Job> = {
  "job-github-antennapod": completedJob(
    "job-github-antennapod",
    antennapod,
    makeGithubGaps(),
    "AntennaPod's GitHub roadmap leaves sleep-timer reliability and durable downloads under-served relative to review volume.",
    142,
  ),
  "job-web-signal": completedJob(
    "job-web-signal",
    signal,
    makeWebGaps(),
    "Signal's public changelog emphasises Stories and usernames while older-Android voice notes dominate negative reviews.",
    28,
  ),
  "job-none-instagram": completedJob(
    "job-none-instagram",
    instagram,
    makeNoneGaps(),
    "No public roadmap was discoverable; needs below are ranked from review evidence alone and marked UNVERIFIED.",
    0,
  ),
  "job-hybrid-newpipe": completedJob(
    "job-hybrid-newpipe",
    newpipe,
    makeHybridGaps(),
    "Hybrid GitHub + web sources confirm live SponsorBlock and tablet playlist gaps.",
    96,
  ),
};

export const MOCK_HEALTH = {
  ok: true,
  store: "mock",
  llm_enabled: true,
  llm_model: "openai/gpt-4o-mini",
  github_token: true,
  embedding_backend: "minilm",
  match_threshold: 0.45,
};

export function searchMockApps(q: string, limit = 25): App[] {
  const needle = q.trim().toLowerCase();
  const list = !needle
    ? MOCK_APPS
    : MOCK_APPS.filter(
        (a) =>
          a.display_name.toLowerCase().includes(needle) ||
          a.package_name.toLowerCase().includes(needle),
      );
  return list.slice(0, limit);
}

export function resolveMockApp(input: {
  app_name: string;
  package_name: string;
  github_repo: string | null;
}): App {
  const found = MOCK_APPS.find(
    (a) =>
      a.package_name === input.package_name ||
      a.display_name.toLowerCase() === input.app_name.toLowerCase(),
  );
  if (found) {
    return {
      ...found,
      github_repo: input.github_repo ?? found.github_repo,
    };
  }
  return {
    id: `app-${input.package_name}`,
    package_name: input.package_name,
    display_name: input.app_name,
    review_count: 0,
    avg_stars: null,
    github_repo: input.github_repo,
    roadmap_source: input.github_repo ? "github" : "none",
    roadmap_item_count: input.github_repo ? 12 : 0,
    sample_review: null,
  };
}

export function jobForApp(appId: string): Job {
  const map: Record<string, string> = {
    "app-antennapod": "job-github-antennapod",
    "app-signal": "job-web-signal",
    "app-instagram": "job-none-instagram",
    "app-newpipe": "job-hybrid-newpipe",
  };
  const jobId = map[appId] ?? "job-none-instagram";
  return structuredClone(MOCK_JOBS[jobId]);
}
