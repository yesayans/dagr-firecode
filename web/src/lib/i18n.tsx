"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export type Locale = "en" | "hy";

const STORAGE_KEY = "dagr-locale";

const dict = {
  en: {
    brandEyebrow: "Silent Stakeholder",
    tagline: "Cross-reference app-store reviews against a product roadmap — for any app, including closed-source ones with no GitHub repo — and surface latent needs the roadmap misses.",
    findApp: "Find an app",
    searchPlaceholder: "Search by name or package — try AntennaPod, Signal, Instagram…",
    searching: "searching…",
    noApps: "No apps matched \"{q}\". Use the form below to upload your own reviews instead.",
    resolving: "Resolving roadmap source…",
    reviews: "Reviews",
    avgStars: "Avg stars",
    roadmapItems: "Roadmap items",
    analyze: "Analyze reviews",
    analyzing: "Starting…",
    newAnalysis: "← New analysis",
    jobLabel: "dagr · job",
    loadingJob: "Loading job…",
    backSearch: "← Back to search",
    analysisFailed: "Analysis failed",
    retry: "Retry analysis",
    retrying: "Retrying…",
    clusters: "Clusters",
    elapsed: "Elapsed",
    degraded: "Degraded run — not full fidelity",
    gapTitle: "Gap ranking — unmet & under-served needs",
    gapTitleNone: "Surfaced Needs (no public roadmap found to verify against)",
    noGaps: "No gaps emitted for this run.",
    chartsTitle: "Review signals",
    chartsSubtitle: "Volume over time, star mix, and need-bearing share for this job.",
    reviewsOverTime: "Reviews by year",
    ratingMix: "Star distribution",
    needBearing: "Need-bearing vs other",
    needBearingLabel: "Need-bearing",
    otherReviews: "Other",
    noChartData: "No date/rating series for this run.",
    chatTitle: "Discuss the evidence",
    chatSubtitle: "Ask about priorities, themes, or strength of evidence. Answers use only this job's gaps and linked review snippets.",
    chatEmpty: "No messages yet — try a suggestion or ask your own question.",
    chatPlaceholder: "Ask what matters in these reviews…",
    chatAsk: "Ask",
    suggestPrioritize: "What should we prioritize?",
    suggestStrongest: "Which need has the strongest evidence?",
    suggestOffline: "Any theme about offline or sync?",
    themeLight: "Light",
    themeDark: "Dark",
    language: "Language",
  },
  hy: {
    brandEyebrow: "Silent Stakeholder",
    tagline: "Համեմատեք հավելվածի կարծիքները ճանապարհային քարտեզի հետ՝ ցանկացած հավելվածի համար՝ ներառյալ փակ կոդովները՝ և գտեք թաքնված կարիքները፩",
    findApp: "Գտնել հավելված",
    searchPlaceholder: "Որոնել անունով կամ փաթեթով — AntennaPod, Signal, Instagram…",
    searching: "որոնում…",
    noApps: "«{q}»-ին համապատասխան հավելված չկա፩ Օգտագործեք ստորևի ձևը፩",
    resolving: "Ճանապարհային քարտեզի աղբյուրը որոշվում է…",
    reviews: "Կարծիքներ",
    avgStars: "Միջին աստղեր",
    roadmapItems: "Քարտեզի կետեր",
    analyze: "Վերլուծել կարծիքները",
    analyzing: "Սկսվում է…",
    newAnalysis: "← Նոր վերլուծություն",
    jobLabel: "dagr · աշխատանք",
    loadingJob: "Բեռնվում է…",
    backSearch: "← Վերադառնալ որոնմանը",
    analysisFailed: "Վերլուծությունը ձախողվեց",
    retry: "Կրկնել վերլուծությունը",
    retrying: "Կրկնվում է…",
    clusters: "Կլաստերներ",
    elapsed: "Տևողություն",
    degraded: "Նվազեցված որակ — ոչ ամբողջական",
    gapTitle: "Բացերի վարկանիշ — չբավարարված կարիքներ",
    gapTitleNone: "Հայտնաբերված կարիքներ (հանրային քարտեզ չի գտնվել)",
    noGaps: "Այս գործարկման համար բացեր չեն ստեղծվել፩",
    chartsTitle: "Կարծիքների ազդանշաններ",
    chartsSubtitle: "Ծավալը ժամանակի ընթացքում՝ աստղերի բաշխումը և կարիք կրող մասնաբաժինը፩",
    reviewsOverTime: "Կարծիքներ ժամանակի ընթացքում",
    ratingMix: "Աստղերի բաշխում",
    needBearing: "Կարիք կրող և այլ",
    needBearingLabel: "Կարիք կրող",
    otherReviews: "Այլ",
    noChartData: "Այս գործարկման համար ժամանակաշար չկա፩",
    chatTitle: "Քննարկել ապացույցները",
    chatSubtitle: "Հարցրեք առաջնահերթությունների կամ ապացույցների մասին፩ Պատասխանները հիմնված են միայն այս աշխատանքի բացերի վրա፩",
    chatEmpty: "Հաղորդագրություններ չկան — փորձեք առաջարկը կամ գրեք հարց፩",
    chatPlaceholder: "Հարցրեք՝ թե ինչն է կարևոր այս կարծիքներում…",
    chatAsk: "Հարցնել",
    suggestPrioritize: "Ի՞նչը առաջնահերթացնել",
    suggestStrongest: "Ո՞ր կարիքն ունի ամենաուժեղ ապացույցը",
    suggestOffline: "Կա՞ թեմա offline/sync-ի մասին",
    themeLight: "Լուսավոր",
    themeDark: "Մուգ",
    language: "Լեզու",
  },
} as const;

export type MessageKey = keyof typeof dict.en;

type I18nContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: MessageKey, vars?: Record<string, string | number>) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

function readLocale(): Locale {
  if (typeof window === "undefined") return "en";
  const raw = window.localStorage.getItem(STORAGE_KEY);
  return raw === "hy" ? "hy" : "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    setLocaleState(readLocale());
  }, []);

  useEffect(() => {
    document.documentElement.lang = locale === "hy" ? "hy" : "en";
  }, [locale]);

  function setLocale(next: Locale) {
    setLocaleState(next);
    window.localStorage.setItem(STORAGE_KEY, next);
  }

  function t(key: MessageKey, vars?: Record<string, string | number>) {
    let text: string = dict[locale][key] ?? dict.en[key] ?? key;
    if (vars) {
      for (const [k, v] of Object.entries(vars)) {
        text = text.replace(`{${k}}`, String(v));
      }
    }
    return text;
  }

  return (
    <I18nContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used within I18nProvider");
  }
  return ctx;
}
