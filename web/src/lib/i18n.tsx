"use client";

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export type Locale = "en" | "hy" | "ru";

const STORAGE_KEY = "dagr-locale";

const dict = {
  en: {
    brandEyebrow: "Silent Stakeholder",
    tagline: "Cross-reference app-store reviews against a product roadmap — for any app, including closed-source ones with no GitHub repo — and surface latent needs the roadmap misses.",
    findApp: "Find an app",
    searchPlaceholder: "Search the catalog — AntennaPod, AcDisplay, Kernel Adiutor…",
    searching: "searching…",
    catalogBrowseHint: "Top apps in catalog · github + closed-source (no public roadmap)",
    closedNoRoadmap: "closed / no roadmap",
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
    themeLabel: "Theme",
    translateAnalysis: "Translate analysis with AI",
    translatingAnalysis: "Translating analysis…",
    showOriginalAnalysis: "Show original (English)",
    translateHint: "UI language is set — translate the written gap analysis and review snippets too.",
    translateError: "Translation failed",
    translatedBadge: "AI-translated",
    noRepoWeb: "No repo linked — web changelog / roadmap pages will be used.",
    closedSourceHint: "Closed-source / no public roadmap — analysis will surface UNVERIFIED needs from reviews alone.",
    selectAppHint: "Select an app to resolve its roadmap source, then run analysis. Catalog covers github, web, hybrid, and none modes.",
    bringOwnData: "or bring your own data",
    customTitle: "Analyze your own data",
    customSubtitle: "Upload a reviews CSV for any app — including closed-source. Optionally add changelog/roadmap URLs or paste a feature list.",
    appNameLabel: "App name *",
    packageLabel: "Package / id (optional)",
    reviewsCsvLabel: "Reviews CSV *",
    reviewsCsvHint: "Auto-detects columns like review/text/body, rating/stars, date. 5-star and under-10-word reviews are dropped.",
    roadmapUrlsLabel: "Roadmap / changelog URLs (optional)",
    roadmapPasteLabel: "Paste roadmap / upcoming features (optional)",
    roadmapPasteHint: "One item per line (or blank-line paragraphs). Used when the app has no public GitHub roadmap.",
    uploadAnalyze: "Upload & analyze",
    uploading: "Uploading & analyzing…",
    appNameRequired: "App name is required.",
    csvRequired: "Upload a reviews CSV.",
  },
  hy: {
    brandEyebrow: "Silent Stakeholder",
    tagline: "Համեմատեք հավելվածի խանութի կարծիքները արտադրանքի ճանապարհային քարտեզի հետ՝ ցանկացած հավելվածի համար՝ ներառյալ փակ կոդովները առանց GitHub րեպոզիտորիայի՝ և բացահայտեք թաքնված կարիքները՝ որոնք քարտեզը բաց է թողնում։",
    findApp: "Գտնել հավելված",
    searchPlaceholder: "Որոնել կատալոգում — AntennaPod, AcDisplay, Kernel Adiutor…",
    searching: "որոնում…",
    catalogBrowseHint: "Կատալոգի լավագույն հավելվածներ · github և փակ կոդ (առանց հանրային քարտեզի)",
    closedNoRoadmap: "փակ / առանց քարտեզի",
    noApps: "«{q}»-ին համապատասխան հավելված չկա։ Օգտագործեք ստորևի ձևը՝ սեփական կարծիքներ վերբեռնելու համար։",
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
    gapTitle: "Բացերի վարկանիշ — չբավարարված և թերծառայված կարիքներ",
    gapTitleNone: "Հայտնաբերված կարիքներ (ստուգման համար հանրային քարտեզ չի գտնվել)",
    noGaps: "Այս գործարկման համար բացեր չեն ստեղծվել։",
    chartsTitle: "Կարծիքների ազդանշաններ",
    chartsSubtitle: "Ծավալը տարիներով՝ աստղերի բաշխումը և կարիք կրող մասնաբաժինը այս աշխատանքի համար։",
    reviewsOverTime: "Կարծիքներ ըստ տարիների",
    ratingMix: "Աստղերի բաշխում",
    needBearing: "Կարիք կրող և այլ",
    needBearingLabel: "Կարիք կրող",
    otherReviews: "Այլ",
    noChartData: "Այս գործարկման համար ժամանակաշար կամ գնահատական չկա։",
    chatTitle: "Քննարկել ապացույցները",
    chatSubtitle: "Հարցրեք առաջնահերթությունների՝ թեմաների կամ ապացույցների ուժի մասին։ Պատասխանները հիմնված են միայն այս աշխատանքի բացերի և կապված մեջբերումների վրա։",
    chatEmpty: "Հաղորդագրություններ չկան — փորձեք առաջարկը կամ գրեք ձեր հարցը։",
    chatPlaceholder: "Հարցրեք՝ թե ինչն է կարևոր այս կարծիքներում…",
    chatAsk: "Հարցնել",
    suggestPrioritize: "Ի՞նչը պետք է առաջնահերթացնել",
    suggestStrongest: "Ո՞ր կարիքն ունի ամենաուժեղ ապացույցը",
    suggestOffline: "Կա՞ թեմա offline կամ sync-ի մասին",
    themeLight: "Լուսավոր",
    themeDark: "Մուգ",
    language: "Լեզու",
    themeLabel: "Թեմա",
    translateAnalysis: "Թարգմանել վերլուծությունը AI-ով",
    translatingAnalysis: "Թարգմանվում է…",
    showOriginalAnalysis: "Ցույց տալ բնօրինակը (անգլերեն)",
    translateHint: "Միջերեսի լեզուն ընտրված է — թարգմանեք նաև գրված վերլուծությունը և կարծիքների մեջբերումները։",
    translateError: "Թարգմանությունը ձախողվեց",
    translatedBadge: "AI թարգմանություն",
    noRepoWeb: "Ռեպոզիտորիա կապված չէ — կօգտագործվեն վեբ changelog / քարտեզի էջերը։",
    closedSourceHint: "Փակ կոդ / հանրային քարտեզ չկա — վերլուծությունը կցուցադրի UNVERIFIED կարիքներ միայն կարծիքներից։",
    selectAppHint: "Ընտրեք հավելված՝ քարտեզի աղբյուրը որոշելու համար՝ ապա գործարկեք վերլուծությունը։ Կատալոգը ներառում է github, web, hybrid և none ռեժիմները։",
    bringOwnData: "կամ բերեք ձեր տվյալները",
    customTitle: "Վերլուծել սեփական տվյալները",
    customSubtitle: "Վերբեռնեք կարծիքների CSV ցանկացած հավելվածի համար — ներառյալ փակ կոդովները։ Կարող եք ավելացնել changelog/քարտեզի URL-ներ կամ տեղադրել հնարավորությունների ցանկ։",
    appNameLabel: "Հավելվածի անուն *",
    packageLabel: "Փաթեթ / id (ոչ պարտադիր)",
    reviewsCsvLabel: "Կարծիքների CSV *",
    reviewsCsvHint: "Ինքնաշխատ հայտնաբերում է սյունակներ՝ review/text/body, rating/stars, date։ 5 աստղանի և 10 բառից փակաս կարծիքները հանվում են։",
    roadmapUrlsLabel: "Քարտեզ / changelog URL-ներ (ոչ պարտադիր)",
    roadmapPasteLabel: "Տեղադրել քարտեզ / սպասվող հնարավորություններ (ոչ պարտադիր)",
    roadmapPasteHint: "Մեկ կետ՝ մեկ տողում (կամ պարագրաֆներ դատարկ տողով)։ Օգտագործվում է՝ երբ հավելվածը չունի հանրային GitHub քարտեզ։",
    uploadAnalyze: "Վերբեռնել և վերլուծել",
    uploading: "Վերբեռնվում և վերլուծվում է…",
    appNameRequired: "Հավելվածի անունը պարտադիր է։",
    csvRequired: "Վերբեռնեք կարծիքների CSV։",
  },
  ru: {
    brandEyebrow: "Silent Stakeholder",
    tagline: "Сопоставляйте отзывы из магазина приложений с продуктовой дорожной картой — для любого приложения, включая закрытые без GitHub-репозитория, — и находите скрытые потребности, которые карта упускает.",
    findApp: "Найти приложение",
    searchPlaceholder: "Поиск в каталоге — AntennaPod, AcDisplay, Kernel Adiutor…",
    searching: "поиск…",
    catalogBrowseHint: "Топ приложений в каталоге · github и закрытый код (без публичной карты)",
    closedNoRoadmap: "закрытое / без карты",
    noApps: "Нет приложений по запросу «{q}». Используйте форму ниже, чтобы загрузить свои отзывы.",
    resolving: "Определяем источник дорожной карты…",
    reviews: "Отзывы",
    avgStars: "Средние звёзды",
    roadmapItems: "Пункты карты",
    analyze: "Анализировать отзывы",
    analyzing: "Запуск…",
    newAnalysis: "← Новый анализ",
    jobLabel: "dagr · задача",
    loadingJob: "Загрузка…",
    backSearch: "← Назад к поиску",
    analysisFailed: "Анализ не удался",
    retry: "Повторить анализ",
    retrying: "Повтор…",
    clusters: "Кластеры",
    elapsed: "Время",
    degraded: "Сниженное качество — неполная точность",
    gapTitle: "Рейтинг пробелов — неудовлетворённые потребности",
    gapTitleNone: "Обнаруженные потребности (нет публичной карты для проверки)",
    noGaps: "Для этого запуска пробелы не сформированы.",
    chartsTitle: "Сигналы отзывов",
    chartsSubtitle: "Объём по годам, распределение звёзд и доля содержащих потребность отзывов.",
    reviewsOverTime: "Отзывы по годам",
    ratingMix: "Распределение звёзд",
    needBearing: "С потребностью и прочие",
    needBearingLabel: "С потребностью",
    otherReviews: "Прочие",
    noChartData: "Нет ряда дат/оценок для этого запуска.",
    chatTitle: "Обсудить доказательства",
    chatSubtitle: "Спрашивайте о приоритетах, темах или силе доказательств. Ответы опираются только на пробелы этой задачи и связанные цитаты отзывов.",
    chatEmpty: "Сообщений пока нет — выберите подсказку или задайте свой вопрос.",
    chatPlaceholder: "Спросите, что важно в этих отзывах…",
    chatAsk: "Спросить",
    suggestPrioritize: "Что стоит приоритизировать?",
    suggestStrongest: "У какой потребности самые сильные доказательства?",
    suggestOffline: "Есть ли тема про offline или sync?",
    themeLight: "Светлая",
    themeDark: "Тёмная",
    language: "Язык",
    themeLabel: "Тема",
    translateAnalysis: "Перевести анализ с ИИ",
    translatingAnalysis: "Перевод анализа…",
    showOriginalAnalysis: "Показать оригинал (английский)",
    translateHint: "Язык интерфейса выбран — переведите также текст анализа и цитаты отзывов.",
    translateError: "Не удалось перевести",
    translatedBadge: "Переведено ИИ",
    noRepoWeb: "Репозиторий не привязан — будут использованы веб changelog / страницы карты.",
    closedSourceHint: "Закрытый код / нет публичной карты — анализ покажет UNVERIFIED потребности только из отзывов.",
    selectAppHint: "Выберите приложение, чтобы определить источник карты, затем запустите анализ. Каталог поддерживает режимы github, web, hybrid и none.",
    bringOwnData: "или загрузите свои данные",
    customTitle: "Анализ своих данных",
    customSubtitle: "Загрузите CSV с отзывами для любого приложения — включая закрытые. При желании добавьте URL changelog/карты или вставьте список функций.",
    appNameLabel: "Название приложения *",
    packageLabel: "Пакет / id (необязательно)",
    reviewsCsvLabel: "CSV с отзывами *",
    reviewsCsvHint: "Автоопределение колонок review/text/body, rating/stars, date. Отзывы на 5★ и короче 10 слов отбрасываются.",
    roadmapUrlsLabel: "URL карты / changelog (необязательно)",
    roadmapPasteLabel: "Вставить карту / будущие функции (необязательно)",
    roadmapPasteHint: "Один пункт на строку (или абзацы через пустую строку). Используется, если у приложения нет публичной GitHub-карты.",
    uploadAnalyze: "Загрузить и анализировать",
    uploading: "Загрузка и анализ…",
    appNameRequired: "Укажите название приложения.",
    csvRequired: "Загрузите CSV с отзывами.",
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
  if (raw === "hy" || raw === "ru" || raw === "en") return raw;
  return "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  useEffect(() => {
    setLocaleState(readLocale());
  }, []);

  useEffect(() => {
    document.documentElement.lang =
      locale === "hy" ? "hy" : locale === "ru" ? "ru" : "en";
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
