"use client";

import { useI18n, type Locale } from "@/lib/i18n";
import { useTheme, type Theme } from "@/lib/theme";

export function SiteControls() {
  const { theme, setTheme } = useTheme();
  const { locale, setLocale, t } = useI18n();

  return (
    <div className="fixed right-4 top-4 z-50 flex items-center gap-2 rounded-xl border border-[var(--border-strong)] bg-[var(--surface)] px-2.5 py-1.5 shadow-[var(--shadow)]">
      <label className="sr-only" htmlFor="theme-select">
        {t("themeLabel")}
      </label>
      <select
        id="theme-select"
        value={theme}
        onChange={(e) => setTheme(e.target.value as Theme)}
        className="rounded-lg border border-[var(--border)] bg-[var(--input-bg)] px-2 py-1 text-xs font-semibold text-[var(--foreground)] outline-none"
      >
        <option value="light">{t("themeLight")}</option>
        <option value="dark">{t("themeDark")}</option>
      </select>
      <label className="sr-only" htmlFor="locale-select">
        {t("language")}
      </label>
      <select
        id="locale-select"
        value={locale}
        onChange={(e) => setLocale(e.target.value as Locale)}
        className="rounded-lg border border-[var(--border)] bg-[var(--input-bg)] px-2 py-1 text-xs font-semibold text-[var(--foreground)] outline-none"
        title={t("language")}
      >
        <option value="en">EN</option>
        <option value="ru">RU</option>
        <option value="hy">ՀԱՅ</option>
      </select>
    </div>
  );
}
