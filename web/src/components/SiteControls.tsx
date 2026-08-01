"use client";

import { useI18n, type Locale } from "@/lib/i18n";
import { useTheme } from "@/lib/theme";

const LOCALES: { id: Locale; short: string; label: string }[] = [
  { id: "en", short: "EN", label: "English" },
  { id: "ru", short: "RU", label: "Русский" },
  { id: "hy", short: "ՀԱՅ", label: "Հայերեն" },
];

function SunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="1.75" />
      <path
        d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.6 5.6l1.6 1.6M16.8 16.8l1.6 1.6M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinecap="round"
      />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M18.5 14.2A7.2 7.2 0 0 1 9.8 5.5 7.4 7.4 0 1 0 18.5 14.2Z"
        stroke="currentColor"
        strokeWidth="1.75"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function SiteControls() {
  const { theme, toggleTheme } = useTheme();
  const { locale, setLocale, t } = useI18n();

  return (
    <div className="pointer-events-none fixed inset-x-0 top-0 z-50 flex justify-end px-4 pt-4 sm:px-6">
      <div className="pointer-events-auto flex items-center gap-2 rounded-2xl border border-[var(--border-strong)] bg-[var(--surface)]/90 p-1.5 shadow-[var(--shadow)] backdrop-blur-md">
        <div
          className="flex items-center rounded-xl bg-[var(--surface-muted)]/80 p-0.5"
          role="group"
          aria-label={t("language")}
        >
          {LOCALES.map((item) => {
            const active = locale === item.id;
            return (
              <button
                key={item.id}
                type="button"
                title={item.label}
                aria-pressed={active}
                onClick={() => setLocale(item.id)}
                className={`min-w-[2.6rem] rounded-lg px-2.5 py-1.5 text-[11px] font-semibold tracking-wide transition ${
                  active
                    ? "bg-[var(--foreground)] text-[var(--background)] shadow-sm"
                    : "text-[var(--muted)] hover:text-[var(--foreground)]"
                }`}
              >
                {item.short}
              </button>
            );
          })}
        </div>

        <div className="h-6 w-px bg-[var(--border)]" aria-hidden />

        <button
          type="button"
          onClick={toggleTheme}
          aria-label={
            theme === "light" ? t("themeDark") : t("themeLight")
          }
          title={theme === "light" ? t("themeDark") : t("themeLight")}
          className="group relative flex h-9 w-9 items-center justify-center rounded-xl border border-[var(--border)] bg-[var(--input-bg)] text-[var(--foreground)] transition hover:border-[var(--border-strong)] hover:bg-[var(--surface-muted)]"
        >
          <span
            className={`absolute transition duration-300 ${
              theme === "light"
                ? "scale-100 opacity-100"
                : "scale-75 opacity-0"
            }`}
          >
            <SunIcon />
          </span>
          <span
            className={`absolute transition duration-300 ${
              theme === "dark"
                ? "scale-100 opacity-100"
                : "scale-75 opacity-0"
            }`}
          >
            <MoonIcon />
          </span>
        </button>
      </div>
    </div>
  );
}
