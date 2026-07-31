import type { Metadata } from "next";
import { IBM_Plex_Mono, Outfit } from "next/font/google";
import { DataModeIndicator } from "@/components/DataModeIndicator";
import { SiteControls } from "@/components/SiteControls";
import { Providers } from "@/components/Providers";
import "./globals.css";

const outfit = Outfit({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "dagr — latent needs from reviews × roadmap",
  description:
    "Cross-reference app-store reviews against any product roadmap to surface unmet user needs — including closed-source apps with no public repo.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('dagr-theme');if(t!=='dark'&&t!=='light')t='light';var r=document.documentElement;r.dataset.theme=t;r.classList.toggle('dark',t==='dark');r.style.colorScheme=t;}catch(e){}})();`,
          }}
        />
      </head>
      <body
        className={`${outfit.variable} ${plexMono.variable} min-h-screen antialiased`}
      >
        <Providers>
          <SiteControls />
          {children}
          <DataModeIndicator />
        </Providers>
      </body>
    </html>
  );
}
