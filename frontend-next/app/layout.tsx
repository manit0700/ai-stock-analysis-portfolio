import type { Metadata } from "next";
import "./globals.css";

// System monospace fonts only: the build must not depend on fetching a font from
// Google at compile time (it fails on machines without that network access), and
// no request leaves the user's browser for a font.

export const metadata: Metadata = {
  title: "MarketVision AI Terminal",
  description: "Stock research, probability forecasts and a simulated $50 trading account. SIMULATED TRADING — NO REAL MONEY.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body
        className="bg-[#040404] text-[#c8c8c8] antialiased"
        style={{ fontFamily: "ui-monospace, 'JetBrains Mono', 'Cascadia Code', Menlo, Consolas, 'Courier New', monospace" }}
      >
        {children}
      </body>
    </html>
  );
}
