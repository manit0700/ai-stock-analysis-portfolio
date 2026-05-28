import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "MarketVision AI Terminal",
  description: "Bloomberg-style AI stock analysis and portfolio intelligence terminal.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${jetbrainsMono.variable} bg-[#040404] text-[#c8c8c8] antialiased`} style={{ fontFamily: "var(--font-mono), 'Courier New', monospace" }}>
        {children}
      </body>
    </html>
  );
}
