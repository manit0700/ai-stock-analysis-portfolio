"use client";

/**
 * Polls the backend's quote endpoint. The response says which provider the price came from,
 * how old it is and whether it is live, delayed or historical — the interface shows exactly
 * that and never invents a number in between polls.
 *
 * This replaces lib/useFinnhubWS.ts, which opened a Finnhub websocket straight from the
 * browser with an API key embedded in the source. The key never belongs in the frontend:
 * quotes now come from the backend, where the key is configuration.
 */
import { useEffect, useRef, useState } from "react";

export type LiveQuote = {
  price: number | null;
  change_pct: number | null;
  freshness: "live" | "delayed" | "historical";
  provider: string;
  age_seconds: number | null;
  market_state: string | null;
  retrieved_at: string | null;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const POLL_MS = 15_000;

export function useLiveQuote(symbol: string | null): { quote: LiveQuote | null; error: string | null } {
  const [quote, setQuote] = useState<LiveQuote | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    setQuote(null);
    setError(null);
    if (!symbol) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v2/stocks/${encodeURIComponent(symbol)}/quote`);
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          throw new Error(body?.error?.message ?? `quote request failed (${response.status})`);
        }
        const body = await response.json();
        if (cancelled) return;
        setQuote({
          price: body.data?.price ?? null,
          change_pct: body.data?.change_pct ?? null,
          freshness: body.freshness ?? "delayed",
          provider: body.provider ?? "unknown",
          age_seconds: body.data?.age_seconds ?? null,
          market_state: body.data?.market_state ?? null,
          retrieved_at: body.retrieved_at ?? null,
        });
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "quote unavailable");
      }
    };

    poll();
    timer.current = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      if (timer.current) clearInterval(timer.current);
    };
  }, [symbol]);

  return { quote, error };
}

/** Label for the data badge: what the price is, never more than the backend claims. */
export function freshnessLabel(quote: LiveQuote | null): string {
  if (!quote || quote.price == null) return "NO DATA";
  if (quote.freshness === "live") return `LIVE · ${quote.provider}`;
  if (quote.freshness === "delayed") return `DELAYED · ${quote.provider}`;
  return `LAST CLOSE · ${quote.provider}`;
}
