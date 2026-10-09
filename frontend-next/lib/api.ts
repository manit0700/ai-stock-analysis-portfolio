import type { AgentReportResponse, BacktestResult, BotExplanationResponse, BotHistoricalSimilarityResult, BotPerformanceResponse, BotScanResponse, CompareResult, CopilotChatResponse, HealthResponse, MacroConditions, MarketOverviewResponse, NewsItem, PortfolioAnalysis, SimulationResponse, SimilarityResult, StockAnalysis, StockOverview, WatchlistResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `API ${path} failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}

export async function fetchMarketOverview(): Promise<MarketOverviewResponse> {
  return apiFetch<MarketOverviewResponse>("/api/market/overview");
}

export async function fetchStockNews(ticker: string, limit = 5): Promise<NewsItem[]> {
  const data = await apiFetch<{ ticker: string; items: NewsItem[] }>(`/api/stocks/${ticker}/news?limit=${limit}`);
  return data.items;
}

export async function fetchStockOverview(ticker: string, period = "6mo"): Promise<StockOverview> {
  return apiFetch<StockOverview>(`/api/stocks/${ticker}/overview?period=${period}`);
}

export async function fetchStockAnalysis(ticker: string, period = "6mo"): Promise<StockAnalysis> {
  return apiFetch<StockAnalysis>(`/api/stocks/${ticker}/analysis?period=${period}`);
}

export async function fetchPrediction(payload: {
  ticker: string;
  period: string;
  interval: string;
  horizon_steps: number;
}): Promise<SimulationResponse> {
  return apiFetch<SimulationResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchMacroConditions(): Promise<MacroConditions> {
  return apiFetch<MacroConditions>("/api/market/macro");
}

export async function fetchExtendedStockNews(ticker: string, limit = 8): Promise<NewsItem[]> {
  const data = await apiFetch<{ ticker: string; items: NewsItem[] }>(`/api/stocks/${ticker}/news/extended?limit=${limit}`);
  return data.items;
}

export async function copilotChat(
  message: string,
  context?: Record<string, unknown>,
  history?: { role: string; content: string }[]
): Promise<CopilotChatResponse> {
  return apiFetch<CopilotChatResponse>("/api/copilot/chat", {
    method: "POST",
    body: JSON.stringify({ message, context, history }),
  });
}

export async function runAgentTradeCandidates(payload: {
  prompt: string;
  universe?: string;
  tickers?: string | null;
  period?: string;
  interval?: string;
  horizon_steps?: number;
  max_candidates?: number;
}): Promise<AgentReportResponse> {
  return apiFetch<AgentReportResponse>("/api/agents/trade-candidates", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchMarketNewsItems(limit = 8): Promise<NewsItem[]> {
  const data = await apiFetch<{ items: NewsItem[] }>(`/api/market/news?limit=${limit}`);
  return data.items;
}

export async function analyzePortfolio(
  holdings: { ticker: string; shares: number; average_cost?: number | null }[],
  period = "1y"
): Promise<PortfolioAnalysis> {
  return apiFetch<PortfolioAnalysis>("/api/portfolio/analyze", {
    method: "POST",
    body: JSON.stringify({ holdings, period }),
  });
}

export async function fetchWatchlist(
  tickers: string[],
  period = "1mo"
): Promise<WatchlistResponse> {
  return apiFetch<WatchlistResponse>("/api/stocks/watchlist", {
    method: "POST",
    body: JSON.stringify({ tickers, period, interval: "1d" }),
  });
}

export async function fetchStockQuote(ticker: string): Promise<{ quote: { current_price: number; change_pct: number } }> {
  return apiFetch(`/api/stocks/${ticker}/quote`);
}

export async function fetchSimilarity(ticker: string, topN = 5): Promise<SimilarityResult> {
  return apiFetch<SimilarityResult>(`/api/stocks/${ticker}/similarity?top_n=${topN}`);
}

export async function fetchBotHistoricalSimilarity(ticker: string, horizonSteps = 12): Promise<BotHistoricalSimilarityResult> {
  const params = new URLSearchParams({ ticker, horizon_steps: String(horizonSteps) });
  return apiFetch<BotHistoricalSimilarityResult>(`/api/bot/historical-similarity?${params.toString()}`);
}

export async function explainBotPrediction(ticker: string, period = "1y", interval = "1d", horizonSteps = 12): Promise<BotExplanationResponse> {
  return apiFetch<BotExplanationResponse>("/api/bot/explain", {
    method: "POST",
    body: JSON.stringify({ ticker, period, interval, horizon_steps: horizonSteps }),
  });
}

export async function compareStocks(tickers: string[], period = "1y"): Promise<CompareResult> {
  return apiFetch<CompareResult>("/api/stocks/compare", {
    method: "POST",
    body: JSON.stringify({ tickers, period, interval: "1d" }),
  });
}

export async function runBacktest(
  ticker: string,
  strategy: string,
  period: string,
  capital: number,
): Promise<BacktestResult> {
  return apiFetch<BacktestResult>(
    `/api/backtest/${ticker}?strategy=${strategy}&period=${period}&capital=${capital}`
  );
}

export async function fetchBacktestStrategies(): Promise<{ strategies: { id: string; label: string }[] }> {
  return apiFetch("/api/backtest/strategies");
}

export async function scanBotSignals(
  tickers: string,
  period = "5d",
  interval = "1d",
  horizonSteps = 12,
  universe = "custom",
): Promise<BotScanResponse> {
  const params = new URLSearchParams({ tickers, period, interval, horizon_steps: String(horizonSteps), universe });
  return apiFetch<BotScanResponse>(`/api/bot/scan?${params.toString()}`);
}

export async function fetchBotPerformance(): Promise<BotPerformanceResponse> {
  return apiFetch<BotPerformanceResponse>("/api/bot/performance");
}

export async function resolveBotOutcomes(): Promise<{ performance: BotPerformanceResponse; resolved: number; skipped: number }> {
  return apiFetch("/api/bot/resolve-outcomes", { method: "POST" });
}

// ── v2: forecasts, the prediction ledger and the simulated $50 account ─────────

export type ForecastResponse = {
  symbol: string;
  status: "model" | "baseline" | "unavailable";
  probability_up: number | null;
  base_rate?: number | null;
  edge_established?: boolean;
  horizon_sessions: number;
  target: string;
  as_of?: string;
  window?: { first_session: string; last_session: string };
  model: { id: string | null; kind: string; error?: string | null };
  expected_return?: { value: number | null; method: string } | null;
  range?: { method: string; p05: number; p10: number; p50: number; p90: number; p95: number; meaning: string } | null;
  message?: string;
  reason?: { code: string; message: string };
  data_source?: { provider: string; freshness: string; retrieved_at: string };
  prediction_id?: string;
  disclaimer: string;
};

export type PredictionRow = {
  prediction_id: string; at: string; symbol: string; as_of: string; horizon_sessions: number;
  window_last_session: string; model_id: string | null; model_kind: string; probability_up: number;
  resolved: boolean;
  outcome: { realized_return: number; outcome_up: boolean; correct: boolean; window_complete: boolean; within_stated_range: boolean | null; at: string } | null;
};

export type PredictionLedgerResponse = {
  history: PredictionRow[];
  performance: {
    issued: number; resolved: number; open: number; resolved_with_complete_window: number;
    accuracy: number | null; majority_class_accuracy?: number; brier: number | null; brier_of_base_rate?: number;
    brier_skill?: number | null; range_coverage?: number | null; note?: string;
    chain?: { intact: boolean; events: number };
  };
};

export type PaperPosition = {
  symbol: string; quantity: string; average_cost: string; cost_basis: string;
  price: string | null; market_value: string | null; unrealized_pnl: string | null;
};

export type PaperAccount = {
  simulation: true; banner: string; account_id: string; created_at: string; paused: boolean;
  starting_cash: string; cash: string; equity: string | null; market_value: string | null;
  realized_pnl: string; unrealized_pnl: string | null; total_pnl: string | null; fees_paid: string; dividends: string;
  positions: PaperPosition[]; orders_submitted: number; marks_missing_live_price: string[]; mark_note?: string | null;
};

export type PaperOrder = {
  order_id: string; client_order_id: string; at: string; side: "buy" | "sell"; symbol: string;
  status: "filled" | "rejected"; reason: string | null; quantity: string | null; fill_price: string | null;
  gross: string | null; fee: string | null; cash_after: string | null; realized: string | null;
};

export type PaperPerformance = {
  banner: string; starting_cash: string; equity: string | null; total_pnl: string | null; realized_pnl: string;
  unrealized_pnl: string | null; fees_paid: string; return_pct: number | null; closed_sales: number;
  winning_sales: number; win_rate_pct: number | null;
};

export type ForecastStatus = {
  champion: string | null; edge_established: boolean; model_state: string;
  baseline?: { base_rate?: number; verdict?: string; from_run?: string } | null;
  schema: { name: string; version: string; hash: string };
};

async function v2<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}/api/v2${path}`, {
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
    ...init,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(body?.error?.message ?? `Request failed with status ${response.status}`);
  }
  return body as T;
}

export function fetchForecast(symbol: string): Promise<ForecastResponse> {
  return v2<ForecastResponse>(`/stocks/${encodeURIComponent(symbol)}/forecast`);
}

export function fetchForecastStatus(): Promise<ForecastStatus> {
  return v2<ForecastStatus>("/forecast/status");
}

export function fetchPredictionLedger(symbol?: string): Promise<PredictionLedgerResponse> {
  const query = symbol ? `?symbol=${encodeURIComponent(symbol)}` : "";
  return v2<PredictionLedgerResponse>(`/predictions${query}`);
}

export function fetchPaperAccount(): Promise<PaperAccount> {
  return v2<PaperAccount>("/paper/account");
}

export function fetchPaperOrders(limit = 50): Promise<{ orders: PaperOrder[] }> {
  return v2<{ orders: PaperOrder[] }>(`/paper/orders?limit=${limit}`);
}

export function fetchPaperPerformance(): Promise<PaperPerformance> {
  return v2<PaperPerformance>("/paper/performance");
}

export function submitPaperOrder(order: { side: "buy" | "sell"; symbol: string; quantity?: number; notional?: number; client_order_id?: string }): Promise<PaperOrder> {
  return v2<PaperOrder>("/paper/orders", { method: "POST", body: JSON.stringify({ ...order, simulation: true }) });
}

export function resetPaperAccount(): Promise<{ account_id: string; starting_cash: string }> {
  return v2<{ account_id: string; starting_cash: string }>("/paper/reset", { method: "POST", body: JSON.stringify({ reason: "reset from the dashboard" }) });
}
