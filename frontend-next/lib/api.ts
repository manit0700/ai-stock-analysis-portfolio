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
