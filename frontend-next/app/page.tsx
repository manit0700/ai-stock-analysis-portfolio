"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import SimulationChart from "@/components/SimulationChart";
import StockChart from "@/components/StockChart";
import {
  analyzePortfolio, compareStocks, copilotChat, fetchBacktestStrategies,
  fetchHealth, fetchMacroConditions, fetchMarketNewsItems, fetchMarketOverview,
  fetchPrediction, fetchSimilarity, fetchStockAnalysis, fetchStockNews,
  fetchStockOverview, fetchWatchlist, runBacktest, scanBotSignals, fetchBotPerformance, resolveBotOutcomes,
} from "@/lib/api";
import { fallbackSimulation, marketCards as mockMarketCards, newsItems as mockNewsItems } from "@/lib/mock-data";
import type {
  BacktestResult, CompareLeader, CompareResult, MacroConditions, MarketCard,
  NewsItem, PortfolioAnalysis, SimulationResponse, SimilarityResult,
  StockAnalysis, StockOverview, WatchlistItem, BotScanResponse, BotPerformanceResponse,
} from "@/lib/types";

type Tab = "overview" | "research" | "simulation" | "scanner" | "proof" | "portfolio" | "copilot" | "macro" | "compare" | "backtest";

// ── Clock ──────────────────────────────────────────────────────────
function useClock() {
  const [time, setTime] = useState("--:--:--");
  useEffect(() => {
    const tick = () =>
      setTime(new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

// ── Helpers ─────────────────────────────────────────────────────────
function pct(n: number | null | undefined, decimals = 2) {
  if (n == null) return "—";
  return `${n > 0 ? "+" : ""}${n.toFixed(decimals)}%`;
}
function usd(n: number | null | undefined) {
  if (n == null) return "—";
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}
function num(n: unknown, decimals = 2) {
  return typeof n === "number" && Number.isFinite(n) ? n.toFixed(decimals) : "—";
}
function yn(v: unknown) {
  if (typeof v !== "boolean") return "—";
  return v ? "YES" : "NO";
}
function colorChange(v: number) {
  return v >= 0 ? "bb-green" : "bb-red";
}

// ── Panel wrapper ────────────────────────────────────────────────────
function Panel({ title, children, className = "", innerClass = "" }: { title: string; children: React.ReactNode; className?: string; innerClass?: string }) {
  return (
    <div className={`t-panel min-w-0 flex flex-col overflow-hidden ${className}`}>
      <div className="t-header shrink-0">{title}</div>
      <div className={`flex-1 overflow-auto ${innerClass || "p-3"}`}>{children}</div>
    </div>
  );
}

// ── Stat box ─────────────────────────────────────────────────────────
function Stat({ label, value, cls = "" }: { label: string; value: string; cls?: string }) {
  return (
    <div className="border border-[#ff8c0012] bg-[#0a0a0a] p-2">
      <div className="text-[9px] bb-dim uppercase tracking-widest mb-1">{label}</div>
      <div className={`text-sm font-mono font-bold ${cls || "bb-white"}`}>{value}</div>
    </div>
  );
}

// ── Sentiment badge ──────────────────────────────────────────────────
function SentBadge({ label }: { label?: string }) {
  const l = (label ?? "neutral").toLowerCase();
  const cls = l === "positive" ? "text-[#00cc55] border-[#00cc5530]" : l === "negative" ? "text-[#ff4444] border-[#ff444430]" : "bb-dim border-[#33333360]";
  return <span className={`border text-[9px] px-1 uppercase tracking-wider ${cls}`}>{l.slice(0, 3)}</span>;
}

// ═══════════════════════════════════════════════════════════════════
//  TAB PANELS
// ═══════════════════════════════════════════════════════════════════

// ── OVERVIEW TAB ─────────────────────────────────────────────────────
function OverviewTab({
  marketCards, marketCardsLoading, watchlistData, watchlistInput,
  setWatchlistInput, loadWatchlist, watchlistLoading, liveNews,
  goResearch,
}: {
  marketCards: MarketCard[]; marketCardsLoading: boolean;
  watchlistData: WatchlistItem[]; watchlistInput: string;
  setWatchlistInput: (v: string) => void; loadWatchlist: () => void;
  watchlistLoading: boolean; liveNews: NewsItem[];
  goResearch: (ticker: string) => void;
}) {
  return (
    <div className="h-full grid grid-rows-[auto_1fr] gap-px bg-[#ff8c0010]">
      {/* Market cards strip */}
      <div className="grid grid-cols-6 gap-px bg-[#ff8c0010]">
        {marketCards.map((card) => (
          <div key={card.name} className={`t-panel p-3 ${marketCardsLoading ? "opacity-50" : ""}`}>
            <div className="text-[10px] bb-dim uppercase tracking-widest">{card.name}</div>
            <div className="text-lg font-bold bb-white mt-1">{card.value}</div>
            <div className={`text-xs font-bold mt-0.5 ${card.change.startsWith("+") ? "bb-green" : "bb-red"}`}>{card.change}</div>
            <div className="text-[10px] bb-dim mt-1">{card.trend} · {card.volatility}</div>
          </div>
        ))}
      </div>

      {/* Watchlist + News */}
      <div className="grid grid-cols-[1fr_380px] gap-px bg-[#ff8c0010]">
        <Panel title="── WATCHLIST ──────────────────────────────────────────────────" innerClass="">
          <div className="t-header flex items-center gap-2 border-b border-[#ff8c0015] px-3 py-2">
            <input
              value={watchlistInput}
              onChange={(e) => setWatchlistInput(e.target.value.toUpperCase())}
              onKeyDown={(e) => { if (e.key === "Enter") loadWatchlist(); }}
              className="bb-input flex-1 text-xs"
              placeholder="AAPL, NVDA, TSLA, MSFT"
            />
            <button onClick={loadWatchlist} disabled={watchlistLoading} className="bb-btn text-[10px]">
              {watchlistLoading ? "LOADING" : "UPDATE"}
            </button>
          </div>
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-[9px] bb-dim uppercase tracking-widest border-b border-[#ff8c0015]">
                <th className="text-left p-2 pl-3">TICKER</th>
                <th className="text-left p-2">COMPANY</th>
                <th className="text-right p-2">CHANGE</th>
                <th className="text-center p-2">STANCE</th>
                <th className="text-right p-2">SCORE</th>
                <th className="p-2"></th>
              </tr>
            </thead>
            <tbody>
              {watchlistData.map((item) => (
                <tr key={item.ticker} className="t-row">
                  <td className="p-2 pl-3 font-bold bb-amber">{item.ticker}</td>
                  <td className="p-2 bb-dim text-[10px] max-w-[120px] truncate">{item.company_name}</td>
                  <td className={`p-2 text-right font-bold ${colorChange(item.daily_change_pct)}`}>{pct(item.daily_change_pct)}</td>
                  <td className="p-2 text-center">
                    <span className={`text-[10px] uppercase tracking-wider border px-1 ${
                      item.stance === "bullish" ? "text-[#00cc55] border-[#00cc5530]"
                      : item.stance === "cautious" ? "text-[#ff4444] border-[#ff444430]"
                      : "bb-dim border-[#33333360]"
                    }`}>{item.stance}</span>
                  </td>
                  <td className="p-2 text-right bb-white">{item.score}/100</td>
                  <td className="p-2">
                    <button onClick={() => goResearch(item.ticker)} className="bb-btn text-[9px] px-2 py-0.5">DRILL</button>
                  </td>
                </tr>
              ))}
              {watchlistData.length === 0 && (
                <tr><td colSpan={6} className="p-4 text-center bb-dim text-xs">NO DATA — UPDATE WATCHLIST</td></tr>
              )}
            </tbody>
          </table>
        </Panel>

        <Panel title="── HEADLINES ──────────────────────────────────────────────────" innerClass="p-0">
          <div className="divide-y divide-[#ffffff06]">
            {liveNews.slice(0, 8).map((item, i) => (
              <div key={i} className="p-3 hover:bg-[#ff8c0005] transition-colors">
                <div className="flex items-start gap-2">
                  <SentBadge label={(item as NewsItem & { sentiment_label?: string }).sentiment_label} />
                  {item.link ? (
                    <a href={item.link} target="_blank" rel="noreferrer" className="text-[11px] leading-4 bb-white hover:bb-amber transition-colors line-clamp-2">
                      {item.title}
                    </a>
                  ) : (
                    <p className="text-[11px] leading-4 bb-white line-clamp-2">{item.title}</p>
                  )}
                </div>
                <div className="flex gap-3 mt-1.5 text-[9px] bb-dim">
                  <span>{item.publisher ?? "—"}</span>
                  {item.published_at && <span>{new Date(item.published_at).toLocaleDateString()}</span>}
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}

// ── RESEARCH TAB ──────────────────────────────────────────────────────
function ResearchTab({
  stockQuery, setStockQuery, stockPeriod, setStockPeriod,
  loadStock, stockLoading, stockError, stockOverview, stockAnalysis, stockNews,
  similarity, similarityLoading,
}: {
  stockQuery: string; setStockQuery: (v: string) => void;
  stockPeriod: string; setStockPeriod: (v: string) => void;
  loadStock: (t?: string, p?: string) => void; stockLoading: boolean;
  stockError: string | null; stockOverview: StockOverview | null;
  stockAnalysis: StockAnalysis | null; stockNews: NewsItem[];
  similarity: SimilarityResult | null; similarityLoading: boolean;
}) {
  return (
    <div className="h-full min-w-0 flex flex-col gap-px bg-[#ff8c0010]">
      {/* Search bar */}
      <div className="t-panel shrink-0 min-w-0 flex items-center gap-3 overflow-x-auto px-4 py-2">
        <span className="bb-amber text-xs tracking-widest">TICKER&gt;</span>
        <input
          value={stockQuery}
          onChange={(e) => setStockQuery(e.target.value.toUpperCase())}
          onKeyDown={(e) => { if (e.key === "Enter") loadStock(); }}
          className="bb-input w-24 text-sm font-bold bb-cursor"
          placeholder="AAPL"
        />
        <select
          value={stockPeriod}
          onChange={(e) => setStockPeriod(e.target.value)}
          className="bb-input text-xs"
        >
          {["1mo","3mo","6mo","1y","2y"].map((p) => <option key={p}>{p}</option>)}
        </select>
        <button onClick={() => loadStock()} disabled={stockLoading} className="bb-btn">
          {stockLoading ? "LOADING..." : "EXECUTE"}
        </button>
        {stockError && <span className="text-xs bb-red ml-4">[ERR] {stockError}</span>}
        {stockOverview && (
          <div className="ml-4 flex items-baseline gap-3">
            <span className="text-lg font-bold bb-white">{stockOverview.ticker}</span>
            <span className="text-xl font-bold bb-amber">{usd(stockOverview.current_price)}</span>
            <span className={`text-sm font-bold ${colorChange(stockOverview.daily_change_pct)}`}>{pct(stockOverview.daily_change_pct)}</span>
            <span className="text-xs bb-dim">{stockOverview.company_name}</span>
          </div>
        )}
      </div>

      {stockOverview ? (
        <div className="flex-1 grid grid-cols-[1fr_320px] gap-px bg-[#ff8c0010] overflow-hidden">
          {/* Left: chart + metrics */}
          <div className="flex flex-col gap-px bg-[#ff8c0010]">
            {/* Key metrics */}
            <div className="grid grid-cols-6 gap-px bg-[#ff8c0010] shrink-0">
              {[
                ["STANCE", stockAnalysis?.stance?.toUpperCase() ?? "—", stockAnalysis?.stance === "bullish" ? "bb-green" : stockAnalysis?.stance === "cautious" ? "bb-red" : "bb-amber"],
                ["SCORE", `${stockAnalysis?.score ?? "—"}/100`, "bb-white"],
                ["RSI 14", stockOverview.indicators.rsi_14?.toFixed(1) ?? "—", "bb-white"],
                ["SMA 20", stockOverview.indicators.sma_20 ? usd(stockOverview.indicators.sma_20) : "—", "bb-white"],
                ["SMA 50", stockOverview.indicators.sma_50 ? usd(stockOverview.indicators.sma_50) : "—", "bb-white"],
                ["VOLATILITY", stockOverview.indicators.annualized_volatility_30d ? `${(stockOverview.indicators.annualized_volatility_30d * 100).toFixed(1)}%` : "—", "bb-white"],
              ].map(([label, value, cls]) => (
                <Stat key={label as string} label={label as string} value={value as string} cls={cls as string} />
              ))}
            </div>
            {/* Chart */}
            <Panel title={`── PRICE CHART · ${stockOverview.ticker} · ${stockPeriod.toUpperCase()} ─────────────────────────────────`} className="flex-1" innerClass="p-0">
              <StockChart history={stockOverview.price_history} ticker={stockOverview.ticker} />
            </Panel>
          </div>

          {/* Right: signals + fundamentals + news */}
          <div className="flex flex-col gap-px bg-[#ff8c0010] overflow-hidden">
            <Panel title="── FUNDAMENTALS ──────────────────────────────────────────────" innerClass="p-0">
              <table className="w-full text-xs font-mono">
                <tbody>
                  {[
                    ["Mkt Cap", stockOverview.market_cap ? `$${(stockOverview.market_cap / 1e9).toFixed(2)}B` : "—"],
                    ["P/E (TTM)", stockOverview.trailing_pe?.toFixed(1) ?? "—"],
                    ["Fwd P/E", stockOverview.forward_pe?.toFixed(1) ?? "—"],
                    ["52W High", usd(stockOverview.fifty_two_week_high)],
                    ["52W Low", usd(stockOverview.fifty_two_week_low)],
                    ["Div Yield", stockOverview.dividend_yield ? pct(stockOverview.dividend_yield * 100) : "—"],
                    ["Sector", stockOverview.sector ?? "—"],
                  ].map(([k, v]) => (
                    <tr key={k as string} className="t-row">
                      <td className="px-3 py-1.5 bb-dim">{k}</td>
                      <td className="px-3 py-1.5 bb-white text-right font-bold">{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>

            <Panel title="── SIGNALS ───────────────────────────────────────────────────" innerClass="p-2">
              {stockAnalysis?.reasons.map((r) => (
                <div key={r} className="flex gap-2 mb-1.5 text-[11px]">
                  <span className="bb-green shrink-0">▲</span>
                  <span className="bb-white">{r}</span>
                </div>
              ))}
              {stockAnalysis?.risks.map((r) => (
                <div key={r} className="flex gap-2 mb-1.5 text-[11px]">
                  <span className="bb-red shrink-0">▼</span>
                  <span className="text-[#999]">{r}</span>
                </div>
              ))}
            </Panel>

            <Panel title="── NEWS ──────────────────────────────────────────────────────" innerClass="p-0">
              <div className="divide-y divide-[#ffffff06]">
                {stockNews.slice(0, 4).map((item, i) => (
                  <div key={i} className="px-3 py-2 hover:bg-[#ff8c0005]">
                    <div className="flex gap-2">
                      <SentBadge label={(item as NewsItem & { sentiment_label?: string }).sentiment_label} />
                      {item.link ? (
                        <a href={item.link} target="_blank" rel="noreferrer" className="text-[10px] leading-4 bb-white hover:bb-amber line-clamp-2">{item.title}</a>
                      ) : (
                        <p className="text-[10px] leading-4 bb-white line-clamp-2">{item.title}</p>
                      )}
                    </div>
                    <p className="text-[9px] bb-dim mt-1">{item.publisher} {item.published_at ? `· ${new Date(item.published_at).toLocaleDateString()}` : ""}</p>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="── HISTORICAL SIMILARITY ENGINE ──────────────────────────────" innerClass="p-0" className="flex-1">
              {similarityLoading && (
                <p className="p-3 text-[10px] bb-amber animate-pulse">SCANNING 5Y HISTORY...</p>
              )}
              {similarity?.available && (
                <>
                  <div className="px-3 py-2 border-b border-[#ff8c0015] flex gap-4 text-[10px]">
                    <span className="bb-dim">DOMINANT:</span>
                    <span className={`font-bold ${similarity.summary.dominant_historical_outcome === "bullish" ? "bb-green" : similarity.summary.dominant_historical_outcome === "bearish" ? "bb-red" : "bb-amber"}`}>
                      {similarity.summary.dominant_historical_outcome.toUpperCase()}
                    </span>
                    <span className="bb-dim">AVG FWD:</span>
                    <span className={`font-bold ${similarity.summary.avg_forward_return_pct >= 0 ? "bb-green" : "bb-red"}`}>
                      {similarity.summary.avg_forward_return_pct > 0 ? "+" : ""}{similarity.summary.avg_forward_return_pct.toFixed(2)}%
                    </span>
                    <span className="bb-dim ml-auto">{similarity.summary.total_matches_scanned} scanned</span>
                  </div>
                  <table className="w-full text-[10px] font-mono">
                    <thead>
                      <tr className="text-[9px] bb-dim border-b border-[#ff8c0015]">
                        <th className="text-left p-2 pl-3">DATE</th>
                        <th className="text-right p-2">MATCH%</th>
                        <th className="text-right p-2">PRICE</th>
                        <th className="text-right p-2">FWD {similarity.setups[0]?.forward_bars}d</th>
                        <th className="text-center p-2">RESULT</th>
                      </tr>
                    </thead>
                    <tbody>
                      {similarity.setups.map((s) => (
                        <tr key={s.date} className="t-row">
                          <td className="p-2 pl-3 bb-dim">{s.date}</td>
                          <td className="p-2 text-right bb-amber font-bold">{s.similarity}%</td>
                          <td className="p-2 text-right bb-white">{usd(s.entry_price)}</td>
                          <td className={`p-2 text-right font-bold ${s.forward_return_pct >= 0 ? "bb-green" : "bb-red"}`}>
                            {s.forward_return_pct > 0 ? "+" : ""}{s.forward_return_pct.toFixed(2)}%
                          </td>
                          <td className="p-2 text-center">
                            <span className={`text-[9px] border px-1 uppercase ${s.outcome === "bullish" ? "text-[#00cc55] border-[#00cc5530]" : s.outcome === "bearish" ? "text-[#ff4444] border-[#ff444430]" : "bb-dim border-[#33333360]"}`}>
                              {s.outcome}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
              {!similarityLoading && !similarity && (
                <p className="p-3 text-[10px] bb-dim">LOAD A STOCK TO RUN PATTERN MATCHING</p>
              )}
              {similarity && !similarity.available && (
                <p className="p-3 text-[10px] bb-red">[ERR] {similarity.error}</p>
              )}
            </Panel>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center bb-dim text-sm tracking-widest">
          ENTER TICKER AND PRESS EXECUTE
        </div>
      )}
    </div>
  );
}

// ── SIMULATION TAB ────────────────────────────────────────────────────
function SimulationTab({
  ticker, setTicker, period, setPeriod, interval, setIntervalValue, loadPrediction, isLoading,
  simulation, dataStatus,
}: {
  ticker: string; setTicker: (v: string) => void; period: string; setPeriod: (v: string) => void;
  interval: string; setIntervalValue: (v: string) => void;
  loadPrediction: (nextTicker?: string, nextPeriod?: string, nextInterval?: string) => void; isLoading: boolean;
  simulation: SimulationResponse; dataStatus: string;
}) {
  const probs = simulation.probabilities;
  const intelligence = simulation.market_intelligence;
  const finalSignal = simulation.final_signal ?? intelligence?.final_signal;
  const signalBot = simulation.ai_signal_bot;
  const technical = simulation.advanced_intelligence?.technical_analysis ?? {};
  const timeframe = simulation.advanced_intelligence?.multi_timeframe_alignment ?? {};
  const similarity = simulation.advanced_intelligence?.historical_similarity ?? {};
  const monteCarlo = simulation.advanced_intelligence?.monte_carlo_simulation ?? {};
  const trends = (timeframe.trends && typeof timeframe.trends === "object" ? timeframe.trends : {}) as Record<string, unknown>;
  const proof80 = finalSignal?.proof?.confidence_buckets?.[">=80%"];
  const proofBuckets = finalSignal?.proof?.confidence_buckets ?? {};
  const eventEntries = Object.entries(intelligence?.event_probabilities ?? {});
  const terminalCone = (monteCarlo.volatility_cone_terminal && typeof monteCarlo.volatility_cone_terminal === "object"
    ? monteCarlo.volatility_cone_terminal
    : {}) as Record<string, unknown>;
  const [railTab, setRailTab] = useState<"signal" | "intel" | "proof" | "similarity" | "risk">("signal");
  return (
    <div className="h-full min-w-0 flex flex-col gap-px bg-[#ff8c0010]">
      {/* Controls */}
      <div className="t-panel shrink-0 min-w-0 flex items-center gap-3 overflow-x-auto px-4 py-2">
        <span className="bb-amber text-xs tracking-widest">SIM&gt;</span>
        <input
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
          onKeyDown={(e) => { if (e.key === "Enter") loadPrediction(); }}
          className="bb-input w-24 text-sm font-bold"
          placeholder="NVDA"
        />
        <select
          value={period}
          onChange={(e) => {
            const next = e.target.value;
            setPeriod(next);
            void loadPrediction(ticker, next, interval);
          }}
          className="bb-input text-xs"
        >
          {["5d","1mo","3mo","6mo","1y","2y","5y"].map((p) => <option key={p}>{p}</option>)}
        </select>
        <select
          value={interval}
          onChange={(e) => {
            const next = e.target.value;
            let nextPeriod = period;
            setIntervalValue(next);
            if (next === "1m" && !["1d", "5d"].includes(period)) {
              nextPeriod = "5d";
              setPeriod(nextPeriod);
            }
            if (["5m", "15m", "30m"].includes(next) && ["6mo", "1y", "2y", "5y"].includes(period)) {
              nextPeriod = "1mo";
              setPeriod(nextPeriod);
            }
            void loadPrediction(ticker, nextPeriod, next);
          }}
          className="bb-input text-xs"
        >
          {["1m","5m","15m","30m","1h","4h","1d","1wk"].map((v) => <option key={v}>{v}</option>)}
        </select>
        <button onClick={loadPrediction} disabled={isLoading} className="bb-btn">
          {isLoading ? "RUNNING..." : "RUN SIMULATION"}
        </button>
        <span className="text-[10px] bb-dim ml-4 truncate">{dataStatus}</span>
        <div className="ml-auto flex gap-4 text-xs">
          <span className="bb-dim">SCENARIO: <span className={`font-bold ${simulation.dominant_scenario === "bullish" ? "bb-green" : simulation.dominant_scenario === "bearish" ? "bb-red" : "bb-amber"}`}>{simulation.dominant_scenario?.toUpperCase()}</span></span>
          {simulation.quality_label && <span className="bb-dim">QUALITY: <span className="bb-amber font-bold">{simulation.quality_label.replaceAll("_", " ").toUpperCase()}</span></span>}
          <span className="bb-dim">CONFIDENCE: <span className="bb-white font-bold">{simulation.confidence?.toFixed(1)}%</span></span>
          <span className="bb-dim">RISK: <span className={`font-bold ${simulation.risk_level === "high" ? "bb-red" : simulation.risk_level === "low" ? "bb-green" : "bb-amber"}`}>{simulation.risk_level?.toUpperCase()}</span></span>
          {simulation.ml_enabled && <span className="bb-green text-[10px]">■ ML:{simulation.model_version?.slice(-8)}</span>}
          {simulation.auxiliary_ml_enabled && <span className="bb-amber text-[10px]">■ AUX</span>}
        </div>
      </div>

      <div className="min-w-0 flex-1 grid grid-cols-[minmax(0,1fr)_280px] gap-px bg-[#ff8c0010] overflow-hidden">
        <Panel title={`── SCENARIO CHART · ${simulation.ticker} · ${simulation.period.toUpperCase()} · ${simulation.interval} ─────────────────────────────────────────`} className="overflow-hidden" innerClass="p-0">
          <SimulationChart simulation={simulation} />
        </Panel>

        <div className="min-h-0 flex flex-col gap-px bg-[#ff8c0010] text-[11px]">
          <Panel title="── PROBABILITIES ──────────────────────────────────────────────" innerClass="p-3">
            {([
              ["BULLISH", probs.bullish, "bb-bar-fill-green"],
              ["BEARISH", probs.bearish, "bb-bar-fill-red"],
              ["SIDEWAYS", probs.sideways, "bb-bar-fill"],
            ] as const).map(([label, val, barCls]) => (
              <div key={label} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="bb-dim">{label}</span>
                  <span className="bb-white font-bold">{(val * 100).toFixed(1)}%</span>
                </div>
                <div className="bb-bar-track">
                  <div className={barCls} style={{ width: `${val * 100}%` }} />
                </div>
              </div>
            ))}
            <div className="mt-3 pt-3 border-t border-[#ff8c0015]">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div><span className="bb-dim">RISK SCORE: </span><span className="bb-white font-bold">{simulation.risk_score}/100</span></div>
                <div><span className="bb-dim">PRICE: </span><span className="bb-amber font-bold">{usd(simulation.current_price)}</span></div>
              </div>
            </div>
          </Panel>

          <div className="t-panel shrink-0 grid grid-cols-5 gap-px bg-[#ff8c0010] p-px">
            {([
              ["signal", "SIGNAL"],
              ["intel", "INTEL"],
              ["proof", "PROOF"],
              ["similarity", "SIM"],
              ["risk", "RISK"],
            ] as const).map(([id, label]) => (
              <button
                key={id}
                type="button"
                onClick={() => setRailTab(id)}
                className={`px-1 py-2 text-[9px] font-bold tracking-wider ${
                  railTab === id ? "bg-[#ff8c0014] bb-amber" : "bg-[#050505] bb-dim hover:text-[#aaa]"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-auto flex flex-col gap-px">

          {railTab === "intel" && (
          <>
          <Panel title="── INTRADAY / VWAP INTELLIGENCE ───────────────────────────────" innerClass="p-3">
            <div className="grid grid-cols-2 gap-2 text-[11px] leading-4">
              <div className="border border-[#ff8c0012] bg-[#050505] p-2">
                <div className="text-[9px] bb-dim uppercase tracking-wider">VWAP</div>
                <div className="bb-white font-bold">{usd(technical.vwap as number | null | undefined)}</div>
              </div>
              <div className="border border-[#ff8c0012] bg-[#050505] p-2">
                <div className="text-[9px] bb-dim uppercase tracking-wider">Distance</div>
                <div className={`${(technical.vwap_distance_pct as number ?? 0) >= 0 ? "bb-green" : "bb-red"} font-bold`}>
                  {num(technical.vwap_distance_pct, 2)}%
                </div>
              </div>
              <div><span className="bb-dim">ABOVE VWAP: </span><span className={(technical.above_vwap as boolean) ? "bb-green font-bold" : "bb-red font-bold"}>{yn(technical.above_vwap)}</span></div>
              <div><span className="bb-dim">VWAP SLOPE: </span><span className="bb-white">{num(technical.vwap_slope, 4)}</span></div>
              <div><span className="bb-dim">RECLAIM: </span><span className={(technical.vwap_reclaim as boolean) ? "bb-green font-bold" : "bb-dim"}>{yn(technical.vwap_reclaim)}</span></div>
              <div><span className="bb-dim">REJECTION: </span><span className={(technical.vwap_rejection as boolean) ? "bb-red font-bold" : "bb-dim"}>{yn(technical.vwap_rejection)}</span></div>
              <div><span className="bb-dim">VOL RATIO: </span><span className="bb-white">{num(technical.volume_ratio_20, 2)}x</span></div>
              <div><span className="bb-dim">ATR %: </span><span className="bb-white">{num((technical.atr_pct as number | undefined) ? (technical.atr_pct as number) * 100 : undefined, 2)}%</span></div>
            </div>
            <div className="mt-3 border-t border-[#ff8c0015] pt-2 text-[10px] leading-4">
              <div><span className="bb-dim">MTF ALIGNMENT: </span><span className="bb-amber font-bold">{String(timeframe.alignment ?? "—").replaceAll("_", " ").toUpperCase()}</span></div>
              <div className="mt-1 grid grid-cols-3 gap-x-2 gap-y-1">
                {["1m", "5m", "15m", "1h", "4h", "1d"].map((tf) => (
                  <div key={tf}><span className="bb-dim">{tf}: </span><span className={trends[tf] === "bullish" ? "bb-green" : trends[tf] === "bearish" ? "bb-red" : "bb-white"}>{String(trends[tf] ?? "—").toUpperCase()}</span></div>
                ))}
              </div>
            </div>
          </Panel>

          {intelligence && (
            <Panel title="── EVENT INTELLIGENCE ─────────────────────────────────────────" innerClass="p-3">
              <div className="grid grid-cols-2 gap-2 mb-3">
                {eventEntries.map(([name, values]) => (
                  <div key={name} className="border border-[#ff8c0012] bg-[#050505] p-2">
                    <div className="text-[9px] text-[#9b9b9b] uppercase tracking-wide leading-3 break-words">{name.replaceAll("_", " ")}</div>
                    <div className={`text-base font-bold leading-5 ${values.probability >= 0.65 ? "bb-green" : values.probability <= 0.35 ? "bb-red" : "bb-amber"}`}>
                      {(values.probability * 100).toFixed(1)}%
                    </div>
                    <div className="text-[9px] text-[#858585]">CONF {(values.confidence * 100).toFixed(0)}%</div>
                  </div>
                ))}
              </div>
              <div className="space-y-1.5 text-[11px] leading-4">
                <div><span className="bb-dim">TOP SIGNAL: </span><span className="bb-amber font-bold">{intelligence.top_event_signal?.replaceAll("_", " ").toUpperCase() ?? "—"}</span></div>
                <div><span className="bb-dim">EXP RETURN: </span><span className={colorChange(intelligence.expected_return_pct ?? 0)}>{pct((intelligence.expected_return_pct ?? 0) * 100)}</span></div>
                <div><span className="bb-dim">RISK ADJ: </span><span className="bb-white">{intelligence.risk_adjusted_return?.toFixed(2) ?? "—"}</span></div>
                {intelligence.expected_price_range && (
                  <div><span className="bb-dim">RANGE: </span><span className="bb-white">{usd(intelligence.expected_price_range.lower)} - {usd(intelligence.expected_price_range.upper)}</span></div>
                )}
              </div>
            </Panel>
          )}
          </>
          )}

          {railTab === "similarity" && (
          <Panel title="── HISTORICAL SIMILARITY ──────────────────────────────────────" innerClass="p-3">
            {similarity.available ? (
              <div className="space-y-2 text-[11px] leading-4">
                <div className="grid grid-cols-2 gap-2">
                  <div className="border border-[#ff8c0012] bg-[#050505] p-2">
                    <div className="text-[9px] bb-dim uppercase tracking-wider">Similarity</div>
                    <div className="bb-white font-bold">{num((similarity.average_similarity_score as number | undefined) ? (similarity.average_similarity_score as number) * 100 : undefined, 1)}%</div>
                  </div>
                  <div className="border border-[#ff8c0012] bg-[#050505] p-2">
                    <div className="text-[9px] bb-dim uppercase tracking-wider">Avg Future</div>
                    <div className={((similarity.average_future_return_pct as number | undefined) ?? 0) >= 0 ? "bb-green font-bold" : "bb-red font-bold"}>
                      {num(similarity.average_future_return_pct, 2)}%
                    </div>
                  </div>
                </div>
                {((similarity.similar_setups as unknown[]) ?? []).slice(0, 3).map((setup, i) => {
                  const row = setup as Record<string, unknown>;
                  return (
                    <div key={`${row.date}-${i}`} className="flex justify-between gap-2 border-t border-[#ff8c0010] pt-1">
                      <span className="bb-dim">{String(row.date ?? "—")}</span>
                      <span className={row.outcome === "bullish" ? "bb-green" : row.outcome === "bearish" ? "bb-red" : "bb-amber"}>{String(row.outcome ?? "—").toUpperCase()}</span>
                      <span className="bb-white">{num(row.future_10_bar_return_pct, 2)}%</span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="bb-dim text-[11px] leading-4">Similarity unavailable: {String(similarity.reason ?? "not enough comparable history")}</div>
            )}
          </Panel>
          )}

          {railTab === "proof" && (
          <Panel title="── MONTE CARLO / CALIBRATION ──────────────────────────────────" innerClass="p-3">
            <div className="space-y-2 text-[11px] leading-4">
              <div className="grid grid-cols-2 gap-2">
                <div><span className="bb-dim">PATHS: </span><span className="bb-white font-bold">{String(monteCarlo.n_paths ?? "—")}</span></div>
                <div><span className="bb-dim">HORIZON: </span><span className="bb-white font-bold">{String(monteCarlo.horizon_steps ?? simulation.horizon_steps)}</span></div>
                <div><span className="bb-dim">P05: </span><span className="bb-red">{usd(terminalCone.p05 as number | null | undefined)}</span></div>
                <div><span className="bb-dim">P95: </span><span className="bb-green">{usd(terminalCone.p95 as number | null | undefined)}</span></div>
              </div>
              <div className="border-t border-[#ff8c0015] pt-2">
                {Object.entries(proofBuckets).slice(0, 4).map(([bucket, value]) => (
                  <div key={bucket} className="mb-1 flex items-center justify-between gap-2">
                    <span className="bb-dim">{bucket}</span>
                    <span className="bb-white">{(((value.accuracy ?? 0) * 100)).toFixed(1)}%</span>
                    <span className="bb-dim">{value.signals ?? 0} sig</span>
                  </div>
                ))}
                {!Object.keys(proofBuckets).length && <div className="bb-dim">No confidence-bucket proof loaded.</div>}
              </div>
            </div>
          </Panel>
          )}

          {railTab === "signal" && (
          <>
          {finalSignal && (
            <Panel title="── FINAL HYBRID SIGNAL ────────────────────────────────────────" innerClass="p-3">
              <div className="space-y-1.5 text-[11px] leading-4">
                <div><span className="bb-dim">SIGNAL: </span><span className="bb-amber font-bold">{finalSignal.name.replaceAll("_", " ").toUpperCase()}</span></div>
                <div><span className="bb-dim">STATE: </span><span className={finalSignal.state.includes("high") ? "bb-green font-bold" : "bb-white font-bold"}>{finalSignal.state.replaceAll("_", " ").toUpperCase()}</span></div>
                <div><span className="bb-dim">LIVE PROB: </span><span className="bb-white font-bold">{(finalSignal.probability * 100).toFixed(1)}%</span></div>
                {proof80 && (
                  <div><span className="bb-dim">PROOF @80: </span><span className="bb-green font-bold">{((proof80.accuracy ?? 0) * 100).toFixed(1)}%</span><span className="bb-dim"> / {proof80.signals} signals</span></div>
                )}
                {finalSignal.proof?.setup_coverage_test !== undefined && (
                  <div><span className="bb-dim">COVERAGE: </span><span className="bb-white">{((finalSignal.proof.setup_coverage_test ?? 0) * 100).toFixed(2)}%</span></div>
                )}
              </div>
            </Panel>
          )}

          {signalBot && (
            <Panel title="── AI SIGNAL BOT ──────────────────────────────────────────────" innerClass="p-3">
              <div className="space-y-1.5 text-[11px] leading-4">
                <div><span className="bb-dim">ACTION: </span><span className={signalBot.action.includes("avoid") ? "bb-red font-bold" : signalBot.action.includes("candidate") ? "bb-green font-bold" : "bb-amber font-bold"}>{signalBot.action.replaceAll("_", " ").toUpperCase()}</span></div>
                {signalBot.quality_label && <div><span className="bb-dim">QUALITY: </span><span className="bb-amber font-bold">{signalBot.quality_label.replaceAll("_", " ").toUpperCase()}</span></div>}
                <div><span className="bb-dim">STANCE: </span><span className="bb-white font-bold">{signalBot.stance.replaceAll("_", " ").toUpperCase()}</span></div>
                <div><span className="bb-dim">MODE: </span><span className="bb-white">{signalBot.mode.replaceAll("_", " ").toUpperCase()}</span></div>
                <div><span className="bb-dim">GATES: </span><span className={signalBot.all_gates_passed ? "bb-green font-bold" : "bb-amber font-bold"}>{signalBot.all_gates_passed ? "PASS" : "WAIT"}</span></div>
                {signalBot.expected_return !== null && (
                  <div><span className="bb-dim">EXP RET: </span><span className={colorChange(signalBot.expected_return)}>{pct(signalBot.expected_return * 100)}</span></div>
                )}
                <div><span className="bb-dim">BOT CONF: </span><span className="bb-white font-bold">{signalBot.confidence.toFixed(1)}%</span></div>
                {signalBot.trade_plan?.available && (
                  <>
                    <div><span className="bb-dim">ENTRY: </span><span className="bb-white">{usd(signalBot.trade_plan.entry_price ?? 0)}</span></div>
                    <div><span className="bb-dim">STOP: </span><span className="bb-red">{usd(signalBot.trade_plan.stop_loss ?? 0)}</span></div>
                    <div><span className="bb-dim">TGT: </span><span className="bb-green">{usd(signalBot.trade_plan.target_1 ?? 0)} / {usd(signalBot.trade_plan.target_2 ?? 0)}</span></div>
                    <div><span className="bb-dim">R/R: </span><span className="bb-white">{signalBot.trade_plan.risk_reward_target_1?.toFixed(2) ?? "—"}</span><span className="bb-dim"> · PAPER SH: {signalBot.trade_plan.paper_position_shares ?? 0}</span></div>
                    <div><span className="bb-dim">COST: </span><span className="bb-white">{num(signalBot.trade_plan.round_trip_cost_pct, 3)}%</span><span className="bb-dim"> · NET REWARD: </span><span className={((signalBot.trade_plan.net_reward_after_cost_pct ?? 0) >= 0) ? "bb-green" : "bb-red"}>{num(signalBot.trade_plan.net_reward_after_cost_pct, 3)}%</span></div>
                    <div><span className="bb-dim">LIQUIDITY: </span><span className={signalBot.trade_plan.liquidity_filter_passed ? "bb-green font-bold" : "bb-red font-bold"}>{signalBot.trade_plan.liquidity_filter_passed ? "PASS" : "FAIL"}</span><span className="bb-dim"> · SLIP: {num(signalBot.trade_plan.slippage_estimate_pct, 3)}%</span></div>
                  </>
                )}
              </div>
            </Panel>
          )}
          </>
          )}

          {railTab === "risk" && (
          <>
          <Panel title="── REASONING ──────────────────────────────────────────────────" innerClass="p-3" className="flex-1">
            <p className="text-[11px] text-[#aaa] leading-5 mb-3">{simulation.reasoning}</p>
            <div className="space-y-1.5">
              {simulation.reasons?.slice(0, 3).map((r) => (
                <div key={r} className="flex gap-2 text-[10px]">
                  <span className="bb-green shrink-0">▲</span><span className="text-[#999]">{r}</span>
                </div>
              ))}
              {simulation.risks?.slice(0, 2).map((r) => (
                <div key={r} className="flex gap-2 text-[10px]">
                  <span className="bb-red shrink-0">▼</span><span className="text-[#777]">{r}</span>
                </div>
              ))}
            </div>
          </Panel>

          <Panel title="── DISCLAIMER ─────────────────────────────────────────────────" innerClass="p-3">
            <p className="text-[9px] bb-dim leading-4">{simulation.disclaimer}</p>
          </Panel>
          </>
          )}

          </div>
        </div>
      </div>
    </div>
  );
}

// ── PORTFOLIO TAB ─────────────────────────────────────────────────────
function PortfolioTab({
  portfolioRows, setPortfolioRows, runPortfolioAnalysis, portfolioLoading,
  portfolioError, portfolioResult,
}: {
  portfolioRows: { ticker: string; shares: string }[];
  setPortfolioRows: (r: { ticker: string; shares: string }[]) => void;
  runPortfolioAnalysis: () => void;
  portfolioLoading: boolean; portfolioError: string | null;
  portfolioResult: PortfolioAnalysis | null;
}) {
  return (
    <div className="h-full grid grid-cols-[320px_1fr] gap-px bg-[#ff8c0010]">
      {/* Input panel */}
      <Panel title="── HOLDINGS INPUT ──────────────────────────────────────────────" innerClass="p-3">
        <div className="space-y-2 mb-4">
          {portfolioRows.map((row, i) => (
            <div key={i} className="flex gap-2 items-center">
              <input
                value={row.ticker}
                onChange={(e) => {
                  const next = [...portfolioRows];
                  next[i] = { ...next[i], ticker: e.target.value.toUpperCase() };
                  setPortfolioRows(next);
                }}
                className="bb-input w-20 text-xs font-bold bb-amber"
                placeholder="AAPL"
              />
              <input
                value={row.shares}
                onChange={(e) => {
                  const next = [...portfolioRows];
                  next[i] = { ...next[i], shares: e.target.value };
                  setPortfolioRows(next);
                }}
                type="number" min="0"
                className="bb-input w-20 text-xs"
                placeholder="shares"
              />
              <button
                onClick={() => setPortfolioRows(portfolioRows.filter((_, j) => j !== i))}
                className="text-[10px] bb-red hover:opacity-70"
              >DEL</button>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setPortfolioRows([...portfolioRows, { ticker: "", shares: "" }])}
            className="bb-btn text-[10px] flex-1"
          >+ ADD ROW</button>
          <button
            onClick={runPortfolioAnalysis}
            disabled={portfolioLoading}
            className="bb-btn bb-btn-green text-[10px] flex-1"
          >{portfolioLoading ? "ANALYSING..." : "RUN ANALYSIS"}</button>
        </div>
        {portfolioError && <p className="mt-3 text-[11px] bb-red">[ERR] {portfolioError}</p>}

        {portfolioResult && (
          <div className="mt-4 pt-4 border-t border-[#ff8c0015]">
            <div className="space-y-1.5 text-xs">
              {[
                ["TOTAL VALUE", `$${portfolioResult.total_market_value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, "bb-amber"],
                ["ANNUAL RETURN", pct(portfolioResult.annualized_return_pct), portfolioResult.annualized_return_pct >= 0 ? "bb-green" : "bb-red"],
                ["VOLATILITY", `${portfolioResult.annualized_volatility_pct.toFixed(1)}%`, "bb-white"],
                ["SHARPE RATIO", portfolioResult.sharpe_ratio.toFixed(2), portfolioResult.sharpe_ratio >= 1 ? "bb-green" : "bb-amber"],
                ["MAX DRAWDOWN", `${portfolioResult.max_drawdown_pct.toFixed(1)}%`, "bb-red"],
                ["DIVERSIFICATION", `${portfolioResult.diversification_score.toFixed(0)}/100`, "bb-white"],
              ].map(([label, value, cls]) => (
                <div key={label as string} className="flex justify-between t-row py-1.5">
                  <span className="bb-dim">{label}</span>
                  <span className={`font-bold ${cls}`}>{value}</span>
                </div>
              ))}
            </div>
            {portfolioResult.warnings.map((w) => (
              <div key={w} className="mt-2 text-[10px] bb-amber border border-[#ff8c0025] p-2">⚠ {w}</div>
            ))}
          </div>
        )}
      </Panel>

      {/* Results panel */}
      <div className="flex flex-col gap-px bg-[#ff8c0010]">
        {portfolioResult ? (
          <>
            <Panel title="── ALLOCATION BREAKDOWN ─────────────────────────────────────────────────────────" innerClass="p-4">
              <div className="space-y-3">
                {portfolioResult.holdings.map((h) => (
                  <div key={h.ticker}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="bb-amber font-bold">{h.ticker}</span>
                      <span className="bb-dim">{h.shares} shares · {usd(h.market_value)} · <span className="bb-white font-bold">{h.weight_pct.toFixed(1)}%</span></span>
                    </div>
                    <div className="bb-bar-track">
                      <div className="bb-bar-fill" style={{ width: `${h.weight_pct}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </Panel>

            <Panel title="── CORRELATION MATRIX ─────────────────────────────────────────────────────────────" innerClass="p-4" className="flex-1">
              <p className="text-[10px] bb-dim mb-3">GREEN = low correlation (diversified) · RED = high correlation (concentrated risk)</p>
              <table className="text-xs font-mono">
                <thead>
                  <tr>
                    <th className="pr-4 pb-2 text-left bb-dim text-[10px]"></th>
                    {Object.keys(portfolioResult.correlation_matrix).map((t) => (
                      <th key={t} className="px-3 pb-2 bb-amber text-[11px]">{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(portfolioResult.correlation_matrix).map(([row, cols]) => (
                    <tr key={row} className="t-row">
                      <td className="pr-4 py-1.5 bb-amber font-bold text-[11px]">{row}</td>
                      {Object.values(cols).map((val, i) => (
                        <td key={i} className={`px-3 py-1.5 text-center font-bold text-xs ${
                          val === 1 ? "bb-dim" : val > 0.6 ? "bb-red" : val > 0.3 ? "bb-amber" : "bb-green"
                        }`}>{val.toFixed(3)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bb-dim text-sm tracking-widest">
            ENTER HOLDINGS AND RUN ANALYSIS
          </div>
        )}
      </div>
    </div>
  );
}

// ── COPILOT TAB ───────────────────────────────────────────────────────
function CopilotTab({
  copilotHistory, copilotInput, setCopilotInput, sendCopilot, copilotLoading,
  stockAnalysis,
}: {
  copilotHistory: { role: string; content: string }[];
  copilotInput: string; setCopilotInput: (v: string) => void;
  sendCopilot: () => void; copilotLoading: boolean;
  stockAnalysis: StockAnalysis | null;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [copilotHistory]);

  return (
    <div className="h-full grid grid-cols-[1fr_280px] gap-px bg-[#ff8c0010]">
      <Panel title="── AI COPILOT TERMINAL · GPT-4o-mini ─────────────────────────────────────────────" innerClass="p-0 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 space-y-3 font-mono">
          {copilotHistory.length === 0 && (
            <div className="text-[11px] bb-dim leading-5">
              <span className="bb-amber">MARKETVISION::COPILOT</span> initialized.<br />
              Type a query below. Context from RESEARCH tab is auto-injected.<br />
              <br />
              <span className="bb-dim">Example queries:</span><br />
              <span className="bb-dim">· Explain the NVDA simulation result</span><br />
              <span className="bb-dim">· What does an inverted yield curve mean?</span><br />
              <span className="bb-dim">· Is AAPL overvalued based on current signals?</span>
            </div>
          )}
          {copilotHistory.map((msg, i) => (
            <div key={i} className={`text-[11px] leading-5 ${msg.role === "user" ? "bb-dim" : "bb-white"}`}>
              <span className={`font-bold ${msg.role === "user" ? "bb-dim" : "bb-amber"}`}>
                {msg.role === "user" ? "USER> " : "COPILOT> "}
              </span>
              {msg.content}
            </div>
          ))}
          {copilotLoading && (
            <div className="text-[11px] bb-amber animate-pulse">COPILOT&gt; processing...</div>
          )}
          <div ref={bottomRef} />
        </div>
        <div className="shrink-0 border-t border-[#ff8c0020] flex items-center gap-2 px-4 py-2">
          <span className="bb-amber text-xs font-bold shrink-0">&gt;</span>
          <input
            value={copilotInput}
            onChange={(e) => setCopilotInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") sendCopilot(); }}
            className="flex-1 bg-transparent text-xs bb-white outline-none placeholder:bb-dim bb-cursor"
            placeholder="Enter query..."
          />
          <button onClick={sendCopilot} disabled={copilotLoading || !copilotInput.trim()} className="bb-btn text-[10px]">SEND</button>
        </div>
      </Panel>

      <Panel title="── CONTEXT ─────────────────────────────────────────────────────" innerClass="p-3">
        <p className="text-[10px] bb-dim mb-3">Auto-injected from RESEARCH tab:</p>
        {stockAnalysis ? (
          <div className="space-y-1.5 text-[11px]">
            <div className="flex justify-between t-row py-1"><span className="bb-dim">TICKER</span><span className="bb-amber font-bold">{stockAnalysis.ticker}</span></div>
            <div className="flex justify-between t-row py-1"><span className="bb-dim">PRICE</span><span className="bb-white">{usd(stockAnalysis.current_price)}</span></div>
            <div className="flex justify-between t-row py-1"><span className="bb-dim">STANCE</span><span className={`font-bold ${stockAnalysis.stance === "bullish" ? "bb-green" : stockAnalysis.stance === "cautious" ? "bb-red" : "bb-amber"}`}>{stockAnalysis.stance?.toUpperCase()}</span></div>
            <div className="flex justify-between t-row py-1"><span className="bb-dim">SCORE</span><span className="bb-white">{stockAnalysis.score}/100</span></div>
            <div className="mt-3 pt-2 border-t border-[#ff8c0015]">
              <p className="bb-dim text-[9px] mb-2">SIGNALS:</p>
              {stockAnalysis.reasons.slice(0, 3).map((r) => <p key={r} className="text-[10px] text-[#999] mb-1">▲ {r}</p>)}
              {stockAnalysis.risks.slice(0, 2).map((r) => <p key={r} className="text-[10px] text-[#666] mb-1">▼ {r}</p>)}
            </div>
          </div>
        ) : (
          <p className="text-[11px] bb-dim">Go to RESEARCH tab, load a stock to inject context here.</p>
        )}
      </Panel>
    </div>
  );
}

// ── MACRO TAB ─────────────────────────────────────────────────────────
function MacroTab({ macroCond }: { macroCond: MacroConditions | null }) {
  if (!macroCond?.available) {
    return <div className="h-full flex items-center justify-center bb-dim text-sm tracking-widest">FRED API UNAVAILABLE</div>;
  }
  return (
    <div className="h-full grid grid-cols-3 gap-px bg-[#ff8c0010]">
      <Panel title="── MONETARY POLICY ─────────────────────────────────────────────" innerClass="p-3">
        <table className="w-full text-xs font-mono">
          <tbody>
            {[
              ["Fed Funds Rate", macroCond.fed_funds_rate != null ? `${macroCond.fed_funds_rate}%` : "—"],
              ["10Y Treasury", macroCond.treasury_10y != null ? `${macroCond.treasury_10y}%` : "—"],
              ["2Y Treasury", macroCond.treasury_2y != null ? `${macroCond.treasury_2y}%` : "—"],
              ["Yield Spread (10-2)", macroCond.yield_spread != null ? `${macroCond.yield_spread > 0 ? "+" : ""}${macroCond.yield_spread}%` : "—"],
            ].map(([k, v]) => (
              <tr key={k as string} className="t-row">
                <td className="py-2 pr-4 bb-dim">{k}</td>
                <td className={`py-2 font-bold text-right ${
                  k === "Yield Spread (10-2)" && macroCond.yield_spread != null && macroCond.yield_spread < 0 ? "bb-red"
                  : "bb-white"
                }`}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-4 pt-3 border-t border-[#ff8c0015]">
          <p className="text-[10px] bb-dim mb-2">YIELD CURVE STATUS:</p>
          <p className={`text-sm font-bold ${macroCond.yield_spread != null && macroCond.yield_spread < 0 ? "bb-red" : "bb-green"}`}>
            {macroCond.yield_spread != null && macroCond.yield_spread < 0 ? "⚠ INVERTED (RECESSION SIGNAL)" : "■ NORMAL"}
          </p>
        </div>
      </Panel>

      <Panel title="── ECONOMIC INDICATORS ─────────────────────────────────────────" innerClass="p-3">
        <table className="w-full text-xs font-mono">
          <tbody>
            {[
              ["Unemployment Rate", macroCond.unemployment != null ? `${macroCond.unemployment}%` : "—"],
              ["CPI (All Urban)", macroCond.cpi != null ? macroCond.cpi.toFixed(3) : "—"],
              ["VIX Index", macroCond.vix != null ? macroCond.vix.toFixed(2) : "—"],
            ].map(([k, v]) => (
              <tr key={k as string} className="t-row">
                <td className="py-2 pr-4 bb-dim">{k}</td>
                <td className={`py-2 font-bold text-right ${
                  k === "VIX Index" && macroCond.vix != null && macroCond.vix > 25 ? "bb-red"
                  : k === "VIX Index" && macroCond.vix != null && macroCond.vix < 15 ? "bb-green"
                  : "bb-white"
                }`}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-4 pt-3 border-t border-[#ff8c0015]">
          <p className="text-[10px] bb-dim mb-2">MARKET VOLATILITY:</p>
          <p className={`text-sm font-bold ${
            macroCond.vix != null && macroCond.vix > 30 ? "bb-red"
            : macroCond.vix != null && macroCond.vix < 15 ? "bb-green"
            : "bb-amber"
          }`}>
            {macroCond.vix != null && macroCond.vix > 30 ? "■ HIGH FEAR" : macroCond.vix != null && macroCond.vix < 15 ? "■ LOW FEAR" : "■ MODERATE"}
          </p>
        </div>
      </Panel>

      <Panel title="── ASSESSMENT ──────────────────────────────────────────────────" innerClass="p-3">
        <div className="mb-4">
          <p className="text-[10px] bb-dim mb-2">OVERALL CONDITIONS:</p>
          <p className={`text-xl font-bold tracking-widest ${
            macroCond.overall_assessment === "positive" ? "bb-green"
            : macroCond.overall_assessment === "cautious" ? "bb-red"
            : "bb-amber"
          }`}>{macroCond.overall_assessment.toUpperCase()}</p>
        </div>
        <div className="space-y-2 mt-4">
          <p className="text-[10px] bb-dim">FLAGS:</p>
          {macroCond.flags.map((f) => <p key={f} className="text-[11px] bb-white border-l-2 border-[#ff8c0040] pl-2">{f}</p>)}
        </div>
        {macroCond.risks.length > 0 && (
          <div className="space-y-2 mt-4">
            <p className="text-[10px] bb-dim">RISKS:</p>
            {macroCond.risks.map((r) => <p key={r} className="text-[11px] bb-red border-l-2 border-[#ff444440] pl-2">{r}</p>)}
          </div>
        )}
        <div className="mt-4 pt-3 border-t border-[#ff8c0015] text-[9px] bb-dim">
          Source: Federal Reserve Economic Data (FRED) · St. Louis Fed
        </div>
      </Panel>
    </div>
  );
}

// ── COMPARE TAB ───────────────────────────────────────────────────────
function CompareTab({
  compareInput, setCompareInput, compareResult, compareLoading, runCompare, comparePeriod, setComparePeriod,
}: {
  compareInput: string; setCompareInput: (v: string) => void;
  compareResult: CompareResult | null; compareLoading: boolean;
  runCompare: () => void; comparePeriod: string; setComparePeriod: (v: string) => void;
}) {
  const leaders = compareResult?.leaders ?? [];
  const metrics: { key: keyof CompareLeader | string; label: string }[] = [
    { key: "current_price", label: "PRICE" },
    { key: "daily_change_pct", label: "DAY %" },
    { key: "score", label: "SCORE" },
    { key: "stance", label: "STANCE" },
    { key: "rsi", label: "RSI 14" },
    { key: "volatility", label: "VOL 30D" },
    { key: "drawdown", label: "MAX DD" },
  ];

  function getVal(leader: CompareLeader, key: string): string {
    if (key === "current_price") return usd(leader.current_price);
    if (key === "daily_change_pct") return pct(leader.daily_change_pct);
    if (key === "score") return `${leader.score}/100`;
    if (key === "stance") return leader.stance.toUpperCase();
    if (key === "rsi") return leader.signals.rsi_14?.toFixed(1) ?? "—";
    if (key === "volatility") return leader.signals.annualized_volatility_30d ? `${(leader.signals.annualized_volatility_30d * 100).toFixed(1)}%` : "—";
    if (key === "drawdown") return leader.signals.max_drawdown_period ? `${(leader.signals.max_drawdown_period * 100).toFixed(1)}%` : "—";
    return "—";
  }

  function getColor(leader: CompareLeader, key: string): string {
    if (key === "daily_change_pct") return leader.daily_change_pct >= 0 ? "bb-green" : "bb-red";
    if (key === "stance") return leader.stance === "bullish" ? "bb-green" : leader.stance === "cautious" ? "bb-red" : "bb-amber";
    if (key === "score") return leader.score >= 70 ? "bb-green" : leader.score >= 50 ? "bb-amber" : "bb-red";
    if (key === "drawdown") return "bb-red";
    return "bb-white";
  }

  return (
    <div className="h-full flex flex-col gap-px bg-[#ff8c0010]">
      {/* Controls */}
      <div className="t-panel shrink-0 flex items-center gap-3 px-4 py-2">
        <span className="bb-amber text-xs tracking-widest">TICKERS&gt;</span>
        <input
          value={compareInput}
          onChange={(e) => setCompareInput(e.target.value.toUpperCase())}
          onKeyDown={(e) => { if (e.key === "Enter") runCompare(); }}
          className="bb-input flex-1 max-w-xs text-xs"
          placeholder="AAPL, NVDA, MSFT, TSLA"
        />
        <select value={comparePeriod} onChange={(e) => setComparePeriod(e.target.value)} className="bb-input text-xs">
          {["1mo","3mo","6mo","1y","2y"].map((p) => <option key={p}>{p}</option>)}
        </select>
        <button onClick={runCompare} disabled={compareLoading} className="bb-btn">
          {compareLoading ? "LOADING..." : "COMPARE"}
        </button>
      </div>

      {leaders.length > 0 ? (
        <div className="flex-1 grid grid-rows-[auto_1fr] gap-px bg-[#ff8c0010] overflow-hidden">
          {/* Side-by-side metrics grid */}
          <Panel title="── METRICS COMPARISON ─────────────────────────────────────────────────────────────────────────────" innerClass="p-0">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-[#ff8c0020]">
                  <th className="text-left p-2 pl-3 text-[9px] bb-dim uppercase tracking-widest w-28">METRIC</th>
                  {leaders.map((l) => (
                    <th key={l.ticker} className="p-2 text-center">
                      <div className="bb-amber font-bold">{l.ticker}</div>
                      <div className="text-[9px] bb-dim truncate max-w-[100px]">{l.company_name.split(" ").slice(0, 2).join(" ")}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metrics.map(({ key, label }) => (
                  <tr key={key} className="t-row">
                    <td className="p-2 pl-3 text-[9px] bb-dim uppercase tracking-wider">{label}</td>
                    {leaders.map((l) => (
                      <td key={l.ticker} className={`p-2 text-center font-bold text-xs ${getColor(l, key)}`}>
                        {getVal(l, key)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>

          {/* Signals breakdown */}
          <div className="grid gap-px bg-[#ff8c0010] overflow-auto" style={{ gridTemplateColumns: `repeat(${leaders.length}, 1fr)` }}>
            {leaders.map((l) => (
              <Panel key={l.ticker} title={`── ${l.ticker} SIGNALS ─────────────────────────────────────────────────────────`} innerClass="p-3">
                <div className="mb-2 flex items-center gap-2">
                  <span className={`text-sm font-bold ${l.stance === "bullish" ? "bb-green" : l.stance === "cautious" ? "bb-red" : "bb-amber"}`}>
                    {l.stance.toUpperCase()}
                  </span>
                  <span className="bb-dim text-xs">{l.score}/100</span>
                </div>
                {l.reasons.map((r) => (
                  <div key={r} className="flex gap-1.5 mb-1 text-[10px]">
                    <span className="bb-green shrink-0">▲</span><span className="text-[#999]">{r}</span>
                  </div>
                ))}
                {l.risks.map((r) => (
                  <div key={r} className="flex gap-1.5 mb-1 text-[10px]">
                    <span className="bb-red shrink-0">▼</span><span className="text-[#666]">{r}</span>
                  </div>
                ))}
              </Panel>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center bb-dim text-sm tracking-widest">
          {compareLoading ? "FETCHING DATA..." : "ENTER TICKERS AND CLICK COMPARE"}
        </div>
      )}
    </div>
  );
}

// ── BACKTEST TAB ──────────────────────────────────────────────────────
function BacktestTab({
  btTicker, setBtTicker, btStrategy, setBtStrategy, btPeriod, setBtPeriod,
  btCapital, setBtCapital, btResult, btLoading, runBt, strategies,
}: {
  btTicker: string; setBtTicker: (v: string) => void;
  btStrategy: string; setBtStrategy: (v: string) => void;
  btPeriod: string; setBtPeriod: (v: string) => void;
  btCapital: string; setBtCapital: (v: string) => void;
  btResult: BacktestResult | null; btLoading: boolean; runBt: () => void;
  strategies: { id: string; label: string }[];
}) {
  const alpha = btResult ? btResult.total_return_pct - btResult.buy_hold_return_pct : 0;
  return (
    <div className="h-full flex flex-col gap-px bg-[#ff8c0010]">
      {/* Controls */}
      <div className="t-panel shrink-0 flex items-center gap-3 px-4 py-2 flex-wrap">
        <span className="bb-amber text-xs tracking-widest">BACKTEST&gt;</span>
        <input value={btTicker} onChange={(e) => setBtTicker(e.target.value.toUpperCase())} onKeyDown={(e) => { if (e.key === "Enter") runBt(); }} className="bb-input w-20 text-sm font-bold" placeholder="AAPL" />
        <select value={btStrategy} onChange={(e) => setBtStrategy(e.target.value)} className="bb-input text-xs">
          {strategies.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
        </select>
        <select value={btPeriod} onChange={(e) => setBtPeriod(e.target.value)} className="bb-input text-xs">
          {["1y","2y","3y","5y"].map((p) => <option key={p}>{p}</option>)}
        </select>
        <span className="bb-dim text-xs">$</span>
        <input value={btCapital} onChange={(e) => setBtCapital(e.target.value)} className="bb-input w-24 text-xs" placeholder="10000" />
        <button onClick={runBt} disabled={btLoading} className="bb-btn bb-btn-green">
          {btLoading ? "RUNNING..." : "RUN BACKTEST"}
        </button>
      </div>

      {btResult?.available ? (
        <div className="flex-1 grid grid-cols-[300px_1fr] gap-px bg-[#ff8c0010] overflow-hidden">
          {/* Left: metrics */}
          <div className="flex flex-col gap-px bg-[#ff8c0010]">
            <Panel title="── PERFORMANCE SUMMARY ─────────────────────────────────────────" innerClass="p-3">
              <div className="mb-2 text-[10px] bb-dim">{btResult.strategy_label} · {btResult.ticker} · {btResult.period}</div>
              <table className="w-full text-xs font-mono">
                <tbody>
                  {[
                    ["Strategy Return", `${btResult.total_return_pct > 0 ? "+" : ""}${btResult.total_return_pct.toFixed(2)}%`, btResult.total_return_pct >= 0 ? "bb-green" : "bb-red"],
                    ["Buy & Hold", `${btResult.buy_hold_return_pct > 0 ? "+" : ""}${btResult.buy_hold_return_pct.toFixed(2)}%`, btResult.buy_hold_return_pct >= 0 ? "bb-green" : "bb-red"],
                    ["Alpha vs B&H", `${alpha > 0 ? "+" : ""}${alpha.toFixed(2)}%`, alpha >= 0 ? "bb-green" : "bb-red"],
                    ["Annual Return", `${btResult.annualized_return_pct > 0 ? "+" : ""}${btResult.annualized_return_pct.toFixed(2)}%`, btResult.annualized_return_pct >= 0 ? "bb-green" : "bb-red"],
                    ["Annualised Vol", `${btResult.annualized_volatility_pct.toFixed(2)}%`, "bb-white"],
                    ["Sharpe Ratio", btResult.sharpe_ratio.toFixed(3), btResult.sharpe_ratio >= 1 ? "bb-green" : btResult.sharpe_ratio >= 0 ? "bb-amber" : "bb-red"],
                    ["Max Drawdown", `${btResult.max_drawdown_pct.toFixed(2)}%`, "bb-red"],
                    ["Final Equity", `$${btResult.final_equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, "bb-amber"],
                  ].map(([label, value, cls]) => (
                    <tr key={label as string} className="t-row">
                      <td className="py-1.5 pr-3 bb-dim">{label}</td>
                      <td className={`py-1.5 text-right font-bold ${cls}`}>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>

            <Panel title="── TRADE STATS ─────────────────────────────────────────────────" innerClass="p-3">
              <table className="w-full text-xs font-mono">
                <tbody>
                  {[
                    ["Total Trades", btResult.total_trades.toString(), "bb-white"],
                    ["Win Rate", `${btResult.win_rate_pct.toFixed(1)}%`, btResult.win_rate_pct >= 50 ? "bb-green" : "bb-red"],
                    ["Avg Win", `+${btResult.avg_win_pct.toFixed(2)}%`, "bb-green"],
                    ["Avg Loss", `${btResult.avg_loss_pct.toFixed(2)}%`, "bb-red"],
                    ["Round Trip Cost", btResult.execution_assumptions ? `${btResult.execution_assumptions.round_trip_cost_pct.toFixed(3)}%` : "—", "bb-amber"],
                    ["Slippage", btResult.execution_assumptions ? `${btResult.execution_assumptions.slippage_estimate_pct.toFixed(3)}%` : "—", "bb-white"],
                  ].map(([label, value, cls]) => (
                    <tr key={label as string} className="t-row">
                      <td className="py-1.5 pr-3 bb-dim">{label}</td>
                      <td className={`py-1.5 text-right font-bold ${cls}`}>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {btResult.execution_assumptions?.note && (
                <p className="mt-3 border-t border-[#ff8c0015] pt-2 text-[10px] leading-4 bb-dim">
                  {btResult.execution_assumptions.note}
                </p>
              )}
            </Panel>

            <Panel title="── EQUITY CURVE ────────────────────────────────────────────────" innerClass="p-3" className="flex-1">
              {btResult.equity_curve.length > 0 && (
                <svg viewBox={`0 0 200 80`} className="w-full h-full" preserveAspectRatio="none">
                  {(() => {
                    const vals = btResult.equity_curve.map((p) => p.equity);
                    const minV = Math.min(...vals);
                    const maxV = Math.max(...vals);
                    const range = maxV - minV || 1;
                    const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * 200},${80 - ((v - minV) / range) * 70}`).join(" ");
                    const isPositive = vals[vals.length - 1] >= vals[0];
                    return (
                      <>
                        <polyline points={pts} fill="none" stroke={isPositive ? "#00cc55" : "#ff4444"} strokeWidth="1.5" />
                        <line x1="0" y1={80 - ((btResult.initial_capital - minV) / range) * 70} x2="200" y2={80 - ((btResult.initial_capital - minV) / range) * 70} stroke="#ff8c0030" strokeWidth="0.5" strokeDasharray="3,3" />
                      </>
                    );
                  })()}
                </svg>
              )}
            </Panel>
          </div>

          {/* Right: trade log */}
          <Panel title="── TRADE LOG ───────────────────────────────────────────────────────────────────────────────────" innerClass="p-0" className="overflow-hidden">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-[9px] bb-dim uppercase tracking-widest border-b border-[#ff8c0015]">
                  <th className="text-left p-2 pl-3">#</th>
                  <th className="text-left p-2">ENTRY DATE</th>
                  <th className="text-left p-2">EXIT DATE</th>
                  <th className="text-right p-2">ENTRY</th>
                  <th className="text-right p-2">EXIT</th>
                  <th className="text-right p-2">P&L %</th>
                  <th className="text-center p-2">RESULT</th>
                </tr>
              </thead>
              <tbody>
                {btResult.trades.map((t, i) => (
                  <tr key={i} className="t-row">
                    <td className="p-2 pl-3 bb-dim">{i + 1}</td>
                    <td className="p-2 bb-dim">{t.entry_date}</td>
                    <td className="p-2 bb-dim">{t.exit_date}</td>
                    <td className="p-2 text-right bb-white">{usd(t.entry_price)}</td>
                    <td className="p-2 text-right bb-white">{usd(t.exit_price)}</td>
                    <td className={`p-2 text-right font-bold ${t.pnl_pct >= 0 ? "bb-green" : "bb-red"}`}>
                      {t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct.toFixed(2)}%
                    </td>
                    <td className="p-2 text-center">
                      <span className={`text-[9px] border px-1 uppercase ${t.outcome === "win" ? "text-[#00cc55] border-[#00cc5530]" : t.outcome === "loss" ? "text-[#ff4444] border-[#ff444430]" : "bb-amber border-[#ff8c0030]"}`}>
                        {t.outcome}
                      </span>
                    </td>
                  </tr>
                ))}
                {btResult.trades.length === 0 && (
                  <tr><td colSpan={7} className="p-4 text-center bb-dim">NO TRADES EXECUTED IN THIS PERIOD</td></tr>
                )}
              </tbody>
            </table>
          </Panel>
        </div>
      ) : btResult && !btResult.available ? (
        <div className="flex-1 flex items-center justify-center bb-red text-sm">[ERR] {btResult.error}</div>
      ) : (
        <div className="flex-1 flex items-center justify-center bb-dim text-sm tracking-widest">
          {btLoading ? "RUNNING BACKTEST..." : "SELECT TICKER, STRATEGY AND RUN BACKTEST"}
        </div>
      )}
    </div>
  );
}

function ScannerTab({
  input, setInput, universe, setUniverse, scanResult, performance, loading, resolving, loadScanner, resolveOutcomes,
}: {
  input: string;
  setInput: (v: string) => void;
  universe: string;
  setUniverse: (v: string) => void;
  scanResult: BotScanResponse | null;
  performance: BotPerformanceResponse | null;
  loading: boolean;
  resolving: boolean;
  loadScanner: () => void;
  resolveOutcomes: () => void;
}) {
  const buckets = scanResult?.quality_buckets ?? {};
  const proof80 = performance?.v12_walk_forward_proof?.confidence_buckets?.[">=80%"];
  return (
    <div className="h-full grid grid-rows-[auto_1fr] gap-px bg-[#ff8c0010]">
      <div className="t-panel shrink-0 flex items-center gap-3 px-4 py-2">
        <span className="bb-amber text-xs tracking-widest">SCAN&gt;</span>
        <select value={universe} onChange={(e) => setUniverse(e.target.value)} className="bb-input text-xs">
          {["custom", "mega_cap_ai", "nasdaq_100_core", "sp500_sample"].map((u) => <option key={u}>{u}</option>)}
        </select>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value.toUpperCase())}
          onKeyDown={(e) => { if (e.key === "Enter") loadScanner(); }}
          className="bb-input flex-1 text-xs"
          placeholder="AAPL,NVDA,MSFT,AMZN,META"
        />
        <button onClick={loadScanner} disabled={loading} className="bb-btn">
          {loading ? "SCANNING..." : "RUN BOT SCAN"}
        </button>
        <button onClick={resolveOutcomes} disabled={resolving} className="bb-btn">
          {resolving ? "RESOLVING..." : "RESOLVE OUTCOMES"}
        </button>
        {scanResult?.cache && <span className={scanResult.cache.hit ? "bb-green text-[10px]" : "bb-dim text-[10px]"}>CACHE:{scanResult.cache.hit ? "HIT" : "MISS"}</span>}
      </div>

      <div className="grid grid-cols-[1fr_330px] gap-px bg-[#ff8c0010] overflow-hidden">
        <Panel title="── BOT SCANNER RANKINGS ─────────────────────────────────────────────────────────────" innerClass="p-0">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-[9px] bb-dim uppercase tracking-widest border-b border-[#ff8c0015]">
                <th className="text-left p-2 pl-3">Ticker</th>
                <th className="text-left p-2">Quality</th>
                <th className="text-left p-2">Action</th>
                <th className="text-right p-2">Conf</th>
                <th className="text-right p-2">Risk</th>
                <th className="text-right p-2">R/R</th>
                <th className="text-left p-2">Failed Gates</th>
              </tr>
            </thead>
            <tbody>
              {(scanResult?.results ?? []).map((row) => (
                <tr key={row.ticker} className="t-row">
                  <td className="p-2 pl-3 bb-amber font-bold">{row.ticker}</td>
                  <td className={`p-2 font-bold ${row.quality_label === "high_confidence_trade_candidate" ? "bb-green" : row.quality_label === "avoid_high_risk" ? "bb-red" : "bb-white"}`}>
                    {(row.quality_label ?? "ERROR").replaceAll("_", " ").toUpperCase()}
                  </td>
                  <td className="p-2 bb-white">{(row.action ?? "—").replaceAll("_", " ").toUpperCase()}</td>
                  <td className="p-2 text-right bb-white">{row.confidence?.toFixed(1) ?? "—"}%</td>
                  <td className={`p-2 text-right font-bold ${(row.risk_score ?? 0) >= 70 ? "bb-red" : "bb-white"}`}>{row.risk_score ?? "—"}</td>
                  <td className="p-2 text-right bb-white">{row.trade_plan?.risk_reward_target_1?.toFixed(2) ?? "—"}</td>
                  <td className="p-2 bb-dim text-[10px]">{row.failed_gates?.join(", ").replaceAll("_", " ") || "PASS"}</td>
                </tr>
              ))}
              {!scanResult && (
                <tr><td colSpan={7} className="p-8 text-center bb-dim">RUN SCANNER TO RANK UNIVERSAL PREDICTIONS</td></tr>
              )}
            </tbody>
          </table>
        </Panel>

        <div className="flex flex-col gap-px bg-[#ff8c0010]">
          <Panel title="── COVERAGE ───────────────────────────────────────────────────" innerClass="p-3">
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(buckets).map(([k, v]) => <Stat key={k} label={k.replaceAll("_", " ")} value={String(v)} cls={k.includes("high") ? "bb-green" : k.includes("avoid") ? "bb-red" : "bb-white"} />)}
              {!Object.keys(buckets).length && <Stat label="scan status" value={loading ? "RUNNING" : "IDLE"} cls="bb-amber" />}
            </div>
          </Panel>
          <Panel title="── PROOF / CALIBRATION ────────────────────────────────────────" innerClass="p-3">
            <div className="space-y-2 text-[10px]">
              <div><span className="bb-dim">LOGGED: </span><span className="bb-white font-bold">{performance?.total_logged_predictions ?? 0}</span></div>
              <div><span className="bb-dim">PROMOTED: </span><span className="bb-green font-bold">{performance?.promoted_signals ?? 0}</span></div>
              <div><span className="bb-dim">PENDING OUTCOMES: </span><span className="bb-amber font-bold">{performance?.pending_outcomes ?? 0}</span></div>
              <div><span className="bb-dim">LIVE WIN RATE: </span><span className="bb-white">{performance?.resolved_win_rate == null ? "PENDING" : `${(performance.resolved_win_rate * 100).toFixed(1)}%`}</span></div>
              {proof80 && <div><span className="bb-dim">V12 PROOF @80: </span><span className="bb-green font-bold">{((proof80.accuracy ?? 0) * 100).toFixed(1)}%</span><span className="bb-dim"> / {proof80.signals} signals</span></div>}
              <p className="bb-dim leading-4 pt-2">{performance?.note ?? "Live calibration appears after future outcomes are resolved."}</p>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function ProofDashboardTab({
  performance, resolving, resolveOutcomes,
}: {
  performance: BotPerformanceResponse | null;
  resolving: boolean;
  resolveOutcomes: () => void;
}) {
  const proofBuckets = performance?.v12_walk_forward_proof?.confidence_buckets ?? {};
  const liveBuckets = performance?.confidence_buckets ?? {};
  const qualityRows = Object.entries(performance?.quality_performance ?? {});
  const actionRows = Object.entries(performance?.action_performance ?? {});
  return (
    <div className="h-full grid grid-rows-[auto_1fr] gap-px bg-[#ff8c0010]">
      <div className="t-panel shrink-0 flex items-center gap-3 px-4 py-2">
        <span className="bb-amber text-xs tracking-widest">PROOF&gt;</span>
        <button onClick={resolveOutcomes} disabled={resolving} className="bb-btn">
          {resolving ? "RESOLVING..." : "RESOLVE LIVE OUTCOMES"}
        </button>
        <span className="text-[10px] bb-dim truncate">{performance?.note ?? "Live proof updates after predictions mature."}</span>
      </div>

      <div className="grid grid-cols-[320px_1fr_360px] gap-px bg-[#ff8c0010] overflow-hidden">
        <div className="flex flex-col gap-px bg-[#ff8c0010]">
          <Panel title="── SIGNAL LEDGER ──────────────────────────────────────────────" innerClass="p-3">
            <div className="grid grid-cols-2 gap-2">
              <Stat label="logged" value={String(performance?.total_logged_predictions ?? 0)} cls="bb-white" />
              <Stat label="promoted" value={String(performance?.promoted_signals ?? 0)} cls="bb-green" />
              <Stat label="pending" value={String(performance?.pending_outcomes ?? 0)} cls="bb-amber" />
              <Stat label="resolved" value={String(performance?.resolved_outcomes ?? 0)} cls="bb-white" />
              <Stat
                label="live win rate"
                value={performance?.resolved_win_rate == null ? "PENDING" : `${(performance.resolved_win_rate * 100).toFixed(1)}%`}
                cls={performance?.resolved_win_rate == null ? "bb-dim" : "bb-green"}
              />
              <Stat label="policy" value="NO GUARANTEES" cls="bb-red" />
            </div>
          </Panel>

          <Panel title="── V12 WALK-FORWARD PROOF ─────────────────────────────────────" innerClass="p-3" className="flex-1">
            <div className="space-y-2 text-xs">
              <div><span className="bb-dim">ACTIVE SETUP ACC: </span><span className="bb-white font-bold">{performance?.v12_walk_forward_proof?.accuracy == null ? "—" : `${(performance.v12_walk_forward_proof.accuracy * 100).toFixed(1)}%`}</span></div>
              <div><span className="bb-dim">SETUP COVERAGE: </span><span className="bb-white">{performance?.v12_walk_forward_proof?.setup_coverage_test == null ? "—" : `${(performance.v12_walk_forward_proof.setup_coverage_test * 100).toFixed(2)}%`}</span></div>
              <p className="pt-2 text-[10px] leading-4 bb-dim">{performance?.accuracy_policy ?? "Only claim high-confidence accuracy when bucket accuracy and coverage are both disclosed."}</p>
            </div>
          </Panel>
        </div>

        <Panel title="── CONFIDENCE CALIBRATION ─────────────────────────────────────────────────────────" innerClass="p-0">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="text-[9px] bb-dim uppercase tracking-widest border-b border-[#ff8c0015]">
                <th className="text-left p-2 pl-3">Bucket</th>
                <th className="text-right p-2">V12 Accuracy</th>
                <th className="text-right p-2">V12 Signals</th>
                <th className="text-right p-2">Live Count</th>
                <th className="text-right p-2">Resolved</th>
                <th className="text-right p-2">Live Win</th>
              </tr>
            </thead>
            <tbody>
              {[">=60%", ">=70%", ">=80%", ">=85%"].map((bucket) => {
                const liveKey = bucket.replace("%", "");
                const proof = proofBuckets[bucket];
                const live = liveBuckets[liveKey];
                return (
                  <tr key={bucket} className="t-row">
                    <td className="p-2 pl-3 bb-amber font-bold">{bucket}</td>
                    <td className="p-2 text-right bb-white">{proof?.accuracy == null ? "—" : `${(proof.accuracy * 100).toFixed(1)}%`}</td>
                    <td className="p-2 text-right bb-dim">{proof?.signals ?? "—"}</td>
                    <td className="p-2 text-right bb-white">{live?.count ?? 0}</td>
                    <td className="p-2 text-right bb-white">{live?.resolved ?? 0}</td>
                    <td className={`p-2 text-right font-bold ${live?.win_rate == null ? "bb-dim" : live.win_rate >= 0.6 ? "bb-green" : "bb-red"}`}>
                      {live?.win_rate == null ? "PENDING" : `${(live.win_rate * 100).toFixed(1)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>

        <div className="flex flex-col gap-px bg-[#ff8c0010] overflow-hidden">
          <Panel title="── QUALITY PERFORMANCE ───────────────────────────────────────" innerClass="p-0" className="flex-1">
            <table className="w-full text-[11px] font-mono">
              <tbody>
                {qualityRows.map(([label, row]) => (
                  <tr key={label} className="t-row">
                    <td className="p-2 bb-dim">{label.replaceAll("_", " ").toUpperCase()}</td>
                    <td className="p-2 text-right bb-white">{row.count}</td>
                    <td className="p-2 text-right bb-amber">{row.resolved}</td>
                    <td className={`p-2 text-right font-bold ${row.win_rate == null ? "bb-dim" : row.win_rate >= 0.6 ? "bb-green" : "bb-red"}`}>{row.win_rate == null ? "—" : `${(row.win_rate * 100).toFixed(1)}%`}</td>
                  </tr>
                ))}
                {!qualityRows.length && <tr><td className="p-4 text-center bb-dim">NO LIVE QUALITY DATA YET</td></tr>}
              </tbody>
            </table>
          </Panel>

          <Panel title="── ACTION PERFORMANCE ────────────────────────────────────────" innerClass="p-0" className="flex-1">
            <table className="w-full text-[11px] font-mono">
              <tbody>
                {actionRows.map(([label, row]) => (
                  <tr key={label} className="t-row">
                    <td className="p-2 bb-dim">{label.replaceAll("_", " ").toUpperCase()}</td>
                    <td className="p-2 text-right bb-white">{row.count}</td>
                    <td className="p-2 text-right bb-amber">{row.resolved}</td>
                    <td className={`p-2 text-right font-bold ${row.win_rate == null ? "bb-dim" : row.win_rate >= 0.6 ? "bb-green" : "bb-red"}`}>{row.win_rate == null ? "—" : `${(row.win_rate * 100).toFixed(1)}%`}</td>
                  </tr>
                ))}
                {!actionRows.length && <tr><td className="p-4 text-center bb-dim">NO LIVE ACTION DATA YET</td></tr>}
              </tbody>
            </table>
          </Panel>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
//  MAIN HOME
// ═══════════════════════════════════════════════════════════════════
export default function Home() {
  const time = useClock();
  const [tab, setTab] = useState<Tab>("overview");

  // ── Simulation state ────────────────────────────────────────────
  const [ticker, setTicker] = useState("NVDA");
  const [period, setPeriod] = useState("1y");
  const [simInterval, setSimInterval] = useState("1d");
  const [simulation, setSimulation] = useState<SimulationResponse>(fallbackSimulation);
  const [isLoading, setIsLoading] = useState(false);
  const [dataStatus, setDataStatus] = useState("Connecting to backend...");
  const simulationRequestRef = useRef({ ticker: "NVDA", period: "1y", interval: "1d" });
  const activeTabRef = useRef<Tab>("overview");

  // ── Live data state ─────────────────────────────────────────────
  const [liveMarketCards, setLiveMarketCards] = useState<MarketCard[]>(mockMarketCards);
  const [liveNews, setLiveNews] = useState<NewsItem[]>(mockNewsItems as NewsItem[]);
  const [mlLoaded, setMlLoaded] = useState(false);
  const [mlVersion, setMlVersion] = useState<string | null>(null);
  const [marketCardsLoading, setMarketCardsLoading] = useState(true);

  // ── Research state ──────────────────────────────────────────────
  const [stockQuery, setStockQuery] = useState("AAPL");
  const [stockPeriod, setStockPeriod] = useState("6mo");
  const [stockOverview, setStockOverview] = useState<StockOverview | null>(null);
  const [stockAnalysis, setStockAnalysis] = useState<StockAnalysis | null>(null);
  const [stockNews, setStockNews] = useState<NewsItem[]>([]);
  const [stockLoading, setStockLoading] = useState(false);
  const [stockError, setStockError] = useState<string | null>(null);

  // ── Similarity state ────────────────────────────────────────
  const [similarity, setSimilarity] = useState<SimilarityResult | null>(null);
  const [similarityLoading, setSimilarityLoading] = useState(false);

  // ── Macro state ─────────────────────────────────────────────────
  const [macroCond, setMacroCond] = useState<MacroConditions | null>(null);

  // ── Copilot state ───────────────────────────────────────────────
  const [copilotHistory, setCopilotHistory] = useState<{ role: string; content: string }[]>([]);
  const [copilotInput, setCopilotInput] = useState("");
  const [copilotLoading, setCopilotLoading] = useState(false);

  // ── Portfolio state ─────────────────────────────────────────────
  const [portfolioRows, setPortfolioRows] = useState([
    { ticker: "AAPL", shares: "10" },
    { ticker: "NVDA", shares: "5" },
    { ticker: "MSFT", shares: "8" },
  ]);
  const [portfolioResult, setPortfolioResult] = useState<PortfolioAnalysis | null>(null);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [portfolioError, setPortfolioError] = useState<string | null>(null);

  // ── Watchlist state ─────────────────────────────────────────────
  const [watchlistInput, setWatchlistInput] = useState("AAPL, NVDA, TSLA, MSFT");
  const [watchlistData, setWatchlistData] = useState<WatchlistItem[]>([]);
  const [watchlistLoading, setWatchlistLoading] = useState(false);

  // ── Compare state ───────────────────────────────────────────────
  const [compareInput, setCompareInput] = useState("AAPL, NVDA, MSFT, TSLA");
  const [comparePeriod, setComparePeriod] = useState("1y");
  const [compareResult, setCompareResult] = useState<CompareResult | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  // ── Backtest state ──────────────────────────────────────────────
  const [btTicker, setBtTicker] = useState("AAPL");
  const [btStrategy, setBtStrategy] = useState("sma_crossover");
  const [btPeriod, setBtPeriod] = useState("3y");
  const [btCapital, setBtCapital] = useState("10000");
  const [btResult, setBtResult] = useState<BacktestResult | null>(null);
  const [btLoading, setBtLoading] = useState(false);
  const [strategies, setStrategies] = useState<{ id: string; label: string }[]>([
    { id: "sma_crossover", label: "SMA Crossover (20/50)" },
    { id: "rsi_mean_reversion", label: "RSI Mean Reversion (30/70)" },
    { id: "macd", label: "MACD Signal Line" },
    { id: "bollinger_bands", label: "Bollinger Bands (20, 2σ)" },
    { id: "buy_and_hold", label: "Buy & Hold (Baseline)" },
  ]);

  // ── Scanner state ───────────────────────────────────────────────
  const [scannerInput, setScannerInput] = useState("AAPL,NVDA,MSFT,AMZN,META,GOOGL,TSLA,AVGO,AMD,CRM");
  const [scannerUniverse, setScannerUniverse] = useState("custom");
  const [scannerResult, setScannerResult] = useState<BotScanResponse | null>(null);
  const [botPerformance, setBotPerformance] = useState<BotPerformanceResponse | null>(null);
  const [scannerLoading, setScannerLoading] = useState(false);
  const [outcomeResolving, setOutcomeResolving] = useState(false);

  // ── API integration status ──────────────────────────────────────
  const [apiStatus, setApiStatus] = useState({ finnhub: false, openai: false, fred: false, newsapi: false });

  // ── Functions ───────────────────────────────────────────────────
  async function loadPrediction(nextTicker = ticker, nextPeriod = period, nextInterval = simInterval) {
    const request = {
      ticker: nextTicker.trim().toUpperCase(),
      period: nextPeriod,
      interval: nextInterval,
    };
    if (!request.ticker) return;
    simulationRequestRef.current = request;
    setTicker(request.ticker);
    setPeriod(request.period);
    setSimInterval(request.interval);
    setIsLoading(true);
    setDataStatus("Running simulation...");
    try {
      const result = await fetchPrediction({ ...request, horizon_steps: 18 });
      const active = simulationRequestRef.current;
      if (active.ticker !== request.ticker || active.period !== request.period || active.interval !== request.interval) return;
      setSimulation(result);
      setDataStatus(`Simulation complete · ${request.interval} candles · ${new Date(result.as_of).toLocaleTimeString()} · ML:${result.ml_enabled ? "ON" : "RULES"}`);
    } catch {
      const active = simulationRequestRef.current;
      if (active.ticker !== request.ticker || active.period !== request.period || active.interval !== request.interval) return;
      setSimulation(fallbackSimulation);
      setDataStatus("Backend unavailable — showing demo data");
    } finally {
      setIsLoading(false);
    }
  }

  async function loadStock(nextTicker = stockQuery, nextPeriod = stockPeriod) {
    const t = nextTicker.trim().toUpperCase();
    if (!t) return;
    setStockLoading(true);
    setStockError(null);
    setSimilarityLoading(true);
    setSimilarity(null);
    try {
      const [overview, analysis, news] = await Promise.all([
        fetchStockOverview(t, nextPeriod),
        fetchStockAnalysis(t, nextPeriod),
        fetchStockNews(t, 6),
      ]);
      setStockOverview(overview);
      setStockAnalysis(analysis);
      setStockNews(news);
      // Load similarity in background (slow — 5y scan)
      fetchSimilarity(t).then(setSimilarity).catch(() => {}).finally(() => setSimilarityLoading(false));
    } catch (err) {
      setStockError(err instanceof Error ? err.message : "Failed to load stock data.");
      setSimilarityLoading(false);
    } finally {
      setStockLoading(false);
    }
  }

  async function sendCopilot() {
    const msg = copilotInput.trim();
    if (!msg || copilotLoading) return;
    const userMsg = { role: "user", content: msg };
    const next = [...copilotHistory, userMsg];
    setCopilotHistory(next);
    setCopilotInput("");
    setCopilotLoading(true);
    try {
      const ctx = stockAnalysis
        ? { ticker: stockAnalysis.ticker, stance: stockAnalysis.stance, score: stockAnalysis.score, current_price: stockAnalysis.current_price, reasons: stockAnalysis.reasons, risks: stockAnalysis.risks }
        : undefined;
      const apiHistory = next.slice(-6).map((m) => ({ role: m.role === "user" ? "user" : "assistant", content: m.content }));
      const res = await copilotChat(msg, ctx as Record<string, unknown> | undefined, apiHistory);
      setCopilotHistory((h) => [...h, { role: "ai", content: res.reply }]);
    } catch {
      setCopilotHistory((h) => [...h, { role: "ai", content: "[ERROR] Copilot unavailable — check OpenAI API key." }]);
    } finally {
      setCopilotLoading(false);
    }
  }

  async function runPortfolioAnalysis() {
    const valid = portfolioRows.filter((r) => r.ticker.trim() && Number(r.shares) > 0);
    if (!valid.length) return;
    setPortfolioLoading(true);
    setPortfolioError(null);
    try {
      const holdings = valid.map((r) => ({ ticker: r.ticker.trim().toUpperCase(), shares: Number(r.shares) }));
      setPortfolioResult(await analyzePortfolio(holdings, "1y"));
    } catch (err) {
      setPortfolioError(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setPortfolioLoading(false);
    }
  }

  async function loadWatchlist() {
    const tickers = watchlistInput.split(/[,\s]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (!tickers.length) return;
    setWatchlistLoading(true);
    try {
      const res = await fetchWatchlist(tickers, "1mo");
      setWatchlistData(res.watchlist);
    } catch { /* keep previous */ } finally {
      setWatchlistLoading(false);
    }
  }

  async function runCompare() {
    const tickers = compareInput.split(/[,\s]+/).map((t) => t.trim().toUpperCase()).filter(Boolean);
    if (!tickers.length) return;
    setCompareLoading(true);
    try {
      setCompareResult(await compareStocks(tickers, comparePeriod));
    } catch { /* keep previous */ } finally {
      setCompareLoading(false);
    }
  }

  async function runBt() {
    const t = btTicker.trim().toUpperCase();
    if (!t) return;
    setBtLoading(true);
    try {
      setBtResult(await runBacktest(t, btStrategy, btPeriod, Number(btCapital) || 10000));
    } catch (err) {
      setBtResult({ available: false, error: err instanceof Error ? err.message : "Backtest failed." } as BacktestResult);
    } finally {
      setBtLoading(false);
    }
  }

  async function loadScanner() {
    const t = scannerInput.trim();
    if (!t || scannerLoading) return;
    setScannerLoading(true);
    try {
      const [scan, perf] = await Promise.all([
        scanBotSignals(t, "5d", "1d", 12, scannerUniverse),
        fetchBotPerformance(),
      ]);
      setScannerResult(scan);
      setBotPerformance(perf);
    } catch {
      // keep previous scanner state
    } finally {
      setScannerLoading(false);
    }
  }

  async function resolveOutcomes() {
    if (outcomeResolving) return;
    setOutcomeResolving(true);
    try {
      const result = await resolveBotOutcomes();
      setBotPerformance(result.performance);
    } catch {
      // keep previous performance state
    } finally {
      setOutcomeResolving(false);
    }
  }

  function goResearch(t: string) {
    setStockQuery(t);
    setTab("research");
    void loadStock(t, stockPeriod);
  }

  // ── Effects ─────────────────────────────────────────────────────
  useEffect(() => {
    simulationRequestRef.current = { ticker: ticker.trim().toUpperCase() || "NVDA", period, interval: simInterval };
  }, [ticker, period, simInterval]);

  useEffect(() => {
    activeTabRef.current = tab;
    window.scrollTo(0, 0);
  }, [tab]);

  useEffect(() => {
    void loadPrediction("NVDA", "1y", "1d");

    fetchHealth()
      .then((h) => {
        setMlLoaded(h.ml_model_loaded);
        setMlVersion(h.ml_model_version === "none" ? null : h.ml_model_version);
        setApiStatus({ finnhub: h.finnhub_connected ?? false, openai: h.copilot_connected ?? false, fred: h.fred_connected ?? false, newsapi: h.newsapi_connected ?? false });
      }).catch(() => {});

    setMarketCardsLoading(true);
    fetchMarketOverview()
      .then((data) => { if (data.cards.length) setLiveMarketCards(data.cards); })
      .catch(() => {})
      .finally(() => setMarketCardsLoading(false));

    fetchMarketNewsItems(8).then((items) => { if (items.length) setLiveNews(items); }).catch(() => {});
    fetchMacroConditions().then(setMacroCond).catch(() => {});
    fetchBacktestStrategies().then((r) => { if (r.strategies.length) setStrategies(r.strategies); }).catch(() => {});
    fetchWatchlist(["AAPL", "NVDA", "TSLA", "MSFT"], "1mo").then((r) => setWatchlistData(r.watchlist)).catch(() => {});
    fetchBotPerformance().then(setBotPerformance).catch(() => {});

    const marketId = setInterval(() => {
      fetchMarketOverview()
        .then((data) => { if (data.cards.length) setLiveMarketCards(data.cards); })
        .catch(() => {});
    }, 7_000);

    // Auto-refresh simulation every 60s when on simulation tab
    const simId = setInterval(() => {
      if (activeTabRef.current !== "simulation") return;
      const request = simulationRequestRef.current;
      fetchPrediction({ ...request, horizon_steps: 18 })
        .then((result) => {
          const active = simulationRequestRef.current;
          if (active.ticker === request.ticker && active.period === request.period && active.interval === request.interval) {
            setSimulation(result);
          }
        })
        .catch(() => {});
    }, 60_000);

    return () => { clearInterval(marketId); clearInterval(simId); };
  }, []);

  // ── Render ──────────────────────────────────────────────────────
  const TABS: [string, string, Tab][] = [
    ["F1", "OVERVIEW", "overview"],
    ["F2", "RESEARCH", "research"],
    ["F3", "SIMULATION", "simulation"],
    ["F4", "SCANNER", "scanner"],
    ["F5", "PROOF", "proof"],
    ["F6", "PORTFOLIO", "portfolio"],
    ["F7", "AI·COPILOT", "copilot"],
    ["F8", "MACRO·DATA", "macro"],
    ["F9", "COMPARE", "compare"],
    ["F10", "BACKTEST", "backtest"],
  ];

  return (
    <main className="scanlines fixed inset-0 h-screen w-screen max-w-screen min-w-0 flex flex-col overflow-hidden bg-[#040404] text-[#c8c8c8]" style={{ fontFamily: "'Courier New', 'Lucida Console', ui-monospace, monospace" }}>

      {/* ── HEADER ──────────────────────────────────────────────── */}
      <header className="shrink-0 min-w-0 flex items-center border-b border-[#ff8c0025] bg-[#050505]">
        <div className="flex items-center gap-2 px-4 py-1.5 border-r border-[#ff8c0020] shrink-0">
          <span className="text-[#ff8c00] font-bold text-xs tracking-[0.25em]">MKTVSN::AI</span>
        </div>
        <div className="min-w-0 flex-1 overflow-hidden py-1.5 px-3">
          <div className="ticker-inner inline-flex gap-8 text-[11px]">
            {[...liveMarketCards, ...liveMarketCards].map((card, i) => (
              <span key={i} className="inline-flex items-center gap-2 shrink-0">
                <span className="bb-amber font-bold">{card.name}</span>
                <span className="bb-white">{card.value}</span>
                <span className={card.change.startsWith("+") ? "bb-green" : "bb-red"}>{card.change}</span>
                <span className="bb-dim">|</span>
              </span>
            ))}
          </div>
        </div>
        <div className="shrink-0 flex items-center gap-4 px-4 py-1.5 border-l border-[#ff8c0020] text-[11px]">
          <span className="bb-amber font-bold tracking-widest">{time}</span>
        </div>
      </header>

      {/* ── TAB BAR ──────────────────────────────────────────────── */}
      <div className="shrink-0 min-w-0 flex items-stretch overflow-x-auto overflow-y-hidden border-b border-[#ff8c0025] bg-[#030303] text-[11px]">
        {TABS.map(([fKey, label, tabId]) => (
          <button
            key={tabId}
            onClick={() => setTab(tabId)}
            className={`flex items-center gap-1.5 px-4 py-2 border-r border-[#ff8c0015] tracking-wider transition-colors ${
              tab === tabId ? "tab-active bg-[#ff8c0010] bb-amber" : "tab-inactive bb-dim hover:text-[#aaa] hover:bg-[#0d0d0d]"
            }`}
          >
            <span className={`text-[9px] ${tab === tabId ? "text-[#ff8c0060]" : "text-[#333]"}`}>{fKey}</span>
            {label}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-3 px-4 text-[9px]">
          {mlLoaded && <span className="bb-green">■ ML:{mlVersion?.slice(-8) ?? "ACTIVE"}</span>}
          <span className={apiStatus.finnhub ? "bb-green" : "bb-red"}>FINNHUB:{apiStatus.finnhub ? "OK" : "NO"}</span>
          <span className={apiStatus.openai ? "bb-green" : "bb-red"}>OPENAI:{apiStatus.openai ? "OK" : "NO"}</span>
          <span className={apiStatus.fred ? "bb-green" : "bb-red"}>FRED:{apiStatus.fred ? "OK" : "NO"}</span>
          <span className={apiStatus.newsapi ? "bb-green" : "bb-red"}>NEWSAPI:{apiStatus.newsapi ? "OK" : "NO"}</span>
        </div>
      </div>

      {/* ── CONTENT ──────────────────────────────────────────────── */}
      <div className="min-w-0 flex-1 overflow-hidden">
        {tab === "overview" && (
          <OverviewTab
            marketCards={liveMarketCards} marketCardsLoading={marketCardsLoading}
            watchlistData={watchlistData} watchlistInput={watchlistInput}
            setWatchlistInput={setWatchlistInput} loadWatchlist={loadWatchlist}
            watchlistLoading={watchlistLoading} liveNews={liveNews}
            goResearch={goResearch}
          />
        )}
        {tab === "research" && (
          <ResearchTab
            stockQuery={stockQuery} setStockQuery={setStockQuery}
            stockPeriod={stockPeriod} setStockPeriod={setStockPeriod}
            loadStock={loadStock} stockLoading={stockLoading}
            stockError={stockError} stockOverview={stockOverview}
            stockAnalysis={stockAnalysis} stockNews={stockNews}
            similarity={similarity} similarityLoading={similarityLoading}
          />
        )}
        {tab === "simulation" && (
          <SimulationTab
            ticker={ticker} setTicker={setTicker}
            period={period} setPeriod={setPeriod}
            interval={simInterval} setIntervalValue={setSimInterval}
            loadPrediction={loadPrediction}
            isLoading={isLoading} simulation={simulation} dataStatus={dataStatus}
          />
        )}
        {tab === "scanner" && (
          <ScannerTab
            input={scannerInput}
            setInput={setScannerInput}
            universe={scannerUniverse}
            setUniverse={setScannerUniverse}
            scanResult={scannerResult}
            performance={botPerformance}
            loading={scannerLoading}
            resolving={outcomeResolving}
            loadScanner={loadScanner}
            resolveOutcomes={resolveOutcomes}
          />
        )}
        {tab === "proof" && (
          <ProofDashboardTab
            performance={botPerformance}
            resolving={outcomeResolving}
            resolveOutcomes={resolveOutcomes}
          />
        )}
        {tab === "portfolio" && (
          <PortfolioTab
            portfolioRows={portfolioRows} setPortfolioRows={setPortfolioRows}
            runPortfolioAnalysis={runPortfolioAnalysis}
            portfolioLoading={portfolioLoading} portfolioError={portfolioError}
            portfolioResult={portfolioResult}
          />
        )}
        {tab === "copilot" && (
          <CopilotTab
            copilotHistory={copilotHistory} copilotInput={copilotInput}
            setCopilotInput={setCopilotInput} sendCopilot={sendCopilot}
            copilotLoading={copilotLoading} stockAnalysis={stockAnalysis}
          />
        )}
        {tab === "macro" && <MacroTab macroCond={macroCond} />}
        {tab === "compare" && (
          <CompareTab
            compareInput={compareInput} setCompareInput={setCompareInput}
            compareResult={compareResult} compareLoading={compareLoading}
            runCompare={runCompare} comparePeriod={comparePeriod} setComparePeriod={setComparePeriod}
          />
        )}
        {tab === "backtest" && (
          <BacktestTab
            btTicker={btTicker} setBtTicker={setBtTicker}
            btStrategy={btStrategy} setBtStrategy={setBtStrategy}
            btPeriod={btPeriod} setBtPeriod={setBtPeriod}
            btCapital={btCapital} setBtCapital={setBtCapital}
            btResult={btResult} btLoading={btLoading}
            runBt={runBt} strategies={strategies}
          />
        )}
      </div>

      {/* ── STATUS BAR ───────────────────────────────────────────── */}
      <footer className="shrink-0 border-t border-[#ff8c0018] bg-[#030303] px-4 py-1 flex gap-6 text-[9px] tracking-wider">
        <span className="bb-amber">{dataStatus}</span>
        <span className="bb-dim">BACKEND: localhost:8000</span>
        <span className="bb-dim ml-auto">MarketVision AI Terminal v2.0 · {new Date().toLocaleDateString("en-US", { year: "numeric", month: "short", day: "2-digit" })}</span>
      </footer>
    </main>
  );
}
