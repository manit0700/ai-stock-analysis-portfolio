import type { SimulationResponse } from "./types";

export const marketCards = [
  { name: "S&P 500", value: "6,214.88", change: "+0.74%", trend: "Bullish", volatility: "Low", points: [24, 26, 25, 31, 33, 37, 36, 42] },
  { name: "Nasdaq", value: "21,058.17", change: "+1.12%", trend: "AI Bid", volatility: "Medium", points: [18, 22, 26, 24, 31, 35, 39, 44] },
  { name: "Dow Jones", value: "44,902.30", change: "-0.18%", trend: "Mixed", volatility: "Low", points: [36, 35, 34, 33, 35, 34, 33, 32] },
  { name: "Bitcoin", value: "$104,820", change: "+2.41%", trend: "Momentum", volatility: "High", points: [28, 33, 31, 42, 39, 46, 51, 58] },
  { name: "VIX", value: "14.22", change: "-3.04%", trend: "Risk-On", volatility: "Falling", points: [48, 46, 41, 39, 37, 32, 30, 28] },
  { name: "Top Sector", value: "Semis", change: "+1.86%", trend: "Leader", volatility: "Medium", points: [22, 25, 29, 34, 33, 39, 45, 49] },
];

export const probabilities = [
  { label: "Bullish", value: 68, color: "#34d399" },
  { label: "Bearish", value: 18, color: "#fb7185" },
  { label: "Sideways", value: 14, color: "#38bdf8" },
  { label: "Risk", value: 42, color: "#f59e0b" },
  { label: "Confidence", value: 76, color: "#8b5cf6" },
];

export const newsItems = [
  { title: "AI chip leaders extend sector strength after guidance revisions", sentiment: "+0.71", impact: "High" },
  { title: "Fed speakers keep rate-cut expectations data dependent", sentiment: "-0.08", impact: "Medium" },
  { title: "Large-cap software earnings show margin expansion", sentiment: "+0.44", impact: "Medium" },
  { title: "Insider filings show selective buying in cyber security names", sentiment: "+0.22", impact: "Low" },
];

export const similaritySetups = [
  { date: "2024 Q2", match: 87, outcome: "+9.4% over 18 sessions", regime: "AI momentum breakout" },
  { date: "2023 Q4", match: 81, outcome: "+6.1% after consolidation", regime: "Fed pause rally" },
  { date: "2021 Q1", match: 74, outcome: "-3.8% volatility shock", regime: "crowded growth trade" },
];

export const copilotMessages = [
  { role: "ai", text: "NVDA shows a bullish dominant scenario, but the model is flagging resistance near the upper confidence band." },
  { role: "user", text: "What would invalidate this setup?" },
  { role: "ai", text: "A close below VWAP with rising sell volume would reduce bullish probability and shift the engine toward sideways or bearish." },
];

export const riskMetrics = [
  { label: "Volatility Risk", value: 42, tone: "medium" },
  { label: "Drawdown Pressure", value: 28, tone: "low" },
  { label: "Position Heat", value: 61, tone: "medium" },
  { label: "Event Risk", value: 74, tone: "high" },
];

const fallbackHistory = Array.from({ length: 70 }, (_, index) => {
  const base = 182 + index * 0.52 + Math.sin(index / 2.7) * 3.5;
  const open = base + Math.sin(index) * 1.2;
  const close = base + Math.cos(index / 1.7) * 1.7;
  return {
    date: new Date(Date.UTC(2026, 1, 1 + index)).toISOString().slice(0, 10),
    open: Number(open.toFixed(2)),
    high: Number((Math.max(open, close) + 2.4).toFixed(2)),
    low: Number((Math.min(open, close) - 2.2).toFixed(2)),
    close: Number(close.toFixed(2)),
    volume: 1000000 + index * 6500,
  };
});

export const fallbackSimulation: SimulationResponse = {
  ticker: "NVDA",
  period: "1y",
  interval: "1d",
  horizon_steps: 18,
  current_price: fallbackHistory[fallbackHistory.length - 1].close,
  probabilities: {
    bullish: 0.68,
    bearish: 0.18,
    sideways: 0.14,
  },
  dominant_scenario: "bullish",
  confidence: 76,
  risk_score: 42,
  risk_level: "medium",
  scenario_paths: {
    bullish: Array.from({ length: 18 }, (_, index) => Number((218 + index * 1.35 + Math.sin(index / 2) * 1.9).toFixed(2))),
    bearish: Array.from({ length: 18 }, (_, index) => Number((218 - index * 0.95 + Math.sin(index / 2) * 1.4).toFixed(2))),
    sideways: Array.from({ length: 18 }, (_, index) => Number((218 + index * 0.08 + Math.sin(index / 2) * 1.6).toFixed(2))),
    high_volatility: Array.from({ length: 18 }, (_, index) => Number((218 + index * 0.25 + Math.sin(index / 1.1) * 5.2).toFixed(2))),
  },
  predicted_prices: Array.from({ length: 18 }, (_, index) => Number((218 + index * 1.35 + Math.sin(index / 2) * 1.9).toFixed(2))),
  confidence_band: {
    upper: Array.from({ length: 18 }, (_, index) => Number((222 + index * 1.45).toFixed(2))),
    lower: Array.from({ length: 18 }, (_, index) => Number((214 + index * 0.95).toFixed(2))),
  },
  reasoning:
    "NVDA currently has a bullish dominant scenario with 76% model confidence. Strong volume accumulation and positive semiconductor sector momentum support continuation. Main risk: resistance near the upper confidence band may limit upside momentum.",
  reasons: ["Price is above the 20-period average.", "MACD histogram is positive.", "Recent volume is elevated."],
  risks: ["Volatility is elevated, so the projected path has higher uncertainty.", "Resistance near the upper band can cap upside."],
  recent_history: fallbackHistory,
  disclaimer: "This is a probability-based simulation for research only, not financial advice.",
  as_of: new Date().toISOString(),
};
