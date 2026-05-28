export type MarketCard = {
  name: string;
  value: string;
  change: string;
  trend: string;
  volatility: string;
  points: number[];
};

export type MarketOverviewResponse = {
  cards: MarketCard[];
  as_of: string;
};

export type HealthResponse = {
  status: string;
  ml_model_loaded: boolean;
  ml_model_version: string;
  finnhub_connected?: boolean;
  copilot_connected?: boolean;
  fred_connected?: boolean;
  newsapi_connected?: boolean;
};

export type MacroConditions = {
  available: boolean;
  source: string;
  fed_funds_rate: number | null;
  cpi: number | null;
  unemployment: number | null;
  treasury_10y: number | null;
  treasury_2y: number | null;
  vix: number | null;
  yield_spread: number | null;
  flags: string[];
  risks: string[];
  overall_assessment: "positive" | "mixed" | "cautious" | "uncertain" | string;
};

export type CopilotChatResponse = {
  reply: string;
  model: string;
};

export type PortfolioHolding = {
  ticker: string;
  shares: number;
  latest_price: number;
  market_value: number;
  weight_pct: number;
};

export type PortfolioAnalysis = {
  total_market_value: number;
  annualized_return_pct: number;
  annualized_volatility_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  diversification_score: number;
  holdings: PortfolioHolding[];
  correlation_matrix: Record<string, Record<string, number>>;
  warnings: string[];
};

export type WatchlistItem = {
  ticker: string;
  company_name: string;
  stance: string;
  score: number;
  daily_change_pct: number;
  headline_reason: string;
  top_risk: string;
};

export type WatchlistResponse = {
  watchlist: WatchlistItem[];
  disclaimer: string;
  period: string;
};

export type SimilarSetup = {
  date: string;
  similarity: number;
  entry_price: number;
  forward_return_pct: number;
  outcome: "bullish" | "bearish" | "sideways";
  forward_bars: number;
};

export type CompareLeader = {
  ticker: string;
  company_name: string;
  stance: string;
  score: number;
  current_price: number;
  daily_change_pct: number;
  signals: {
    sma_20: number | null;
    sma_50: number | null;
    rsi_14: number | null;
    annualized_volatility_30d: number | null;
    max_drawdown_period: number | null;
  };
  reasons: string[];
  risks: string[];
};

export type CompareResult = {
  period: string;
  leaders: CompareLeader[];
};

export type BacktestTrade = {
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  pnl_pct: number;
  gross_pnl_pct?: number;
  estimated_round_trip_cost_pct?: number;
  outcome: "win" | "loss" | "open";
};

export type BacktestResult = {
  available: boolean;
  ticker: string;
  strategy: string;
  strategy_label: string;
  period: string;
  initial_capital: number;
  total_return_pct: number;
  buy_hold_return_pct: number;
  annualized_return_pct: number;
  annualized_volatility_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  total_trades: number;
  win_rate_pct: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  final_equity: number;
  trades: BacktestTrade[];
  equity_curve: { date: string; equity: number }[];
  execution_assumptions?: {
    commission_pct_per_trade: number;
    spread_estimate_pct: number;
    slippage_estimate_pct: number;
    one_way_cost_pct: number;
    round_trip_cost_pct: number;
    note: string;
  };
  error?: string;
};

export type SimilarityResult = {
  ticker: string;
  available: boolean;
  setups: SimilarSetup[];
  summary: {
    dominant_historical_outcome: string;
    avg_forward_return_pct: number;
    bullish_count: number;
    bearish_count: number;
    sideways_count: number;
    total_matches_scanned: number;
  };
};

export type NewsItem = {
  title: string;
  publisher: string | null;
  link: string | null;
  published_at: string | null;
  summary: string | null;
};

export type PriceHistoryPoint = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type StockIndicators = {
  sma_20: number | null;
  sma_50: number | null;
  rsi_14: number | null;
  annualized_volatility_30d: number | null;
  max_drawdown_period: number | null;
};

export type StockOverview = {
  ticker: string;
  company_name: string;
  currency: string;
  current_price: number;
  previous_close: number;
  daily_change_pct: number;
  market_cap: number | null;
  trailing_pe: number | null;
  forward_pe: number | null;
  dividend_yield: number | null;
  sector: string | null;
  industry: string | null;
  website: string | null;
  business_summary: string | null;
  fifty_two_week_high: number | null;
  fifty_two_week_low: number | null;
  indicators: StockIndicators;
  price_history: PriceHistoryPoint[];
  as_of: string;
};

export type StockAnalysis = {
  ticker: string;
  company_name: string;
  stance: "bullish" | "cautious" | "watch" | string;
  score: number;
  current_price: number;
  daily_change_pct: number;
  signals: StockIndicators;
  reasons: string[];
  risks: string[];
  disclaimer: string;
  as_of: string;
};

export type SimulationResponse = {
  ticker: string;
  period: string;
  interval: string;
  horizon_steps: number;
  current_price: number;
  probabilities: {
    bullish: number;
    bearish: number;
    sideways: number;
  };
  dominant_scenario: "bullish" | "bearish" | "sideways" | string;
  confidence: number;
  risk_score: number;
  risk_level: "low" | "medium" | "high" | string;
  scenario_paths: {
    bullish: number[];
    bearish: number[];
    sideways: number[];
    high_volatility: number[];
  };
  predicted_prices: number[];
  predicted_candles?: (Omit<PriceHistoryPoint, "volume"> & {
    source?: string;
    scenario?: string;
    confidence?: number;
    risk?: number;
    prediction_type?: string;
  })[];
  predicted_candle_model?: {
    source: string;
    model_version: string | null;
    uses: string[];
  };
  confidence_band: {
    upper: number[];
    lower: number[];
  };
  reasoning: string;
  reasons: string[];
  risks: string[];
  recent_history: PriceHistoryPoint[];
  disclaimer: string;
  as_of: string;
  ml_enabled?: boolean;
  auxiliary_ml_enabled?: boolean;
  model_version?: string | null;
  quality_label?: string;
  coverage_level?: string;
  final_signal?: {
    name: string;
    state: string;
    probability: number;
    confidence: number;
    proof?: {
      all_active_setup_accuracy?: number | null;
      confidence_buckets?: Record<string, {
        accuracy?: number;
        coverage?: number;
        signals?: number;
      }>;
      setup_coverage_test?: number | null;
      baseline_win_rate_test?: number | null;
      proof_rule?: string | null;
    };
    accuracy_claim_policy?: string | null;
    disclaimer?: string;
  };
  ai_signal_bot?: {
    name: string;
    mode: string;
    action: string;
    stance: string;
    quality_label?: string;
    coverage_level?: string;
    dominant_scenario: string;
    expected_return: number | null;
    confidence: number;
    risk_score: number;
    gates?: Record<string, boolean>;
    failed_gates?: string[];
    all_gates_passed?: boolean;
    inputs_used: string[];
    trade_plan?: {
      available: boolean;
      eligible_for_paper_trade?: boolean;
      direction?: string;
      entry_type?: string;
      entry_price?: number;
      stop_loss?: number;
      target_1?: number;
      target_2?: number;
      risk_per_share?: number;
      reward_per_share_target_1?: number;
      risk_reward_target_1?: number;
      spread_estimate_pct?: number;
      slippage_estimate_pct?: number;
      round_trip_cost_pct?: number;
      net_reward_after_cost_pct?: number;
      liquidity_filter_passed?: boolean;
      paper_risk_budget?: number;
      paper_position_shares?: number;
      paper_notional?: number;
      time_stop?: string;
      rules?: string[];
      reason?: string;
    };
    reasons: string[];
    risk_controls: string[];
    disclaimer: string;
  };
  advanced_intelligence?: {
    technical_analysis?: Record<string, unknown>;
    market_structure?: Record<string, unknown>;
    multi_timeframe_alignment?: Record<string, unknown>;
    market_regime?: Record<string, unknown>;
    sector_rotation?: Record<string, unknown>;
    macro?: Record<string, unknown>;
    event_flags?: Record<string, unknown>;
    historical_similarity?: Record<string, unknown>;
    monte_carlo_simulation?: Record<string, unknown>;
    coverage?: {
      trained_or_computed_now?: string[];
      partial?: string[];
      missing?: string[];
    };
  };
  market_intelligence?: {
    event_probabilities: Record<string, {
      probability: number;
      confidence: number;
    }>;
    top_event_signal: string | null;
    expected_return_pct: number | null;
    expected_price_range: {
      upper: number;
      lower: number;
      upper_pct: number;
      lower_pct: number;
    } | null;
    risk_adjusted_return: number | null;
    final_signal?: SimulationResponse["final_signal"];
    pretrained_layers?: {
      enabled_now: string[];
      recommended_next: string[];
    };
    model_version?: string | null;
    disclaimer: string;
  } | null;
};

export type BotScanItem = {
  ticker: string;
  quality_label?: string;
  coverage_level?: string;
  action?: string;
  all_gates_passed?: boolean;
  failed_gates?: string[];
  eligible_for_paper_trade?: boolean;
  confidence?: number;
  risk_score?: number;
  expected_return?: number | null;
  gates?: Record<string, boolean>;
  trade_plan?: SimulationResponse["ai_signal_bot"] extends infer B
    ? B extends { trade_plan?: infer T } ? T : never
    : never;
  error?: string;
};

export type BotScanResponse = {
  period: string;
  interval: string;
  universe?: string;
  available_universes?: string[];
  scanned: number;
  quality_buckets: Record<string, number>;
  passed: BotScanItem[];
  results: BotScanItem[];
  cache?: { hit: boolean; ttl_seconds: number };
  disclaimer: string;
};

export type BotPerformanceResponse = {
  total_logged_predictions: number;
  promoted_signals: number;
  pending_outcomes: number;
  resolved_outcomes: number;
  resolved_win_rate: number | null;
  quality_counts: Record<string, number>;
  action_counts: Record<string, number>;
  quality_performance?: Record<string, { count: number; resolved: number; wins: number; win_rate: number | null }>;
  action_performance?: Record<string, { count: number; resolved: number; wins: number; win_rate: number | null }>;
  confidence_buckets: Record<string, { count: number; resolved: number; wins: number; win_rate: number | null }>;
  v12_walk_forward_proof?: {
    accuracy?: number;
    confidence_buckets?: Record<string, { accuracy?: number; coverage?: number; signals?: number }>;
    setup_coverage_test?: number;
  };
  accuracy_policy?: string;
  note: string;
  disclaimer: string;
};
