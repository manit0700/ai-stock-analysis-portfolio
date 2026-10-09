"use client";

/**
 * The simulated $50 account and the honest forecast panel.
 *
 * Everything on this tab is virtual. The banner is rendered from the backend's own response,
 * forecasts show the probability the backend actually serves (the labelled base rate until a
 * model passes the promotion rule), and the prediction table shows resolved outcomes only as
 * what they are. Nothing here ever shows a number the backend did not send.
 */
import { useCallback, useEffect, useState } from "react";
import {
  fetchForecast,
  fetchForecastStatus,
  fetchPaperAccount,
  fetchPaperOrders,
  fetchPaperPerformance,
  fetchPredictionLedger,
  resetPaperAccount,
  submitPaperOrder,
  type ForecastResponse,
  type ForecastStatus,
  type PaperAccount,
  type PaperOrder,
  type PaperPerformance,
  type PredictionLedgerResponse,
} from "@/lib/api";

function Panel({ title, children, innerClass = "p-3" }: { title: string; children: React.ReactNode; innerClass?: string }) {
  return (
    <div className="t-panel flex flex-col min-h-0">
      <div className="t-header shrink-0">{title}</div>
      <div className={`min-h-0 overflow-auto ${innerClass}`}>{children}</div>
    </div>
  );
}

const pctText = (value: number | null | undefined, decimals = 1) => (value == null ? "—" : `${(value * 100).toFixed(decimals)}%`);

export default function PaperAccountTab() {
  const [account, setAccount] = useState<PaperAccount | null>(null);
  const [orders, setOrders] = useState<PaperOrder[]>([]);
  const [performance, setPerformance] = useState<PaperPerformance | null>(null);
  const [ledger, setLedger] = useState<PredictionLedgerResponse | null>(null);
  const [modelStatus, setModelStatus] = useState<ForecastStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [orderSide, setOrderSide] = useState<"buy" | "sell">("buy");
  const [orderSymbol, setOrderSymbol] = useState("AAPL");
  const [orderNotional, setOrderNotional] = useState("5");
  const [orderBusy, setOrderBusy] = useState(false);
  const [orderMessage, setOrderMessage] = useState<string | null>(null);

  const [forecastSymbol, setForecastSymbol] = useState("AAPL");
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [forecastBusy, setForecastBusy] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const [acct, hist, perf, pred, status] = await Promise.all([
        fetchPaperAccount(),
        fetchPaperOrders(50),
        fetchPaperPerformance(),
        fetchPredictionLedger(),
        fetchForecastStatus(),
      ]);
      setAccount(acct);
      setOrders(hist.orders);
      setPerformance(perf);
      setLedger(pred);
      setModelStatus(status);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The backend is not reachable.");
    }
  }, []);

  useEffect(() => {
    reload();
    const id = setInterval(reload, 30_000);
    return () => clearInterval(id);
  }, [reload]);

  async function placeOrder() {
    setOrderBusy(true);
    setOrderMessage(null);
    try {
      const amount = Number(orderNotional);
      const placed = await submitPaperOrder({ side: orderSide, symbol: orderSymbol.trim().toUpperCase(), notional: amount });
      setOrderMessage(`${placed.status.toUpperCase()}: ${placed.side} ${placed.quantity} ${placed.symbol} @ ${placed.fill_price}`);
      await reload();
    } catch (err) {
      setOrderMessage(`REJECTED: ${err instanceof Error ? err.message : "unknown error"}`);
    } finally {
      setOrderBusy(false);
    }
  }

  async function runForecast() {
    setForecastBusy(true);
    setForecastError(null);
    try {
      setForecast(await fetchForecast(forecastSymbol.trim().toUpperCase()));
      await reload();
    } catch (err) {
      setForecast(null);
      setForecastError(err instanceof Error ? err.message : "forecast unavailable");
    } finally {
      setForecastBusy(false);
    }
  }

  async function onReset() {
    if (!window.confirm("Archive this virtual account and start a fresh one with $50.00? History is kept.")) return;
    await resetPaperAccount();
    await reload();
  }

  const ledgerPerf = ledger?.performance;

  return (
    <div className="h-full min-w-0 grid grid-rows-[auto_1fr] gap-px bg-[#ff8c0010]">
      <div className="t-panel flex items-center gap-4 px-4 py-2 text-[11px]">
        <span className="border border-[#ffcc00] bg-[#1a1400] px-2 py-0.5 font-bold tracking-widest text-[#ffcc00]">
          {account?.banner ?? "SIMULATED TRADING — NO REAL MONEY"}
        </span>
        {account && (
          <>
            <span className="bb-dim">EQUITY: <span className="bb-white font-bold">${account.equity ?? "—"}</span></span>
            <span className="bb-dim">CASH: <span className="bb-white">${account.cash}</span></span>
            <span className="bb-dim">P&L: <span className={Number(account.total_pnl ?? 0) >= 0 ? "bb-green" : "bb-red"}>${account.total_pnl ?? "—"}</span></span>
            <span className="bb-dim">RETURN: <span className={performance && (performance.return_pct ?? 0) >= 0 ? "bb-green" : "bb-red"}>{performance?.return_pct == null ? "—" : `${performance.return_pct.toFixed(2)}%`}</span></span>
            {account.paused && <span className="bb-red font-bold">TRADING PAUSED</span>}
          </>
        )}
        <button onClick={onReset} className="ml-auto border border-[#ff8c0040] px-2 py-0.5 text-[10px] bb-amber hover:border-[#ff8c00]">RESET ($50)</button>
      </div>

      {error && <div className="t-panel p-3 text-[11px] bb-red">[BACKEND] {error}</div>}

      <div className="grid min-h-0 grid-cols-1 gap-px lg:grid-cols-3">
        <div className="flex min-h-0 flex-col gap-px">
          <Panel title="── FORECAST (HONEST) ──────────────────────────────">
            <div className="flex items-center gap-2 pb-2">
              <input value={forecastSymbol} onChange={(e) => setForecastSymbol(e.target.value.toUpperCase())} onKeyDown={(e) => e.key === "Enter" && runForecast()} className="bb-input w-24 text-xs font-bold" />
              <button onClick={runForecast} disabled={forecastBusy} className="border border-[#ff8c0040] px-3 py-1 text-[10px] bb-amber hover:border-[#ff8c00]">
                {forecastBusy ? "..." : "RUN"}
              </button>
              <span className="bb-dim text-[9px]">{modelStatus?.edge_established ? `MODEL: ${modelStatus.champion}` : "NO MODEL PROMOTED — BASE RATE SERVED"}</span>
            </div>
            {forecastError && <p className="text-[10px] bb-red">[ERR] {forecastError}</p>}
            {forecast && (
              <div className="space-y-1 text-[11px] leading-5">
                <div>
                  <span className="bb-dim">P(UP, {forecast.horizon_sessions} SESSIONS): </span>
                  <span className="bb-white text-lg font-bold">{pctText(forecast.probability_up)}</span>
                  <span className={`ml-2 border px-1 text-[9px] ${forecast.status === "model" ? "border-[#00cc55] bb-green" : forecast.status === "baseline" ? "border-[#ff8c00] bb-amber" : "border-[#ff4444] bb-red"}`}>
                    {forecast.status === "model" ? "CALIBRATED MODEL" : forecast.status === "baseline" ? "BASE RATE (NO EDGE ESTABLISHED)" : "UNAVAILABLE"}
                  </span>
                </div>
                {forecast.status === "unavailable" && forecast.reason && <div className="bb-red text-[10px]">{forecast.reason.message}</div>}
                {forecast.window && <div className="bb-dim text-[10px]">WINDOW: {forecast.window.first_session} → {forecast.window.last_session} · AS OF {forecast.as_of}</div>}
                {forecast.range && (
                  <div className="bb-dim text-[10px]">
                    80% RANGE: <span className="bb-red">{pctText(forecast.range.p10)}</span> … <span className="bb-white">{pctText(forecast.range.p50)}</span> … <span className="bb-green">{pctText(forecast.range.p90)}</span>
                    <span className="ml-1">({forecast.range.method})</span>
                  </div>
                )}
                {forecast.expected_return && <div className="bb-dim text-[10px]">EXPECTED RETURN: <span className="bb-white">{pctText(forecast.expected_return.value, 2)}</span> ({forecast.expected_return.method})</div>}
                {forecast.data_source && <div className="bb-dim text-[9px]">DATA: {forecast.data_source.provider} · {forecast.data_source.freshness} · {forecast.data_source.retrieved_at}</div>}
                {forecast.message && <div className="bb-amber text-[9px] leading-4">{forecast.message}</div>}
                <div className="bb-dim text-[9px]">{forecast.disclaimer} A forecast never places an order.</div>
              </div>
            )}
          </Panel>
          <Panel title="── PLACE SIMULATED ORDER ──────────────────────────">
            <div className="flex flex-wrap items-center gap-2 text-[11px]">
              <select value={orderSide} onChange={(e) => setOrderSide(e.target.value as "buy" | "sell")} className="bb-input text-xs">
                <option value="buy">BUY</option>
                <option value="sell">SELL</option>
              </select>
              <input value={orderSymbol} onChange={(e) => setOrderSymbol(e.target.value.toUpperCase())} className="bb-input w-20 text-xs font-bold" />
              <span className="bb-dim">$</span>
              <input value={orderNotional} onChange={(e) => setOrderNotional(e.target.value)} className="bb-input w-16 text-xs" />
              <button onClick={placeOrder} disabled={orderBusy} className="border border-[#ff8c0040] px-3 py-1 text-[10px] bb-amber hover:border-[#ff8c00]">
                {orderBusy ? "..." : "SUBMIT (VIRTUAL)"}
              </button>
            </div>
            {orderMessage && <p className={`mt-2 text-[10px] ${orderMessage.startsWith("REJECTED") ? "bb-red" : "bb-green"}`}>{orderMessage}</p>}
            <p className="mt-2 text-[9px] bb-dim leading-4">
              Market orders on virtual money. Fractional shares; no margin, no short selling, no negative cash. Orders are rejected when the
              quote is stale, the market is closed, or a position or exposure cap would be breached — rejections appear in the history with their reason.
            </p>
          </Panel>
          <Panel title="── ACCOUNT PERFORMANCE ────────────────────────────">
            {performance ? (
              <div className="grid grid-cols-2 gap-y-1 text-[11px]">
                <span className="bb-dim">STARTING CASH</span><span className="bb-white">${performance.starting_cash}</span>
                <span className="bb-dim">EQUITY</span><span className="bb-white">${performance.equity ?? "—"}</span>
                <span className="bb-dim">REALISED P&L</span><span className={Number(performance.realized_pnl) >= 0 ? "bb-green" : "bb-red"}>${performance.realized_pnl}</span>
                <span className="bb-dim">UNREALISED P&L</span><span className="bb-white">${performance.unrealized_pnl ?? "—"}</span>
                <span className="bb-dim">FEES PAID</span><span className="bb-white">${performance.fees_paid}</span>
                <span className="bb-dim">CLOSED SALES</span><span className="bb-white">{performance.closed_sales} ({performance.win_rate_pct == null ? "—" : `${performance.win_rate_pct}% wins`})</span>
              </div>
            ) : <p className="text-[10px] bb-dim">LOADING...</p>}
          </Panel>
        </div>

        <div className="flex min-h-0 flex-col gap-px">
          <Panel title="── HOLDINGS (VIRTUAL) ─────────────────────────────" innerClass="p-0">
            <table className="w-full text-[10px]">
              <thead><tr className="t-header text-left"><th className="p-2">SYM</th><th className="p-2">QTY</th><th className="p-2">AVG COST</th><th className="p-2">PRICE</th><th className="p-2">VALUE</th><th className="p-2">UNRL P&L</th></tr></thead>
              <tbody>
                {account?.positions.map((row) => (
                  <tr key={row.symbol} className="border-b border-[#ff8c0010]">
                    <td className="p-2 bb-amber font-bold">{row.symbol}</td>
                    <td className="p-2 bb-white">{row.quantity}</td>
                    <td className="p-2 bb-white">${row.average_cost}</td>
                    <td className="p-2 bb-white">{row.price ? `$${row.price}` : "—"}</td>
                    <td className="p-2 bb-white">{row.market_value ? `$${row.market_value}` : "—"}</td>
                    <td className={`p-2 ${Number(row.unrealized_pnl ?? 0) >= 0 ? "bb-green" : "bb-red"}`}>{row.unrealized_pnl ? `$${row.unrealized_pnl}` : "—"}</td>
                  </tr>
                ))}
                {account && account.positions.length === 0 && <tr><td colSpan={6} className="p-4 text-center bb-dim">NO POSITIONS — $50.00 IN VIRTUAL CASH</td></tr>}
              </tbody>
            </table>
            {account?.mark_note && <p className="p-2 text-[9px] bb-amber">{account.mark_note}</p>}
          </Panel>
          <Panel title="── TRADE HISTORY ──────────────────────────────────" innerClass="p-0">
            <table className="w-full text-[10px]">
              <thead><tr className="t-header text-left"><th className="p-2">AT</th><th className="p-2">SIDE</th><th className="p-2">SYM</th><th className="p-2">QTY</th><th className="p-2">FILL</th><th className="p-2">STATUS</th></tr></thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.order_id} className="border-b border-[#ff8c0010]" title={order.reason ?? ""}>
                    <td className="p-2 bb-dim">{order.at.slice(0, 19).replace("T", " ")}</td>
                    <td className={`p-2 font-bold ${order.side === "buy" ? "bb-green" : "bb-red"}`}>{order.side.toUpperCase()}</td>
                    <td className="p-2 bb-amber">{order.symbol}</td>
                    <td className="p-2 bb-white">{order.quantity ?? "—"}</td>
                    <td className="p-2 bb-white">{order.fill_price ?? "—"}</td>
                    <td className={`p-2 ${order.status === "filled" ? "bb-green" : "bb-red"}`}>{order.status.toUpperCase()}</td>
                  </tr>
                ))}
                {orders.length === 0 && <tr><td colSpan={6} className="p-4 text-center bb-dim">NO ORDERS YET</td></tr>}
              </tbody>
            </table>
          </Panel>
        </div>

        <div className="flex min-h-0 flex-col gap-px">
          <Panel title="── PREDICTION LEDGER (APPEND-ONLY) ────────────────">
            {ledgerPerf ? (
              <div className="grid grid-cols-2 gap-y-1 text-[11px]">
                <span className="bb-dim">ISSUED / RESOLVED / OPEN</span><span className="bb-white">{ledgerPerf.issued} / {ledgerPerf.resolved} / {ledgerPerf.open}</span>
                <span className="bb-dim">ACCURACY (RESOLVED)</span><span className="bb-white">{ledgerPerf.accuracy == null ? "NO RESOLVED PREDICTIONS YET" : pctText(ledgerPerf.accuracy)}</span>
                <span className="bb-dim">ALWAYS-UP WOULD SCORE</span><span className="bb-white">{pctText(ledgerPerf.majority_class_accuracy ?? null)}</span>
                <span className="bb-dim">BRIER (MODEL vs BASE)</span><span className="bb-white">{ledgerPerf.brier ?? "—"} vs {ledgerPerf.brier_of_base_rate ?? "—"}</span>
                <span className="bb-dim">80% RANGE COVERAGE</span><span className="bb-white">{pctText(ledgerPerf.range_coverage ?? null)}</span>
                <span className="bb-dim">CHAIN</span><span className={ledgerPerf.chain?.intact ? "bb-green" : "bb-red"}>{ledgerPerf.chain?.intact ? `INTACT (${ledgerPerf.chain.events} EVENTS)` : "BROKEN"}</span>
              </div>
            ) : <p className="text-[10px] bb-dim">LOADING...</p>}
            <p className="mt-2 text-[9px] bb-dim leading-4">
              Each forecast is hash-chained when issued and scored later on exactly its own window. Open predictions are never counted as wins.
            </p>
          </Panel>
          <Panel title="── RECENT PREDICTIONS ─────────────────────────────" innerClass="p-0">
            <table className="w-full text-[10px]">
              <thead><tr className="t-header text-left"><th className="p-2">AS OF</th><th className="p-2">SYM</th><th className="p-2">P(UP)</th><th className="p-2">KIND</th><th className="p-2">OUTCOME</th></tr></thead>
              <tbody>
                {ledger?.history.slice(0, 25).map((row) => (
                  <tr key={row.prediction_id} className="border-b border-[#ff8c0010]">
                    <td className="p-2 bb-dim">{row.as_of}</td>
                    <td className="p-2 bb-amber">{row.symbol}</td>
                    <td className="p-2 bb-white">{pctText(row.probability_up)}</td>
                    <td className="p-2 bb-dim">{row.model_kind.toUpperCase()}</td>
                    <td className="p-2">
                      {!row.resolved && <span className="bb-amber">OPEN UNTIL {row.window_last_session}</span>}
                      {row.resolved && row.outcome && (
                        <span className={row.outcome.correct ? "bb-green" : "bb-red"}>
                          {row.outcome.outcome_up ? "UP" : "DOWN"} {pctText(row.outcome.realized_return, 2)} · {row.outcome.correct ? "CORRECT" : "WRONG"}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
                {ledger && ledger.history.length === 0 && <tr><td colSpan={5} className="p-4 text-center bb-dim">NO PREDICTIONS RECORDED YET</td></tr>}
              </tbody>
            </table>
          </Panel>
        </div>
      </div>
    </div>
  );
}
