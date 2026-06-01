"use client";

import { useEffect, useRef, useState } from "react";
import {
  AreaSeries,
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  LineSeries,
  LastPriceAnimationMode,
  createChart,
} from "lightweight-charts";
import type { SimulationResponse } from "@/lib/types";
import { fetchStockQuote } from "@/lib/api";
import { useFinnhubWS } from "@/lib/useFinnhubWS";

type Time = import("lightweight-charts").Time;

const CANDLE_INTERVAL_SECONDS: Record<string, number> = {
  "1m": 60,
  "2m": 120,
  "5m": 300,
  "15m": 900,
  "30m": 1800,
  "60m": 3600,
  "90m": 5400,
  "1h": 3600,
  "4h": 14400,
  "1d": 86400,
  "5d": 432000,
  "1wk": 604800,
  "1w": 604800,
};

function getIntervalSeconds(interval?: string): number {
  return CANDLE_INTERVAL_SECONDS[(interval ?? "1d").toLowerCase()] ?? 86400;
}

function alignToInterval(unixTs: number, intervalSeconds: number): number {
  return Math.floor(unixTs / intervalSeconds) * intervalSeconds;
}

type LinePoint = { time: Time; value: number };
type CandlePoint = { time: Time; open: number; high: number; low: number; close: number };

function toUnixTime(value: string | number | undefined, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Math.floor(new Date(value).getTime() / 1000);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function cleanLineData(points: Array<{ time: Time | number; value: number }>): LinePoint[] {
  const byTime = new Map<number, LinePoint>();
  for (const point of points) {
    const time = Number(point.time);
    const value = Number(point.value);
    if (!Number.isFinite(time) || !Number.isFinite(value)) continue;
    byTime.set(time, { time: time as Time, value });
  }
  return [...byTime.values()].sort((a, b) => Number(a.time) - Number(b.time));
}

function cleanCandleData(points: Array<{ time: Time | number; open: number; high: number; low: number; close: number }>): CandlePoint[] {
  const byTime = new Map<number, CandlePoint>();
  for (const point of points) {
    const time = Number(point.time);
    const open = Number(point.open);
    const high = Number(point.high);
    const low = Number(point.low);
    const close = Number(point.close);
    if (![time, open, high, low, close].every(Number.isFinite)) continue;
    byTime.set(time, {
      time: time as Time,
      open,
      high: Math.max(high, open, close),
      low: Math.min(low, open, close),
      close,
    });
  }
  return [...byTime.values()].sort((a, b) => Number(a.time) - Number(b.time));
}

export default function SimulationChart({ simulation }: { simulation: SimulationResponse }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const candleSeriesRef = useRef<ReturnType<ReturnType<typeof createChart>["addSeries"]> | null>(null);
  const candleDataRef = useRef<{ time: number; open: number; high: number; low: number; close: number }[]>([]);
  const visibleRangeRef = useRef<{ from: number; to: number } | null>(null);

  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [liveChange, setLiveChange] = useState<number | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [tickCount, setTickCount] = useState(0);
  const [layers, setLayers] = useState({
    candles: true,
    scenarios: true,
    band: true,
    cone: true,
    cloud: true,
    stress: false,
    ml: true,
  });

  // Real Finnhub WebSocket
  const { lastTick, connected } = useFinnhubWS(simulation.ticker || null);

  function resetView() {
    if (!chartRef.current || !candleDataRef.current.length) return;
    const intervalSeconds = getIntervalSeconds(simulation.interval);
    const lastHistory = candleDataRef.current[candleDataRef.current.length - 1]?.time;
    const backCandles = Math.min(Math.max(candleDataRef.current.length, 18), 55);
    const firstHistory = lastHistory ? lastHistory - intervalSeconds * backCandles : undefined;
    const horizon = simulation.horizon_steps || simulation.scenario_paths?.bullish?.length || 18;
    if (firstHistory && lastHistory) {
      const nextRange = {
        from: firstHistory as Time,
        to: (lastHistory + intervalSeconds * Math.max(horizon + 2, 12)) as Time,
      };
      visibleRangeRef.current = { from: nextRange.from as number, to: nextRange.to as number };
      chartRef.current.timeScale().setVisibleRange(nextRange);
    } else {
      chartRef.current.timeScale().fitContent();
    }
  }

  function getRange() {
    const live = chartRef.current?.timeScale().getVisibleRange();
    if (live?.from && live?.to) {
      visibleRangeRef.current = { from: live.from as number, to: live.to as number };
    }
    return visibleRangeRef.current;
  }

  function setRange(from: number, to: number) {
    if (!chartRef.current) return;
    visibleRangeRef.current = { from, to };
    chartRef.current.timeScale().setVisibleRange({ from: from as Time, to: to as Time });
  }

  function panChart(direction: -1 | 1) {
    const range = getRange();
    if (!range) return;
    const width = range.to - range.from;
    const shift = width * 0.18 * direction;
    setRange(range.from + shift, range.to + shift);
  }

  function zoomChart(direction: "in" | "out") {
    const range = getRange();
    if (!range) return;
    const center = (range.from + range.to) / 2;
    const width = range.to - range.from;
    const nextWidth = direction === "in" ? width * 0.72 : width * 1.38;
    setRange(center - nextWidth / 2, center + nextWidth / 2);
  }

  function handleTrackpad(event: React.WheelEvent<HTMLDivElement>) {
    if (!chartRef.current) return;
    event.preventDefault();
    const absX = Math.abs(event.deltaX);
    const absY = Math.abs(event.deltaY);

    if (event.ctrlKey || event.metaKey || event.altKey) {
      zoomChart(event.deltaY > 0 ? "out" : "in");
      return;
    }

    const range = getRange();
    if (!range) return;
    const width = range.to - range.from;
    const rawDelta = absX > absY ? event.deltaX : event.deltaY;
    const shift = width * Math.max(Math.min(rawDelta / 900, 0.35), -0.35);
    setRange(range.from + shift, range.to + shift);
  }

  // ── Build + mount chart ─────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || !simulation.recent_history?.length) return;

    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "#070707" },
        textColor: "rgba(255,140,0,0.7)",
        fontSize: 11,
        fontFamily: "'JetBrains Mono','Courier New',monospace",
      },
      grid: {
        vertLines: { color: "rgba(255,140,0,0.06)" },
        horzLines: { color: "rgba(255,140,0,0.06)" },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: false,
      },
      handleScale: {
        axisPressedMouseMove: true,
        axisDoubleClickReset: true,
        mouseWheel: true,
        pinch: true,
      },
      kineticScroll: {
        mouse: true,
        touch: true,
      },
      rightPriceScale: {
        autoScale: false,
        borderColor: "rgba(255,140,0,0.15)",
        textColor: "rgba(255,140,0,0.6)",
        scaleMargins: { top: 0.06, bottom: 0.06 },
      },
      timeScale: {
        borderColor: "rgba(255,140,0,0.15)",
        textColor: "rgba(255,140,0,0.5)",
        timeVisible: true,
        secondsVisible: getIntervalSeconds(simulation.interval) < 3600,
        fixLeftEdge: false,
        fixRightEdge: false,
        lockVisibleTimeRangeOnResize: false,
        rightBarStaysOnScroll: false,
        shiftVisibleRangeOnNewBar: false,
        rightOffset: 8,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(255,140,0,0.35)", labelBackgroundColor: "#1a0a00" },
        horzLine: { color: "rgba(255,140,0,0.35)", labelBackgroundColor: "#1a0a00" },
      },
    });
    chartRef.current = chart;
    chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
      if (range?.from && range?.to) {
        visibleRangeRef.current = { from: range.from as number, to: range.to as number };
      }
    });

    // Historical candles
    const candles = cleanCandleData(simulation.recent_history.map((p) => ({
      time: Math.floor(new Date(p.date).getTime() / 1000) as Time,
      open: p.open, high: p.high, low: p.low, close: p.close,
    })));
    if (!candles.length) return;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#00cc55",
      downColor: "#ff4444",
      borderUpColor: "#00cc55",
      borderDownColor: "#ff4444",
      wickUpColor: "#00cc5560",
      wickDownColor: "#ff444460",
      borderVisible: false,
      lastPriceAnimation: LastPriceAnimationMode.Continuous,
    });
    candleSeries.setData(candles);
    candleSeriesRef.current = candleSeries;
    candleDataRef.current = candles.map((p) => ({
      time: Number(p.time),
      open: p.open, high: p.high, low: p.low, close: p.close,
    }));

    const lastTs = candles[candles.length - 1].time as number;
    const lastClose = simulation.recent_history[simulation.recent_history.length - 1].close;
    const anchorPrice = Number.isFinite(simulation.current_price) && simulation.current_price > 0
      ? simulation.current_price
      : lastClose;
    setLivePrice(anchorPrice);
    const horizon = simulation.horizon_steps || simulation.scenario_paths.bullish.length;
    const intervalSeconds = getIntervalSeconds(simulation.interval);

    const futureTimes: Time[] = [];
    for (let i = 1; i <= horizon; i++) {
      futureTimes.push((lastTs + intervalSeconds * i) as Time);
    }

    function withAnchor(prices: number[]) {
      return cleanLineData([
        { time: lastTs as Time, value: anchorPrice },
        ...prices.slice(0, horizon).map((v, i) => ({ time: futureTimes[i], value: v })),
      ]);
    }

    function chartPoints(points?: { time: string | number; value: number }[]) {
      if (!points?.length) return [];
      let previous = lastTs;
      const projected = points.slice(0, horizon).map((point, i) => {
        const rawTime = toUnixTime(point.time, Number(futureTimes[i]));
        const time = rawTime <= previous ? previous + intervalSeconds : rawTime;
        previous = time;
        return {
          time: time as Time,
          value: point.value,
        };
      });
      return cleanLineData([
        { time: lastTs as Time, value: anchorPrice },
        ...projected,
      ]);
    }

    function buildProjectedCandles() {
      const source = simulation.predicted_candles?.length
        ? simulation.predicted_candles
        : simulation.predicted_prices?.map((close, i, arr) => {
            const open = i === 0 ? anchorPrice : arr[i - 1];
            const bodyHigh = Math.max(open, close);
            const bodyLow = Math.min(open, close);
            const wick = Math.max(Math.abs(close / open - 1) * 0.35, 0.003);
            return {
              date: new Date((futureTimes[i] as number) * 1000).toISOString(),
              open,
              high: bodyHigh * (1 + wick),
              low: bodyLow * (1 - wick),
              close,
            };
          });
      let previous = lastTs;
      return cleanCandleData((source ?? []).slice(0, horizon).map((c, i) => {
        const rawTime = toUnixTime(c.date, Number(futureTimes[i]));
        const time = rawTime <= previous ? previous + intervalSeconds : rawTime;
        previous = time;
        return {
          time: time as Time,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        };
      }));
    }

    // Confidence band
    if (layers.band && simulation.confidence_band?.upper?.length && simulation.confidence_band?.lower?.length) {
      const upper = chart.addSeries(AreaSeries, {
        topColor: "rgba(255,140,0,0.10)",
        bottomColor: "rgba(255,140,0,0.01)",
        lineColor: "rgba(255,140,0,0.20)",
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      upper.setData(withAnchor(simulation.confidence_band.upper));

      const lower = chart.addSeries(AreaSeries, {
        topColor: "rgba(0,0,0,0)",
        bottomColor: "rgba(0,0,0,0)",
        lineColor: "rgba(255,140,0,0.12)",
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      lower.setData(withAnchor(simulation.confidence_band.lower));
    }

    if (layers.cone && simulation.monte_carlo_chart?.volatility_cone_p95?.length && simulation.monte_carlo_chart?.volatility_cone_p05?.length) {
      const upperCone = chart.addSeries(AreaSeries, {
        topColor: "rgba(0,204,255,0.08)",
        bottomColor: "rgba(0,204,255,0.01)",
        lineColor: "rgba(0,204,255,0.22)",
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        title: "VOL CONE",
      });
      upperCone.setData(chartPoints(simulation.monte_carlo_chart.volatility_cone_p95));

      const lowerCone = chart.addSeries(LineSeries, {
        color: "rgba(0,204,255,0.18)",
        lineWidth: 1,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      lowerCone.setData(chartPoints(simulation.monte_carlo_chart.volatility_cone_p05));
    }

    // Scenario paths
    const paths = simulation.scenario_paths;

    if (layers.scenarios && paths?.bullish?.length) {
      const s = chart.addSeries(LineSeries, {
        color: "#00cc55", lineWidth: 2, lineStyle: 0,
        priceLineVisible: false, lastValueVisible: true, title: "BULL",
        crosshairMarkerRadius: 5, crosshairMarkerBackgroundColor: "#00cc55",
      });
      s.setData(withAnchor(paths.bullish));
    }

    if (layers.scenarios && paths?.bearish?.length) {
      const s = chart.addSeries(LineSeries, {
        color: "#ff4444", lineWidth: 2, lineStyle: 0,
        priceLineVisible: false, lastValueVisible: true, title: "BEAR",
        crosshairMarkerRadius: 5, crosshairMarkerBackgroundColor: "#ff4444",
      });
      s.setData(withAnchor(paths.bearish));
    }

    if (layers.scenarios && paths?.sideways?.length) {
      const s = chart.addSeries(LineSeries, {
        color: "#ff8c00", lineWidth: 2, lineStyle: 1,
        priceLineVisible: false, lastValueVisible: true, title: "SIDE",
        crosshairMarkerRadius: 5, crosshairMarkerBackgroundColor: "#ff8c00",
      });
      s.setData(withAnchor(paths.sideways));
    }

    if (layers.cloud && simulation.monte_carlo_chart?.main_predicted_path?.length) {
      const s = chart.addSeries(LineSeries, {
        color: "rgba(0,204,255,0.55)", lineWidth: 2, lineStyle: 3,
        priceLineVisible: false, lastValueVisible: false, title: "PROB CLOUD",
        crosshairMarkerVisible: false,
      });
      s.setData(chartPoints(simulation.monte_carlo_chart.main_predicted_path));
    }

    if (layers.stress && paths?.high_volatility?.length) {
      const s = chart.addSeries(LineSeries, {
        color: "#666666", lineWidth: 1, lineStyle: 2,
        priceLineVisible: false, lastValueVisible: false, title: "HVOL",
        crosshairMarkerVisible: false,
        // Keep the stress path visible but stop it from pushing model candles off-screen.
        autoscaleInfoProvider: () => null,
      });
      s.setData(withAnchor(paths.high_volatility));
    }

    const projectedCandles = buildProjectedCandles();
    if (layers.candles && projectedCandles.length) {
      const projected = chart.addSeries(CandlestickSeries, {
        upColor: "rgba(0,204,85,0.34)",
        downColor: "rgba(255,68,68,0.34)",
        borderUpColor: "rgba(0,204,85,0.82)",
        borderDownColor: "rgba(255,68,68,0.82)",
        wickUpColor: "rgba(0,204,85,0.48)",
        wickDownColor: "rgba(255,68,68,0.48)",
        borderVisible: true,
        priceLineVisible: false,
        lastValueVisible: false,
        title: "AI CANDLES",
      });
      projected.setData(projectedCandles);
    }

    if (layers.ml && simulation.predicted_prices?.length) {
      const s = chart.addSeries(LineSeries, {
        color: "rgba(255,255,255,0.45)", lineWidth: 1, lineStyle: 3,
        priceLineVisible: false, lastValueVisible: true, title: "ML",
        crosshairMarkerVisible: false,
      });
      s.setData(withAnchor(simulation.predicted_prices));
    }

    const visibleBackCandles = Math.min(Math.max(candles.length, 18), 55);
    const visibleFrom = (lastTs - intervalSeconds * visibleBackCandles) as number;
    const visibleTo = (lastTs + intervalSeconds * Math.max(horizon + 2, 12)) as number;
    if (visibleFrom && visibleTo) {
      chart.timeScale().setVisibleRange({ from: visibleFrom as Time, to: visibleTo as Time });
      visibleRangeRef.current = { from: visibleFrom, to: visibleTo };
    } else {
      chart.timeScale().fitContent();
    }
    // Set the initial view once. After this, user panning/zooming stays free
    // until they explicitly click FIT.
    requestAnimationFrame(() => resetView());

    return () => {
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, [simulation, layers]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        panChart(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        panChart(1);
      } else if (event.key === "+" || event.key === "=") {
        event.preventDefault();
        zoomChart("in");
      } else if (event.key === "-" || event.key === "_") {
        event.preventDefault();
        zoomChart("out");
      } else if (event.key.toLowerCase() === "f") {
        event.preventDefault();
        resetView();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [simulation]);

  // ── Apply real WS tick ──────────────────────────────────────────
  useEffect(() => {
    if (!lastTick || !candleSeriesRef.current || !candleDataRef.current.length) return;
    const price = lastTick.price;
    const intervalSeconds = getIntervalSeconds(simulation.interval);
    const tickTime = alignToInterval(Math.floor((lastTick.timestamp || Date.now()) / 1000), intervalSeconds);
    const prev = candleDataRef.current[candleDataRef.current.length - 1];
    const last = tickTime > prev.time
      ? { time: tickTime, open: prev.close, high: price, low: price, close: price }
      : { ...prev, close: price, high: Math.max(prev.high, price), low: Math.min(prev.low, price) };
    if (tickTime > prev.time) candleDataRef.current.push(last);
    else candleDataRef.current[candleDataRef.current.length - 1] = last;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (candleSeriesRef.current as any).update({ ...last, time: last.time as Time });
    setLivePrice(price);
    setIsLive(true);
    setLastUpdated(new Date().toLocaleTimeString("en-US", { hour12: false }));
    setTickCount((n) => n + 1);
  }, [lastTick, simulation.interval]);

  // ── Fallback: REST poll every 3s when WS not connected ──────────
  useEffect(() => {
    if (connected) return;
    if (!simulation.ticker || !simulation.recent_history?.length) return;

    const poll = async () => {
      try {
        const q = await fetchStockQuote(simulation.ticker);
        const price = q.quote.current_price;
        setLivePrice(price);
        setLiveChange(q.quote.change_pct);
        setIsLive(true);
        setLastUpdated(new Date().toLocaleTimeString("en-US", { hour12: false }));
        if (candleSeriesRef.current && candleDataRef.current.length) {
          const intervalSeconds = getIntervalSeconds(simulation.interval);
          const pollTime = alignToInterval(Math.floor(Date.now() / 1000), intervalSeconds);
          const prev = candleDataRef.current[candleDataRef.current.length - 1];
          const last = pollTime > prev.time
            ? { time: pollTime, open: prev.close, high: price, low: price, close: price }
            : { ...prev, close: price, high: Math.max(prev.high, price), low: Math.min(prev.low, price) };
          if (pollTime > prev.time) candleDataRef.current.push(last);
          else candleDataRef.current[candleDataRef.current.length - 1] = last;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (candleSeriesRef.current as any).update({ ...last, time: last.time as Time });
        }
      } catch { setIsLive(false); }
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, [connected, simulation.ticker, simulation.recent_history, simulation.interval]);

  // ── Fallback: simulated micro-tick when WS not connected ────────
  useEffect(() => {
    if (connected) return;
    if (!simulation.recent_history?.length) return;
    const id = setInterval(() => {
      if (!candleSeriesRef.current || !candleDataRef.current.length) return;
      const intervalSeconds = getIntervalSeconds(simulation.interval);
      const simTime = alignToInterval(Math.floor(Date.now() / 1000), intervalSeconds);
      const prev = candleDataRef.current[candleDataRef.current.length - 1];
      const noise = prev.close * 0.0003 * (Math.random() - 0.495);
      const newClose = Math.max(prev.close + noise, prev.low * 0.998);
      const last = simTime > prev.time
        ? { time: simTime, open: prev.close, high: newClose, low: newClose, close: newClose }
        : { ...prev, close: newClose, high: Math.max(prev.high, newClose), low: Math.min(prev.low, newClose) };
      if (simTime > prev.time) candleDataRef.current.push(last);
      else candleDataRef.current[candleDataRef.current.length - 1] = last;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (candleSeriesRef.current as any).update({ ...last, time: last.time as Time });
      setLivePrice(newClose);
    }, 1000);
    return () => clearInterval(id);
  }, [connected, simulation.recent_history, simulation.interval]);

  const { dominant_scenario, confidence, risk_level, probabilities } = simulation;
  const isPositive = (liveChange ?? 0) >= 0;

  return (
    <div className="relative h-full flex flex-col bg-[#070707]">
      {/* ── Top bar ── */}
      <div className="shrink-0 flex items-center gap-3 px-3 py-1.5 border-b border-[#ff8c0015] bg-[#050505] text-[10px] font-mono flex-wrap">
        <span className="bb-amber font-bold tracking-widest">{simulation.ticker}</span>

        {livePrice != null && (
          <span className={`text-base font-bold ${isPositive ? "bb-green" : "bb-red"}`}>
            ${livePrice.toFixed(2)}
          </span>
        )}
        {liveChange != null && (
          <span className={`text-[11px] font-bold ${isPositive ? "bb-green" : "bb-red"}`}>
            {liveChange >= 0 ? "+" : ""}{liveChange.toFixed(2)}%
          </span>
        )}

        <span className="bb-dim">|</span>
        <span className={`font-bold ${dominant_scenario === "bullish" ? "bb-green" : dominant_scenario === "bearish" ? "bb-red" : "bb-amber"}`}>
          ▶ {dominant_scenario?.toUpperCase()}
        </span>
        <span className="bb-dim">CONF:<span className="bb-white font-bold ml-1">{confidence?.toFixed(1)}%</span></span>
        <span className="bb-dim">RISK:<span className={`font-bold ml-1 ${risk_level === "high" ? "bb-red" : risk_level === "low" ? "bb-green" : "bb-amber"}`}>{risk_level?.toUpperCase()}</span></span>

        {/* Legend */}
        <div className="ml-auto flex items-center gap-3 text-[9px]">
          <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5 bg-[#00cc55]"></span>BULL {((probabilities?.bullish ?? 0) * 100).toFixed(0)}%</span>
          <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5 bg-[#ff4444]"></span>BEAR {((probabilities?.bearish ?? 0) * 100).toFixed(0)}%</span>
          <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5 bg-[#ff8c00]"></span>SIDE {((probabilities?.sideways ?? 0) * 100).toFixed(0)}%</span>
          <span className="flex items-center gap-1"><span className="inline-block w-4 h-0.5 bg-[#ffffff50]"></span>ML</span>
          <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 border border-[#00cc55aa] bg-[#00cc5545]"></span>MODEL CANDLE</span>
          {connected && <span className="bb-dim">TICKS:<span className="bb-white ml-1">{tickCount}</span></span>}
        <span className={`flex items-center gap-1 ${connected ? "bb-green" : isLive ? "bb-amber" : "bb-dim"}`}>
            <span className={`inline-block w-1.5 h-1.5 rounded-full ${connected ? "bg-[#00cc55] animate-pulse" : isLive ? "bg-[#ff8c00] animate-pulse" : "bg-[#444]"}`}></span>
            {connected ? `WS·LIVE ${lastUpdated}` : isLive ? `POLL ${lastUpdated}` : "STATIC"}
          </span>
        </div>
      </div>

      {/* ── Chart ── */}
      <div
        ref={containerRef}
        onWheel={handleTrackpad}
        className="flex-1 w-full min-h-0 touch-none cursor-grab active:cursor-grabbing"
      />
      <div className="absolute left-3 top-12 z-20 flex max-w-[calc(100%-24px)] flex-wrap gap-2">
        <button
          type="button"
          onClick={resetView}
          className="border border-[#ff8c0050] bg-[#050505e6] px-3 py-1 text-[10px] font-bold tracking-widest text-[#ff8c00] hover:border-[#ff8c00] hover:bg-[#140a00]"
          title="Reset chart view"
        >
          FIT
        </button>
        <button type="button" onClick={() => panChart(-1)} className="border border-[#ffffff18] bg-[#050505e6] px-2 py-1 text-[10px] font-bold text-[#c8c8c8] hover:border-[#ff8c00]" title="Pan left">←</button>
        <button type="button" onClick={() => panChart(1)} className="border border-[#ffffff18] bg-[#050505e6] px-2 py-1 text-[10px] font-bold text-[#c8c8c8] hover:border-[#ff8c00]" title="Pan right">→</button>
        <button type="button" onClick={() => zoomChart("in")} className="border border-[#ffffff18] bg-[#050505e6] px-2 py-1 text-[10px] font-bold text-[#c8c8c8] hover:border-[#ff8c00]" title="Zoom in">+</button>
        <button type="button" onClick={() => zoomChart("out")} className="border border-[#ffffff18] bg-[#050505e6] px-2 py-1 text-[10px] font-bold text-[#c8c8c8] hover:border-[#ff8c00]" title="Zoom out">−</button>
        {([
          ["candles", "CANDLES"],
          ["scenarios", "LINES"],
          ["band", "BAND"],
          ["cone", "CONE"],
          ["cloud", "CLOUD"],
          ["stress", "STRESS"],
          ["ml", "ML"],
        ] as const).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setLayers((current) => ({ ...current, [key]: !current[key] }))}
            className={`border px-2 py-1 text-[10px] font-bold tracking-wider hover:border-[#ff8c00] ${
              layers[key]
                ? "border-[#ff8c0050] bg-[#140a00e6] text-[#ff8c00]"
                : "border-[#ffffff12] bg-[#050505e6] text-[#777]"
            }`}
            title={`Toggle ${label.toLowerCase()}`}
          >
            {label}
          </button>
        ))}
        <span className="border border-[#ffffff12] bg-[#050505cc] px-2 py-1 text-[10px] text-[#9a9a9a]">
          TRACKPAD: 2-FINGER PAN · OPTION/CTRL + SCROLL ZOOM · KEYS: ← → + − F
        </span>
      </div>
    </div>
  );
}
