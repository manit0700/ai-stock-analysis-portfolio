"use client";

import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  ColorType,
  CrosshairMode,
  LastPriceAnimationMode,
  createChart,
} from "lightweight-charts";
import type { PriceHistoryPoint } from "@/lib/types";
import { useFinnhubWS } from "@/lib/useFinnhubWS";

type Time = import("lightweight-charts").Time;

function computeSMA(data: { time: string; value: number }[], window: number) {
  return data
    .map((_, i) => {
      if (i < window - 1) return null;
      const slice = data.slice(i - window + 1, i + 1);
      return { time: data[i].time, value: slice.reduce((s, d) => s + d.value, 0) / window };
    })
    .filter(Boolean) as { time: string; value: number }[];
}

export default function StockChart({ history, ticker }: { history: PriceHistoryPoint[]; ticker: string }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const candleSeriesRef = useRef<any>(null);
  const candleDataRef = useRef<{ time: number; open: number; high: number; low: number; close: number }[]>([]);

  const [livePrice, setLivePrice] = useState<number | null>(null);
  const [prevClose, setPrevClose] = useState<number | null>(null);
  const [ohlcv, setOhlcv] = useState<{ o: number; h: number; l: number; c: number; v: number } | null>(null);
  const [tickCount, setTickCount] = useState(0);

  // Real Finnhub WebSocket ticks
  const { lastTick, connected } = useFinnhubWS(ticker || null);

  // ── Build + mount chart ─────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || history.length === 0) return;

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
      rightPriceScale: {
        borderColor: "rgba(255,140,0,0.15)",
        textColor: "rgba(255,140,0,0.7)",
        scaleMargins: { top: 0.08, bottom: 0.25 },
      },
      timeScale: {
        borderColor: "rgba(255,140,0,0.15)",
        textColor: "rgba(255,140,0,0.5)",
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(255,140,0,0.4)", labelBackgroundColor: "#1a0a00" },
        horzLine: { color: "rgba(255,140,0,0.4)", labelBackgroundColor: "#1a0a00" },
      },
    });

    // Candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#00cc55",
      downColor: "#ff4444",
      borderUpColor: "#00cc55",
      borderDownColor: "#ff4444",
      wickUpColor: "#00cc5580",
      wickDownColor: "#ff444480",
      borderVisible: false,
      lastPriceAnimation: LastPriceAnimationMode.Continuous,
    });

    const candles = history.map((p) => ({
      time: p.date as Time,
      open: p.open, high: p.high, low: p.low, close: p.close,
    }));
    candleSeries.setData(candles);
    candleSeriesRef.current = candleSeries;
    candleDataRef.current = history.map((p) => ({
      time: Math.floor(new Date(p.date).getTime() / 1000),
      open: p.open, high: p.high, low: p.low, close: p.close,
    }));

    const last = history[history.length - 1];
    const prev = history[history.length - 2]?.close ?? last.close;
    setLivePrice(last.close);
    setPrevClose(prev);
    setOhlcv({ o: last.open, h: last.high, l: last.low, c: last.close, v: last.volume });

    // SMA lines
    const closeData = history.map((p) => ({ time: p.date, value: p.close }));
    const sma20 = computeSMA(closeData, 20);
    const sma50 = computeSMA(closeData, 50);

    const sma20s = chart.addSeries(LineSeries, {
      color: "rgba(255,140,0,0.8)", lineWidth: 1,
      priceLineVisible: false, lastValueVisible: true, title: "SMA20",
    });
    sma20s.setData(sma20 as import("lightweight-charts").LineData[]);

    const sma50s = chart.addSeries(LineSeries, {
      color: "rgba(0,150,255,0.7)", lineWidth: 1,
      priceLineVisible: false, lastValueVisible: true, title: "SMA50",
    });
    sma50s.setData(sma50 as import("lightweight-charts").LineData[]);

    // Volume
    const volSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } });
    volSeries.setData(
      history.map((p) => ({
        time: p.date as Time,
        value: p.volume,
        color: p.close >= p.open ? "rgba(0,204,85,0.3)" : "rgba(255,68,68,0.3)",
      }))
    );

    // Crosshair OHLCV
    chart.subscribeCrosshairMove((param) => {
      if (!param.time) return;
      const d = param.seriesData.get(candleSeries) as { open?: number; high?: number; low?: number; close?: number } | undefined;
      const v = param.seriesData.get(volSeries) as { value?: number } | undefined;
      if (d && "open" in d) {
        setOhlcv({ o: d.open!, h: d.high!, l: d.low!, c: d.close!, v: v?.value ?? 0 });
      }
    });

    chart.timeScale().fitContent();

    return () => {
      chart.remove();
      candleSeriesRef.current = null;
    };
  }, [history]);

  // ── Apply real WS tick ──────────────────────────────────────────
  useEffect(() => {
    if (!lastTick || !candleSeriesRef.current || !candleDataRef.current.length) return;

    const price = lastTick.price;
    const last = { ...candleDataRef.current[candleDataRef.current.length - 1] };
    last.close = price;
    last.high = Math.max(last.high, price);
    last.low = Math.min(last.low, price);
    candleSeriesRef.current.update({ ...last, time: last.time as Time });
    setLivePrice(price);
    setTickCount((n) => n + 1);
  }, [lastTick]);

  // ── Fallback: simulated tick when WS not connected ──────────────
  useEffect(() => {
    if (connected) return; // real ticks are flowing — no simulation needed
    if (!candleSeriesRef.current || history.length === 0) return;

    const id = setInterval(() => {
      if (!candleSeriesRef.current || !candleDataRef.current.length) return;
      const last = { ...candleDataRef.current[candleDataRef.current.length - 1] };
      const noise = last.close * 0.0003 * (Math.random() - 0.495);
      const newClose = Math.max(last.close + noise, last.low * 0.998);
      last.close = newClose;
      last.high = Math.max(last.high, newClose);
      last.low = Math.min(last.low, newClose);
      candleSeriesRef.current.update({ ...last, time: last.time as Time });
      setLivePrice(newClose);
    }, 1000);

    return () => clearInterval(id);
  }, [connected, history]);

  const isPositive = livePrice != null && prevClose != null ? livePrice >= prevClose : true;
  const changePct = livePrice != null && prevClose != null && prevClose !== 0
    ? ((livePrice - prevClose) / prevClose) * 100 : null;

  return (
    <div className="relative h-full flex flex-col overflow-hidden bg-[#070707]">
      {/* ── Top bar ── */}
      <div className="shrink-0 flex items-center gap-4 px-3 py-1.5 border-b border-[#ff8c0015] bg-[#050505] text-[10px] font-mono">
        <span className="text-xs font-bold bb-amber tracking-widest">{ticker}</span>
        {livePrice != null && (
          <span className={`text-lg font-bold ${isPositive ? "bb-green" : "bb-red"}`}>
            ${livePrice.toFixed(2)}
          </span>
        )}
        {changePct != null && (
          <span className={`text-xs font-bold ${isPositive ? "bb-green" : "bb-red"}`}>
            {changePct >= 0 ? "+" : ""}{changePct.toFixed(2)}%
          </span>
        )}
        {ohlcv && (
          <div className="flex gap-3 text-[10px] ml-2">
            <span><span className="bb-dim">O </span><span className="bb-white">{ohlcv.o.toFixed(2)}</span></span>
            <span><span className="bb-dim">H </span><span className="bb-green">{ohlcv.h.toFixed(2)}</span></span>
            <span><span className="bb-dim">L </span><span className="bb-red">{ohlcv.l.toFixed(2)}</span></span>
            <span><span className="bb-dim">C </span><span className="bb-white">{ohlcv.c.toFixed(2)}</span></span>
            <span><span className="bb-dim">V </span><span className="bb-dim">
              {ohlcv.v > 1e6 ? `${(ohlcv.v / 1e6).toFixed(1)}M` : ohlcv.v > 1e3 ? `${(ohlcv.v / 1e3).toFixed(0)}K` : ohlcv.v}
            </span></span>
          </div>
        )}
        <div className="ml-auto flex items-center gap-3 text-[9px]">
          <span><span className="inline-block w-3 h-px bg-[#ff8c00] mr-1 align-middle"></span>SMA20</span>
          <span><span className="inline-block w-3 h-px bg-[#0096ff] mr-1 align-middle"></span>SMA50</span>
          {connected && (
            <span className="bb-dim">TICKS: <span className="bb-white">{tickCount}</span></span>
          )}
          <span className={`flex items-center gap-1 ${connected ? "bb-green" : "bb-amber"}`}>
            <span className={`inline-block w-1.5 h-1.5 rounded-full ${connected ? "bg-[#00cc55] animate-pulse" : "bg-[#ff8c00] animate-pulse"}`}></span>
            {connected ? "WS·LIVE" : "POLLING"}
          </span>
        </div>
      </div>

      {/* ── Chart ── */}
      <div ref={containerRef} className="flex-1 w-full" />
    </div>
  );
}
