"use client";

import { useEffect, useRef, useCallback, useState } from "react";

export type Tick = { price: number; volume: number; timestamp: number };

type Callback = (tick: Tick) => void;

const FINNHUB_KEY = "d1kvra9r01qt8foqkdfgd1kvra9r01qt8foqkdg0";
const WS_URL = `wss://ws.finnhub.io?token=${FINNHUB_KEY}`;

// Singleton WebSocket shared across all hook instances
let ws: WebSocket | null = null;
let wsReady = false;
const subscribers = new Map<string, Set<Callback>>();
const pendingSubs: string[] = [];

function ensureWS() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  ws = new WebSocket(WS_URL);
  wsReady = false;

  ws.onopen = () => {
    wsReady = true;
    // Subscribe all pending + current
    const all = new Set([...pendingSubs, ...subscribers.keys()]);
    all.forEach((sym) => ws!.send(JSON.stringify({ type: "subscribe", symbol: sym })));
    pendingSubs.length = 0;
  };

  ws.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data as string);
      if (msg.type !== "trade" || !Array.isArray(msg.data)) return;
      for (const trade of msg.data) {
        const sym: string = trade.s;
        const tick: Tick = { price: trade.p, volume: trade.v, timestamp: trade.t };
        subscribers.get(sym)?.forEach((cb) => cb(tick));
      }
    } catch { /* ignore malformed */ }
  };

  ws.onerror = () => { wsReady = false; };
  ws.onclose = () => {
    wsReady = false;
    ws = null;
    // Reconnect after 3s
    setTimeout(ensureWS, 3000);
  };
}

function subscribe(symbol: string, cb: Callback) {
  if (!subscribers.has(symbol)) subscribers.set(symbol, new Set());
  subscribers.get(symbol)!.add(cb);

  if (wsReady && ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "subscribe", symbol }));
  } else {
    if (!pendingSubs.includes(symbol)) pendingSubs.push(symbol);
    ensureWS();
  }
}

function unsubscribe(symbol: string, cb: Callback) {
  const set = subscribers.get(symbol);
  if (!set) return;
  set.delete(cb);
  if (set.size === 0) {
    subscribers.delete(symbol);
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "unsubscribe", symbol }));
    }
  }
}

export function useFinnhubWS(symbol: string | null) {
  const [lastTick, setLastTick] = useState<Tick | null>(null);
  const [connected, setConnected] = useState(false);
  const cbRef = useRef<Callback | null>(null);

  const onTick = useCallback((tick: Tick) => {
    setLastTick(tick);
    setConnected(true);
  }, []);

  useEffect(() => {
    if (!symbol) return;

    cbRef.current = onTick;
    subscribe(symbol, onTick);

    // Monitor connection state
    const id = setInterval(() => {
      setConnected(ws?.readyState === WebSocket.OPEN && wsReady);
    }, 2000);

    return () => {
      clearInterval(id);
      if (cbRef.current) unsubscribe(symbol, cbRef.current);
    };
  }, [symbol, onTick]);

  return { lastTick, connected };
}
