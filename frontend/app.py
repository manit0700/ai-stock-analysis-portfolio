from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]


def api_get(path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def build_price_chart(price_history: list[dict]) -> go.Figure:
    frame = pd.DataFrame(price_history)
    frame["date"] = pd.to_datetime(frame["date"])

    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=frame["date"],
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Price",
        )
    )
    figure.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return figure


def build_simulation_chart(simulation: dict) -> go.Figure:
    history = pd.DataFrame(simulation["recent_history"])
    history["date"] = pd.to_datetime(history["date"])
    last_date = history["date"].iloc[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=simulation["horizon_steps"],
        freq="D",
    )

    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=history["date"],
            open=history["open"],
            high=history["high"],
            low=history["low"],
            close=history["close"],
            name="Historical Price",
        )
    )

    scenario_styles = {
        "bullish": "#22c55e",
        "bearish": "#ef4444",
        "sideways": "#38bdf8",
        "high_volatility": "#f59e0b",
    }
    for scenario, path in simulation["scenario_paths"].items():
        figure.add_trace(
            go.Scatter(
                x=future_dates,
                y=path,
                mode="lines",
                name=scenario.replace("_", " ").title(),
                line=dict(color=scenario_styles.get(scenario, "#94a3b8"), width=3 if scenario == simulation["dominant_scenario"] else 1.5),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=simulation["confidence_band"]["upper"] + simulation["confidence_band"]["lower"][::-1],
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band",
        )
    )
    figure.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=520,
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h"),
    )
    return figure


def render_stock_tab() -> None:
    st.subheader("Single Stock Research")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    period = st.selectbox("Period", options=["1mo", "3mo", "6mo", "1y"], index=2)

    if st.button("Analyze Stock", use_container_width=True):
        with st.spinner("Loading stock research..."):
            overview = api_get(f"/api/stocks/{ticker}/overview", params={"period": period})
            analysis = api_get(f"/api/stocks/{ticker}/analysis", params={"period": period})
            news = api_get(f"/api/stocks/{ticker}/news", params={"limit": 5})

        metric_cols = st.columns(4)
        metric_cols[0].metric("Price", f"${overview['current_price']}")
        metric_cols[1].metric("Daily Move", f"{overview['daily_change_pct']}%")
        metric_cols[2].metric("Research Score", analysis["score"])
        metric_cols[3].metric("Stance", analysis["stance"].title())

        st.plotly_chart(build_price_chart(overview["price_history"]), use_container_width=True)

        info_cols = st.columns(2)
        with info_cols[0]:
            st.markdown("**Signals**")
            st.json(analysis["signals"])
        with info_cols[1]:
            st.markdown("**Company Snapshot**")
            st.write(f"Sector: {overview.get('sector') or 'N/A'}")
            st.write(f"Industry: {overview.get('industry') or 'N/A'}")
            st.write(f"Trailing P/E: {overview.get('trailing_pe') or 'N/A'}")
            st.write(f"Market Cap: {overview.get('market_cap') or 'N/A'}")

        st.markdown("**Why it looks this way**")
        for reason in analysis["reasons"]:
            st.write(f"- {reason}")

        st.markdown("**Main risks**")
        for risk in analysis["risks"]:
            st.write(f"- {risk}")

        st.markdown("**Recent news**")
        items = news.get("items", [])
        if not items:
            st.write("No recent news was returned.")
        for item in items:
            title = item["title"] or "Untitled article"
            link = item.get("link")
            if link:
                st.markdown(f"- [{title}]({link})")
            else:
                st.write(f"- {title}")


def render_simulation_tab() -> None:
    st.subheader("AI Future Chart Simulation")
    ticker = st.text_input("Simulation Ticker", value="NVDA").upper().strip()
    period = st.selectbox("Simulation History", options=["3mo", "6mo", "1y", "2y"], index=2)
    horizon_steps = st.slider("Future Steps", min_value=4, max_value=30, value=12)

    if st.button("Run Simulation", use_container_width=True):
        with st.spinner("Building probability paths..."):
            simulation = api_post(
                "/api/predict",
                {
                    "ticker": ticker,
                    "period": period,
                    "interval": "1d",
                    "horizon_steps": horizon_steps,
                },
            )

        metric_cols = st.columns(5)
        metric_cols[0].metric("Current Price", f"${simulation['current_price']}")
        metric_cols[1].metric("Dominant Scenario", simulation["dominant_scenario"].title())
        metric_cols[2].metric("Confidence", f"{simulation['confidence']}%")
        metric_cols[3].metric("Risk Level", simulation["risk_level"].title())
        metric_cols[4].metric("Risk Score", simulation["risk_score"])

        st.plotly_chart(build_simulation_chart(simulation), use_container_width=True)

        probability_frame = pd.DataFrame(
            [{"scenario": key.title(), "probability_pct": round(value * 100, 1)} for key, value in simulation["probabilities"].items()]
        )
        st.markdown("**Scenario Probabilities**")
        st.dataframe(probability_frame, use_container_width=True, hide_index=True)

        st.markdown("**AI Market Reasoning**")
        st.write(simulation["reasoning"])

        info_cols = st.columns(2)
        with info_cols[0]:
            st.markdown("**Supporting Signals**")
            for reason in simulation["reasons"]:
                st.write(f"- {reason}")
        with info_cols[1]:
            st.markdown("**Risk Notes**")
            for risk in simulation["risks"]:
                st.write(f"- {risk}")

        st.caption(simulation["disclaimer"])


def render_watchlist_tab() -> None:
    st.subheader("Watchlist Ranking")
    tickers_input = st.text_input("Tickers", value=", ".join(DEFAULT_WATCHLIST))
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    period = st.selectbox("Watchlist Period", options=["1mo", "3mo", "6mo", "1y"], index=2)

    if st.button("Build Watchlist", use_container_width=True):
        with st.spinner("Ranking watchlist..."):
            result = api_post("/api/stocks/watchlist", {"tickers": tickers, "period": period, "interval": "1d"})

        frame = pd.DataFrame(result["watchlist"])
        st.dataframe(frame, use_container_width=True, hide_index=True)


def render_compare_tab() -> None:
    st.subheader("Compare Stocks")
    tickers_input = st.text_input("Compare Tickers", value="AAPL, MSFT, NVDA")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    if st.button("Compare", use_container_width=True):
        with st.spinner("Comparing stocks..."):
            result = api_post("/api/stocks/compare", {"tickers": tickers, "period": "6mo", "interval": "1d"})

        frame = pd.DataFrame(result["leaders"])
        st.dataframe(
            frame[["ticker", "company_name", "stance", "score", "daily_change_pct"]],
            use_container_width=True,
            hide_index=True,
        )


def parse_holdings(text: str) -> list[dict]:
    holdings = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        ticker, shares = [part.strip() for part in clean.split(",")]
        holdings.append({"ticker": ticker.upper(), "shares": float(shares)})
    return holdings


def render_portfolio_tab() -> None:
    st.subheader("Portfolio Risk Analyzer")
    holdings_text = st.text_area(
        "Enter one holding per line as TICKER,SHARES",
        value="AAPL,10\nMSFT,5\nNVDA,2",
        height=140,
    )
    period = st.selectbox("Portfolio Period", options=["3mo", "6mo", "1y"], index=1)

    if st.button("Analyze Portfolio", use_container_width=True):
        holdings = parse_holdings(holdings_text)
        with st.spinner("Analyzing portfolio..."):
            result = api_post("/api/portfolio/analyze", {"holdings": holdings, "period": period})

        metric_cols = st.columns(5)
        metric_cols[0].metric("Value", f"${result['total_market_value']}")
        metric_cols[1].metric("Return", f"{result['annualized_return_pct']}%")
        metric_cols[2].metric("Volatility", f"{result['annualized_volatility_pct']}%")
        metric_cols[3].metric("Sharpe", result["sharpe_ratio"])
        metric_cols[4].metric("Diversification", result["diversification_score"])

        st.markdown("**Holdings**")
        st.dataframe(pd.DataFrame(result["holdings"]), use_container_width=True, hide_index=True)

        st.markdown("**Warnings**")
        for warning in result["warnings"]:
            st.write(f"- {warning}")

        st.markdown("**Correlation Matrix**")
        st.dataframe(pd.DataFrame(result["correlation_matrix"]), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Stock Research Dashboard", layout="wide")
    st.title("Stock Research Dashboard")
    st.caption("Built for investor support. Informational only, not personalized investment advice.")
    st.write(f"Backend API: `{API_BASE_URL}`")

    tabs = st.tabs(["Stock", "Simulation", "Watchlist", "Compare", "Portfolio"])
    with tabs[0]:
        render_stock_tab()
    with tabs[1]:
        render_simulation_tab()
    with tabs[2]:
        render_watchlist_tab()
    with tabs[3]:
        render_compare_tab()
    with tabs[4]:
        render_portfolio_tab()


if __name__ == "__main__":
    main()
