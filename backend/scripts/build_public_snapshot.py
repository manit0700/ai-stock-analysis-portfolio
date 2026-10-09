"""Build a data snapshot from public historical price files hosted on GitHub.

Why this exists: the model evaluation in this repository must run on genuine prices and be
repeatable by anyone, including on machines that cannot reach Yahoo Finance. Each source is
pinned to a commit, downloaded, hashed, converted to the snapshot format and written through
``SnapshotStore``; the manifest records every URL and SHA-256.

    python scripts/build_public_snapshot.py            # writes <data dir>/snapshots/public-daily-<hash>/
    python scripts/build_public_snapshot.py --cache /tmp/src   # reuse downloaded files

Sources (all genuine market data; none is redistributed by this repository)
    QuantConnect LEAN sample data   daily bars 1998 to 2021-03-31 with split and dividend factors
    TidyTuesday 2023-02-07          Yahoo Finance daily bars for 14 technology stocks, 2010 to 2022
    DJIA 30 stock time series       daily bars 2006 to 2017 for Dow Jones members (split-adjusted,
                                    not dividend-adjusted)
    datasets/finance-vix            CBOE VIX daily, 1990 onwards

Known limits of this snapshot, repeated in the manifest
    * The stock list was chosen with hindsight: companies that were large when the files were
      made. Results on it are flattered by survivorship bias.
    * SPY ends on 2021-03-31, so no feature row exists after that date.
    * The Dow Jones extract has no dividend adjustment; its returns are price returns.
    * There is no continuous QQQ history, no sector ETFs and no daily Treasury yields, so the
      ``qqq``, ``sector`` and ``macro`` feature groups are absent.

To evaluate on current data instead, run ``scripts/build_yahoo_snapshot.py`` on a machine that
can reach Yahoo Finance and pass the new snapshot id to ``python -m app.forecast``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.marketdata.quality import assess_ohlcv, normalise_ohlcv  # noqa: E402
from app.marketdata.snapshots import SnapshotStore  # noqa: E402

LEAN_COMMIT = "80e7843f645673bcbeaab963049f76f20f6785e1"
TIDY_COMMIT = "78474f0d3496a1f855deeceb63e250f494cbe938"
DJIA_COMMIT = "8da42f14ae64fca79b9b8b51208e98674fc3d4e9"
VIX_COMMIT = "3dbea23c70cc2afbeeec8ac232c9ec72ac26ac21"

LEAN_BASE = f"https://raw.githubusercontent.com/QuantConnect/Lean/{LEAN_COMMIT}/Data/equity/usa"
TIDY_URL = f"https://raw.githubusercontent.com/rfordatascience/tidytuesday/{TIDY_COMMIT}/data/2023/2023-02-07/big_tech_stock_prices.csv"
DJIA_URL = f"https://raw.githubusercontent.com/szrlee/Stock-Time-Series-Analysis/{DJIA_COMMIT}/data/all_stocks_2006-01-01_to_2018-01-01.csv"
VIX_URL = f"https://raw.githubusercontent.com/datasets/finance-vix/{VIX_COMMIT}/data/vix-daily.csv"

#: symbol in the snapshot -> LEAN file stem
#: LEAN files are keyed by ticker, not by company, so three of them are left out or trimmed:
#:   qqq   has no bars from 2004-12-01 to 2011-03-22 (the years the fund traded as QQQQ)
#:   wm    is Washington Mutual until 2008 and Waste Management afterwards
#:   goog  shows the April 2014 share-class split as a 50% one-day fall
#:   eem   holds another security's bars before the fund was listed in April 2003 (trimmed below)
LEAN_SYMBOLS = {"SPY": "spy", "IWM": "iwm", "EEM": "eem", "AAPL": "aapl", "AIG": "aig", "BAC": "bac", "IBM": "ibm"}
LEAN_FIRST_DATE = {"EEM": "2003-04-14"}
TIDY_SYMBOLS = ["ADBE", "AMZN", "CRM", "CSCO", "GOOGL", "INTC", "META", "MSFT", "NFLX", "NVDA", "ORCL", "TSLA"]
DJIA_SYMBOLS = ["AXP", "BA", "CAT", "CVX", "DIS", "GE", "GS", "HD", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "NKE", "PFE", "PG", "TRV", "UNH", "UTX", "VZ", "WMT", "XOM"]
CONTEXT_ONLY = ["SPY", "^VIX"]


def fetch(url: str, cache: Path | None) -> bytes:
    name = hashlib.sha256(url.encode()).hexdigest()[:16] + "-" + url.rsplit("/", 1)[-1]
    if cache is not None and (cache / name).exists():
        return (cache / name).read_bytes()
    import requests

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / name).write_bytes(response.content)
    return response.content


def lean_frame(stem: str, cache: Path | None, sources: list[dict]) -> pd.DataFrame:
    """LEAN daily bars -> split-adjusted OHLC, dividend-and-split-adjusted ``Adj Close``, split-adjusted volume.

    LEAN stores raw traded prices multiplied by 10,000 and a factor file whose row dated ``d``
    applies to every bar after the previous row's date up to and including ``d``.
    """
    bars_url, factor_url = f"{LEAN_BASE}/daily/{stem}.zip", f"{LEAN_BASE}/factor_files/{stem}.csv"
    raw_zip, raw_factors = fetch(bars_url, cache), fetch(factor_url, cache)
    sources.append({"url": bars_url, "sha256": hashlib.sha256(raw_zip).hexdigest()})
    sources.append({"url": factor_url, "sha256": hashlib.sha256(raw_factors).hexdigest()})
    archive = zipfile.ZipFile(io.BytesIO(raw_zip))
    bars = pd.read_csv(io.BytesIO(archive.read(archive.namelist()[0])), header=None, names=["stamp", "Open", "High", "Low", "Close", "Volume"])
    bars.index = pd.to_datetime(bars["stamp"].str.slice(0, 8), format="%Y%m%d")
    bars = bars.drop(columns="stamp").astype(float)
    bars[["Open", "High", "Low", "Close"]] /= 10_000.0

    factors = pd.read_csv(io.BytesIO(raw_factors), header=None).iloc[:, :3]
    factors.columns = ["date", "price_factor", "split_factor"]
    factors["date"] = pd.to_datetime(factors["date"].astype(str), format="%Y%m%d", errors="coerce")
    factors = factors.dropna(subset=["date"]).sort_values("date")
    position = np.clip(factors["date"].searchsorted(bars.index, side="left"), 0, len(factors) - 1)
    price_factor = factors["price_factor"].to_numpy(float)[position]
    split_factor = factors["split_factor"].to_numpy(float)[position]

    out = pd.DataFrame(index=bars.index)
    for column in ("Open", "High", "Low", "Close"):
        out[column] = bars[column] * split_factor
    out["Adj Close"] = bars["Close"] * split_factor * price_factor
    out["Volume"] = (bars["Volume"] / split_factor).round()
    return out


def tidy_frames(cache: Path | None, sources: list[dict]) -> dict[str, pd.DataFrame]:
    raw = fetch(TIDY_URL, cache)
    sources.append({"url": TIDY_URL, "sha256": hashlib.sha256(raw).hexdigest()})
    table = pd.read_csv(io.BytesIO(raw), parse_dates=["date"])
    out = {}
    for symbol in TIDY_SYMBOLS:
        part = table[table["stock_symbol"] == symbol].set_index("date").sort_index()
        out[symbol] = part.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "adj_close": "Adj Close", "volume": "Volume"})[
            ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        ].astype(float)
    return out


def djia_frames(cache: Path | None, sources: list[dict]) -> dict[str, pd.DataFrame]:
    raw = fetch(DJIA_URL, cache)
    sources.append({"url": DJIA_URL, "sha256": hashlib.sha256(raw).hexdigest()})
    table = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"])
    out = {}
    for symbol in DJIA_SYMBOLS:
        part = table[table["Name"] == symbol].set_index("Date").sort_index()
        out[symbol] = part[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return out


def vix_frame(cache: Path | None, sources: list[dict]) -> pd.DataFrame:
    raw = fetch(VIX_URL, cache)
    sources.append({"url": VIX_URL, "sha256": hashlib.sha256(raw).hexdigest()})
    table = pd.read_csv(io.BytesIO(raw), parse_dates=["DATE"]).set_index("DATE").sort_index()
    frame = table.rename(columns={"OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close"})[["Open", "High", "Low", "Close"]].astype(float)
    frame["Volume"] = 0.0  # an index has no volume
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=None, help="folder to keep the downloaded source files in")
    parser.add_argument("--name", default="public-daily")
    args = parser.parse_args()

    sources: list[dict] = []
    frames: dict[str, pd.DataFrame] = {}
    origin: dict[str, str] = {}
    for symbol, stem in LEAN_SYMBOLS.items():
        frame = lean_frame(stem, args.cache, sources)
        frames[symbol] = frame[frame.index >= LEAN_FIRST_DATE[symbol]] if symbol in LEAN_FIRST_DATE else frame
        origin[symbol] = "QuantConnect LEAN sample (split- and dividend-adjusted through factor files)"
    for symbol, frame in tidy_frames(args.cache, sources).items():
        frames[symbol] = frame
        origin[symbol] = "TidyTuesday 2023-02-07 (Yahoo Finance; adjusted close included)"
    for symbol, frame in djia_frames(args.cache, sources).items():
        frames[symbol] = frame
        origin[symbol] = "DJIA 30 stock time series (split-adjusted; no dividend adjustment)"
    frames["^VIX"] = vix_frame(args.cache, sources)
    origin["^VIX"] = "datasets/finance-vix (CBOE)"

    quality: dict[str, dict] = {}
    cleaned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        tidy, report = normalise_ohlcv(frame)
        report = assess_ohlcv(tidy, report=report, check_staleness=False)
        cleaned[symbol] = tidy
        quality[symbol] = {"status": report.status, "issues": [f"{i.code}:{i.count}" for i in report.issues if i.code != "zero_volume"]}

    store = SnapshotStore(get_settings().snapshot_dir)
    manifest = store.create(
        args.name,
        cleaned,
        sources=sources,
        description="Public daily bars for model evaluation: 40 US stocks, IWM, EEM, SPY and the VIX.",
        metadata={
            "origin": origin,
            "quality": quality,
            "context_only": CONTEXT_ONLY,
            "limits": [
                "Stock list chosen with hindsight (survivorship bias).",
                "SPY ends on 2021-03-31; no feature row exists after that date.",
                "DJIA extract is not dividend-adjusted.",
                "No continuous QQQ history, no sector ETFs and no daily Treasury yields: the qqq, sector and macro feature groups are absent.",
            ],
        },
    )
    print(f"snapshot {manifest['id']}: {manifest['n_symbols']} symbols, {manifest['start']} to {manifest['end']}")
    worst = {s: q for s, q in quality.items() if q["issues"]}
    for symbol, q in sorted(worst.items()):
        print(f"  {symbol:6s} {q['status']:9s} {', '.join(q['issues'])}")
    print(f"written to {store.path(manifest['id'])}")


if __name__ == "__main__":
    main()
