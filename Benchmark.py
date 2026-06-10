"""
benchmark.py — Bot vs baselines, from inception.

Baselines:
  1. SPY buy-and-hold (100% invested)
  2. Exposure-matched: 40% SPY + 60% cash
  3. Equal-weight buy-and-hold of the 20-stock watchlist

Run manually anytime: python benchmark.py
Outputs a summary table and saves benchmark.png
"""

import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import contextlib
import io
import os
from datetime import datetime
from dotenv import load_dotenv

# ── Config ─────────────────────────────────────────────────────────────────
load_dotenv()
if not os.getenv("ALPACA_API_KEY"):
    load_dotenv(r"C:\Users\ryanc\OneDrive\Desktop\algo-trading-10k\.env")

API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

START_DATE     = "2026-06-15"   # <-- SET THIS to the bot's first trading day
EXPOSURE_MATCH = 0.40           # matches the bot's max exposure cap
TRADING_DAYS   = 252

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "V",    "JPM",   "ORCL", "COST",
    "ADBE", "CRM",  "AMD",   "NFLX", "PYPL",
    "MA",   "UNH",  "HD",    "BAC",  "QCOM",
]

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# ── Bot equity curve from Alpaca ───────────────────────────────────────────
def get_bot_equity():
    hist = api.get_portfolio_history(
        date_start = START_DATE,
        timeframe  = "1D"
    )
    ts     = pd.to_datetime(hist.timestamp, unit="s").tz_localize(None).normalize()
    equity = pd.Series(hist.equity, index=ts, name="Bot", dtype=float)
    equity = equity[equity > 0].dropna()
    return equity

# ── Price data via yfinance ────────────────────────────────────────────────
def get_prices(tickers, start):
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(tickers, start=start, progress=False, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers if isinstance(tickers, str) else tickers[0])
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df.dropna(how="all")

# ── Metrics ────────────────────────────────────────────────────────────────
def metrics(equity):
    rets = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    if len(rets) > 1 and rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * np.sqrt(TRADING_DAYS)
    else:
        sharpe = float("nan")
    running_max = equity.cummax()
    max_dd      = ((equity - running_max) / running_max).min()
    return total_return, sharpe, max_dd

# ── Build baselines ────────────────────────────────────────────────────────
print(f"Pulling bot equity since {START_DATE}...")
bot = get_bot_equity()
if bot.empty or len(bot) < 2:
    raise SystemExit("Not enough bot history yet — needs at least 2 trading days.")

start_value = bot.iloc[0]
print(f"Bot inception value: ${start_value:,.2f} on {bot.index[0].date()}")

print("Pulling SPY...")
spy_px = get_prices("SPY", START_DATE)["SPY"]
spy    = (spy_px / spy_px.iloc[0]) * start_value
spy.name = "SPY 100%"

# Exposure-matched: 40% in SPY, 60% flat cash
spy_matched = start_value * (1 - EXPOSURE_MATCH) + (spy / start_value) * start_value * EXPOSURE_MATCH
spy_matched.name = f"SPY {EXPOSURE_MATCH:.0%} + cash"

print("Pulling watchlist (20 stocks)...")
wl_px = get_prices(WATCHLIST, START_DATE)
wl_norm = wl_px / wl_px.iloc[0]                      # each stock starts at 1.0
eq_weight = wl_norm.mean(axis=1) * start_value       # equal-weight, no rebalance drift modeled
eq_weight.name = "Equal-weight watchlist"

# ── Align all series on common dates ───────────────────────────────────────
combined = pd.concat([bot, spy, spy_matched, eq_weight], axis=1).dropna()
if len(combined) < 2:
    raise SystemExit("Date alignment produced <2 rows — check START_DATE.")

# ── Report ─────────────────────────────────────────────────────────────────
print(f"\n{'Strategy':<28}{'Return':>10}{'Sharpe':>10}{'Max DD':>10}{'Value':>14}")
print("─" * 72)
for col in combined.columns:
    tr, sh, dd = metrics(combined[col])
    print(f"{col:<28}{tr:>9.2%}{sh:>10.2f}{dd:>9.2%}{combined[col].iloc[-1]:>13,.2f}")

bot_tr, _, _     = metrics(combined["Bot"])
matched_tr, _, _ = metrics(combined[f"SPY {EXPOSURE_MATCH:.0%} + cash"])
spy_tr, _, _     = metrics(combined["SPY 100%"])

print("\n── Diagnosis ──────────────────────────────")
print(f"  vs exposure-matched : {bot_tr - matched_tr:+.2%}  (stock-picking skill)")
print(f"  vs SPY 100%         : {bot_tr - spy_tr:+.2%}  (includes cash drag)")

# ── Chart ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
for col in combined.columns:
    ax.plot(combined.index, combined[col], label=col,
            linewidth=2.2 if col == "Bot" else 1.2)
ax.set_title(f"Bot vs Baselines since {combined.index[0].date()}")
ax.set_ylabel("Portfolio value ($)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("benchmark.png", dpi=150)
print(f"\n  ✓ Chart saved to benchmark.png")