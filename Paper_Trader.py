import alpaca_trade_api as tradeapi
import yfinance as yf
import pandas as pd
import ta
import xgboost as xgb
import joblib
import time
import warnings
import contextlib
import io
import requests
import base64
from datetime import datetime
from dotenv import load_dotenv
import os

# ── Load environment variables ─────────────────────────────────────────────
load_dotenv()
if not os.getenv("ALPACA_API_KEY"):
    # NOTE: this is the NEW $10k project folder — update if your path differs
    load_dotenv(r"C:\Users\ryanc\OneDrive\Desktop\algo-trading-10k\.env")

# ── Credentials ────────────────────────────────────────────────────────────
API_KEY    = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL   = "https://paper-api.alpaca.markets"

# ── Strategy constants ─────────────────────────────────────────────────────
MAX_HOLD_DAYS    = 5     # trading days since last buy fill before time-exit
MAX_POSITION_PCT = 0.08  # per-ticker cap as % of portfolio value
MIN_ADD_DOLLARS  = 10    # skip adds smaller than this

# ── Signal log config ──────────────────────────────────────────────────────
SIGNAL_LOG_PATH = "logs/signal_log.csv"
LOG_HEADER      = "date,time,ticker,signal,confidence,price,action,dollars"
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")        # fine-grained PAT, contents:write
GITHUB_REPO     = os.getenv("GITHUB_REPO")         # e.g. "ryanappel42/algo-trading-10k"
run_log         = []

def log_row(ticker, signal, confidence, price, action, dollars=0.0):
    now = datetime.now()
    run_log.append({
        "date"      : now.strftime("%Y-%m-%d"),
        "time"      : now.strftime("%H:%M:%S"),
        "ticker"    : ticker,
        "signal"    : signal,
        "confidence": f"{confidence:.4f}",
        "price"     : f"{price:.2f}",
        "action"    : action,
        "dollars"   : f"{dollars:.2f}",
    })

def save_log():
    """Append this run's rows to logs/signal_log.csv locally and on GitHub."""
    if not run_log:
        print("\n  No signal rows to log")
        return
    new_lines = "\n".join(
        ",".join(r[k] for k in ["date","time","ticker","signal",
                                "confidence","price","action","dollars"])
        for r in run_log
    )

    # Local copy (useful for manual runs; ephemeral on Railway)
    os.makedirs("logs", exist_ok=True)
    is_new = not os.path.exists(SIGNAL_LOG_PATH)
    with open(SIGNAL_LOG_PATH, "a") as f:
        if is_new:
            f.write(LOG_HEADER + "\n")
        f.write(new_lines + "\n")
    print(f"\n  ✓ Logged {len(run_log)} signal rows locally")

    # Persistent copy via GitHub Contents API
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("  ⚠ GITHUB_TOKEN / GITHUB_REPO not set — log NOT persisted to GitHub")
        return
    try:
        url     = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{SIGNAL_LOG_PATH}"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept"       : "application/vnd.github+json",
        }
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            existing = base64.b64decode(r.json()["content"]).decode()
            sha      = r.json()["sha"]
        elif r.status_code == 404:
            existing = LOG_HEADER + "\n"
            sha      = None
        else:
            print(f"  ✗ GitHub GET failed ({r.status_code}): {r.text[:200]}")
            return
        content = existing.rstrip("\n") + "\n" + new_lines + "\n"
        payload = {
            "message": f"signal log {datetime.now().strftime('%Y-%m-%d')}",
            "content": base64.b64encode(content.encode()).decode(),
        }
        if sha:
            payload["sha"] = sha
        r = requests.put(url, headers=headers, json=payload, timeout=30)
        if r.status_code in (200, 201):
            print(f"  ✓ Signal log pushed to GitHub ({GITHUB_REPO}/{SIGNAL_LOG_PATH})")
        else:
            print(f"  ✗ GitHub PUT failed ({r.status_code}): {r.text[:200]}")
    except Exception as e:
        print(f"  ✗ GitHub log push failed: {e}")

print("API_KEY found:", API_KEY is not None)
print("SECRET_KEY found:", SECRET_KEY is not None)

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version="v2")

# ── Verify connection ──────────────────────────────────────────────────────
account         = api.get_account()
portfolio_value = float(account.portfolio_value)
print(f"Account status : {account.status}")
print(f"Portfolio value: ${portfolio_value:,.2f}")
print(f"Buying power   : ${float(account.buying_power):,.2f}")
print(f"Cash           : ${float(account.cash):,.2f}")

# ── Feature engineering ────────────────────────────────────────────────────
def get_features(ticker):
    for attempt in range(3):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = yf.download(
                        ticker,
                        period="1y",
                        progress=False,
                        auto_adjust=True
                    )
            if df.empty:
                raise ValueError("Empty dataframe")
            df.columns = df.columns.get_level_values(0)
            df["rsi"]         = ta.momentum.RSIIndicator(df["Close"]).rsi()
            df["macd"]        = ta.trend.MACD(df["Close"]).macd()
            df["macd_sig"]    = ta.trend.MACD(df["Close"]).macd_signal()
            df["bb_high"]     = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
            df["bb_low"]      = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()
            df["vol_ma"]      = df["Volume"].rolling(20).mean()
            df["returns_1d"]  = df["Close"].pct_change(1)
            df["returns_5d"]  = df["Close"].pct_change(5)
            df["returns_20d"] = df["Close"].pct_change(20)
            df["volatility"]  = df["returns_1d"].rolling(20).std()
            df["ma_20"]       = df["Close"].rolling(20).mean()
            df["ma_50"]       = df["Close"].rolling(50).mean()
            df["ma_cross"]    = (df["ma_20"] > df["ma_50"]).astype(int)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            print(f"  ⚠ Attempt {attempt+1}/3 failed for {ticker}: {e}")
            time.sleep(5)
    return pd.DataFrame()

# ── Load model ─────────────────────────────────────────────────────────────
features = ["rsi","macd","macd_sig","bb_high","bb_low",
            "vol_ma","returns_1d","returns_5d","returns_20d",
            "volatility","ma_20","ma_50","ma_cross"]

model = joblib.load("models/aapl_xgb_model.joblib")
print("\nModel loaded successfully")

# ── Get signal ─────────────────────────────────────────────────────────────
def get_signal(ticker):
    try:
        df = get_features(ticker)
        if df.empty or len(df) < 50:
            print(f"  ⚠ Not enough data for {ticker}, skipping")
            return None
        latest     = df[features].iloc[-1:]
        pred       = model.predict(latest)[0]
        prob       = model.predict_proba(latest)[0]
        price      = df["Close"].iloc[-1]
        confidence = prob[1] if pred == 1 else prob[0]
        return {
            "ticker"    : ticker,
            "signal"    : "BUY" if pred == 1 else "SELL",
            "confidence": confidence,
            "price"     : price,
            "time"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"  ✗ Error getting signal for {ticker}: {e}")
        return None

# ── Position sizing (returns DOLLARS — fractional shares via notional) ─────
def get_position_size(confidence, portfolio_value):
    if confidence >= 0.80:
        pct = 0.05
    elif confidence >= 0.70:
        pct = 0.03
    elif confidence >= 0.65:
        pct = 0.02
    else:
        pct = 0.01
    return round(portfolio_value * pct, 2)

# ── Portfolio exposure ─────────────────────────────────────────────────────
def get_portfolio_exposure():
    positions      = api.list_positions()
    total_invested = sum(float(p.market_value) for p in positions)
    return total_invested

# ── Holding age (stateless — derived from Alpaca order history) ───────────
def get_last_buy_fill(ticker):
    """Timestamp of the most recent filled BUY order for this ticker."""
    try:
        orders = api.list_orders(
            status    = "closed",
            limit     = 500,
            direction = "desc",
            symbols   = [ticker]
        )
        for o in orders:
            if o.side == "buy" and o.filled_at is not None:
                return pd.Timestamp(o.filled_at)
    except Exception as e:
        print(f"  ⚠ Could not fetch order history for {ticker}: {e}")
    return None

def trading_days_held(ticker):
    """Completed trading days since the last buy fill. None if unknown."""
    entry = get_last_buy_fill(ticker)
    if entry is None:
        return None
    try:
        cal = api.get_calendar(
            start = entry.date().isoformat(),
            end   = datetime.now().date().isoformat()
        )
        return max(0, len(cal) - 1)  # exclude the entry day itself
    except Exception as e:
        print(f"  ⚠ Calendar lookup failed for {ticker}: {e}")
        return None

# ── Execute sell ───────────────────────────────────────────────────────────
def execute_sell(ticker, confidence=0.0, reason="MODEL_SELL"):
    try:
        position      = api.get_position(ticker)
        qty           = float(position.qty)
        unrealized_pl = float(position.unrealized_pl)
        current_price = float(position.current_price)
        value         = float(position.market_value)
        order = api.close_position(ticker)
        print(f"  ✓ SELL {qty:g} shares of {ticker} | P&L: ${unrealized_pl:+.2f}")
        log_row(ticker, "SELL", confidence, current_price, reason, value)
        return order
    except Exception as e:
        print(f"  ✗ Sell failed for {ticker}: {e}")
        log_row(ticker, "SELL", confidence, 0.0, "SELL_FAILED")
        return None

# ── Execute buy ────────────────────────────────────────────────────────────
def execute_buy(ticker, confidence, price, portfolio_value):
    try:
        # ── Confidence gate — hard 60% minimum ────────────────────────────
        if confidence < 0.60:
            print(f"  — Confidence too low ({confidence:.1%}) — skipping {ticker}")
            return None

        total_invested = get_portfolio_exposure()
        exposure_pct   = total_invested / portfolio_value
        max_exposure   = 0.40

        if exposure_pct >= max_exposure:
            print(f"  — Portfolio at {exposure_pct:.1%} exposure (max 40%) — skipping {ticker}")
            log_row(ticker, "BUY", confidence, price, "SKIPPED_EXPOSURE_CAP")
            return None

        try:
            position       = api.get_position(ticker)
            has_position   = True
            unrealized_pl  = float(position.unrealized_pl)
            position_value = float(position.market_value)
        except:
            has_position   = False
            unrealized_pl  = 0
            position_value = 0

        dollars = get_position_size(confidence, portfolio_value)

        # ── Per-position cap — trim adds to remaining room ────────────────
        trimmed      = False
        position_cap = portfolio_value * MAX_POSITION_PCT
        room         = position_cap - position_value
        if has_position:
            if room < MIN_ADD_DOLLARS:
                print(f"  — {ticker} at ${position_value:,.0f} vs ${position_cap:,.0f} cap ({MAX_POSITION_PCT:.0%}) — skipping add")
                log_row(ticker, "BUY", confidence, price, "SKIPPED_POSITION_CAP")
                return None
            if dollars > room:
                print(f"  — Trimming {ticker} add from ${dollars:,.0f} to ${room:,.2f} (position cap)")
                dollars = round(room, 2)
                trimmed = True

        if not has_position:
            order = api.submit_order(
                symbol        = ticker,
                notional      = dollars,
                side          = "buy",
                type          = "market",
                time_in_force = "day"
            )
            print(f"  ✓ BUY ~${dollars:,.0f} of {ticker} | Confidence: {confidence:.1%} | Exposure: {exposure_pct:.1%}")
            log_row(ticker, "BUY", confidence, price, "BUY_NEW", dollars)
            return order

        elif unrealized_pl > 0 or confidence >= 0.65:
            order = api.submit_order(
                symbol        = ticker,
                notional      = dollars,
                side          = "buy",
                type          = "market",
                time_in_force = "day"
            )
            print(f"  ✓ Adding ~${dollars:,.0f} to {ticker} | Confidence: {confidence:.1%} | Exposure: {exposure_pct:.1%}")
            log_row(ticker, "BUY", confidence, price,
                    "BUY_TRIMMED" if trimmed else "BUY_ADD", dollars)
            return order

        else:
            print(f"  — Position at loss, confidence below 65% — holding {ticker}")
            log_row(ticker, "BUY", confidence, price, "HELD_AT_LOSS")
            return None

    except Exception as e:
        print(f"  ✗ Buy failed for {ticker}: {e}")
        log_row(ticker, "BUY", confidence, price, "BUY_FAILED")
        return None

# ── Portfolio summary ──────────────────────────────────────────────────────
def print_portfolio():
    print("\n── Current Positions ─────────────────────")
    positions = api.list_positions()
    if not positions:
        print("  No open positions")
    total_pnl      = 0
    total_invested = 0
    for p in positions:
        pnl            = float(p.unrealized_pl)
        qty            = float(p.qty)
        value          = float(p.market_value)
        total_pnl     += pnl
        total_invested += value
        print(f"  {p.symbol:<6} {qty:g} shares | Value: ${value:,.2f} | P&L: ${pnl:+.2f}")
    exposure = (total_invested / portfolio_value) * 100
    print(f"\n  Total invested    : ${total_invested:,.2f} ({exposure:.1f}% of portfolio)")
    print(f"  Total P&L         : ${total_pnl:+.2f}")
    print(f"  Cash remaining    : ${float(account.cash):,.2f}")
    print(f"  Max exposure limit: ${portfolio_value * 0.40:,.2f} (40%)")
    print("\n── Recent Orders ──────────────────────────")
    orders = api.list_orders(status="all", limit=10)
    for o in orders:
        qty_display = o.qty if o.qty is not None else f"${float(o.notional):,.0f}"
        print(f"  {o.symbol} {o.side.upper()} {qty_display} — {o.status} @ {o.created_at}")

# ── Watchlist ──────────────────────────────────────────────────────────────
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "V",    "JPM",   "ORCL", "COST",
    "ADBE", "CRM",  "AMD",   "NFLX", "PYPL",
    "MA",   "UNH",  "HD",    "BAC",  "QCOM",
]

# ── STEP 1 — Collect all signals ───────────────────────────────────────────
print("\n── Collecting Signals for All 20 Stocks ───")
sell_signals = []
buy_signals  = []

for ticker in WATCHLIST:
    print(f"\nAnalyzing {ticker}...")
    signal_data = get_signal(ticker)
    if signal_data is None:
        print(f"  — Skipping {ticker}")
        continue
    print(f"  Signal    : {signal_data['signal']}")
    print(f"  Confidence: {signal_data['confidence']:.1%}")
    print(f"  Price     : ${signal_data['price']:.2f}")

    if signal_data["signal"] == "SELL":
        sell_signals.append(signal_data)
    elif signal_data["confidence"] >= 0.60:
        # Only add to buy list if above 60% threshold
        buy_signals.append(signal_data)
    else:
        print(f"  — Below 60% confidence threshold — not queued for buying")
        log_row(ticker, "BUY", signal_data["confidence"], signal_data["price"], "SKIPPED_LOW_CONF")

# ── STEP 2 — Execute all sells first ──────────────────────────────────────
print(f"\n── Pass 1: Executing {len(sell_signals)} Sell Signal(s) ───")
if not sell_signals:
    print("  No sell signals today")
else:
    for s in sell_signals:
        ticker = s["ticker"]
        print(f"\n  Processing SELL for {ticker}...")
        try:
            api.get_position(ticker)
            execute_sell(ticker, confidence=s["confidence"], reason="MODEL_SELL")
        except:
            print(f"  — No position in {ticker}, nothing to sell")
            log_row(ticker, "SELL", s["confidence"], s["price"], "SELL_NO_POSITION")

# Small pause to let sell orders settle
if sell_signals:
    print("\n  Waiting 10 seconds for sell orders to settle...")
    time.sleep(10)

# ── STEP 2.5 — Time exits: close positions held ≥ MAX_HOLD_DAYS ───────────
print(f"\n── Pass 1b: Time Exits (held ≥ {MAX_HOLD_DAYS} trading days, no fresh BUY) ───")
buy_tickers_today = {b["ticker"] for b in buy_signals}
time_exits = 0

for p in api.list_positions():
    sym = p.symbol
    if sym in buy_tickers_today:
        print(f"  — {sym}: fresh BUY signal today — clock renewed, holding")
        continue
    days = trading_days_held(sym)
    if days is None:
        print(f"  ⚠ {sym}: holding age unknown — holding (will not exit blind)")
        continue
    if days >= MAX_HOLD_DAYS:
        print(f"\n  {sym} held {days} trading days — time exit")
        execute_sell(sym, reason="TIME_EXIT")
        time_exits += 1
    else:
        print(f"  — {sym}: held {days}/{MAX_HOLD_DAYS} trading days — holding")

if time_exits == 0:
    print("  No time exits today")
else:
    print(f"\n  Waiting 10 seconds for time-exit orders to settle...")
    time.sleep(10)

# ── STEP 3 — Rank buys by confidence, execute highest first ───────────────
buy_signals_sorted = sorted(buy_signals, key=lambda x: x["confidence"], reverse=True)

print(f"\n── Pass 2: Executing {len(buy_signals_sorted)} Buy Signal(s) — Ranked by Confidence ───")
if not buy_signals_sorted:
    print("  No buy signals above 60% confidence today")
else:
    print("\n  Buy signal rankings:")
    for i, b in enumerate(buy_signals_sorted, 1):
        print(f"  #{i} {b['ticker']:<6} Confidence: {b['confidence']:.1%} | Price: ${b['price']:.2f}")

    print("\n  Executing buys...")
    for b in buy_signals_sorted:
        total_invested = get_portfolio_exposure()
        exposure_pct   = total_invested / portfolio_value
        if exposure_pct >= 0.40:
            remaining = buy_signals_sorted[buy_signals_sorted.index(b):]
            print(f"\n  — Portfolio at {exposure_pct:.1%} exposure — cap reached, stopping buys")
            print(f"  — Skipped: {[x['ticker'] for x in remaining]}")
            for x in remaining:
                log_row(x["ticker"], "BUY", x["confidence"], x["price"], "SKIPPED_EXPOSURE_CAP")
            break
        print(f"\n  Processing BUY for {b['ticker']}...")
        execute_buy(b["ticker"], b["confidence"], b["price"], portfolio_value)

save_log()
print_portfolio()
print("\nDone. Run this script daily to execute your strategy.")