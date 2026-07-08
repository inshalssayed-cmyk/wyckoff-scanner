"""
Wyckoff Scalping Scanner — v3
==============================
Scans ALL liquid KuCoin USDT pairs (dynamic watchlist) for Wyckoff
accumulation signals on a 5-min chart, using 48 hours of context to judge
where price sits in the broader range. Scores each setup 0-100 and only
alerts on the strongest ones, with entry/stop/target levels, a per-symbol
cooldown, spread/liquidity filtering, better CVD windowing, and a simple
outcome tracker — all layered on top of the original scanner.

No on-chain / Nansen dependency yet — that's a separate future layer.

SETUP REQUIRED (set these as environment variables, not hardcoded):
  - TELEGRAM_BOT_TOKEN : create a bot via @BotFather on Telegram
  - TELEGRAM_CHAT_ID   : your chat/channel ID (message @userinfobot to get yours)

  Locally (before running):
    export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
    export TELEGRAM_CHAT_ID="your_chat_id"
    python3 wyckoff_scanner.py

  On Render: add these under your service's "Environment" tab instead.
  Start command: python3 -u wyckoff_scanner.py (the -u flag avoids Render
  swallowing print() output due to buffering).

Install deps:
  pip install requests pandas --break-system-packages
"""

import time
import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone

# ============================================================
# CONFIG — reads secrets from environment variables.
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("[WARNING] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
          "alerts will fail to send until these are configured.")

# --- Dynamic watchlist settings ---
MIN_VOLUME_USDT = 2_000_000       # 24h volume filter — keeps the symbol list to liquid pairs only
SYMBOL_REFRESH_SECONDS = 3600      # refresh the tradable symbol list every hour

# --- Timeframe / context settings ---
KLINE_TYPE = "5min"                # scalping-speed candles
CANDLES_PER_HOUR = 12              # 5min candles = 12 per hour
CONTEXT_HOURS = 48
CONTEXT_CANDLES = CONTEXT_HOURS * CANDLES_PER_HOUR   # 576 candles = 48h of 5min data

# --- Range / accumulation detection settings ---
RANGE_WINDOW = 20                  # candles used to define the short-term trading range
VOL_AVG_WINDOW = 20                # rolling window for average volume
TIGHT_RANGE_PCT = 0.012            # 1.2% — how tight a "sideways" range must be
ACCUM_LOOKBACK = 10                # candles checked for the tight-range accumulation window
MACRO_POSITION_MAX_PCT = 55        # only treat as accumulation if in lower 55% of 48h range

# --- CVD windowing settings ---
CVD_WINDOW_MINUTES = ACCUM_LOOKBACK * 5   # match the CVD window to the accumulation lookback (50 min)
CVD_MAX_PAGES = 5                          # safety cap on trade-history pagination per symbol

# --- Spread / liquidity settings ---
MAX_SPREAD_PCT = 0.15              # skip alert if bid/ask spread exceeds this % — too costly to scalp

# --- Trade level settings ---
MIN_RISK_REWARD = 1.5              # minimum reward multiple of risk used when projecting target

# --- Cooldown settings ---
COOLDOWN_SECONDS = 1800            # 30 min — don't re-alert the same symbol before this passes

# --- Scoring / alerting settings ---
MIN_ALERT_SCORE = 60               # only alert on setups scoring 60+ out of 100
SCAN_INTERVAL_SECONDS = 300        # how often to re-scan (5 min, matches candle close)

# --- Outcome tracker settings ---
OUTCOME_LOG_PATH = "outcome_log.jsonl"
OUTCOME_CHECK_MINUTES = [15, 30, 60]   # check price this many minutes after each alert

KUCOIN_BASE = "https://api.kucoin.com"

# --- In-memory state (resets on restart — fine for now, DB comes later) ---
last_alert_time = {}     # symbol -> unix timestamp of last alert sent
pending_outcomes = []    # list of dicts tracking alerts awaiting outcome checks


# ============================================================
# DYNAMIC SYMBOL LIST
# ============================================================
def get_all_kucoin_symbols(min_volume_usdt=MIN_VOLUME_USDT):
    """
    Pulls all active USDT spot pairs from KuCoin, filtered by 24h volume
    so we skip illiquid/dead coins that are unscalpable anyway.
    """
    url = f"{KUCOIN_BASE}/api/v1/market/allTickers"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    tickers = r.json().get("data", {}).get("ticker", [])

    symbols = []
    for t in tickers:
        symbol = t.get("symbol", "")
        if not symbol.endswith("-USDT"):
            continue
        vol_value = float(t.get("volValue", 0) or 0)  # 24h volume in USDT
        if vol_value >= min_volume_usdt:
            symbols.append(symbol)
    return symbols


# ============================================================
# DATA FETCHING
# ============================================================
def get_kucoin_klines(symbol, kline_type=KLINE_TYPE, limit=CONTEXT_CANDLES):
    """Fetch OHLCV candles from KuCoin public API (no key needed)."""
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    params = {"symbol": symbol, "type": kline_type}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return None
    # KuCoin returns newest first: [time, open, close, high, low, volume, turnover]
    df = pd.DataFrame(
        data,
        columns=["time", "open", "close", "high", "low", "volume", "turnover"],
    )
    df = df.astype(
        {"time": "int64", "open": "float", "close": "float",
         "high": "float", "low": "float", "volume": "float", "turnover": "float"}
    )
    df = df.sort_values("time").reset_index(drop=True)  # oldest -> newest
    return df.tail(limit).reset_index(drop=True)


def get_kucoin_trade_history_window(symbol, window_minutes=CVD_WINDOW_MINUTES, max_pages=CVD_MAX_PAGES):
    """
    Paginates KuCoin's public trade-history endpoint backward (using the
    'before' cursor) until the trades collected cover roughly `window_minutes`,
    or `max_pages` is hit — whichever comes first.

    This replaces a fixed "last 100 trades" pull, which was inconsistent:
    on busy pairs like BTC that's only 2-3 minutes of tape, on slower pairs
    it could be an hour+. Now CVD is measured over a consistent time window
    that matches the price-range lookback, on any pair.

    NOTE: KuCoin's public /market/histories endpoint pagination behavior
    should be double-checked against current KuCoin API docs if this stops
    returning expected pages — public endpoints occasionally change cursor
    field names.
    """
    url = f"{KUCOIN_BASE}/api/v1/market/histories"
    all_trades = []
    before_cursor = None
    window_start_ms = (time.time() - window_minutes * 60) * 1000

    for _ in range(max_pages):
        params = {"symbol": symbol}
        if before_cursor:
            params["before"] = before_cursor
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json().get("data", [])
        except Exception:
            break
        if not data:
            break

        all_trades.extend(data)
        oldest_time_ns = int(data[-1].get("time", 0))
        oldest_time_ms = oldest_time_ns / 1_000_000  # KuCoin trade time is in nanoseconds
        if oldest_time_ms <= window_start_ms:
            break
        before_cursor = data[-1].get("sequence") or data[-1].get("time")
        time.sleep(0.1)

    trimmed = [t for t in all_trades if int(t.get("time", 0)) / 1_000_000 >= window_start_ms]
    return trimmed if trimmed else all_trades[:100]


def get_orderbook_snapshot(symbol):
    """Fetch best bid/ask for spread + current price checks."""
    url = f"{KUCOIN_BASE}/api/v1/market/orderbook/level1"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json().get("data") or {}


def get_spread_pct(symbol):
    """Bid/ask spread as a percentage of bid price. None if unavailable."""
    data = get_orderbook_snapshot(symbol)
    bid = float(data.get("bestBid", 0) or 0)
    ask = float(data.get("bestAsk", 0) or 0)
    if bid <= 0 or ask <= 0:
        return None
    return round((ask - bid) / bid * 100, 4)


def get_current_price(symbol):
    """Current traded price, used by the outcome tracker."""
    data = get_orderbook_snapshot(symbol)
    price = float(data.get("price", 0) or 0)
    return price if price > 0 else None


# ============================================================
# WYCKOFF DETECTION LOGIC
# ============================================================
def compute_range(df, window=RANGE_WINDOW):
    df["range_high"] = df["high"].rolling(window=window).max()
    df["range_low"] = df["low"].rolling(window=window).min()
    df["range_width_pct"] = (df["range_high"] - df["range_low"]) / df["range_low"]
    return df


def compute_macro_context(df, window=CONTEXT_CANDLES):
    """
    Uses the full 48h window to define the broader trading range.
    Tells us where current price sits in that range — accumulation
    should be happening in the LOWER portion, not near the top
    (near the top = more likely distribution, not accumulation).
    """
    window_df = df.tail(window)
    macro_high = window_df["high"].max()
    macro_low = window_df["low"].min()

    current_price = df["close"].iloc[-1]
    if macro_high == macro_low:
        position_pct = 50.0
    else:
        position_pct = (current_price - macro_low) / (macro_high - macro_low) * 100

    return {
        "macro_high": macro_high,
        "macro_low": macro_low,
        "position_pct": round(position_pct, 1),  # 0 = at 48h low, 100 = at 48h high
    }


def detect_spring(df):
    """Spring: wick below range low, closes back inside, on relatively low volume."""
    avg_vol = df["volume"].rolling(VOL_AVG_WINDOW).mean()
    last = df.iloc[-1]
    prev_range_low = df["range_low"].iloc[-2]
    if pd.isna(prev_range_low) or pd.isna(avg_vol.iloc[-1]):
        return False
    broke_support = last["low"] < prev_range_low * 0.998
    closed_back_inside = last["close"] > prev_range_low
    low_volume = last["volume"] < avg_vol.iloc[-1]
    return bool(broke_support and closed_back_inside and low_volume)


def detect_sos(df):
    """Sign of Strength: close above range high, on volume expansion, strong candle."""
    avg_vol = df["volume"].rolling(VOL_AVG_WINDOW).mean()
    last = df.iloc[-1]
    prev_range_high = df["range_high"].iloc[-2]
    if pd.isna(prev_range_high) or pd.isna(avg_vol.iloc[-1]):
        return False
    broke_resistance = last["close"] > prev_range_high
    volume_expansion = last["volume"] > avg_vol.iloc[-1] * 1.5
    strong_candle = (last["close"] - last["open"]) / last["open"] > 0.01
    return bool(broke_resistance and volume_expansion and strong_candle)


def detect_lps(df, sos_lookback=5):
    """Last Point of Support: pullback to former resistance after a recent SOS, on light volume."""
    avg_vol = df["volume"].rolling(VOL_AVG_WINDOW).mean()
    last = df.iloc[-1]
    sos_recent = any(detect_sos(df.iloc[: i + 1]) for i in range(len(df) - sos_lookback, len(df) - 1))
    prev_range_high = df["range_high"].iloc[-2]
    if pd.isna(prev_range_high) or pd.isna(avg_vol.iloc[-1]):
        return False
    pulled_back = last["low"] <= prev_range_high * 1.01
    light_volume = last["volume"] < avg_vol.iloc[-1]
    return bool(sos_recent and pulled_back and light_volume)


def detect_sideways_accumulation(df, trades, tight_range_pct=TIGHT_RANGE_PCT, lookback=ACCUM_LOOKBACK):
    """
    Detects a small sideways range where buying (accumulation) is quietly happening.
    - Price has stayed inside a TIGHT range for the last `lookback` candles
    - CVD over that same window is positive (buyers absorbing supply)
    """
    if len(df) < lookback + 2:
        return False, None

    recent = df.tail(lookback)
    range_high = recent["high"].max()
    range_low = recent["low"].min()
    range_pct = (range_high - range_low) / range_low

    is_tight = range_pct <= tight_range_pct

    cvd = compute_cvd(trades) if trades else 0
    is_accumulating = cvd > 0

    return bool(is_tight and is_accumulating), {
        "range_pct": round(range_pct * 100, 3),
        "cvd": round(cvd, 2),
        "range_high": range_high,
        "range_low": range_low,
    }


def compute_cvd(trades):
    """Cumulative Volume Delta from a trade tape window."""
    delta = 0.0
    for t in trades:
        size = float(t.get("size", 0))
        if t.get("side") == "buy":
            delta += size
        else:
            delta -= size
    return delta


# ============================================================
# TRADE LEVELS (entry / stop / target)
# ============================================================
def compute_trade_levels(df, accum_info):
    """
    Builds a simple entry/stop/target plan off the same range data the
    scanner already computed — no separate indicator needed.
    - Entry: current close
    - Stop: just below the recent tight range (or short-term range) low
    - Target: the recent range high, or a 1.5R projection if the range
      itself is too thin to offer a decent reward
    """
    last = df.iloc[-1]
    entry = float(last["close"])

    if accum_info:
        local_low = accum_info["range_low"]
        local_high = accum_info["range_high"]
    else:
        local_low = df["range_low"].iloc[-1]
        local_high = df["range_high"].iloc[-1]

    stop_loss = local_low * 0.995  # small buffer below support
    risk = entry - stop_loss

    if risk <= 0:
        # fallback if data is weird (e.g. entry already below "support")
        risk = entry * 0.005
        stop_loss = entry - risk

    take_profit = max(local_high, entry + risk * MIN_RISK_REWARD)
    reward = take_profit - entry
    rr_ratio = round(reward / risk, 2) if risk > 0 else None

    return {
        "entry": round(entry, 6),
        "stop_loss": round(stop_loss, 6),
        "take_profit": round(take_profit, 6),
        "rr_ratio": rr_ratio,
    }


# ============================================================
# SCORING
# ============================================================
def score_signal(spring, sos, lps, accum, accum_info, macro, cvd):
    """
    Scores a setup 0-100. Higher = stronger scalp candidate.
    Weighted toward tight-range accumulation happening low in the 48h range,
    since that's the core pattern being screened for.
    """
    score = 0.0

    if accum and accum_info:
        score += 30
        tightness_bonus = max(0, 20 - (accum_info["range_pct"] / (TIGHT_RANGE_PCT * 100) * 20))
        score += tightness_bonus
        position_bonus = max(0, 20 - (macro["position_pct"] / MACRO_POSITION_MAX_PCT * 20))
        score += position_bonus

    if spring:
        score += 15
    if sos:
        score += 15
    if lps:
        score += 10

    if cvd is not None and cvd > 0:
        cvd_bonus = min(10, cvd / 1000)  # scale based on typical CVD sizes — tune as needed
        score += cvd_bonus

    return round(min(score, 100), 1)


# ============================================================
# TELEGRAM ALERTING
# ============================================================
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"[Telegram error] {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[Telegram error] {e}")


# ============================================================
# OUTCOME TRACKER
# ============================================================
def log_signal_for_outcome_tracking(symbol, score, levels, ts):
    pending_outcomes.append({
        "symbol": symbol,
        "entry_price": levels["entry"],
        "score": score,
        "alert_time": time.time(),
        "alert_time_str": ts,
        "checked_minutes": [],
    })


def check_pending_outcomes():
    """
    Revisits each logged alert at 15/30/60 min marks, records the price move
    at that point, and appends the result to a local JSONL file so signal
    quality can be reviewed later instead of guessed at.
    """
    if not pending_outcomes:
        return

    now = time.time()
    still_pending = []
    for entry in pending_outcomes:
        elapsed_minutes = (now - entry["alert_time"]) / 60
        due = [m for m in OUTCOME_CHECK_MINUTES
               if elapsed_minutes >= m and m not in entry["checked_minutes"]]

        for m in due:
            try:
                current_price = get_current_price(entry["symbol"])
                if current_price:
                    pct_move = round(
                        (current_price - entry["entry_price"]) / entry["entry_price"] * 100, 3
                    )
                    record = {
                        "symbol": entry["symbol"],
                        "score": entry["score"],
                        "alert_time": entry["alert_time_str"],
                        "entry_price": entry["entry_price"],
                        "check_minutes": m,
                        "price_at_check": current_price,
                        "pct_move": pct_move,
                    }
                    with open(OUTCOME_LOG_PATH, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    print(f"[Outcome] {entry['symbol']} +{m}min: {pct_move}%")
            except Exception as e:
                print(f"[Error checking outcome for {entry['symbol']}] {e}")
            entry["checked_minutes"].append(m)

        if len(entry["checked_minutes"]) < len(OUTCOME_CHECK_MINUTES):
            still_pending.append(entry)

    pending_outcomes[:] = still_pending


# ============================================================
# MAIN SCAN LOGIC
# ============================================================
def scan_symbol(symbol):
    # --- Cooldown check first: cheapest possible early exit ---
    last_time = last_alert_time.get(symbol, 0)
    if time.time() - last_time < COOLDOWN_SECONDS:
        return

    df = get_kucoin_klines(symbol, limit=CONTEXT_CANDLES)
    if df is None or len(df) < RANGE_WINDOW + 2:
        return

    df = compute_range(df)
    macro = compute_macro_context(df)

    spring = detect_spring(df)
    sos = detect_sos(df)
    lps = detect_lps(df)

    trades = get_kucoin_trade_history_window(symbol)
    accum, accum_info = detect_sideways_accumulation(df, trades)

    in_accumulation_zone = macro["position_pct"] <= MACRO_POSITION_MAX_PCT
    accum = accum and in_accumulation_zone

    if not (spring or sos or lps or accum):
        return  # nothing interesting, skip silently

    cvd = compute_cvd(trades) if trades else None
    score = score_signal(spring, sos, lps, accum, accum_info, macro, cvd)

    if score < MIN_ALERT_SCORE:
        return  # setup exists but too weak — skip

    # --- Spread/liquidity check: skip if too costly to actually scalp ---
    spread_pct = get_spread_pct(symbol)
    if spread_pct is not None and spread_pct > MAX_SPREAD_PCT:
        print(f"[Skip] {symbol} score {score} but spread {spread_pct}% > {MAX_SPREAD_PCT}% max")
        return

    levels = compute_trade_levels(df, accum_info)

    confirmations = []
    if cvd is not None and cvd > 0:
        confirmations.append("CVD positive")
    if accum and accum_info:
        confirmations.append(f"tight range {accum_info['range_pct']}%")
        confirmations.append(f"48h position {macro['position_pct']}%")
    if spread_pct is not None:
        confirmations.append(f"spread {spread_pct}%")

    confidence = "HIGH" if score >= 80 else "MEDIUM"

    signal_type = []
    if spring:
        signal_type.append("SPRING")
    if sos:
        signal_type.append("SOS")
    if lps:
        signal_type.append("LPS")
    if accum:
        signal_type.append("ACCUM")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    message = (
        f"*Wyckoff Signal — {symbol}*\n"
        f"Score: {score}/100\n"
        f"Type: {', '.join(signal_type)}\n"
        f"Confidence: {confidence}\n\n"
        f"Entry: {levels['entry']}\n"
        f"Stop Loss: {levels['stop_loss']}\n"
        f"Take Profit: {levels['take_profit']}\n"
        f"Risk:Reward: 1:{levels['rr_ratio']}\n\n"
        f"48h Range: {macro['macro_low']} - {macro['macro_high']} (currently at {macro['position_pct']}%)\n"
        f"Confirmations: {', '.join(confirmations) if confirmations else 'none (price/volume only)'}\n"
        f"Time: {ts}"
    )
    print(message)
    send_telegram_alert(message)

    last_alert_time[symbol] = time.time()
    log_signal_for_outcome_tracking(symbol, score, levels, ts)


def build_startup_message(symbol_count):
    return (
        "🟢 *Wyckoff Scalping Scanner — STARTED*\n\n"
        "*Strategy:* Wyckoff Accumulation Detection (scalping mode)\n"
        "*Signals tracked:* Spring, Sign of Strength (SOS), Last Point of Support (LPS), "
        "Sideways Accumulation (ACCUM)\n"
        "*Confirmation:* CVD (windowed to match lookback) + spread/liquidity filter\n"
        "*Data source:* KuCoin (OHLCV + trade history + order book)\n\n"
        f"*Watchlist:* dynamic — all KuCoin USDT pairs with 24h volume ≥ ${MIN_VOLUME_USDT:,.0f} "
        f"({symbol_count} pairs currently)\n"
        f"*Timeframe:* {KLINE_TYPE} candles, {CONTEXT_HOURS}h of context\n"
        f"*Scan interval:* every {SCAN_INTERVAL_SECONDS // 60} min\n"
        f"*Min alert score:* {MIN_ALERT_SCORE}/100\n"
        f"*Cooldown:* {COOLDOWN_SECONDS // 60} min per symbol\n"
        f"*Max spread:* {MAX_SPREAD_PCT}%\n\n"
        "Each alert now includes entry/stop/target levels, and outcomes get "
        "logged automatically at 15/30/60 min for later review.\n\n"
        "_Note: this flags statistical setups, not guaranteed outcomes — always your own judgment on entries._"
    )


def run_scanner():
    print("Starting Wyckoff scalping scanner... (Ctrl+C to stop)")

    symbols_cache = []
    last_symbol_refresh = 0

    try:
        symbols_cache = get_all_kucoin_symbols()
        last_symbol_refresh = time.time()
        print(f"[Info] Loaded {len(symbols_cache)} symbols to scan")
    except Exception as e:
        print(f"[Error loading initial symbol list] {e}")

    send_telegram_alert(build_startup_message(len(symbols_cache)))

    while True:
        now = time.time()
        if now - last_symbol_refresh > SYMBOL_REFRESH_SECONDS or not symbols_cache:
            try:
                symbols_cache = get_all_kucoin_symbols()
                last_symbol_refresh = now
                print(f"[Info] Refreshed symbol list — {len(symbols_cache)} pairs to scan")
            except Exception as e:
                print(f"[Error refreshing symbol list] {e}")

        for symbol in symbols_cache:
            try:
                scan_symbol(symbol)
            except Exception as e:
                print(f"[Error scanning {symbol}] {e}")
            time.sleep(0.3)  # be polite to the API between symbols

        check_pending_outcomes()
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_scanner()
