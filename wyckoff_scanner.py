"""
Wyckoff Scalping Scanner — v3 (market-entry, 1:1.5 R:R)
======================================================
Scans ALL liquid KuCoin USDT pairs (dynamic watchlist) for Wyckoff
accumulation signals on a 5-min chart, using 48 hours of context to judge
where price sits in the broader range. Scores each setup 0-100 and alerts
on the strongest ones.

CHANGED IN THIS VERSION (based on live outcome data from 115 trades):
  - ENTRY is now at MARKET price (instant fill). The previous wall-anchored
    limit entry left ~50% of signals unfilled (expired) — market entry gets
    us into every qualifying trade so the data is complete.
  - Risk:Reward is 1:1.5. Default stop is a fixed 1% and target 1.5%.
    (A structural, price-action stop is available via STOP_MODE below.)
  - Single target, full exit — no more TP1/TP2 partial-close machinery,
    since a 1.5% move has no room to split.
  - Every trade now records MFE / MAE (max favourable / adverse excursion):
    how far price actually ran up and down before the trade resolved. This
    is the measurement that tells you whether 1.5% is well-calibrated.

Order-book wall detection is retained as an optional FILTER / confirmation
(see REQUIRE_WALL) and as a reference for the structural stop.

ENV VARS (set on Render, not hardcoded):
  - TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  - DATABASE_URL (optional — enables the dashboard; Telegram-only without it)

Start command on Render:  python3 -u wyckoff_scanner.py
Install deps:  pip install requests pandas psycopg2-binary --break-system-packages
"""

import time
import os
import json
import threading
import requests
import pandas as pd
from datetime import datetime, timezone

# Optional Postgres layer — scanner runs fine without it (Telegram + JSONL only).
try:
    import db
    DB_ENABLED = db.is_available()
except Exception:
    db = None
    DB_ENABLED = False

# ============================================================
# CONFIG
# ============================================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("[WARNING] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — "
          "alerts will fail to send until these are configured.")

SCANNER_NAME = "Inshal Crypto Scanner"
SCANNER_VERSION = "v3.0"

# --- Dynamic watchlist ---
MIN_VOLUME_USDT = 2_000_000
SYMBOL_REFRESH_SECONDS = 3600

# --- Timeframe / context ---
KLINE_TYPE = "5min"
CANDLES_PER_HOUR = 12
CONTEXT_HOURS = 48
CONTEXT_CANDLES = CONTEXT_HOURS * CANDLES_PER_HOUR

# --- Range / accumulation detection ---
RANGE_WINDOW = 20
VOL_AVG_WINDOW = 20
TIGHT_RANGE_PCT = 0.012
ACCUM_LOOKBACK = 10
MACRO_POSITION_MAX_PCT = 55

# --- CVD windowing ---
CVD_WINDOW_MINUTES = ACCUM_LOOKBACK * 5
CVD_MAX_PAGES = 5

# --- Spread / liquidity ---
MAX_SPREAD_PCT = 0.15

# --- ENTRY / STOP / TARGET (the core of this version) ---
ENTRY_MODE = "market"          # enter at current price — always fills
RR_RATIO = 1.5                 # target distance = RR_RATIO x stop distance (1:1.5)

# Stop mode: "fixed" honours exactly what you described (1% stop -> 1.5% target).
#            "structural" instead places the stop below the recent swing low /
#            wall / range support (ATR-buffered) and clamps it to a sane band,
#            so the stop respects price action; target stays RR_RATIO x risk.
STOP_MODE = "fixed"            # "fixed" or "structural"
FIXED_SL_PCT = 1.0             # used when STOP_MODE == "fixed"

# structural-mode parameters (ignored in fixed mode):
ATR_PERIOD = 14
ATR_STOP_MULTIPLE = 0.5
SWING_LOOKBACK = 12
MIN_SL_PCT = 0.6               # tightest structural stop
MAX_SL_PCT = 1.5              # widest structural stop

# --- Order-book wall (now a filter/confirmation, not the entry anchor) ---
REQUIRE_WALL = False           # if True, only alert when a real buy wall sits below price
WALL_SCAN_DEPTH_PCT = 2.0
WALL_MIN_MULTIPLE = 3.0

# --- Cooldown / scoring / cadence ---
COOLDOWN_SECONDS = 1800
MIN_ALERT_SCORE = 60
SCAN_INTERVAL_SECONDS = 300

# --- Outcome tracker ---
OUTCOME_LOG_PATH = "outcome_log.jsonl"
OUTCOME_TIMEOUT_HOURS = 4

KUCOIN_BASE = "https://api.kucoin.com"

# --- In-memory state ---
last_alert_time = {}
pending_outcomes = []
scanner_stats = {
    "start_time": time.time(),
    "last_cycle_time": None,
    "symbols_count": 0,
    "cycles_completed": 0,
    "alerts_sent": 0,
}


# ============================================================
# DYNAMIC SYMBOL LIST
# ============================================================
def get_all_kucoin_symbols(min_volume_usdt=MIN_VOLUME_USDT):
    url = f"{KUCOIN_BASE}/api/v1/market/allTickers"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    tickers = r.json().get("data", {}).get("ticker", [])
    symbols = []
    for t in tickers:
        symbol = t.get("symbol", "")
        if not symbol.endswith("-USDT"):
            continue
        vol_value = float(t.get("volValue", 0) or 0)
        if vol_value >= min_volume_usdt:
            symbols.append(symbol)
    return symbols


# ============================================================
# DATA FETCHING
# ============================================================
def get_kucoin_klines(symbol, kline_type=KLINE_TYPE, limit=CONTEXT_CANDLES):
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    params = {"symbol": symbol, "type": kline_type}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return None
    df = pd.DataFrame(
        data,
        columns=["time", "open", "close", "high", "low", "volume", "turnover"],
    )
    df = df.astype(
        {"time": "int64", "open": "float", "close": "float",
         "high": "float", "low": "float", "volume": "float", "turnover": "float"}
    )
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(limit).reset_index(drop=True)


def get_kucoin_trade_history_window(symbol, window_minutes=CVD_WINDOW_MINUTES, max_pages=CVD_MAX_PAGES):
    """Paginate the trade tape back ~window_minutes for a consistent CVD window."""
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
        oldest_time_ms = int(data[-1].get("time", 0)) / 1_000_000
        if oldest_time_ms <= window_start_ms:
            break
        before_cursor = data[-1].get("sequence") or data[-1].get("time")
        time.sleep(0.1)

    trimmed = [t for t in all_trades if int(t.get("time", 0)) / 1_000_000 >= window_start_ms]
    return trimmed if trimmed else all_trades[:100]


def get_orderbook_snapshot(symbol):
    url = f"{KUCOIN_BASE}/api/v1/market/orderbook/level1"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json().get("data") or {}


def get_orderbook_depth(symbol):
    url = f"{KUCOIN_BASE}/api/v1/market/orderbook/level2_100"
    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return r.json().get("data") or {}


def find_bid_wall(symbol, current_price, scan_depth_pct=WALL_SCAN_DEPTH_PCT,
                  min_multiple=WALL_MIN_MULTIPLE):
    """Find a genuine buy wall below price: a level several x the average bid size."""
    try:
        book = get_orderbook_depth(symbol)
    except Exception:
        return None
    bids = book.get("bids") or []
    if not bids:
        return None

    floor_price = current_price * (1 - scan_depth_pct / 100)
    zone = []
    for level in bids:
        try:
            p = float(level[0]); s = float(level[1])
        except (ValueError, IndexError, TypeError):
            continue
        if floor_price <= p < current_price:
            zone.append((p, p * s))
    if len(zone) < 5:
        return None

    avg_value = sum(v for _, v in zone) / len(zone)
    best_price, best_value = max(zone, key=lambda x: x[1])
    if avg_value <= 0 or best_value < avg_value * min_multiple:
        return None
    return {
        "price": best_price,
        "value_usdt": round(best_value, 2),
        "strength": round(best_value / avg_value, 2),
    }


def get_spread_pct(symbol):
    data = get_orderbook_snapshot(symbol)
    bid = float(data.get("bestBid", 0) or 0)
    ask = float(data.get("bestAsk", 0) or 0)
    if bid <= 0 or ask <= 0:
        return None
    return round((ask - bid) / bid * 100, 4)


def get_current_price(symbol):
    data = get_orderbook_snapshot(symbol)
    price = float(data.get("price", 0) or 0)
    return price if price > 0 else None


# ============================================================
# WYCKOFF DETECTION
# ============================================================
def compute_range(df, window=RANGE_WINDOW):
    df["range_high"] = df["high"].rolling(window=window).max()
    df["range_low"] = df["low"].rolling(window=window).min()
    df["range_width_pct"] = (df["range_high"] - df["range_low"]) / df["range_low"]
    return df


def compute_macro_context(df, window=CONTEXT_CANDLES):
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
        "position_pct": round(position_pct, 1),
    }


def detect_spring(df):
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
    avg_vol = df["volume"].rolling(VOL_AVG_WINDOW).mean()
    last = df.iloc[-1]
    sos_recent = any(detect_sos(df.iloc[: i + 1]) for i in range(len(df) - sos_lookback, len(df) - 1))
    prev_range_high = df["range_high"].iloc[-2]
    if pd.isna(prev_range_high) or pd.isna(avg_vol.iloc[-1]):
        return False
    pulled_back = last["low"] <= prev_range_high * 1.01
    light_volume = last["volume"] < avg_vol.iloc[-1]
    return bool(sos_recent and pulled_back and light_volume)


def compute_cvd(trades):
    delta = 0.0
    for t in trades:
        size = float(t.get("size", 0))
        if t.get("side") == "buy":
            delta += size
        else:
            delta -= size
    return delta


def detect_sideways_accumulation(df, trades, tight_range_pct=TIGHT_RANGE_PCT, lookback=ACCUM_LOOKBACK):
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


# ============================================================
# TRADE LEVELS  (market entry, 1:1.5 R:R)
# ============================================================
def compute_atr(df, period=ATR_PERIOD):
    high = df["high"]; low = df["low"]; prev_close = df["close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]


def compute_trade_levels(df, accum_info, symbol, wall):
    """
    Market entry with a 1:1.5 risk:reward.

    STOP_MODE == "fixed":       stop = entry - FIXED_SL_PCT%  (your 1% / 1.5% plan)
    STOP_MODE == "structural":  stop below the lowest of (swing low, wall, range
                                support) minus an ATR buffer, clamped to
                                MIN_SL_PCT..MAX_SL_PCT so it respects price action
                                without being wickable by noise or oversized.
    Target is always RR_RATIO x the stop distance.
    """
    entry = float(df["close"].iloc[-1])  # market entry

    if STOP_MODE == "structural":
        atr = compute_atr(df)
        if pd.isna(atr) or atr <= 0:
            atr = entry * 0.003
        swing_low = float(df["low"].tail(SWING_LOOKBACK).min())
        candidates = [swing_low]
        if wall:
            candidates.append(wall["price"])
        if accum_info:
            candidates.append(accum_info["range_low"])
        structural_low = min(candidates)
        stop_loss = structural_low - atr * ATR_STOP_MULTIPLE
        min_stop = entry * (1 - MAX_SL_PCT / 100)
        max_stop = entry * (1 - MIN_SL_PCT / 100)
        stop_loss = max(min_stop, min(stop_loss, max_stop))
        anchor = "structure"
    else:  # fixed
        stop_loss = entry * (1 - FIXED_SL_PCT / 100)
        anchor = "fixed"

    risk = entry - stop_loss
    if risk <= 0:
        risk = entry * (FIXED_SL_PCT / 100)
        stop_loss = entry - risk

    take_profit = entry + risk * RR_RATIO

    sl_pct = round((entry - stop_loss) / entry * 100, 2)
    tp_pct = round((take_profit - entry) / entry * 100, 2)

    return {
        "entry": round(entry, 6),
        "last_market_price": round(entry, 6),
        "anchor": anchor,
        "wall": wall,                 # confirmation info only, not the entry anchor
        "is_pullback_order": False,   # market entry: always fills now
        "stop_loss": round(stop_loss, 6),
        "sl_pct": sl_pct,
        "take_profit": round(take_profit, 6),
        "tp_pct": tp_pct,
        # kept for db/dashboard compatibility (single target duplicated):
        "take_profit_1": round(take_profit, 6),
        "take_profit_2": round(take_profit, 6),
        "rr": RR_RATIO,
    }


# ============================================================
# SCORING
# ============================================================
def score_signal(spring, sos, lps, accum, accum_info, macro, cvd):
    score = 0.0
    if accum and accum_info:
        score += 30
        score += max(0, 20 - (accum_info["range_pct"] / (TIGHT_RANGE_PCT * 100) * 20))
        score += max(0, 20 - (macro["position_pct"] / MACRO_POSITION_MAX_PCT * 20))
    if spring:
        score += 15
    if sos:
        score += 15
    if lps:
        score += 10
    if cvd is not None and cvd > 0:
        score += min(10, cvd / 1000)
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
# TELEGRAM COMMANDS
# ============================================================
def _stop_desc():
    if STOP_MODE == "fixed":
        return f"{FIXED_SL_PCT}% fixed"
    return f"price action ({MIN_SL_PCT}-{MAX_SL_PCT}%)"


def handle_status_command():
    results_logged = 0
    if os.path.exists(OUTCOME_LOG_PATH):
        try:
            with open(OUTCOME_LOG_PATH) as f:
                results_logged = sum(1 for line in f if line.strip())
        except Exception:
            pass
    open_positions = len([p for p in pending_outcomes if p["stage"] == "open"])
    return (
        f"🤖 *{SCANNER_NAME} {SCANNER_VERSION} Status*\n\n"
        f"✅ Running: True\n"
        f"🔄 Scans Completed: {scanner_stats['cycles_completed']}\n"
        f"📈 Active Positions: {open_positions}\n"
        f"👀 Coins in Watchlist: {scanner_stats['symbols_count']}\n"
        f"📋 Total Results Logged: {results_logged}\n"
        f"🎯 Threshold: Score ≥{MIN_ALERT_SCORE} | HIGH ≥80\n"
        f"🧱 Wall Filter: {'REQUIRED' if REQUIRE_WALL else 'confirmation only'}\n"
        f"💰 Entry: MARKET | R:R 1:{RR_RATIO} | SL: {_stop_desc()}"
    )


def handle_positions_command():
    active = [p for p in pending_outcomes if p["stage"] == "open"]
    if not active:
        return "📭 No active positions right now."
    lines = [f"📈 *Active Positions ({len(active)})*\n"]
    for p in active:
        try:
            price = get_current_price(p["symbol"])
        except Exception:
            price = None
        if price:
            move = round((price - p["entry_price"]) / p["entry_price"] * 100, 2)
            sign = "+" if move >= 0 else ""
            cur = f"  Current: ${price:g} ({sign}{move}%)\n"
        else:
            cur = "  Current: (unavailable)\n"
        lines.append(
            f"*{p['symbol']}*\n"
            f"  Entry: ${p['entry_price']:g}\n"
            f"{cur}"
            f"  Target: ${p['target']:g} (+{p['tp_pct']}%)\n"
            f"  SL: ${p['stop_loss']:g} (-{p['sl_pct']}%)\n"
            f"  Peak so far: +{round(p['mfe'],2)}% | Dip: {round(p['mae'],2)}%\n"
        )
    return "\n".join(lines)


def handle_results_command():
    if not os.path.exists(OUTCOME_LOG_PATH):
        return "📭 No completed trades logged yet."
    records = []
    try:
        with open(OUTCOME_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        return f"⚠️ Could not read results log: {e}"
    if not records:
        return "📭 No completed trades logged yet."

    total = len(records)
    tp = [r for r in records if r["outcome"] == "TP_HIT"]
    sl = [r for r in records if r["outcome"] == "STOPPED"]
    to = [r for r in records if r["outcome"] == "TIMEOUT_CLOSE"]

    def pct(n):
        return round(n / total * 100, 1) if total else 0.0

    resolved = len(tp) + len(sl) + len(to)
    win_rate = round(len(tp) / resolved * 100, 1) if resolved else 0.0
    avg_win = round(sum(r["realized_pct"] for r in tp) / len(tp), 2) if tp else 0.0
    avg_loss = round(sum(r["realized_pct"] for r in sl) / len(sl), 2) if sl else 0.0

    mfes = [r.get("mfe", 0) for r in records]
    maes = [r.get("mae", 0) for r in records]
    avg_mfe = round(sum(mfes) / len(mfes), 2) if mfes else 0.0
    avg_mae = round(sum(maes) / len(maes), 2) if maes else 0.0

    anchor_stats = {}
    for r in records:
        a = r.get("anchor", "?").capitalize()
        s = anchor_stats.setdefault(a, {"win": 0, "tot": 0})
        s["tot"] += 1
        if r["outcome"] == "TP_HIT":
            s["win"] += 1
    anchor_lines = "\n".join(
        f"• {a}: {s['win']}/{s['tot']} TP ({round(s['win']/s['tot']*100,0):.0f}%)"
        for a, s in sorted(anchor_stats.items())
    ) or "• (none yet)"

    short = {"TP_HIT": "TP", "STOPPED": "SL", "TIMEOUT_CLOSE": "TIMEOUT"}
    log_lines = "\n".join(
        f"  {r['symbol']} → {short.get(r['outcome'], r['outcome'])}  "
        f"{'+' if r['realized_pct']>=0 else ''}{r['realized_pct']:.2f}%  "
        f"(ran +{r.get('mfe',0):.1f}/{r.get('mae',0):.1f})"
        for r in reversed(records[-10:])
    )
    bar = "━" * 20
    return (
        f"📊 *Scanner Performance Report*\n"
        f"🗓 All time | {total} trade(s)\n{bar}\n"
        f"🏆 Target Hit:  {len(tp)}  ({pct(len(tp))}%)\n"
        f"🛑 Stop Loss:   {len(sl)}  ({pct(len(sl))}%)\n"
        f"⏱ Timeout:     {len(to)}  ({pct(len(to))}%)\n{bar}\n"
        f"✅ Win Rate:   {win_rate}%  (breakeven need: 40%)\n"
        f"📈 Avg Win:    +{avg_win}%\n"
        f"📉 Avg Loss:   {avg_loss}%\n{bar}\n"
        f"🔎 *Calibration* (how far price actually ran):\n"
        f"   Avg peak up:  +{avg_mfe}%\n"
        f"   Avg dip down: {avg_mae}%\n"
        f"   → target is +{records[-1].get('tp_pct', RR_RATIO)}%; "
        f"if avg peak ≥ target, aim is reachable\n{bar}\n"
        f"*BY STOP TYPE:*\n{anchor_lines}\n{bar}\n"
        f"*TRADE LOG:*\n{log_lines}"
    )


def handle_help_command():
    return (
        "🤖 *Commands*\n\n"
        "/status — health, uptime, active positions\n"
        "/positions — live trades with peak/dip so far\n"
        "/results — win rate + calibration (how far price ran)\n"
        "/help — this message"
    )


COMMAND_HANDLERS = {
    "/status": handle_status_command,
    "/positions": handle_positions_command,
    "/position": handle_positions_command,
    "/results": handle_results_command,
    "/result": handle_results_command,
    "/help": handle_help_command,
    "/start": handle_help_command,
}


def poll_telegram_commands():
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    offset = None
    while True:
        try:
            params = {"timeout": 30}
            if offset is not None:
                params["offset"] = offset
            r = requests.get(url, params=params, timeout=40)
            for update in r.json().get("result", []):
                offset = update["update_id"] + 1
                msg = update.get("message") or {}
                chat_id = str((msg.get("chat") or {}).get("id", ""))
                text = (msg.get("text") or "").strip()
                if chat_id != str(TELEGRAM_CHAT_ID):
                    continue
                command = text.split()[0].split("@")[0].lower() if text else ""
                handler = COMMAND_HANDLERS.get(command)
                if handler:
                    try:
                        send_telegram_alert(handler())
                    except Exception as e:
                        send_telegram_alert(f"⚠️ Command failed: {e}")
        except Exception as e:
            print(f"[Telegram polling error] {e}")
            time.sleep(5)


# ============================================================
# OUTCOME TRACKER  (market fill -> TP / SL / timeout, with MFE/MAE)
# ============================================================
def log_signal_for_outcome_tracking(symbol, score, levels, ts):
    pending_outcomes.append({
        "symbol": symbol,
        "entry_price": levels["entry"],
        "stop_loss": levels["stop_loss"],
        "target": levels["take_profit"],
        "sl_pct": levels["sl_pct"],
        "tp_pct": levels["tp_pct"],
        "anchor": levels["anchor"],
        "score": score,
        "alert_time": time.time(),
        "alert_time_str": ts,
        "stage": "open",     # filled instantly at market
        "mfe": 0.0,          # max favourable excursion (%)
        "mae": 0.0,          # max adverse excursion (%)
    })


def _log_outcome_record(entry, outcome_label, exit_price, realized_pct):
    record = {
        "symbol": entry["symbol"],
        "score": entry["score"],
        "anchor": entry.get("anchor", "unknown"),
        "alert_time": entry["alert_time_str"],
        "entry_price": entry["entry_price"],
        "exit_price": round(exit_price, 6),
        "outcome": outcome_label,
        "realized_pct": round(realized_pct, 3),
        "tp_pct": entry["tp_pct"],
        "sl_pct": entry["sl_pct"],
        "mfe": round(entry["mfe"], 3),
        "mae": round(entry["mae"], 3),
    }
    with open(OUTCOME_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Outcome] {entry['symbol']} {outcome_label}: {round(realized_pct,3)}% "
          f"(peak +{record['mfe']}%, dip {record['mae']}%)")
    if DB_ENABLED:
        try:
            db.save_outcome(record)
            db.close_position(entry["symbol"], entry["entry_price"])
        except Exception as e:
            print(f"[DB error saving outcome] {e}")


def check_pending_outcomes():
    if not pending_outcomes:
        return
    now = time.time()
    still_pending = []
    for entry in pending_outcomes:
        ep = entry["entry_price"]
        try:
            price = get_current_price(entry["symbol"])
        except Exception as e:
            print(f"[Error checking outcome for {entry['symbol']}] {e}")
            still_pending.append(entry)
            continue
        if not price:
            still_pending.append(entry)
            continue

        move_pct = (price - ep) / ep * 100
        entry["mfe"] = max(entry["mfe"], move_pct)
        entry["mae"] = min(entry["mae"], move_pct)

        resolved = False
        if price >= entry["target"]:
            _log_outcome_record(entry, "TP_HIT", price, (entry["target"] - ep) / ep * 100)
            resolved = True
        elif price <= entry["stop_loss"]:
            _log_outcome_record(entry, "STOPPED", price, (entry["stop_loss"] - ep) / ep * 100)
            resolved = True
        elif (now - entry["alert_time"]) / 3600 >= OUTCOME_TIMEOUT_HOURS:
            _log_outcome_record(entry, "TIMEOUT_CLOSE", price, move_pct)
            resolved = True

        if not resolved:
            still_pending.append(entry)
    pending_outcomes[:] = still_pending


# ============================================================
# MAIN SCAN
# ============================================================
def scan_symbol(symbol):
    if time.time() - last_alert_time.get(symbol, 0) < COOLDOWN_SECONDS:
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
    accum = accum and macro["position_pct"] <= MACRO_POSITION_MAX_PCT

    if not (spring or sos or lps or accum):
        return

    cvd = compute_cvd(trades) if trades else None
    score = score_signal(spring, sos, lps, accum, accum_info, macro, cvd)
    if score < MIN_ALERT_SCORE:
        return

    spread_pct = get_spread_pct(symbol)
    if spread_pct is not None and spread_pct > MAX_SPREAD_PCT:
        print(f"[Skip] {symbol} score {score} but spread {spread_pct}% > {MAX_SPREAD_PCT}%")
        return

    last_price = float(df["close"].iloc[-1])
    wall = find_bid_wall(symbol, last_price)
    if REQUIRE_WALL and not wall:
        print(f"[Skip] {symbol} score {score} but no buy wall (REQUIRE_WALL on)")
        return

    levels = compute_trade_levels(df, accum_info, symbol, wall)

    confirmations = []
    if cvd is not None and cvd > 0:
        confirmations.append("CVD positive")
    if accum and accum_info:
        confirmations.append(f"tight range {accum_info['range_pct']}%")
        confirmations.append(f"48h position {macro['position_pct']}%")
    if wall:
        confirmations.append(f"buy wall {wall['strength']}x @ {wall['price']}")
    if spread_pct is not None:
        confirmations.append(f"spread {spread_pct}%")

    confidence = "HIGH" if score >= 80 else "MEDIUM"

    signal_type = []
    if spring: signal_type.append("SPRING")
    if sos: signal_type.append("SOS")
    if lps: signal_type.append("LPS")
    if accum: signal_type.append("ACCUM")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    message = (
        f"*Wyckoff Signal — {symbol}*\n"
        f"Score: {score}/100\n"
        f"Type: {', '.join(signal_type)}\n"
        f"Confidence: {confidence}\n\n"
        f"Entry: {levels['entry']} — MARKET (enter now)\n"
        f"Stop Loss: {levels['stop_loss']} (-{levels['sl_pct']}%)\n"
        f"Target: {levels['take_profit']} (+{levels['tp_pct']}%, R:R 1:{RR_RATIO})\n\n"
        f"*Plan:* single target, full exit at +{levels['tp_pct']}%. "
        f"Hard stop at -{levels['sl_pct']}%. No averaging down.\n\n"
        f"48h Range: {macro['macro_low']} - {macro['macro_high']} "
        f"(currently at {macro['position_pct']}%)\n"
        f"Confirmations: {', '.join(confirmations) if confirmations else 'price/volume only'}\n"
        f"Time: {ts}"
    )
    print(message)
    send_telegram_alert(message)
    scanner_stats["alerts_sent"] += 1
    last_alert_time[symbol] = time.time()
    log_signal_for_outcome_tracking(symbol, score, levels, ts)

    if DB_ENABLED:
        try:
            db.save_signal(symbol, score, signal_type, confidence, levels, macro, confirmations)
            db.open_position(symbol, levels, score, "open")
        except Exception as e:
            print(f"[DB error saving signal] {e}")


def build_startup_message(symbol_count):
    return (
        f"🟢 *{SCANNER_NAME} {SCANNER_VERSION} — STARTED*\n\n"
        "*Strategy:* Wyckoff accumulation, scalping mode\n"
        "*Signals:* Spring, SOS, LPS, Sideways Accumulation (ACCUM)\n"
        f"*Watchlist:* dynamic — KuCoin USDT pairs, 24h vol ≥ ${MIN_VOLUME_USDT:,.0f} "
        f"({symbol_count} pairs)\n"
        f"*Timeframe:* {KLINE_TYPE}, {CONTEXT_HOURS}h context\n"
        f"*Scan interval:* {SCAN_INTERVAL_SECONDS // 60} min | cooldown {COOLDOWN_SECONDS//60} min\n"
        f"*Min score:* {MIN_ALERT_SCORE}/100\n\n"
        f"*Entry:* MARKET price (fills every signal)\n"
        f"*Risk:Reward:* 1:{RR_RATIO} | Stop: {_stop_desc()}\n"
        f"Each trade records peak/dip (MFE/MAE) so you can see if the "
        f"target is well-calibrated. Outcomes logged to {OUTCOME_LOG_PATH}.\n\n"
        f"*Commands:* /status /positions /results /help\n\n"
        "_Statistical setups, not guaranteed outcomes — your own judgment on entries._"
    )


def run_scanner():
    print("Starting Wyckoff scalping scanner... (Ctrl+C to stop)")

    if DB_ENABLED:
        try:
            db.init_db()
            print("[Info] Database connected — signals saved for dashboard")
        except Exception as e:
            print(f"[DB init error] {e}")
    else:
        print("[Info] No DATABASE_URL — running Telegram-only")

    threading.Thread(target=poll_telegram_commands, daemon=True).start()
    print("[Info] Telegram command listener started")

    symbols_cache = []
    last_refresh = 0
    try:
        symbols_cache = get_all_kucoin_symbols()
        last_refresh = time.time()
        scanner_stats["symbols_count"] = len(symbols_cache)
        print(f"[Info] Loaded {len(symbols_cache)} symbols")
    except Exception as e:
        print(f"[Error loading symbols] {e}")

    send_telegram_alert(build_startup_message(len(symbols_cache)))

    while True:
        now = time.time()
        if now - last_refresh > SYMBOL_REFRESH_SECONDS or not symbols_cache:
            try:
                symbols_cache = get_all_kucoin_symbols()
                last_refresh = now
                scanner_stats["symbols_count"] = len(symbols_cache)
                print(f"[Info] Refreshed — {len(symbols_cache)} pairs")
            except Exception as e:
                print(f"[Error refreshing symbols] {e}")

        for symbol in symbols_cache:
            try:
                scan_symbol(symbol)
            except Exception as e:
                print(f"[Error scanning {symbol}] {e}")
            time.sleep(0.3)

        check_pending_outcomes()
        scanner_stats["last_cycle_time"] = time.time()
        scanner_stats["cycles_completed"] += 1
        if DB_ENABLED:
            try:
                db.update_heartbeat(scanner_stats)
            except Exception as e:
                print(f"[DB heartbeat error] {e}")
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_scanner()
