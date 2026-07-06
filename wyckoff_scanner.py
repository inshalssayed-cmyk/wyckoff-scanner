"""
Wyckoff Scanner (Simple Version)
=================================
Scans KuCoin markets for Wyckoff accumulation signals (Spring, SOS, LPS),
confirms with order flow (CVD from trade tape), then alerts to Telegram.

No on-chain / Etherscan dependency — just KuCoin data + Telegram.

SETUP REQUIRED (set these as environment variables, not hardcoded):
  - TELEGRAM_BOT_TOKEN : create a bot via @BotFather on Telegram
  - TELEGRAM_CHAT_ID   : your chat/channel ID (message @userinfobot to get yours)

  Locally (before running):
    export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
    export TELEGRAM_CHAT_ID="your_chat_id"
    python3 wyckoff_scanner.py

  On Render: add these under your service's "Environment" tab instead —
  no need to export them yourself, Render injects them automatically.

  Also edit directly in the code below (not a secret):
  - WATCHLIST : KuCoin symbols to scan (must be spot pairs, e.g. "BTC-USDT")

Install deps:
  pip install requests pandas --break-system-packages

Run:
  python3 wyckoff_scanner.py
"""

import time
import os
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

# KuCoin spot symbols to scan
WATCHLIST = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    # add more KuCoin pairs here
]

# Scanner tuning
RANGE_WINDOW = 20          # candles used to define the trading range
VOL_AVG_WINDOW = 20        # rolling window for average volume
SCAN_INTERVAL_SECONDS = 300  # how often to re-scan (5 min default)
KLINE_TYPE = "1hour"       # KuCoin candle size: 1min,5min,15min,1hour,4hour,1day

KUCOIN_BASE = "https://api.kucoin.com"


# ============================================================
# DATA FETCHING
# ============================================================
def get_kucoin_klines(symbol, kline_type=KLINE_TYPE, limit=100):
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


def get_kucoin_trade_history(symbol, limit=100):
    """Fetch recent trade tape for CVD calculation."""
    url = f"{KUCOIN_BASE}/api/v1/market/histories"
    params = {"symbol": symbol}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    return data[:limit]  # each item has 'side' (buy/sell), 'size', 'price', 'time'


# ============================================================
# WYCKOFF DETECTION LOGIC
# ============================================================
def compute_range(df, window=RANGE_WINDOW):
    df["range_high"] = df["high"].rolling(window=window).max()
    df["range_low"] = df["low"].rolling(window=window).min()
    df["range_width_pct"] = (df["range_high"] - df["range_low"]) / df["range_low"]
    return df


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


def compute_cvd(trades):
    """Cumulative Volume Delta from recent trade tape."""
    delta = 0.0
    for t in trades:
        size = float(t.get("size", 0))
        if t.get("side") == "buy":
            delta += size
        else:
            delta -= size
    return delta


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
# MAIN SCAN LOOP
# ============================================================
def scan_symbol(symbol):
    df = get_kucoin_klines(symbol)
    if df is None or len(df) < RANGE_WINDOW + 2:
        return

    df = compute_range(df)

    spring = detect_spring(df)
    sos = detect_sos(df)
    lps = detect_lps(df)

    if not (spring or sos or lps):
        return  # nothing interesting, skip silently

    trades = get_kucoin_trade_history(symbol)
    cvd = compute_cvd(trades) if trades else None

    confirmations = []
    if cvd is not None and cvd > 0:
        confirmations.append("CVD positive")

    confidence = "HIGH" if confirmations else "MEDIUM"

    signal_type = []
    if spring:
        signal_type.append("SPRING")
    if sos:
        signal_type.append("SOS")
    if lps:
        signal_type.append("LPS")

    price = df["close"].iloc[-1]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    message = (
        f"*Wyckoff Signal — {symbol}*\n"
        f"Type: {', '.join(signal_type)}\n"
        f"Confidence: {confidence}\n"
        f"Price: {price}\n"
        f"Confirmations: {', '.join(confirmations) if confirmations else 'none (price/volume only)'}\n"
        f"Time: {ts}"
    )
    print(message)
    send_telegram_alert(message)


def build_startup_message():
    return (
        "🟢 *Wyckoff Scanner — STARTED*\n\n"
        "*Strategy:* Wyckoff Accumulation Detection\n"
        "*Signals tracked:* Spring, Sign of Strength (SOS), Last Point of Support (LPS)\n"
        "*Confirmation:* CVD (Cumulative Volume Delta) from live trade tape\n"
        "*Data source:* KuCoin (OHLCV + trade history)\n\n"
        f"*Watchlist:* {', '.join(WATCHLIST)}\n"
        f"*Timeframe:* {KLINE_TYPE}\n"
        f"*Scan interval:* every {SCAN_INTERVAL_SECONDS // 60} min\n\n"
        "You'll get an alert here the moment a Spring, SOS, or LPS is detected "
        "on any watchlist coin — confidence tagged HIGH or MEDIUM based on CVD confirmation.\n\n"
        "_Note: this flags statistical setups, not guaranteed outcomes — always your own judgment on entries._"
    )


def run_scanner():
    print("Starting Wyckoff scanner... (Ctrl+C to stop)")
    send_telegram_alert(build_startup_message())
    while True:
        for symbol in WATCHLIST:
            try:
                scan_symbol(symbol)
            except Exception as e:
                print(f"[Error scanning {symbol}] {e}")
            time.sleep(1)  # be polite to the API between symbols
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_scanner()
