"""
XAUUSD Scalp Signal Bot (5M/15M/30M/1H)
==========================================================
Data mənbəyi: Twelve Data (TAM, hər şey bir mənbədən - etibarlı,
bulud serverlərindən bloklanmır)

DXY əvəzedicisi: EUR/USD (Twelve Data-da mövcuddur, DXY-nin özü
pulsuz planda olmaya bilər). EUR ~57.6% çəki daşıyır və DXY ilə
əks-korrelyasiyadadır.

Strategiya:
- EMA Ribbon: EMA20 və EMA50 (qiymət hər ikisinin üzərində/altında)
- HAHO: VWMA(20) - hacim ağırlıqlı hərəkətli ortalama
- Yuxarıdakı 2 indikator 5M, 15M, 30M, 1H timeframe-lərində yoxlanılır
- DXY proxy (EUR/USD): bullish EUR/USD -> qızıl üçün bullish təzyiq
- RSI momentum (50 xəttinə görə) əlavə təsdiq
- Price action: Break of Structure (5M) əlavə təsdiq
- Score 100 üzərindən, 60+ olanda siqnal
- Risk/Reward: sabit 1:1 (scalp üçün)

QURULUM: pip install pandas numpy requests
Secrets: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TWELVEDATA_API_KEY
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

# ============ TƏNZİMLƏMƏLƏR ============
SYMBOL_TD = "XAU/USD"
DXY_PROXY_TD = "EUR/USD"   # DXY-nin əvəzedicisi (EUR ~57.6% çəki, əks-korrelyasiya)

MIN_SCORE_TO_SIGNAL = 60
SL_TP_ATR_MULTIPLIER = 1.0   # scalp üçün sabit, RR 1:1 deməkdir
MIN_ROOM_ATR = 1.0            # hədəfə qədər yol açıq olmalıdır (əks halda keç)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "").strip() or None

NEWS_BLOCK_MINUTES_BEFORE = 15
NEWS_BLOCK_MINUTES_AFTER = 15
HIGH_IMPACT_KEYWORDS = ["CPI", "NFP", "FOMC", "Interest Rate", "PCE", "GDP", "Powell", "Fed"]


# ============ İNDİKATOR FUNKSİYALARI (xarici kitabxana lazım deyil) ============
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def atr_value(df: pd.DataFrame, length: int = 14) -> float:
    return float(atr_series(df["high"], df["low"], df["close"], length).iloc[-1])


def vwma_or_sma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    if df["volume"].sum() <= 0:
        return df["close"].rolling(length).mean()
    pv = df["close"] * df["volume"]
    return pv.rolling(length).sum() / df["volume"].rolling(length).sum()


# ============ TELEGRAM ============
def send_telegram_message(text: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=15)
    if r.status_code != 200:
        print(f"Telegram xətası: {r.text}", file=sys.stderr)
        return False
    return True


# ============ DATA: TWELVE DATA (yalnız 5min çəkilir) ============
def fetch_live_price(symbol_td: str) -> Optional[float]:
    """Twelve Data-nın real-time 'price' endpoint-i - canlı bazar qiyməti,
    5 dəqiqəlik şamın bağlanmasını gözləmir."""
    if not TWELVEDATA_API_KEY:
        return None
    try:
        url = "https://api.twelvedata.com/price"
        params = {"symbol": symbol_td, "apikey": TWELVEDATA_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "price" in data:
            return float(data["price"])
        print(f"Canlı qiymət alınmadı: {data}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Canlı qiymət xətası: {e}", file=sys.stderr)
        return None


def fetch_td_series(symbol_td: str, interval_td: str, outputsize: int = 1000) -> pd.DataFrame:
    if not TWELVEDATA_API_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY verilməyib")
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol_td, "interval": interval_td, "outputsize": outputsize,
               "apikey": TWELVEDATA_API_KEY, "order": "ASC"}
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data xətası ({symbol_td}, {interval_td}): {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float) if "volume" in df.columns else 0.0
    df = df.sort_values("datetime").set_index("datetime")
    return df[["open", "high", "low", "close", "volume"]].dropna()


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna()


def get_scalp_timeframes(symbol_td: str) -> dict:
    """Twelve Data-dan yalnız 5min çəkilir (1 API sorğusu), qalanı resample edilir."""
    m5 = fetch_td_series(symbol_td, "5min", outputsize=1000)
    return {
        "5m": m5,
        "15m": resample_ohlc(m5, "15min"),
        "30m": resample_ohlc(m5, "30min"),
        "1h": resample_ohlc(m5, "1h"),
    }


# ============ DXY ƏVƏZEDİCİSİ: EUR/USD (Twelve Data, etibarlı) ============
def get_dxy_pressure() -> str:
    """
    EUR/USD, DXY-nin əks-korrelyasiya proksisidir (EUR ~57.6% çəki daşıyır).
    EUR/USD yüksəlirsə -> DXY düşür -> qızıl üçün bullish təzyiq.
    EUR/USD düşürsə -> DXY yüksəlir -> qızıl üçün bearish təzyiq.
    """
    try:
        df = fetch_td_series(DXY_PROXY_TD, "1h", outputsize=300)
        direction = ema_direction(df)
        if direction == "bullish":   # EUR/USD yüksəlir -> DXY düşür
            return "bullish_pressure"
        if direction == "bearish":   # EUR/USD düşür -> DXY yüksəlir
            return "bearish_pressure"
        return "neutral"
    except Exception as e:
        print(f"DXY-proxy (EUR/USD) xətası (filtr neytral qalır): {e}", file=sys.stderr)
        return "neutral"


# ============ İSTİQAMƏT FUNKSİYALARI ============
def ema_direction(df: pd.DataFrame) -> str:
    ema20 = ema(df["close"], 20)
    ema50 = ema(df["close"], 50)
    if pd.isna(ema20.iloc[-1]) or pd.isna(ema50.iloc[-1]):
        return "neutral"
    close = df["close"].iloc[-1]
    if ema20.iloc[-1] > ema50.iloc[-1] and close > ema20.iloc[-1]:
        return "bullish"
    if ema20.iloc[-1] < ema50.iloc[-1] and close < ema20.iloc[-1]:
        return "bearish"
    return "neutral"


def haho_direction(df: pd.DataFrame) -> str:
    haho = vwma_or_sma(df, 20)
    if pd.isna(haho.iloc[-1]):
        return "neutral"
    return "bullish" if df["close"].iloc[-1] > haho.iloc[-1] else "bearish"


def rsi_momentum(df: pd.DataFrame) -> str:
    r = rsi(df["close"], 14).iloc[-1]
    if pd.isna(r):
        return "neutral"
    if r > 55:
        return "bullish"
    if r < 45:
        return "bearish"
    return "neutral"


def find_swings(df: pd.DataFrame, window: int = 10):
    highs = df["high"]
    lows = df["low"]
    sh = highs[(highs == highs.rolling(window, center=True).max())].dropna().index
    sl = lows[(lows == lows.rolling(window, center=True).min())].dropna().index
    return highs.loc[sh], lows.loc[sl]


def detect_bos(df: pd.DataFrame) -> str:
    """Break of Structure - qiymət son swing high/low-u keçibmi."""
    swing_highs, swing_lows = find_swings(df.iloc[:-1], window=8)
    if swing_highs.empty or swing_lows.empty:
        return "neutral"
    last_close = df["close"].iloc[-1]
    if last_close > swing_highs.iloc[-1]:
        return "bullish"
    if last_close < swing_lows.iloc[-1]:
        return "bearish"
    return "neutral"


def nearest_level_distance(df: pd.DataFrame, direction: str) -> float:
    swing_highs, swing_lows = find_swings(df.iloc[:-1], window=8)
    last_close = df["close"].iloc[-1]
    atr = atr_value(df)
    if atr == 0 or pd.isna(atr):
        return 999
    if direction == "LONG" and not swing_highs.empty:
        above = swing_highs[swing_highs > last_close]
        if not above.empty:
            return (above.min() - last_close) / atr
    if direction == "SHORT" and not swing_lows.empty:
        below = swing_lows[swing_lows < last_close]
        if not below.empty:
            return (last_close - below.max()) / atr
    return 999


# ============ XƏBƏR FİLTRİ ============
def is_news_blackout() -> bool:
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=10)
        events = r.json()
    except Exception as e:
        print(f"Xəbər kalendarı gəlmədi, filtr keçilir: {e}", file=sys.stderr)
        return False

    now = datetime.now(timezone.utc)
    for ev in events:
        try:
            if ev.get("country") != "USD" or ev.get("impact") != "High":
                continue
            if not any(k.lower() in ev.get("title", "").lower() for k in HIGH_IMPACT_KEYWORDS):
                continue
            ev_time = pd.to_datetime(ev["date"]).tz_convert("UTC")
            delta_min = abs((now - ev_time).total_seconds()) / 60
            if delta_min <= max(NEWS_BLOCK_MINUTES_BEFORE, NEWS_BLOCK_MINUTES_AFTER):
                print(f"Xəbər blackout: {ev.get('title')} @ {ev_time}")
                return True
        except Exception:
            continue
    return False


# ============ SCORE SİSTEMİ ============
def build_score(tfs: dict, dxy_pressure: str) -> dict:
    votes = {}

    for tf in ["5m", "15m", "30m", "1h"]:
        votes[f"ema_{tf}"] = ema_direction(tfs[tf])
        votes[f"haho_{tf}"] = haho_direction(tfs[tf])

    votes["rsi"] = rsi_momentum(tfs["5m"])
    votes["bos"] = detect_bos(tfs["5m"])
    votes["dxy"] = "bullish" if dxy_pressure == "bullish_pressure" else ("bearish" if dxy_pressure == "bearish_pressure" else "neutral")

    weights = {
        "ema_1h": 15, "ema_30m": 10, "ema_15m": 8, "ema_5m": 5,
        "haho_1h": 12, "haho_30m": 8, "haho_15m": 6, "haho_5m": 4,
        "dxy": 12,
        "rsi": 8,
        "bos": 12,
    }

    bull_score = sum(w for k, w in weights.items() if votes.get(k) == "bullish")
    bear_score = sum(w for k, w in weights.items() if votes.get(k) == "bearish")

    direction, score = None, 0
    if bull_score >= MIN_SCORE_TO_SIGNAL and bull_score > bear_score:
        direction, score = "LONG", bull_score
    elif bear_score >= MIN_SCORE_TO_SIGNAL and bear_score > bull_score:
        direction, score = "SHORT", bear_score

    print(f"  -> bull_score={bull_score}, bear_score={bear_score}, seçilən={direction or 'yoxdur'}")
    return {"direction": direction, "score": score, "votes": votes}


# ============ ENTRY / SL / TP (RR 1:1) ============
def calc_levels(df_5m: pd.DataFrame, direction: str, live_price: Optional[float] = None):
    # Entry: canlı qiymət varsa onu istifadə et (real bazar qiyməti),
    # olmasa son bağlanmış 5M şamın qiymətinə keç (fallback)
    entry = live_price if live_price is not None else df_5m["close"].iloc[-1]
    atr = atr_value(df_5m)
    risk = atr * SL_TP_ATR_MULTIPLIER

    if direction == "LONG":
        sl = entry - risk
        tp = entry + risk
    else:
        sl = entry + risk
        tp = entry - risk

    return entry, sl, tp, risk


# ============ MESAJ FORMATI ============
CHECK_LABELS = {
    "ema_1h": "EMA Ribbon (1H)", "ema_30m": "EMA Ribbon (30M)",
    "ema_15m": "EMA Ribbon (15M)", "ema_5m": "EMA Ribbon (5M)",
    "haho_1h": "HAHO (1H)", "haho_30m": "HAHO (30M)",
    "haho_15m": "HAHO (15M)", "haho_5m": "HAHO (5M)",
    "dxy": "DXY Filtri", "rsi": "RSI Momentum", "bos": "Break of Structure",
}


def format_signal(direction, score, votes, entry, sl, tp):
    dir_label = "BUY" if direction == "LONG" else "SELL"
    want = "bullish" if direction == "LONG" else "bearish"
    confirms = [f"✅ {label}" for key, label in CHECK_LABELS.items() if votes.get(key) == want]
    confidence = "HIGH" if score >= 85 else ("MEDIUM" if score >= 70 else "LOW")

    lines = [
        f"⚡ <b>XAUUSD SCALP {dir_label}</b>\n",
        f"⭐ Signal Score: {score}/100",
        f"📊 Timeframe: 5M SCALP",
        f"📍 Entry: {entry:.2f}",
        f"🛑 Stop Loss: {sl:.2f}",
        f"🎯 TP (1:1): {tp:.2f}\n",
    ] + confirms + [f"\nConfidence: {confidence}", "⚠️ Scalp siqnalıdır, sürətli izləyin."]

    return "\n".join(lines)


# ============ ƏSAS AXIN ============
def main():
    if is_news_blackout():
        print("Yüksək təsirli xəbər pəncərəsi aktivdir, siqnal axtarışı dayandırıldı.")
        return

    try:
        tfs = get_scalp_timeframes(SYMBOL_TD)
    except Exception as e:
        print(f"Data xətası: {e}", file=sys.stderr)
        return

    dxy_pressure = get_dxy_pressure()
    result = build_score(tfs, dxy_pressure)
    direction = result["direction"]

    if not direction:
        print("Kifayət qədər score yoxdur, siqnal yoxdur.")
        return

    live_price = fetch_live_price(SYMBOL_TD)
    entry, sl, tp, risk = calc_levels(tfs["5m"], direction, live_price)

    dist = nearest_level_distance(tfs["5m"], direction)
    if dist < MIN_ROOM_ATR:
        print(f"Hədəfə qədər yol bağlıdır ({dist:.2f} ATR), siqnal keçilir.")
        return

    msg = format_signal(direction, result["score"], result["votes"], entry, sl, tp)
    if send_telegram_message(msg):
        print(f"SİQNAL göndərildi: {direction} (score={result['score']})")
    else:
        print("Telegram-a göndərilmədi.")


if __name__ == "__main__":
    main()
