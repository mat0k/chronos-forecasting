# pip install pandas requests holidays gdeltdoc

import os
import time
import math
import requests
import pandas as pd
import holidays

from gdeltdoc import GdeltDoc, Filters

# ---------------------------
# CONFIG
# ---------------------------
START = "2024-01-01"   # <<< change as needed (1s for years = enormous)
END   = "2024-01-03"   # <<< change as needed

SYMBOL = "BTCUSDT"
INTERVAL = "1s"

# Binance public market-data-only base endpoint
BINANCE_BASE = "https://data-api.binance.vision"

OUTPUT_CSV = "btc_dataset_1s.csv"
WRITE_HEADER_IF_NEW = True

# Optional GDELT covariate: news tone about a politician (proxy)
USE_GDELT = True
POLITICIAN_QUERY = "donald trump"   # change to the politician name you mean

# Chunk sizes / pacing
KLINES_LIMIT = 1000     # Binance max is 1000 for many endpoints
SLEEP_SEC = 0.05        # be nice to the API


# ---------------------------
# BINANCE: fetch 1s klines (streaming)
# ---------------------------
def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def klines_to_df(rows):
    """
    Binance /api/v3/klines response:
    [
      [
        openTime, open, high, low, close, volume, closeTime,
        quoteAssetVolume, numberOfTrades, takerBuyBase, takerBuyQuote, ignore
      ],
      ...
    ]
    """
    if not rows:
        return pd.DataFrame()

    cols = [
        "open_time_ms",
        "btc_open",
        "btc_high",
        "btc_low",
        "btc_close",
        "btc_volume",
        "close_time_ms",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)

    # types
    df["open_time_ms"] = df["open_time_ms"].astype("int64")
    df["close_time_ms"] = df["close_time_ms"].astype("int64")
    num_cols = ["btc_open","btc_high","btc_low","btc_close","btc_volume","quote_volume","taker_buy_base","taker_buy_quote"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype("int64")

    # timestamp at second granularity
    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True).dt.tz_convert(None)

    # extra microstructure covariates
    df["taker_buy_ratio"] = (df["taker_buy_base"] / df["btc_volume"]).replace([math.inf, -math.inf], 0).fillna(0)

    # final columns to keep (add id later)
    keep = [
        "timestamp",
        "btc_open","btc_high","btc_low","btc_close","btc_volume",
        "trades","taker_buy_base","taker_buy_quote","taker_buy_ratio",
    ]
    return df[keep]


# ---------------------------
# GDELT tone (proxy sentiment) and upsample to seconds
# ---------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df
    for c in ["datetime", "date", "day", "time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.tz_localize(None)
            df = df.set_index(c)
            return df
    # fallback: first column
    first = df.columns[0]
    maybe_dt = pd.to_datetime(df[first], errors="coerce")
    if maybe_dt.notna().any():
        df[first] = maybe_dt.dt.tz_localize(None)
        return df.set_index(first)
    raise RuntimeError("Could not find datetime column in GDELT output.")

def _pick_value_col(df: pd.DataFrame) -> str:
    # prefer 'value' if present, else first numeric column
    lower = {c.lower(): c for c in df.columns}
    if "value" in lower:
        return lower["value"]
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]
    raise RuntimeError(f"No numeric value column in GDELT df columns: {list(df.columns)}")

def gdelt_tone_daily(keyword: str, start: str, end: str) -> pd.DataFrame:
    gd = GdeltDoc()
    f = Filters(keyword=keyword, start_date=start, end_date=end)

    tone = gd.timeline_search("timelinetone", f)
    vol  = gd.timeline_search("timelinevolraw", f)

    if tone is None or len(tone) == 0:
        return pd.DataFrame(columns=["pol_tone","pol_news_volume"])

    tone = _ensure_datetime_index(tone)
    vol  = _ensure_datetime_index(vol) if vol is not None and len(vol) > 0 else pd.DataFrame(index=tone.index)

    tone_val = _pick_value_col(tone)
    out = pd.DataFrame(index=tone.index)
    out["pol_tone"] = pd.to_numeric(tone[tone_val], errors="coerce")

    if not vol.empty:
        vol_val = _pick_value_col(vol)
        out["pol_news_volume"] = pd.to_numeric(vol[vol_val], errors="coerce")
    else:
        out["pol_news_volume"] = 0.0

    # normalize to date-only index (daily)
    out.index = pd.to_datetime(out.index.date)
    out = out.groupby(out.index).mean()
    out.index.name = "date"
    return out

def upsample_daily_to_seconds(daily_df: pd.DataFrame, sec_index: pd.DatetimeIndex) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(index=sec_index, columns=["pol_tone","pol_news_volume"]).fillna(0.0)

    # map each second -> its date, then join and forward fill
    tmp = pd.DataFrame(index=sec_index)
    tmp["date"] = pd.to_datetime(tmp.index.date)
    out = tmp.join(daily_df, on="date").drop(columns=["date"])
    out = out.ffill().fillna(0.0)
    return out


# ---------------------------
# Calendar features (future-known)
# ---------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    us_hols = holidays.US()
    idx = df["timestamp"]
    out = df.copy()
    out["dow"] = idx.dt.dayofweek
    out["hour"] = idx.dt.hour
    out["minute"] = idx.dt.minute
    out["second"] = idx.dt.second
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_us_holiday"] = idx.dt.date.map(lambda d: 1 if d in us_hols else 0).astype(int)
    return out


# ---------------------------
# MAIN: stream klines to CSV
# ---------------------------
def main():
    start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)

    # prepare output
    file_exists = os.path.exists(OUTPUT_CSV)
    write_header = (WRITE_HEADER_IF_NEW and (not file_exists))

    # optional GDELT daily series fetched once
    gdelt_daily = None
    if USE_GDELT:
        gdelt_daily = gdelt_tone_daily(POLITICIAN_QUERY, START, END)

    cur = start_ms
    total_rows = 0

    while cur < end_ms:
        rows = fetch_klines(SYMBOL, INTERVAL, cur, end_ms, limit=KLINES_LIMIT)
        if not rows:
            break

        df = klines_to_df(rows)
        if df.empty:
            break

        # add id
        df.insert(0, "id", 1)

        # add calendar features
        df = add_calendar_features(df)

        # add GDELT covariates (upsampled to seconds)
        if USE_GDELT and gdelt_daily is not None:
            pol_sec = upsample_daily_to_seconds(gdelt_daily, df["timestamp"])
            df = pd.concat([df, pol_sec.reset_index(drop=True)], axis=1)

        # write chunk
        df.to_csv(OUTPUT_CSV, mode="a", index=False, header=write_header)
        write_header = False

        total_rows += len(df)
        # advance start: next millisecond after last open_time (1s step -> +1000ms)
        last_ts = df["timestamp"].iloc[-1]
        cur = int(pd.Timestamp(last_ts, tz="UTC").timestamp() * 1000) + 1000

        time.sleep(SLEEP_SEC)

    print(f"Done. Wrote ~{total_rows} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()