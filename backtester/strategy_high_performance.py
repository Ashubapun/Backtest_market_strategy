"""
High Performance Trader Strategy
====================================
Adapted from the HighPerformanceTrader class for the backtester app.

ORIGINAL LOGIC (preserved exactly):
  Step 1 - 4H Bias      : BULL if 4H close > 4H open, else BEAR
  Step 2 - Asia Range   : High and Low of 00:00-07:00 UTC session
  Step 3 - Sweep        : BEAR bias -> price sweeps Asia HIGH
                          BULL bias -> price sweeps Asia LOW
  Step 4 - BOS confirm  : SELL: close < previous bar's low
                          BUY:  close > previous bar's high
  Step 5 - Entry window : London Open 07:00-10:30 UTC
                          Max 1 high-quality trade per day

NOTE ON DATA:
  Original strategy was designed for 5m bars. This backtester version uses
  1H bars (required by the app's intraday mode). The entry logic is
  identical; only bar granularity differs. For finer 5m entries, the
  standalone HighPerformanceTrader class should be used directly.

NOTE ON SL/TP:
  Original uses dynamic SL = Asia High/Low +/- 2 pip buffer with 1:2 RR.
  This backtester version uses the fixed 1%/2% SL/TP set in the sidebar.
  Entry logic is unchanged.

HOW TO USE IN THE BACKTESTER APP:
  Market      -> Foreign Markets -> Forex -> EUR/USD
  Mode        -> intraday  (1H bars required)
  Stop Loss   -> 1.0%
  Take Profit -> 2.0%
  Allow Short -> ON
  Paste the HighPerformanceTraderStrategy class into Strategy Code box.
  Click Run Backtest.
"""

import pandas as pd
import numpy as np
from backtester import Strategy


class HighPerformanceTraderStrategy(Strategy):
    """
    High Performance Trader - 4H bias + Asia sweep + London BOS entry.
    1 trade per day max. Use on 1H EUR/USD with SL=1%, TP=2%, Short=ON.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mode: str = "intraday",
        custom_step=None,

        # Asia session (UTC hours, start inclusive, end exclusive)
        asia_start_hour: int = 0,
        asia_end_hour:   int = 7,

        # London entry window (UTC hours, inclusive)
        entry_start_hour: int = 7,
        entry_end_hour:   int = 10,   # 10:00 UTC ~ 10:30 original

        # Max entries per day (original: 1 high-quality trade)
        max_trades_per_day: int = 1,

        # Max bars to hold if SL/TP not hit (backtester handles the actual exit)
        max_hold_bars: int = 16,      # 16 hours then flatten
    ):
        super().__init__(data)
        self.asia_start    = asia_start_hour
        self.asia_end      = asia_end_hour
        self.entry_start   = entry_start_hour
        self.entry_end     = entry_end_hour
        self.max_trades_pd = max_trades_per_day
        self.max_hold_bars = max_hold_bars

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def generate_signals(self) -> pd.Series:
        df = self._prepare(self.data.copy())
        return self._build_signals(df)

    # ------------------------------------------------------------------ #
    # Preparation                                                          #
    # ------------------------------------------------------------------ #

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:

        # ── 4H Bias ──────────────────────────────────────────────────────
        # Resample 1H to 4H candles, compare close vs open (original logic).
        ohlc_4h = df["Close"].resample("4h").ohlc()
        bias_4h = pd.Series(
            np.where(ohlc_4h["close"] > ohlc_4h["open"], 1, -1),
            index=ohlc_4h.index,
        )
        df["bias"] = bias_4h.reindex(df.index, method="ffill")

        # ── Session helpers ───────────────────────────────────────────────
        df["_hour"] = df.index.hour
        df["_date"] = df.index.normalize()

        # ── Asia Range ────────────────────────────────────────────────────
        asia_mask = (df["_hour"] >= self.asia_start) & (df["_hour"] < self.asia_end)
        asia_range = (
            df[asia_mask]
            .groupby("_date")
            .agg(asia_high=("High", "max"), asia_low=("Low", "min"))
        )
        df = df.join(asia_range, on="_date")
        df["asia_high"] = df["asia_high"].ffill()
        df["asia_low"]  = df["asia_low"].ffill()

        return df

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def _build_signals(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index, dtype=float)

        daily_trades = {}   # date -> int
        swept_high   = {}   # date -> bool  (Asia High was swept)
        swept_low    = {}   # date -> bool  (Asia Low was swept)

        position  = 0
        entry_bar = -9999

        idx  = list(df.index)
        rows = df.to_dict("index")

        for i in range(1, len(idx)):
            ts   = idx[i]
            row  = rows[ts]
            prev = rows[idx[i - 1]]
            date = row["_date"]

            # ── Exit after max hold (SL/TP handled by backtester) ─────────
            if position != 0 and (i - entry_bar) >= self.max_hold_bars:
                position = 0

            signal[ts] = position

            # ── Track Asia sweeps at any hour throughout the day ──────────
            ah = row.get("asia_high")
            al = row.get("asia_low")
            if pd.notna(ah) and row["High"] >= ah:
                swept_high[date] = True
            if pd.notna(al) and row["Low"] <= al:
                swept_low[date] = True

            # ── Entry only inside London window ───────────────────────────
            hour = row["_hour"]
            if not (self.entry_start <= hour <= self.entry_end):
                continue

            # ── Max 1 trade per day ───────────────────────────────────────
            if daily_trades.get(date, 0) >= self.max_trades_pd:
                continue

            bias = row.get("bias")
            if pd.isna(bias) or pd.isna(ah) or pd.isna(al):
                continue

            # ── SELL ──────────────────────────────────────────────────────
            # BEAR 4H bias + Asia High swept + BOS down (close < prev low)
            if (
                bias == -1
                and swept_high.get(date, False)
                and row["Close"] < prev["Low"]
                and position != -1
            ):
                position  = -1
                entry_bar = i
                daily_trades[date] = daily_trades.get(date, 0) + 1
                signal[ts] = position

            # ── BUY ───────────────────────────────────────────────────────
            # BULL 4H bias + Asia Low swept + BOS up (close > prev high)
            elif (
                bias == 1
                and swept_low.get(date, False)
                and row["Close"] > prev["High"]
                and position != 1
            ):
                position  = 1
                entry_bar = i
                daily_trades[date] = daily_trades.get(date, 0) + 1
                signal[ts] = position

        return signal
