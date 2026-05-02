"""
Punjabi Liquidity Sweep Strategy
====================================
Umar Punjabi's Asia-to-London transition method.
Adapted from the standalone PunjabiStrategy class for the backtester app.

ORIGINAL LOGIC (preserved exactly):
  Step 1 - 4H Bias   : Bullish if 4H close > 4H open, else Bearish
  Step 2 - Asia Range : High and Low of Asia session (00:00-07:00 UTC)
  Step 3 - Sweep      : Bearish bias -> wait for price to sweep Asia HIGH
                        Bullish bias -> wait for price to sweep Asia LOW
  Step 4 - BOS        : After sweep, look for Break of Structure
                        SHORT: candle close < previous candle low  (engulf down)
                        LONG:  candle close > previous candle high (engulf up)
  Step 5 - Entry      : Only during London pre-open window (07:00-11:00 UTC)
                        Max 2 trades per day (Punjabi rule)

HOW TO USE IN THE BACKTESTER APP:
  Market      -> Foreign Markets -> Forex -> EUR/USD
  Mode        -> intraday  (1H bars required)
  Stop Loss   -> 1.0%
  Take Profit -> 2.0%
  Allow Short -> ON
  Paste the PunjabiLiquiditySweepStrategy class into Strategy Code box.
  Click Run Backtest.

NOTE ON SL/TP:
  The original strategy uses dynamic SL based on Asia High/Low + buffer.
  In this backtester version the fixed 1%/2% SL/TP set in the app sidebar
  are used instead. The entry logic is identical to the original.
"""

import pandas as pd
import numpy as np
from backtester import Strategy


class PunjabiLiquiditySweepStrategy(Strategy):
    """
    Punjabi Liquidity Sweep - Asia session to London open.
    Original Umar Punjabi logic adapted to the backtester signal format.
    Use on 1H EUR/USD data with SL=1%, TP=2%, Allow Short=ON.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mode: str = "intraday",
        custom_step=None,

        # Asia session definition (UTC hours, inclusive)
        asia_start_hour: int = 0,
        asia_end_hour:   int = 6,    # up to but not including 07:00

        # Entry window (London pre-open / open)
        entry_start_hour: int = 7,
        entry_end_hour:   int = 11,

        # Punjabi rule: max 2 entries per day
        max_trades_per_day: int = 2,

        # How long to hold if SL/TP not hit (bars on 1H = hours)
        max_hold_bars: int = 16,     # hold up to 16 hours then flatten
    ):
        super().__init__(data)
        self.asia_start      = asia_start_hour
        self.asia_end        = asia_end_hour
        self.entry_start     = entry_start_hour
        self.entry_end       = entry_end_hour
        self.max_trades_pd   = max_trades_per_day
        self.max_hold_bars   = max_hold_bars

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

        # ── 4H Bias (Punjabi Step 1) ─────────────────────────────────────
        # Resample 1H bars to 4H candles, then compare close vs open.
        ohlc_4h = df["Close"].resample("4h").ohlc()
        bias_4h = pd.Series(
            np.where(ohlc_4h["close"] > ohlc_4h["open"], 1, -1),
            index=ohlc_4h.index,
        )
        # Forward-fill 4H bias onto every 1H bar
        df["bias"] = bias_4h.reindex(df.index, method="ffill")

        # ── Session labels ───────────────────────────────────────────────
        df["_hour"] = df.index.hour
        df["_date"] = df.index.normalize()

        # ── Asia session high / low (Punjabi Step 2) ─────────────────────
        asia_mask = (df["_hour"] >= self.asia_start) & (df["_hour"] < self.asia_end)
        asia_range = (
            df[asia_mask]
            .groupby("_date")
            .agg(asia_high=("High", "max"), asia_low=("Low", "min"))
        )
        df = df.join(asia_range, on="_date")

        # Forward-fill so London bars can read today's Asia range
        df["asia_high"] = df["asia_high"].ffill()
        df["asia_low"]  = df["asia_low"].ffill()

        return df

    # ------------------------------------------------------------------ #
    # Signal generation                                                    #
    # ------------------------------------------------------------------ #

    def _build_signals(self, df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index, dtype=float)

        daily_trades = {}   # date -> int  (trade count per day)
        swept_high   = {}   # date -> bool (Asia High was swept today)
        swept_low    = {}   # date -> bool (Asia Low was swept today)

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

            # ── Track Asia sweeps throughout the whole day ────────────────
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

            # ── Punjabi rule: max 2 trades per day ────────────────────────
            if daily_trades.get(date, 0) >= self.max_trades_pd:
                continue

            bias = row.get("bias")
            if pd.isna(bias) or pd.isna(ah) or pd.isna(al):
                continue

            # ── SHORT (Punjabi Steps 3-4-5) ───────────────────────────────
            # Bearish 4H bias  +  Asia High swept  +  BOS down (close < prev low)
            if (
                bias == -1
                and swept_high.get(date, False)
                and row["Close"] < prev["Low"]
                and position != -1              # not already short
            ):
                position  = -1
                entry_bar = i
                daily_trades[date] = daily_trades.get(date, 0) + 1
                signal[ts] = position

            # ── LONG (Punjabi Steps 3-4-5) ────────────────────────────────
            # Bullish 4H bias  +  Asia Low swept  +  BOS up (close > prev high)
            elif (
                bias == 1
                and swept_low.get(date, False)
                and row["Close"] > prev["High"]
                and position != 1               # not already long
            ):
                position  = 1
                entry_bar = i
                daily_trades[date] = daily_trades.get(date, 0) + 1
                signal[ts] = position

        return signal
