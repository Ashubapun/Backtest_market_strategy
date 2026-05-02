"""
EUR/USD Liquidity Sweep Strategy
=================================
Based on Umar Punjabi's Asia-to-London transition method.

HOW TO RUN IN THE APP:
  1. Market       -> Foreign Markets -> Forex -> EUR/USD
  2. Mode         -> intraday  (IMPORTANT: needs 1H bars)
  3. Stop Loss    -> 1.0%
  4. Take Profit  -> 2.0%   (gives 1:2 Risk-to-Reward)
  5. Paste this entire file's class into the Strategy Code box
  6. Click Run Backtest

STRATEGY LOGIC (5 steps):
  Step 1 - 4H Bias   : bullish if 4H close > 4H SMA(20), else bearish
  Step 2 - Asia Range : identify the high & low of the Asia session (00:00-07:00 UTC)
  Step 3 - Sweep      : bearish bias -> wait for price to sweep Asia HIGH
                        bullish bias -> wait for price to sweep Asia LOW
  Step 4 - Confirm    : after the sweep, look for a Bullish or Bearish Engulfing candle
                        in the direction of bias (acts as BOS proxy on 1H)
  Step 5 - Entry      : only enter during pre-London / London open window (06:00-12:00 UTC)
                        max 2 trades per day
                        skip if Asia session was too sideways (range < min_pips)

NOTE: yfinance returns EUR/USD 1H data in UTC. This strategy is built around UTC hours.
"""

import pandas as pd
import numpy as np
from backtester import Strategy


class EURUSDLiquiditySweepStrategy(Strategy):
    """
    EUR/USD Liquidity Sweep - Asia to London Session
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mode: str = "intraday",          # accepted for app compatibility, not used
        custom_step=None,                 # accepted for app compatibility, not used
        bias_lookback: int = 80,          # 4H SMA lookback in 1H bars (20 x 4H = 80 x 1H)
        asia_start_hour: int = 0,         # UTC hour Asia session starts
        asia_end_hour: int = 7,           # UTC hour Asia session ends (inclusive)
        entry_start_hour: int = 6,        # UTC hour entry window opens
        entry_end_hour: int = 12,         # UTC hour entry window closes
        min_asia_range_pips: float = 5,   # skip day if Asia range < this (in pips, 1 pip = 0.0001)
        max_trades_per_day: int = 2,      # Punjabi recommends max 2 entries per day
    ):
        super().__init__(data)
        self.bias_lookback      = bias_lookback
        self.asia_start         = asia_start_hour
        self.asia_end           = asia_end_hour
        self.entry_start        = entry_start_hour
        self.entry_end          = entry_end_hour
        self.min_range          = min_asia_range_pips * 0.0001
        self.max_trades_per_day = max_trades_per_day

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def generate_signals(self) -> pd.Series:
        df = self._prepare(self.data.copy())
        return self._build_signals(df)

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bias, Asia range, and session columns."""

        # ── 4H Bias ──────────────────────────────────────────────────────
        # Smooth close over 4 bars to mimic 4H candle, then compare to SMA
        df["close_4h"] = df["Close"].rolling(4, min_periods=4).mean()
        df["sma_bias"] = df["close_4h"].rolling(self.bias_lookback, min_periods=self.bias_lookback).mean()
        df["bias"]     = np.where(df["close_4h"] > df["sma_bias"], 1, -1)

        # ── Session labels ───────────────────────────────────────────────
        df["_hour"] = df.index.hour
        df["_date"] = df.index.normalize()   # midnight of each day

        # ── Asia session high / low ───────────────────────────────────────
        asia_mask = (df["_hour"] >= self.asia_start) & (df["_hour"] <= self.asia_end)
        asia_ohlc = (
            df[asia_mask]
            .groupby("_date")
            .agg(asia_high=("High", "max"), asia_low=("Low", "min"))
        )
        df = df.join(asia_ohlc, on="_date")

        # Forward-fill so London bars can see the Asia range
        df["asia_high"] = df["asia_high"].ffill()
        df["asia_low"]  = df["asia_low"].ffill()

        return df

    def _is_bullish_engulf(self, prev, curr) -> bool:
        """Bullish engulfing: current green candle body engulfs previous red body."""
        prev_bear = prev["Close"] < prev["Open"]
        curr_bull = curr["Close"] > curr["Open"]
        engulfs   = curr["Open"] <= prev["Close"] and curr["Close"] >= prev["Open"]
        return prev_bear and curr_bull and engulfs

    def _is_bearish_engulf(self, prev, curr) -> bool:
        """Bearish engulfing: current red candle body engulfs previous green body."""
        prev_bull = prev["Close"] > prev["Open"]
        curr_bear = curr["Close"] < curr["Open"]
        engulfs   = curr["Open"] >= prev["Close"] and curr["Close"] <= prev["Open"]
        return prev_bull and curr_bear and engulfs

    def _build_signals(self, df: pd.DataFrame) -> pd.Series:
        signal              = pd.Series(0, index=df.index, dtype=float)
        daily_trade_count   = {}
        swept_high_today    = {}   # date -> bool (Asia high was swept)
        swept_low_today     = {}   # date -> bool (Asia low was swept)

        rows = df.to_dict("index")
        idx  = list(df.index)

        for i in range(1, len(idx)):
            ts   = idx[i]
            row  = rows[ts]
            prev = rows[idx[i - 1]]
            date = row["_date"]

            # ── Track sweeps anywhere in the day ─────────────────────────
            ah = row.get("asia_high")
            al = row.get("asia_low")

            if pd.notna(ah) and row["High"] >= ah:
                swept_high_today[date] = True
            if pd.notna(al) and row["Low"] <= al:
                swept_low_today[date]  = True

            # ── Entry window only ─────────────────────────────────────────
            if not (self.entry_start <= row["_hour"] <= self.entry_end):
                continue

            # ── Max trades per day ────────────────────────────────────────
            if daily_trade_count.get(date, 0) >= self.max_trades_per_day:
                continue

            bias     = row.get("bias")
            asia_h   = row.get("asia_high")
            asia_l   = row.get("asia_low")

            if pd.isna(bias) or pd.isna(asia_h) or pd.isna(asia_l):
                continue

            # ── Skip sideways Asia sessions ───────────────────────────────
            if (asia_h - asia_l) < self.min_range:
                continue

            # ── Bearish setup ─────────────────────────────────────────────
            # Bias bearish + Asia HIGH was swept + bearish engulfing confirmation
            if (
                bias == -1
                and swept_high_today.get(date, False)
                and self._is_bearish_engulf(prev, row)
            ):
                signal.loc[ts] = -1
                daily_trade_count[date] = daily_trade_count.get(date, 0) + 1

            # ── Bullish setup ─────────────────────────────────────────────
            # Bias bullish + Asia LOW was swept + bullish engulfing confirmation
            elif (
                bias == 1
                and swept_low_today.get(date, False)
                and self._is_bullish_engulf(prev, row)
            ):
                signal.loc[ts] = 1
                daily_trade_count[date] = daily_trade_count.get(date, 0) + 1

        return signal
