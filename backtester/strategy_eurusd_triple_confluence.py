"""
EUR/USD Trend Pullback Strategy v3
=====================================
Root-cause analysis of previous failures:

  V1 (-41%, 885 trades, 1.2% win):
    No trend filter. Block B (timing) fired every Tue/Thu session window.
    The strategy was SHORTING a market that rose from 1.10 to 1.20.

  V2 (-23%, 450 trades, 3.8% win):
    EMA21/50 whipsaws in ranging markets caused the "exit on EMA break"
    condition to fire on almost every bar, closing trades at avg -$27
    (commission + tiny adverse move) BEFORE the 2% TP could be reached.
    Only 17 of 450 trades actually hit TP.

  ROOT CAUSE OF BOTH: Commission drag + premature exits.
    450 trades x $40 commission = $18,000 in fees alone (-18% of capital).
    Fixing direction alone won't help if exits kill winners early.

V3 DESIGN PRINCIPLES:
  1. EMA50 vs EMA200 for trend - stable, only 3-5 regime changes per year.
     Prevents the EMA21/50 whipsaw that caused 450 phantom exits.

  2. ZERO early exits in strategy code.
     SL (1%) and TP (2%) are the ONLY exits. Strategy adds only a
     5-day max-hold as a last resort for stalled trades.
     By letting TP get hit, each winning trade contributes the full +$360
     instead of +$50 after a premature exit.

  3. RSI pullback entry in confirmed trend.
     In uptrend (EMA50 > EMA200): enter after RSI dips below 50 and
     recovers. This is buying the dip in a bull trend - high probability.
     Symmetric for shorts.

  4. MACD histogram as momentum gate.
     Only enter when momentum is already aligned with the trade direction.

  5. 12-bar cooldown (not 24). Shorter cooldown captures more pullbacks
     while still preventing same-day churn.

TARGET PERFORMANCE:
  With 20% position size and 1%/2% SL/TP, the math limits maximum return.
  For best results, increase Position Size per Trade to 40-50% in the app.
  At 40% position, 80 trades with 60% win rate -> ~17% annual return.

HOW TO USE:
  Market      -> Foreign Markets -> Forex -> EUR/USD
  Mode        -> intraday (1H bars required)
  Stop Loss   -> 1.0%
  Take Profit -> 2.0%
  Allow Short -> ON
  Position    -> 40-50% recommended (see TARGET PERFORMANCE above)
  Paste the EURUSDTripleConfluenceStrategy class into Strategy Code.
"""

import pandas as pd
import numpy as np
from backtester import Strategy


class EURUSDTripleConfluenceStrategy(Strategy):
    """
    EUR/USD Trend Pullback v3 - EMA50/200 trend + RSI dip entries.
    NO early exits. SL=1% and TP=2% handle all position closes.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        mode: str = "intraday",
        custom_step=None,

        # -- Trend EMAs (slow pair avoids whipsaw) --
        ema_trend_fast: int = 50,
        ema_trend_slow: int = 200,

        # -- RSI pullback detection --
        rsi_period: int = 14,
        rsi_dip_threshold:   float = 50.0,  # dip below this = pullback occurred (long)
        rsi_spike_threshold: float = 50.0,  # spike above this = bounce occurred (short)
        rsi_lookback: int = 8,              # bars to scan for the dip/spike
        rsi_min_long:  float = 44.0,        # RSI must be back above this to enter long
        rsi_max_short: float = 56.0,        # RSI must be back below this to enter short

        # -- MACD --
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,

        # -- Session and day filters --
        session_start: int = 7,
        session_end:   int = 17,
        best_days: tuple = (0, 1, 2, 3),   # Mon=0 Tue=1 Wed=2 Thu=3

        # -- Position management --
        max_hold_bars: int = 120,   # 5-day safety net; SL/TP exit before this
        cooldown_bars: int = 12,    # 12-hour cooldown between trades
    ):
        super().__init__(data)
        self.ema_trend_fast      = ema_trend_fast
        self.ema_trend_slow      = ema_trend_slow
        self.rsi_period          = rsi_period
        self.rsi_dip_threshold   = rsi_dip_threshold
        self.rsi_spike_threshold = rsi_spike_threshold
        self.rsi_lookback        = rsi_lookback
        self.rsi_min_long        = rsi_min_long
        self.rsi_max_short       = rsi_max_short
        self.macd_fast           = macd_fast
        self.macd_slow           = macd_slow
        self.macd_signal         = macd_signal
        self.session_start       = session_start
        self.session_end         = session_end
        self.best_days           = best_days
        self.max_hold_bars       = max_hold_bars
        self.cooldown_bars       = cooldown_bars

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def generate_signals(self) -> pd.Series:
        df = self._compute_indicators(self.data.copy())
        return self._apply_rules(df)

    # ------------------------------------------------------------------ #
    # Indicators                                                           #
    # ------------------------------------------------------------------ #

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:

        # Trend filter: EMA50 vs EMA200
        # Stable - only 3-5 regime changes per year in EUR/USD
        df["ema_fast"] = df["Close"].ewm(span=self.ema_trend_fast, adjust=False).mean()
        df["ema_slow"] = df["Close"].ewm(span=self.ema_trend_slow, adjust=False).mean()
        df["uptrend"]  = df["ema_fast"] > df["ema_slow"]
        df["downtrend"]= df["ema_fast"] < df["ema_slow"]

        # RSI (Wilder's EWM smoothing)
        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI pullback window: did RSI touch the dip/spike level recently?
        df["rsi_min_window"] = df["rsi"].rolling(self.rsi_lookback).min()
        df["rsi_max_window"] = df["rsi"].rolling(self.rsi_lookback).max()

        # RSI momentum direction (rising or falling)
        df["rsi_rising"]  = df["rsi"] > df["rsi"].shift(2)
        df["rsi_falling"] = df["rsi"] < df["rsi"].shift(2)

        # MACD histogram
        ema_f           = df["Close"].ewm(span=self.macd_fast,   adjust=False).mean()
        ema_s           = df["Close"].ewm(span=self.macd_slow,   adjust=False).mean()
        macd_line       = ema_f - ema_s
        sig_line        = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = macd_line - sig_line

        # ATR (volatility filter - skip dead market hours)
        tr              = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"]  - df["Close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"]       = tr.rolling(14).mean()
        df["atr_floor"] = df["atr"].rolling(200).mean() * 0.4  # 40% of avg ATR

        # Session and day
        df["hour"]       = df.index.hour
        df["dow"]        = df.index.dayofweek
        df["in_session"] = (df["hour"] >= self.session_start) & (df["hour"] <= self.session_end)
        df["good_day"]   = df["dow"].isin(self.best_days)

        return df

    # ------------------------------------------------------------------ #
    # Rules                                                                #
    # ------------------------------------------------------------------ #

    def _apply_rules(self, df: pd.DataFrame) -> pd.Series:
        signal    = pd.Series(0, index=df.index, dtype=float)
        position  = 0       # +1 long, -1 short, 0 flat
        entry_bar = -9999
        last_exit = -9999

        for i in range(200, len(df)):
            row = df.iloc[i]

            # ── EXIT: only via max hold ────────────────────────────
            # The backtester's SL (1%) and TP (2%) are the primary exits.
            # This block is just the 5-day safety net for stalled trades.
            if position != 0:
                if (i - entry_bar) >= self.max_hold_bars:
                    position  = 0
                    last_exit = i

            # ── EMIT SIGNAL ────────────────────────────────────────
            signal.iloc[i] = position

            # ── GATE: must be flat and past cooldown ───────────────
            if position != 0 or (i - last_exit) < self.cooldown_bars:
                continue

            # ── GATE: indicators must be ready ─────────────────────
            if pd.isna(row["rsi"]) or pd.isna(row["macd_hist"]):
                continue

            # ── GATE: active trading session only ──────────────────
            if not (row["in_session"] and row["good_day"]):
                continue

            # ── GATE: skip near-zero volatility (weekends, holidays) ─
            if pd.notna(row["atr_floor"]) and row["atr"] < row["atr_floor"]:
                continue

            rsi  = row["rsi"]
            macd = row["macd_hist"]

            # ── LONG ENTRY ─────────────────────────────────────────
            # Conditions (all required):
            #   1. EMA50 > EMA200  -> confirmed uptrend (stable filter)
            #   2. RSI dipped below 50 in last 8 bars -> pullback occurred
            #   3. RSI now >= 44   -> pullback is recovering
            #   4. RSI currently rising -> momentum turning up
            #   5. MACD histogram > 0  -> trend momentum confirmed
            if (
                bool(row["uptrend"])
                and row["rsi_min_window"] < self.rsi_dip_threshold
                and rsi >= self.rsi_min_long
                and bool(row["rsi_rising"])
                and macd > 0
            ):
                position  = 1
                entry_bar = i

            # ── SHORT ENTRY ────────────────────────────────────────
            # Symmetric logic:
            #   1. EMA50 < EMA200  -> confirmed downtrend
            #   2. RSI spiked above 50 in last 8 bars -> bounce occurred
            #   3. RSI now <= 56   -> bounce is reversing
            #   4. RSI currently falling -> momentum turning down
            #   5. MACD histogram < 0   -> trend momentum confirmed
            elif (
                bool(row["downtrend"])
                and row["rsi_max_window"] > self.rsi_spike_threshold
                and rsi <= self.rsi_max_short
                and bool(row["rsi_falling"])
                and macd < 0
            ):
                position  = -1
                entry_bar = i

        return signal
