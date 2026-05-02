"""
Golden Setup Strategy
=====================
Round-number breakout strategy.
- Intraday  : 500pt levels, daily reference open
- Positional: 1000pt levels, weekly reference open

Long  when price HIGH crosses above the nearest upper round level
Short when price LOW  crosses below the nearest lower round level
"""

import pandas as pd
import numpy as np
from backtester import Strategy


class GoldenSetupStrategy(Strategy):

    def __init__(self, data: pd.DataFrame, mode: str = "intraday"):
        super().__init__(data)
        if mode not in ("intraday", "positional"):
            raise ValueError("mode must be 'intraday' or 'positional'")
        self.mode = mode
        self._cfg = {
            "intraday":   {"step": 500,  "freq": "D"},
            "positional": {"step": 1000, "freq": "W"},
        }[mode]

    def generate_signals(self) -> pd.Series:
        df   = self.data.copy()
        step = self._cfg["step"]
        freq = self._cfg["freq"]

        # Period opening price — first Open of each day (D) or week (W)
        period_groups = df.index.to_period(freq)
        period_open   = df["Open"].groupby(period_groups).transform("first")

        # Nearest round levels
        upper_trigger = (period_open // step + 1) * step   # first level ABOVE open
        lower_trigger = (period_open // step)       * step  # first level AT/BELOW open

        # Signals  — long takes priority when both triggered on same bar
        signal = pd.Series(0, index=df.index, dtype=float)
        signal[df["Low"]  <= lower_trigger] = -1   # short
        signal[df["High"] >= upper_trigger] =  1   # long (overrides)

        # Carry forward: stay in trade until opposite signal fires
        # (remove this block if you want single-bar signals only)
        signal = signal.replace(0, np.nan).ffill().fillna(0)

        return signal
