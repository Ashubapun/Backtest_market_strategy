"""
Trading Strategy Backtester
===========================
Drop your strategy into strategy.py (implement generate_signals),
then run: python backtester.py

Dependencies: pip install yfinance pandas numpy matplotlib seaborn tabulate
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    direction: str        # "long" or "short"
    shares: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_reason: str = ""


@dataclass
class BacktestConfig:
    symbol: str = "RELIANCE.NS"
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10      # fraction of equity per trade
    commission_pct: float = 0.001        # 0.1% per side
    slippage_pct: float = 0.0005         # 0.05% per side
    allow_short: bool = False
    max_open_positions: int = 1
    stop_loss_pct: Optional[float] = None   # e.g. 0.05 = 5% stop
    take_profit_pct: Optional[float] = None # e.g. 0.10 = 10% target
    interval: str = "1d"                 # yfinance interval


# ---------------------------------------------------------------------------
# Data fetcher
# ---------------------------------------------------------------------------

def fetch_data(config: BacktestConfig) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    print(f"Fetching {config.symbol} from {config.start_date} to {config.end_date} ...")
    ticker = yf.Ticker(config.symbol)
    df = ticker.history(
        start=config.start_date,
        end=config.end_date,
        interval=config.interval,
        auto_adjust=True,
    )
    if df.empty:
        raise ValueError(f"No data returned for {config.symbol}. Check symbol/dates.")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    print(f"  Loaded {len(df)} bars  ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data: pd.DataFrame = pd.DataFrame()
        self.signals: pd.Series = pd.Series(dtype=float)
        self.equity_curve: pd.Series = pd.Series(dtype=float)
        self.trades: list[Trade] = []
        self._cash = config.initial_capital
        self._open_trades: list[Trade] = []
        self._portfolio_values: list[float] = []
        self._dates: list[pd.Timestamp] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, signals: pd.Series) -> "BacktestResult":
        """
        signals: pd.Series aligned to self.data index
                 +1 = buy/long,  -1 = sell/short,  0 = flat/close
        """
        self.signals = signals.reindex(self.data.index).fillna(0)
        self._cash = self.config.initial_capital
        self._open_trades = []
        self._portfolio_values = []
        self._dates = []
        self.trades = []

        for i, (date, row) in enumerate(self.data.iterrows()):
            price = row["Close"]
            sig = self.signals.loc[date] if date in self.signals.index else 0

            # 1. Check stops/targets on open trades
            self._check_exits(date, row)

            # 2. Process signal
            if sig == 1 and len(self._open_trades) < self.config.max_open_positions:
                self._open_long(date, row)
            elif sig == -1:
                if self.config.allow_short and len(self._open_trades) < self.config.max_open_positions:
                    self._open_short(date, row)
                else:
                    self._close_all(date, row, reason="signal")
            elif sig == 0:
                self._close_all(date, row, reason="signal")

            # 3. Record portfolio value
            equity = self._cash + sum(
                t.shares * price * (1 if t.direction == "long" else -1)
                for t in self._open_trades
            )
            self._portfolio_values.append(equity)
            self._dates.append(date)

        # Close any remaining positions at last price
        if self._open_trades:
            last_date = self.data.index[-1]
            last_row = self.data.iloc[-1]
            self._close_all(last_date, last_row, reason="end-of-data")

        self.equity_curve = pd.Series(self._portfolio_values, index=self._dates)
        return BacktestResult(self.config, self.data, self.equity_curve, self.trades)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fill_price(self, price: float, side: str) -> float:
        slip = self.config.slippage_pct
        if side == "buy":
            return price * (1 + slip)
        return price * (1 - slip)

    def _commission(self, value: float) -> float:
        return value * self.config.commission_pct

    def _open_long(self, date, row):
        price = self._fill_price(row["Close"], "buy")
        capital = self._cash * self.config.position_size_pct
        shares = capital / price
        cost = shares * price
        comm = self._commission(cost)
        if cost + comm > self._cash:
            return  # not enough cash
        self._cash -= cost + comm
        self._open_trades.append(Trade(
            entry_date=date, entry_price=price,
            direction="long", shares=shares
        ))

    def _open_short(self, date, row):
        price = self._fill_price(row["Close"], "sell")
        capital = self._cash * self.config.position_size_pct
        shares = capital / price
        proceeds = shares * price
        comm = self._commission(proceeds)
        self._cash += proceeds - comm
        self._open_trades.append(Trade(
            entry_date=date, entry_price=price,
            direction="short", shares=shares
        ))

    def _close_trade(self, trade: Trade, date, price: float, reason: str):
        fill = self._fill_price(price, "sell" if trade.direction == "long" else "buy")
        comm = self._commission(trade.shares * fill)
        if trade.direction == "long":
            proceeds = trade.shares * fill - comm
            self._cash += proceeds
            trade.pnl = proceeds - trade.shares * trade.entry_price
        else:
            cost = trade.shares * fill + comm
            self._cash -= cost
            trade.pnl = trade.shares * (trade.entry_price - fill) - comm
        entry_value = trade.shares * trade.entry_price
        trade.pnl_pct = trade.pnl / entry_value if entry_value else 0
        trade.exit_date = date
        trade.exit_price = fill
        trade.exit_reason = reason
        self.trades.append(trade)

    def _close_all(self, date, row, reason: str):
        for t in list(self._open_trades):
            self._close_trade(t, date, row["Close"], reason)
        self._open_trades.clear()

    def _check_exits(self, date, row):
        sl = self.config.stop_loss_pct
        tp = self.config.take_profit_pct
        remaining = []
        for t in self._open_trades:
            if t.direction == "long":
                pnl_pct = (row["Low"] - t.entry_price) / t.entry_price
                if sl and pnl_pct <= -sl:
                    self._close_trade(t, date, row["Low"], "stop-loss")
                    continue
                pnl_pct_high = (row["High"] - t.entry_price) / t.entry_price
                if tp and pnl_pct_high >= tp:
                    self._close_trade(t, date, row["High"], "take-profit")
                    continue
            else:
                pnl_pct = (t.entry_price - row["High"]) / t.entry_price
                if sl and pnl_pct <= -sl:
                    self._close_trade(t, date, row["High"], "stop-loss")
                    continue
                pnl_pct_tp = (t.entry_price - row["Low"]) / t.entry_price
                if tp and pnl_pct_tp >= tp:
                    self._close_trade(t, date, row["Low"], "take-profit")
                    continue
            remaining.append(t)
        self._open_trades = remaining


# ---------------------------------------------------------------------------
# Result & metrics
# ---------------------------------------------------------------------------

class BacktestResult:
    def __init__(self, config, data, equity_curve, trades):
        self.config = config
        self.data = data
        self.equity_curve = equity_curve
        self.trades = trades
        self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> dict:
        eq = self.equity_curve
        cfg = self.config
        initial = cfg.initial_capital
        final = eq.iloc[-1] if len(eq) else initial

        total_return = (final - initial) / initial
        daily_returns = eq.pct_change().dropna()
        ann_factor = 252

        ann_return = (1 + total_return) ** (ann_factor / max(len(eq), 1)) - 1
        ann_vol = daily_returns.std() * np.sqrt(ann_factor)
        sharpe = (daily_returns.mean() * ann_factor) / (daily_returns.std() * np.sqrt(ann_factor)) if daily_returns.std() else 0

        # Max drawdown
        roll_max = eq.cummax()
        drawdown = (eq - roll_max) / roll_max
        max_dd = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        # Trade stats
        n = len(self.trades)
        winners = [t for t in self.trades if (t.pnl or 0) > 0]
        losers  = [t for t in self.trades if (t.pnl or 0) <= 0]
        win_rate = len(winners) / n if n else 0
        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl for t in losers]) if losers else 0
        profit_factor = (sum(t.pnl for t in winners) / abs(sum(t.pnl for t in losers))
                         if losers and sum(t.pnl for t in losers) != 0 else float("inf"))
        avg_hold = (np.mean([(t.exit_date - t.entry_date).days for t in self.trades
                             if t.exit_date]) if self.trades else 0)

        # Buy-and-hold benchmark
        bh_return = (self.data["Close"].iloc[-1] / self.data["Close"].iloc[0]) - 1

        return dict(
            initial_capital=initial,
            final_equity=final,
            total_return_pct=total_return * 100,
            annualized_return_pct=ann_return * 100,
            annualized_volatility_pct=ann_vol * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd * 100,
            calmar_ratio=calmar,
            total_trades=n,
            win_rate_pct=win_rate * 100,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_hold_days=avg_hold,
            buy_and_hold_return_pct=bh_return * 100,
        )

    def print_summary(self):
        try:
            from tabulate import tabulate
            rows = [(k.replace("_", " ").title(), f"{v:.2f}" if isinstance(v, float) else v)
                    for k, v in self.metrics.items()]
            print("\n" + "=" * 50)
            print(f"  BACKTEST RESULTS - {self.config.symbol}")
            print("=" * 50)
            print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))
        except ImportError:
            print("\n=== BACKTEST RESULTS ===")
            for k, v in self.metrics.items():
                label = k.replace("_", " ").title()
                val = f"{v:.2f}" if isinstance(v, float) else str(v)
                print(f"  {label:<35} {val}")

        if self.trades:
            print(f"\n  Last 5 trades:")
            for t in self.trades[-5:]:
                print(f"    {t.entry_date.date()} -> {t.exit_date.date() if t.exit_date else '?'}"
                      f"  {t.direction.upper():<5}  PnL: {t.pnl:+.2f}  ({t.exit_reason})")

    def plot(self, save_path: Optional[str] = None):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("Install matplotlib to see charts: pip install matplotlib")
            return

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Backtest - {self.config.symbol}  "
                     f"({self.config.start_date} to {self.config.end_date})", fontsize=13)

        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

        # 1. Price + entry/exit markers
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.data.index, self.data["Close"], color="steelblue", linewidth=1, label="Close")
        long_entries  = [t for t in self.trades if t.direction == "long"]
        short_entries = [t for t in self.trades if t.direction == "short"]
        if long_entries:
            ax1.scatter([t.entry_date for t in long_entries],
                        [t.entry_price for t in long_entries],
                        marker="^", color="green", s=60, zorder=5, label="Long entry")
            ax1.scatter([t.exit_date for t in long_entries if t.exit_date],
                        [t.exit_price for t in long_entries if t.exit_price],
                        marker="v", color="red", s=60, zorder=5, label="Long exit")
        if short_entries:
            ax1.scatter([t.entry_date for t in short_entries],
                        [t.entry_price for t in short_entries],
                        marker="v", color="orange", s=60, zorder=5, label="Short entry")
        ax1.set_title("Price + Trade Entries/Exits")
        ax1.legend(fontsize=8)
        ax1.set_ylabel("Price")

        # 2. Equity curve vs buy-and-hold
        ax2 = fig.add_subplot(gs[1, :])
        bh = self.data["Close"] / self.data["Close"].iloc[0] * self.config.initial_capital
        ax2.plot(self.equity_curve.index, self.equity_curve, color="green", linewidth=1.5, label="Strategy")
        ax2.plot(bh.index, bh, color="gray", linewidth=1, linestyle="--", label="Buy & Hold")
        ax2.axhline(self.config.initial_capital, color="black", linewidth=0.8, linestyle=":")
        ax2.set_title("Equity Curve vs Buy & Hold")
        ax2.legend(fontsize=8)
        ax2.set_ylabel("Portfolio Value (₹)")

        # 3. Drawdown
        ax3 = fig.add_subplot(gs[2, 0])
        roll_max = self.equity_curve.cummax()
        dd = (self.equity_curve - roll_max) / roll_max * 100
        ax3.fill_between(dd.index, dd, 0, color="red", alpha=0.4)
        ax3.set_title("Drawdown (%)")
        ax3.set_ylabel("%")

        # 4. Monthly returns heatmap
        ax4 = fig.add_subplot(gs[2, 1])
        monthly = self.equity_curve.resample("ME").last().pct_change().dropna() * 100
        colors = ["red" if r < 0 else "green" for r in monthly]
        ax4.bar(range(len(monthly)), monthly.values, color=colors, alpha=0.7)
        ax4.axhline(0, color="black", linewidth=0.8)
        ax4.set_title("Monthly Returns (%)")
        ax4.set_xticks(range(len(monthly)))
        ax4.set_xticklabels([d.strftime("%b %y") for d in monthly.index], rotation=90, fontsize=6)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Chart saved to {save_path}")
        else:
            plt.show()

    def to_csv(self, path: str):
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        trades_df.to_csv(path, index=False)
        print(f"Trades exported to {path}")


# ---------------------------------------------------------------------------
# Strategy base class — inherit this in strategy.py
# ---------------------------------------------------------------------------

class Strategy:
    """
    Subclass this and implement generate_signals().
    self.data is available as a DataFrame with OHLCV columns.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def generate_signals(self) -> pd.Series:
        """
        Return a pd.Series indexed like self.data with values:
          +1  → enter long (or close short and go long)
          -1  → enter short (or close long and go short)
           0  → flat / close all positions
        """
        raise NotImplementedError("Implement generate_signals() in your Strategy subclass.")


# ---------------------------------------------------------------------------
# Built-in example: Simple Moving Average Crossover
# ---------------------------------------------------------------------------

class SMACrossoverStrategy(Strategy):
    """
    Classic dual-SMA crossover.
    Buy when fast SMA crosses above slow SMA; sell when it crosses below.
    """

    def __init__(self, data: pd.DataFrame, fast: int = 20, slow: int = 50):
        super().__init__(data)
        self.fast = fast
        self.slow = slow

    def generate_signals(self) -> pd.Series:
        close = self.data["Close"]
        sma_fast = close.rolling(self.fast).mean()
        sma_slow = close.rolling(self.slow).mean()
        signal = pd.Series(0, index=self.data.index, dtype=float)
        signal[sma_fast > sma_slow] = 1
        signal[sma_fast <= sma_slow] = 0
        # Only fire on crossover, not every bar — optional: remove to stay in trend
        # crosses = signal.diff().ne(0) & signal.ne(0)
        return signal


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_backtest(
    strategy_cls,
    config: Optional[BacktestConfig] = None,
    strategy_kwargs: Optional[dict] = None,
    plot: bool = True,
    export_csv: bool = False,
):
    if config is None:
        config = BacktestConfig()

    data = fetch_data(config)

    strategy = strategy_cls(data, **(strategy_kwargs or {}))
    signals = strategy.generate_signals()

    engine = Backtester(config)
    engine.data = data
    result = engine.run(signals)

    result.print_summary()

    if plot:
        chart_path = os.path.join(os.path.dirname(__file__), "backtest_chart.png")
        result.plot(save_path=chart_path)

    if export_csv:
        csv_path = os.path.join(os.path.dirname(__file__), "trades.csv")
        result.to_csv(csv_path)

    return result


# ---------------------------------------------------------------------------
# Entry point — edit config here, or import run_backtest from your own script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = BacktestConfig(
        symbol="RELIANCE.NS",       # NSE: RELIANCE.NS  |  BSE: RELIANCE.BO  |  US: AAPL
        start_date="2020-01-01",
        end_date="2024-12-31",
        initial_capital=100_000,
        position_size_pct=0.20,     # 20% of equity per trade
        commission_pct=0.001,       # 0.1% per side (Zerodha-ish)
        slippage_pct=0.0005,
        allow_short=False,
        stop_loss_pct=0.05,         # 5% stop loss
        take_profit_pct=0.15,       # 15% take profit
    )

    # Swap SMACrossoverStrategy for your own strategy class
    run_backtest(
        strategy_cls=SMACrossoverStrategy,
        config=config,
        strategy_kwargs={"fast": 20, "slow": 50},
        plot=True,
        export_csv=True,
    )
