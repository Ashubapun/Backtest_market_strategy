"""
Run your strategy against the backtester.
Edit config below, then: python run_my_strategy.py
"""

from backtester import BacktestConfig, run_backtest
from strategy import MyStrategy


config = BacktestConfig(
    symbol="RELIANCE.NS",       # NSE suffix: .NS  |  BSE: .BO  |  US stocks: no suffix
    start_date="2020-01-01",
    end_date="2024-12-31",
    initial_capital=100_000,    # INR
    position_size_pct=0.20,     # 20% of equity per trade
    commission_pct=0.001,       # 0.1% per side
    slippage_pct=0.0005,        # 0.05% per side
    allow_short=False,
    max_open_positions=1,
    stop_loss_pct=0.05,         # 5% hard stop — set None to disable
    take_profit_pct=0.15,       # 15% target — set None to disable
)

result = run_backtest(
    strategy_cls=MyStrategy,
    config=config,
    strategy_kwargs={
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
    },
    plot=True,        # saves backtest_chart.png
    export_csv=True,  # saves trades.csv
)
