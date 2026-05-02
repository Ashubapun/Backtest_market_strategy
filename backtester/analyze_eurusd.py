import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("FETCHING EUR/USD DATA")
print("=" * 60)

df_d = yf.Ticker("EURUSD=X").history(period="5y", interval="1d", auto_adjust=True)
df_d.index = pd.to_datetime(df_d.index).tz_localize(None)
df_d = df_d[["Open","High","Low","Close","Volume"]].dropna()
print(f"Daily bars : {len(df_d)}  ({df_d.index[0].date()} to {df_d.index[-1].date()})")

df_h = yf.Ticker("EURUSD=X").history(period="730d", interval="1h", auto_adjust=True)
df_h.index = pd.to_datetime(df_h.index).tz_localize(None)
df_h = df_h[["Open","High","Low","Close"]].dropna()
print(f"Hourly bars: {len(df_h)}  ({df_h.index[0].date()} to {df_h.index[-1].date()})")

print()
print("=" * 60)
print("1. BASIC PRICE STATS (DAILY)")
print("=" * 60)
print(f"  Price range   : {df_d['Close'].min():.4f} - {df_d['Close'].max():.4f}")
print(f"  Current price : {df_d['Close'].iloc[-1]:.4f}")
print(f"  5Y return     : {(df_d['Close'].iloc[-1]/df_d['Close'].iloc[0]-1)*100:+.2f}%")
daily_ret = df_d['Close'].pct_change().dropna()
print(f"  Daily vol     : {daily_ret.std()*100:.4f}%")
print(f"  Ann vol       : {daily_ret.std()*np.sqrt(252)*100:.2f}%")
print(f"  Avg daily rng : {((df_d['High']-df_d['Low'])/df_d['Close']).mean()*10000:.1f} pips")
print(f"  Median daily rng: {((df_d['High']-df_d['Low'])/df_d['Close']).median()*10000:.1f} pips")

print()
print("=" * 60)
print("2. TREND ANALYSIS")
print("=" * 60)
df_d['sma20']  = df_d['Close'].rolling(20).mean()
df_d['sma50']  = df_d['Close'].rolling(50).mean()
df_d['sma200'] = df_d['Close'].rolling(200).mean()
above_20  = (df_d['Close'] > df_d['sma20']).mean() * 100
above_50  = (df_d['Close'] > df_d['sma50']).mean() * 100
above_200 = (df_d['Close'] > df_d['sma200']).mean() * 100
print(f"  Days above SMA20  : {above_20:.1f}%")
print(f"  Days above SMA50  : {above_50:.1f}%")
print(f"  Days above SMA200 : {above_200:.1f}%")
up_days = daily_ret > 0
streak = (up_days == up_days.shift(1)).mean() * 100
print(f"  Trend persistence : {streak:.1f}%  (same direction next day)")
big_up   = daily_ret[daily_ret >  0.005]
big_down = daily_ret[daily_ret < -0.005]
next_big_up   = daily_ret.shift(-1)[big_up.index]
next_big_down = daily_ret.shift(-1)[big_down.index]
print(f"  After big UP   -> next day avg: {next_big_up.mean()*10000:+.1f} pips  mean-revert: {(next_big_up<0).mean()*100:.0f}%")
print(f"  After big DOWN -> next day avg: {next_big_down.mean()*10000:+.1f} pips mean-revert: {(next_big_down>0).mean()*100:.0f}%")

print()
print("=" * 60)
print("3. DAY OF WEEK PATTERNS")
print("=" * 60)
df_d['dow']     = df_d.index.day_name()
df_d['ret']     = daily_ret
df_d['abs_rng'] = (df_d['High'] - df_d['Low']) * 10000
dow_stats = df_d.groupby('dow').agg(
    avg_ret_pips=('ret', lambda x: round(x.mean()*10000, 2)),
    win_rate_pct=('ret', lambda x: round((x>0).mean()*100, 1)),
    avg_range_pips=('abs_rng', lambda x: round(x.mean(), 1)),
).reindex(['Monday','Tuesday','Wednesday','Thursday','Friday'])
print(dow_stats.to_string())

print()
print("=" * 60)
print("4. MONTHLY SEASONALITY")
print("=" * 60)
df_d['month'] = df_d.index.month
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
monthly = df_d.groupby('month').agg(
    avg_ret_pips=('ret', lambda x: round(x.mean()*10000, 2)),
    win_rate_pct=('ret', lambda x: round((x>0).mean()*100, 1)),
    avg_range_pips=('abs_rng', lambda x: round(x.mean(), 1)),
)
monthly.index = monthly.index.map(month_names)
print(monthly.to_string())

print()
print("=" * 60)
print("5. SESSION ANALYSIS (HOURLY)")
print("=" * 60)
df_h['hour']    = df_h.index.hour
df_h['ret']     = df_h['Close'].pct_change()
df_h['rng_pip'] = (df_h['High'] - df_h['Low']) * 10000
hour_stats = df_h.groupby('hour').agg(
    avg_range_pip=('rng_pip', lambda x: round(x.mean(), 2)),
    avg_ret_pip  =('ret',     lambda x: round(x.mean()*10000, 3)),
    volatility   =('ret',     lambda x: round(x.std()*10000, 3)),
    win_rate_pct =('ret',     lambda x: round((x>0).mean()*100, 1)),
).round(2)
print(hour_stats.to_string())

print()
print("=" * 60)
print("6. VOLATILITY REGIMES (ATR14)")
print("=" * 60)
df_d['atr14'] = (df_d['High'] - df_d['Low']).rolling(14).mean() * 10000
q33 = df_d['atr14'].quantile(0.33)
q66 = df_d['atr14'].quantile(0.66)
print(f"  Low vol  (ATR14 < {q33:.1f} pips): {(df_d['atr14']<q33).mean()*100:.0f}% of days")
print(f"  Med vol  ({q33:.1f}-{q66:.1f} pips)  : {((df_d['atr14']>=q33)&(df_d['atr14']<q66)).mean()*100:.0f}% of days")
print(f"  High vol (ATR14 > {q66:.1f} pips): {(df_d['atr14']>=q66).mean()*100:.0f}% of days")
low_vol_next = df_d['atr14'].shift(-1)[df_d['atr14'] < q33]
print(f"  After low-vol day -> next ATR: {low_vol_next.mean():.1f} pips  expansion: {(low_vol_next > q33).mean()*100:.0f}%")

print()
print("=" * 60)
print("7. RSI MEAN REVERSION TEST")
print("=" * 60)
delta = df_d['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
rs    = gain / loss.replace(0, np.nan)
df_d['rsi'] = 100 - (100 / (1 + rs))
oversold   = df_d[df_d['rsi'] < 35]
overbought = df_d[df_d['rsi'] > 65]
fwd5_os = df_d['Close'].pct_change(5).shift(-5)[oversold.index]
fwd5_ob = df_d['Close'].pct_change(5).shift(-5)[overbought.index]
print(f"  RSI < 35 (n={len(oversold)})  -> 5d fwd: {fwd5_os.mean()*10000:+.1f} pips  win%: {(fwd5_os>0).mean()*100:.0f}%")
print(f"  RSI > 65 (n={len(overbought)}) -> 5d fwd: {fwd5_ob.mean()*10000:+.1f} pips  win%: {(fwd5_ob>0).mean()*100:.0f}%")

print()
print("=" * 60)
print("8. MACD SIGNAL TEST")
print("=" * 60)
exp12 = df_d['Close'].ewm(span=12).mean()
exp26 = df_d['Close'].ewm(span=26).mean()
macd  = exp12 - exp26
signal_line = macd.ewm(span=9).mean()
df_d['macd_cross'] = np.where(
    (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1)), 1,
    np.where((macd < signal_line) & (macd.shift(1) >= signal_line.shift(1)), -1, 0)
)
bull_cross = df_d[df_d['macd_cross'] == 1]
bear_cross = df_d[df_d['macd_cross'] == -1]
fwd10_bull = df_d['Close'].pct_change(10).shift(-10)[bull_cross.index]
fwd10_bear = df_d['Close'].pct_change(10).shift(-10)[bear_cross.index]
print(f"  Bullish MACD cross (n={len(bull_cross)}) -> 10d fwd: {fwd10_bull.mean()*10000:+.1f} pips  win%: {(fwd10_bull>0).mean()*100:.0f}%")
print(f"  Bearish MACD cross (n={len(bear_cross)}) -> 10d fwd: {fwd10_bear.mean()*10000:+.1f} pips  win%: {(fwd10_bear>0).mean()*100:.0f}%")

print()
print("=" * 60)
print("9. LONDON BREAKOUT TEST (HOURLY)")
print("=" * 60)
df_h['date'] = df_h.index.normalize()
df_h['hour'] = df_h.index.hour
pre_london = df_h[df_h['hour'].between(5, 7)].groupby('date').agg(pre_high=('High','max'), pre_low=('Low','min'))
df_h2 = df_h[df_h['hour'].between(8, 16)].copy()
df_h2 = df_h2.join(pre_london, on='date')
breakout_up   = df_h2[df_h2['High'] > df_h2['pre_high']]
breakout_down = df_h2[df_h2['Low']  < df_h2['pre_low']]
fwd_up   = df_h2['Close'].pct_change(4).shift(-4)[breakout_up.index]
fwd_down = df_h2['Close'].pct_change(4).shift(-4)[breakout_down.index]
print(f"  Pre-London range breakout UP   (n={len(breakout_up)})  -> 4h fwd: {fwd_up.mean()*10000:+.1f} pips   win%: {(fwd_up>0).mean()*100:.0f}%")
print(f"  Pre-London range breakout DOWN (n={len(breakout_down)}) -> 4h fwd: {fwd_down.mean()*10000:+.1f} pips  win%: {(fwd_down<0).mean()*100:.0f}%")

print()
print("=" * 60)
print("10. EMA TREND FILTER ON HOURLY")
print("=" * 60)
df_h['ema50'] = df_h['Close'].ewm(span=50).mean()
df_h['ema200']= df_h['Close'].ewm(span=200).mean()
with_trend  = df_h[df_h['ema50'] > df_h['ema200']]
against_trend=df_h[df_h['ema50'] < df_h['ema200']]
fwd_with  = with_trend['Close'].pct_change(8).shift(-8)
fwd_ag    = against_trend['Close'].pct_change(8).shift(-8)
print(f"  EMA50 > EMA200 (uptrend hours): 8h fwd avg: {fwd_with.mean()*10000:+.1f} pips  win%: {(fwd_with>0).mean()*100:.0f}%")
print(f"  EMA50 < EMA200 (downtrend hrs): 8h fwd avg: {fwd_ag.mean()*10000:+.1f} pips   win%: {(fwd_ag<0).mean()*100:.0f}%")

print()
print("=" * 60)
print("11. ATR-BASED VOLATILITY BREAKOUT")
print("=" * 60)
df_d['atr_prev'] = df_d['atr14'].shift(1)
df_d['open_to_high'] = (df_d['High'] - df_d['Open']) * 10000
df_d['open_to_low']  = (df_d['Open'] - df_d['Low']) * 10000
print(f"  Avg open->high : {df_d['open_to_high'].mean():.1f} pips")
print(f"  Avg open->low  : {df_d['open_to_low'].mean():.1f} pips")
print(f"  When ATR > {q66:.0f} pips (high vol):")
hv = df_d[df_d['atr_prev'] > q66]
print(f"    open->high: {hv['open_to_high'].mean():.1f} pips  open->low: {hv['open_to_low'].mean():.1f} pips")

print()
print("=" * 60)
print("12. MULTI-DAY CONSECUTIVE ANALYSIS")
print("=" * 60)
df_d['direction'] = np.sign(df_d['ret'])
for n in [2, 3, 4]:
    consec_up   = (df_d['direction'].rolling(n).sum() == n)
    consec_down = (df_d['direction'].rolling(n).sum() == -n)
    fwd_cup  = df_d['ret'].shift(-1)[consec_up]
    fwd_cdown= df_d['ret'].shift(-1)[consec_down]
    print(f"  After {n} consec UP days   -> next: {fwd_cup.mean()*10000:+.1f} pips  reversal: {(fwd_cup<0).mean()*100:.0f}%")
    print(f"  After {n} consec DOWN days -> next: {fwd_cdown.mean()*10000:+.1f} pips reversal: {(fwd_cdown>0).mean()*100:.0f}%")

print()
print("ANALYSIS COMPLETE")
