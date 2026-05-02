"""
Trading Strategy Backtester — Professional Dashboard
Run: streamlit run app.py
"""

import sys, os, importlib, textwrap, tempfile, math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from backtester import BacktestConfig, Backtester, fetch_data, Strategy

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategy Backtester Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"],
.main, .main > div {
    background: #070b14 !important;
}
.block-container {
    padding: 2rem 2rem 3rem 2rem !important;
    max-width: 1400px;
    background: #070b14 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c1120 !important;
    border-right: 1px solid #2d3a52 !important;
}
[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* All sidebar text brighter */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] input {
    color: #f1f5f9 !important;
    background: #1e293b !important;
}
/* Slider value labels */
[data-testid="stSidebar"] [data-testid="stTickBar"],
[data-testid="stSidebar"] .st-emotion-cache-1xw8zd1 {
    color: #94a3b8 !important;
}

/* Sidebar section headers */
.sb-header {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 2px;
    text-transform: uppercase; color: #60a5fa;
    margin: 22px 0 10px 0; padding-bottom: 7px;
    border-bottom: 1px solid #2d3a52;
}

/* ── Global text overrides ── */
p, span, div, label { color: #e2e8f0; }
.stCaption, [data-testid="stCaptionContainer"] { color: #94a3b8 !important; }

/* ── Top banner ── */
.top-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #3730a3;
    border-radius: 16px; padding: 28px 32px;
    margin-bottom: 24px; position: relative; overflow: hidden;
}
.top-banner::before {
    content: "";
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px; border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
}
.banner-title {
    font-size: 1.7rem; font-weight: 800;
    color: #f8fafc; line-height: 1.2; margin-bottom: 8px;
}
.banner-subtitle { font-size: 0.88rem; color: #cbd5e1; }
.banner-badge {
    display: inline-block; background: #1e293b;
    border: 1px solid #334155; border-radius: 20px;
    padding: 5px 14px; font-size: 0.76rem; color: #cbd5e1;
    margin-right: 8px; margin-top: 10px;
}
.banner-badge span { color: #60a5fa; font-weight: 700; }

/* ── Score ring card ── */
.score-wrapper {
    background: linear-gradient(145deg, #111827, #1a1f35);
    border: 1px solid #2d3a52; border-radius: 16px;
    padding: 24px 16px; text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.score-title {
    font-size: 0.7rem; font-weight: 800; letter-spacing: 2.5px;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 14px;
}

/* ── Metric cards ── */
.mcard {
    background: #111827;
    border: 1px solid #2d3a52; border-radius: 12px;
    padding: 14px 18px; position: relative; overflow: hidden;
}
.mcard::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent, #3b82f6);
}
.mcard-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 5px;
}
.mcard-value { font-size: 1.55rem; font-weight: 800; color: #f8fafc; line-height: 1.1; }
.mcard-sub   { font-size: 0.74rem; color: #94a3b8; margin-top: 4px; font-weight: 500; }

/* ── Section titles ── */
.sec-title {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 2px;
    text-transform: uppercase; color: #94a3b8;
    margin: 28px 0 14px 0; display: flex; align-items: center; gap: 10px;
}
.sec-title::after { content: ""; flex: 1; height: 1px; background: #2d3a52; }

/* ── Score breakdown bars ── */
.breakdown-row {
    display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
}
.breakdown-label {
    font-size: 0.77rem; color: #cbd5e1; min-width: 170px; font-weight: 600;
}
.breakdown-bar-bg {
    flex: 1; height: 9px; background: #1e293b; border-radius: 5px; overflow: hidden;
}
.breakdown-bar-fill {
    height: 100%; border-radius: 5px;
    background: linear-gradient(90deg, var(--bar-color), var(--bar-color-light));
}
.breakdown-val {
    font-size: 0.8rem; color: #f1f5f9; font-weight: 800;
    min-width: 38px; text-align: right;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0c1120 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid #2d3a52 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #1e293b !important;
    color: #f8fafc !important;
}

/* ── Info pill ── */
.info-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0f172a; border: 1px solid #2d3a52;
    border-radius: 8px; padding: 8px 14px;
    font-size: 0.8rem; color: #cbd5e1; margin-bottom: 8px; font-weight: 500;
}
.info-pill b { color: #60a5fa; }

/* ── Stat boxes ── */
.stat-box {
    background: #111827; border: 1px solid #2d3a52;
    border-radius: 10px; padding: 16px; text-align: center;
}
.stat-box-val  { font-size: 1.35rem; font-weight: 800; color: #f1f5f9; }
.stat-box-lbl  { font-size: 0.7rem; color: #94a3b8; letter-spacing: 1px;
                 text-transform: uppercase; margin-top: 4px; font-weight: 600; }

/* ── Run button ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
    color: #ffffff !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 800 !important; letter-spacing: 0.5px !important;
    font-size: 0.92rem !important; padding: 12px !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.35) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb, #4f46e5) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.5) !important;
}

/* ── Streamlit widget text ── */
[data-testid="stMetric"] { background: #111827; border-radius: 10px; padding: 12px; }
[data-testid="stMetric"] label { color: #94a3b8 !important; font-weight: 600 !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
.stDataFrame { background: #111827 !important; }
div[data-testid="stVerticalBlock"] > div { background: transparent; }

/* Radio & checkbox labels */
.stRadio label span, .stCheckbox label span { color: #cbd5e1 !important; font-weight: 500 !important; }
/* Slider labels */
[data-testid="stSlider"] label { color: #cbd5e1 !important; font-weight: 600 !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] .st-emotion-cache-1xw8zd1 { color: #60a5fa !important; font-weight: 700 !important; }
/* Number input label */
[data-testid="stNumberInput"] label { color: #cbd5e1 !important; font-weight: 600 !important; }
/* Text input label */
[data-testid="stTextInput"] label { color: #cbd5e1 !important; font-weight: 600 !important; }
/* Text area label */
[data-testid="stTextArea"] label { color: #cbd5e1 !important; font-weight: 600 !important; }
/* Toggle */
[data-testid="stToggle"] label span { color: #cbd5e1 !important; font-weight: 500 !important; }
/* Error boxes */
[data-testid="stException"] { border: 1px solid #7f1d1d !important; background: #1a0a0a !important; }
[data-testid="stException"] p { color: #fca5a5 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────────────────────

CIRC = 2 * math.pi * 52   # SVG ring circumference (r=52)

GRADE_CFG = [
    (85, "Excellent", "#10b981", "#059669"),
    (70, "Strong",    "#22c55e", "#16a34a"),
    (55, "Moderate",  "#f59e0b", "#d97706"),
    (40, "Weak",      "#f97316", "#ea580c"),
    (0,  "Poor",      "#ef4444", "#dc2626"),
]

def grade(score):
    for t, lbl, c1, c2 in GRADE_CFG:
        if score >= t:
            return lbl, c1, c2
    return "Poor", "#ef4444", "#dc2626"

def clamp(v, lo, hi): return max(lo, min(hi, v))

def strategy_score(m):
    s_sharpe = clamp((m["sharpe_ratio"] + 1) / 4 * 100, 0, 100)
    s_win    = clamp(m["win_rate_pct"], 0, 100)
    s_pf     = clamp(m["profit_factor"] / 3 * 100, 0, 100)
    rel      = m["total_return_pct"] - m["buy_and_hold_return_pct"]
    s_rel    = clamp((rel + 50) / 100 * 100, 0, 100)
    s_dd     = clamp((1 + m["max_drawdown_pct"] / 20) * 100, 0, 100)
    return round(0.30*s_sharpe + 0.25*s_win + 0.20*s_pf + 0.15*s_rel + 0.10*s_dd, 1)

def pct_color(val, good_above=0):
    return "#10b981" if val >= good_above else "#ef4444"

def score_ring_svg(score, color, color2, label):
    filled   = CIRC * score / 100
    empty    = CIRC - filled
    gradient = f"url(#rg_{label.lower()[:3]})"
    return f"""
    <svg viewBox="0 0 130 130" width="150" height="150" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="rg_{label.lower()[:3]}" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%"   stop-color="{color2}"/>
          <stop offset="100%" stop-color="{color}"/>
        </linearGradient>
      </defs>
      <!-- track -->
      <circle cx="65" cy="65" r="52" fill="none"
              stroke="#1e293b" stroke-width="10"/>
      <!-- progress -->
      <circle cx="65" cy="65" r="52" fill="none"
              stroke="{gradient}" stroke-width="10"
              stroke-linecap="round"
              stroke-dasharray="{filled:.1f} {empty:.1f}"
              transform="rotate(-90 65 65)"/>
      <!-- score text -->
      <text x="65" y="58" text-anchor="middle"
            font-size="24" font-weight="800" fill="{color}"
            font-family="Inter,sans-serif">{score}%</text>
      <!-- grade label -->
      <text x="65" y="76" text-anchor="middle"
            font-size="9.5" font-weight="700" fill="#64748b"
            font-family="Inter,sans-serif" letter-spacing="1.5">{label.upper()}</text>
    </svg>"""

def mcard(label, value_html, sub="", accent="#3b82f6"):
    return f"""
    <div class="mcard" style="--accent:{accent};">
      <div class="mcard-label">{label}</div>
      <div class="mcard-value">{value_html}</div>
      {'<div class="mcard-sub">' + sub + '</div>' if sub else ''}
    </div>"""

def breakdown_bar(label, val, color, color_light):
    return f"""
    <div class="breakdown-row">
      <div class="breakdown-label">{label}</div>
      <div class="breakdown-bar-bg">
        <div class="breakdown-bar-fill"
             style="width:{val:.0f}%;--bar-color:{color};--bar-color-light:{color_light};"></div>
      </div>
      <div class="breakdown-val">{val:.0f}</div>
    </div>"""

def load_strategy(code):
    tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8")
    tmp.write("import pandas as pd\nimport numpy as np\nfrom backtester import Strategy\n\n")
    tmp.write(textwrap.dedent(code.encode("utf-8", errors="replace").decode("utf-8")))
    tmp.close()
    spec = importlib.util.spec_from_file_location("user_strategy", tmp.name)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    classes = [v for v in vars(mod).values()
               if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy]
    if not classes:
        raise ValueError("No Strategy subclass found — make sure your class inherits from Strategy.")
    return classes[-1]

CHART_LAYOUT = dict(
    paper_bgcolor="#070b14", plot_bgcolor="#0c1120",
    font=dict(color="#94a3b8", family="Inter,sans-serif", size=11),
    legend=dict(bgcolor="#111827", bordercolor="#1e293b", borderwidth=1,
                font_size=11, orientation="h", y=1.06),
    margin=dict(l=8, r=8, t=32, b=8),
    xaxis=dict(gridcolor="#1e293b", showgrid=True, zeroline=False,
               linecolor="#1e293b", tickcolor="#475569"),
    yaxis=dict(gridcolor="#1e293b", showgrid=True, zeroline=False,
               linecolor="#1e293b", tickcolor="#475569"),
    hovermode="x unified",
)

GOLDEN_SETUP_CODE = '''\
class GoldenSetupStrategy(Strategy):
    """
    Golden Setup - Round Number Breakout
    Long  : HIGH crosses the nearest round level above open
    Short : LOW  crosses the nearest round level below open
    """

    def __init__(self, data, mode="intraday", custom_step=None):
        super().__init__(data)
        self.mode = mode
        base = {"intraday": 500, "positional": 1000}[mode]
        self._cfg = {
            "step": custom_step if custom_step else base,
            "freq": "D" if mode == "intraday" else "W",
        }

    def generate_signals(self):
        df   = self.data.copy()
        step = self._cfg["step"]
        freq = self._cfg["freq"]

        period_groups = df.index.to_period(freq)
        period_open   = df["Open"].groupby(period_groups).transform("first")

        upper_trigger = (period_open // step + 1) * step
        lower_trigger = (period_open // step)       * step

        signal = pd.Series(0, index=df.index, dtype=float)
        signal[df["Low"]  <= lower_trigger] = -1
        signal[df["High"] >= upper_trigger] =  1
        signal = signal.replace(0, np.nan).ffill().fillna(0)
        return signal
'''

# ── Market configs ────────────────────────────────────────────
# step = round-number level size for Golden Setup
# sl_def/tp_def = default stop/take-profit %
# interval = yfinance bar size

MARKET_CFG = {
    # Indian indices
    "^NSEI":       {"label":"NIFTY 50",        "step_i":500,   "step_p":1000,  "sl":2.0, "tp":4.0, "interval":"1d",  "currency":"INR"},
    "^NSEBANK":    {"label":"Bank NIFTY",       "step_i":500,   "step_p":1000,  "sl":2.0, "tp":4.0, "interval":"1d",  "currency":"INR"},
    "^CNXIT":      {"label":"NIFTY IT",         "step_i":200,   "step_p":500,   "sl":2.0, "tp":4.0, "interval":"1d",  "currency":"INR"},
    # Indian stocks — generic small step
    "_INDIAN_STK": {"label":"Indian Stock",     "step_i":50,    "step_p":100,   "sl":2.0, "tp":5.0, "interval":"1d",  "currency":"INR"},
    # Forex  (intraday uses 1h bars so Asia/London sessions are visible)
    "EURUSD=X":    {"label":"EUR / USD",        "step_i":0.005, "step_p":0.01,  "sl":1.0, "tp":2.0, "interval":"1h",  "currency":"USD"},
    "GBPUSD=X":    {"label":"GBP / USD",        "step_i":0.005, "step_p":0.01,  "sl":1.0, "tp":2.0, "interval":"1h",  "currency":"USD"},
    "USDJPY=X":    {"label":"USD / JPY",        "step_i":0.5,   "step_p":1.0,   "sl":1.0, "tp":2.0, "interval":"1h",  "currency":"JPY"},
    "GBPINR=X":    {"label":"GBP / INR",        "step_i":1.0,   "step_p":2.0,   "sl":1.0, "tp":2.0, "interval":"1h",  "currency":"INR"},
    "USDINR=X":    {"label":"USD / INR",        "step_i":0.5,   "step_p":1.0,   "sl":1.0, "tp":2.0, "interval":"1h",  "currency":"INR"},
    "AUDUSD=X":    {"label":"AUD / USD",        "step_i":0.005, "step_p":0.01,  "sl":1.0, "tp":2.0, "interval":"1h",  "currency":"USD"},
    # US indices
    "^GSPC":       {"label":"S&P 500",          "step_i":50,    "step_p":100,   "sl":2.0, "tp":4.0, "interval":"1d",  "currency":"USD"},
    "^DJI":        {"label":"Dow Jones",        "step_i":100,   "step_p":500,   "sl":2.0, "tp":4.0, "interval":"1d",  "currency":"USD"},
    "^IXIC":       {"label":"NASDAQ",           "step_i":50,    "step_p":100,   "sl":2.0, "tp":4.0, "interval":"1d",  "currency":"USD"},
    # US stocks — generic
    "_US_STK":     {"label":"US Stock",         "step_i":5,     "step_p":10,    "sl":2.0, "tp":5.0, "interval":"1d",  "currency":"USD"},
    # Crypto
    "BTC-USD":     {"label":"Bitcoin / USD",    "step_i":500,   "step_p":1000,  "sl":3.0, "tp":6.0, "interval":"1d",  "currency":"USD"},
    "ETH-USD":     {"label":"Ethereum / USD",   "step_i":50,    "step_p":100,   "sl":3.0, "tp":6.0, "interval":"1d",  "currency":"USD"},
    "SOL-USD":     {"label":"Solana / USD",     "step_i":5,     "step_p":10,    "sl":3.0, "tp":6.0, "interval":"1d",  "currency":"USD"},
    # Custom
    "_CUSTOM":     {"label":"Custom",           "step_i":50,    "step_p":100,   "sl":2.0, "tp":4.0, "interval":"1d",  "currency":""},
}


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:0 4px 16px 4px;">
      <div style="font-size:1.15rem;font-weight:800;color:#f1f5f9;">📈 Backtester Pro</div>
      <div style="font-size:0.72rem;color:#94a3b8;margin-top:2px;">Golden Setup · Any Market</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Market type ───────────────────────────────────
    st.markdown('<div class="sb-header">1 · Market</div>', unsafe_allow_html=True)
    market_type = st.radio(
        "Market", ["🇮🇳  Indian Markets", "🌍  Foreign Markets"],
        label_visibility="collapsed",
    )

    # ── Step 2: Instrument selection ──────────────────────────
    st.markdown('<div class="sb-header">2 · Instrument</div>', unsafe_allow_html=True)

    if market_type == "🇮🇳  Indian Markets":
        instrument_cat = st.radio(
            "Instrument type", ["Indices", "Stocks"],
            horizontal=True, label_visibility="collapsed",
        )
        if instrument_cat == "Indices":
            index_map = {
                "NIFTY 50  (^NSEI)":      "^NSEI",
                "Bank NIFTY  (^NSEBANK)": "^NSEBANK",
                "NIFTY IT  (^CNXIT)":     "^CNXIT",
            }
            chosen = st.selectbox("Select Index", list(index_map.keys()),
                                  label_visibility="collapsed")
            symbol   = index_map[chosen]
            mkey     = symbol
        else:
            stock_map = {
                "-- Type custom symbol --": "",
                "Reliance  (RELIANCE.NS)":  "RELIANCE.NS",
                "TCS  (TCS.NS)":            "TCS.NS",
                "HDFC Bank  (HDFCBANK.NS)": "HDFCBANK.NS",
                "Infosys  (INFY.NS)":       "INFY.NS",
                "ICICI Bank  (ICICIBANK.NS)":"ICICIBANK.NS",
                "Wipro  (WIPRO.NS)":        "WIPRO.NS",
                "Bajaj Finance  (BAJFINANCE.NS)":"BAJFINANCE.NS",
                "SBI  (SBIN.NS)":           "SBIN.NS",
            }
            chosen = st.selectbox("Select Stock", list(stock_map.keys()),
                                  label_visibility="collapsed")
            if chosen == "-- Type custom symbol --":
                symbol = st.text_input("Symbol (e.g. TATASTEEL.NS)", value="",
                                       label_visibility="collapsed",
                                       placeholder="e.g. TATASTEEL.NS")
            else:
                symbol = stock_map[chosen]
            mkey = "_INDIAN_STK"

    else:  # Foreign Markets
        foreign_cat = st.radio(
            "Category",
            ["Forex", "US Indices", "US Stocks", "Crypto", "Custom"],
            label_visibility="collapsed",
        )

        if foreign_cat == "Forex":
            forex_map = {
                "EUR / USD  (EURUSD=X)":  "EURUSD=X",
                "GBP / USD  (GBPUSD=X)":  "GBPUSD=X",
                "USD / JPY  (USDJPY=X)":  "USDJPY=X",
                "USD / INR  (USDINR=X)":  "USDINR=X",
                "GBP / INR  (GBPINR=X)":  "GBPINR=X",
                "AUD / USD  (AUDUSD=X)":  "AUDUSD=X",
            }
            chosen = st.selectbox("Select Pair", list(forex_map.keys()),
                                  label_visibility="collapsed")
            symbol = forex_map[chosen]
            mkey   = symbol

        elif foreign_cat == "US Indices":
            us_idx_map = {
                "S&P 500  (^GSPC)":   "^GSPC",
                "Dow Jones  (^DJI)":  "^DJI",
                "NASDAQ  (^IXIC)":    "^IXIC",
            }
            chosen = st.selectbox("Select Index", list(us_idx_map.keys()),
                                  label_visibility="collapsed")
            symbol = us_idx_map[chosen]
            mkey   = symbol

        elif foreign_cat == "US Stocks":
            us_stk_map = {
                "-- Type custom symbol --": "",
                "Apple  (AAPL)":    "AAPL",
                "Tesla  (TSLA)":    "TSLA",
                "Microsoft  (MSFT)":"MSFT",
                "NVIDIA  (NVDA)":   "NVDA",
                "Amazon  (AMZN)":   "AMZN",
                "Google  (GOOGL)":  "GOOGL",
                "Meta  (META)":     "META",
            }
            chosen = st.selectbox("Select Stock", list(us_stk_map.keys()),
                                  label_visibility="collapsed")
            if chosen == "-- Type custom symbol --":
                symbol = st.text_input("Symbol (e.g. AAPL)", value="",
                                       label_visibility="collapsed",
                                       placeholder="e.g. AAPL")
            else:
                symbol = us_stk_map[chosen]
            mkey = "_US_STK"

        elif foreign_cat == "Crypto":
            crypto_map = {
                "Bitcoin  (BTC-USD)":  "BTC-USD",
                "Ethereum  (ETH-USD)": "ETH-USD",
                "Solana  (SOL-USD)":   "SOL-USD",
            }
            chosen = st.selectbox("Select Crypto", list(crypto_map.keys()),
                                  label_visibility="collapsed")
            symbol = crypto_map[chosen]
            mkey   = symbol

        else:  # Custom
            symbol = st.text_input("Enter any Yahoo Finance symbol", value="",
                                   label_visibility="collapsed",
                                   placeholder="e.g. GC=F (Gold), CL=F (Oil)")
            mkey = "_CUSTOM"

    # Resolve market config
    mcfg = MARKET_CFG.get(mkey, MARKET_CFG["_CUSTOM"])

    # Show selected symbol pill
    if symbol:
        st.markdown(f"""
        <div class="info-pill" style="margin-top:6px;">
          <b style="color:#10b981;">Selected:</b>&nbsp; {symbol}
          &nbsp;·&nbsp; {mcfg['label']}
        </div>
        """, unsafe_allow_html=True)

    # ── Step 3: Test period ───────────────────────────────────
    st.markdown('<div class="sb-header">3 · Test Period</div>', unsafe_allow_html=True)
    period_choice = st.radio("Duration", ["6 Months", "1 Year", "Custom"],
                             horizontal=True, label_visibility="collapsed")
    today = date.today()
    if period_choice == "6 Months":
        start_dt, end_dt = today - timedelta(days=182), today
    elif period_choice == "1 Year":
        start_dt, end_dt = today - timedelta(days=365), today
    else:
        col_a, col_b = st.columns(2)
        start_dt = col_a.date_input("From", value=today - timedelta(days=365))
        end_dt   = col_b.date_input("To",   value=today)

    # ── Step 4: Golden Setup mode ─────────────────────────────
    st.markdown('<div class="sb-header">4 · Strategy Mode</div>', unsafe_allow_html=True)
    gs_mode = st.radio("Mode", ["intraday", "positional"],
                       horizontal=True, label_visibility="collapsed")

    # Auto step size from market config
    auto_step = mcfg["step_i"] if gs_mode == "intraday" else mcfg["step_p"]
    bar_interval = mcfg["interval"]
    currency_lbl = mcfg.get("currency", "")

    st.markdown(f"""
    <div class="info-pill">
      {'⚡' if gs_mode == 'intraday' else '📅'}&nbsp;
      <b>Round Level Step:</b> {auto_step} &nbsp;·&nbsp;
      <b>Interval:</b> {bar_interval}
    </div>
    """, unsafe_allow_html=True)

    # ── Step 5: Capital & Risk ────────────────────────────────
    st.markdown('<div class="sb-header">5 · Capital & Risk</div>', unsafe_allow_html=True)
    cap_label = f"Initial Capital ({currency_lbl})" if currency_lbl else "Initial Capital"
    capital   = st.number_input(cap_label, value=100_000, step=10_000, min_value=1_000)
    pos_size  = st.slider("Position Size per Trade", 5, 100, 20, step=5, format="%d%%") / 100
    commission= st.slider("Commission (per side)", 0.0, 0.5, 0.1, step=0.05, format="%.2f%%") / 100
    allow_short = st.toggle("Allow Short Selling", value=True)

    col_sl, col_tp = st.columns(2)
    sl_on = col_sl.checkbox("Stop Loss",   value=True)
    tp_on = col_tp.checkbox("Take Profit", value=True)
    sl_pct = st.slider("Stop Loss %",   0.5, 10.0, float(mcfg["sl"]), 0.5,
                        format="%.1f%%") / 100 if sl_on else None
    tp_pct = st.slider("Take Profit %", 0.5, 15.0, float(mcfg["tp"]), 0.5,
                        format="%.1f%%") / 100 if tp_on else None

    # ── Strategy code ─────────────────────────────────────────
    st.markdown('<div class="sb-header">Strategy Code</div>', unsafe_allow_html=True)
    strategy_code = st.text_area(
        "code", value=GOLDEN_SETUP_CODE, height=240, label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Idle / landing state
# ─────────────────────────────────────────────────────────────

if not run_btn:
    # ── Hero banner ───────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 50%,#0f172a 100%);
         border:1px solid #312e81; border-radius:16px; padding:36px 40px; margin-bottom:28px;">
      <div style="font-size:2rem; font-weight:800; color:#f1f5f9; margin-bottom:8px;">
        📈 Strategy Backtester Pro
      </div>
      <div style="font-size:0.95rem; color:#94a3b8; margin-bottom:16px;">
        Test your trading strategy on real NSE/BSE market data &nbsp;·&nbsp;
        Get a <b style="color:#3b82f6;">0–100% performance score</b> &nbsp;·&nbsp;
        Analyse every single trade
      </div>
      <div>
        <span style="background:#1e293b;border:1px solid #334155;border-radius:20px;
              padding:5px 16px;font-size:0.78rem;color:#94a3b8;margin-right:8px;">
          <b style="color:#3b82f6;">^NSEI</b> NIFTY 50
        </span>
        <span style="background:#1e293b;border:1px solid #334155;border-radius:20px;
              padding:5px 16px;font-size:0.78rem;color:#94a3b8;margin-right:8px;">
          <b style="color:#3b82f6;">^NSEBANK</b> BankNIFTY
        </span>
        <span style="background:#1e293b;border:1px solid #334155;border-radius:20px;
              padding:5px 16px;font-size:0.78rem;color:#94a3b8;margin-right:8px;">
          <b style="color:#10b981;">500 / 1000 pt</b> Round Levels
        </span>
        <span style="background:#1e293b;border:1px solid #334155;border-radius:20px;
              padding:5px 16px;font-size:0.78rem;color:#94a3b8;">
          <b style="color:#f59e0b;">Golden Setup</b> Strategy
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score preview (what you'll get) ───────────────────────
    st.markdown("""
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
         color:#475569;margin-bottom:14px;">WHAT YOU WILL SEE AFTER RUNNING</div>
    """, unsafe_allow_html=True)

    prev_c1, prev_c2, prev_c3, prev_c4 = st.columns(4)
    for col, icon, label, val, color, desc in [
        (prev_c1, "🎯", "Strategy Score",   "0–100%",  "#3b82f6", "Composite performance grade"),
        (prev_c2, "✅", "Win Rate",          "X%",      "#10b981", "% of profitable trades"),
        (prev_c3, "📊", "Profit Factor",     "X.XX",    "#f59e0b", "Gross win ÷ Gross loss"),
        (prev_c4, "📉", "Max Drawdown",      "–X%",     "#ef4444", "Worst peak-to-trough loss"),
    ]:
        col.markdown(f"""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:14px;
                    padding:20px 18px;text-align:center;">
          <div style="font-size:1.6rem;margin-bottom:6px;">{icon}</div>
          <div style="font-size:1.4rem;font-weight:800;color:{color};">{val}</div>
          <div style="font-size:0.72rem;font-weight:700;color:#f1f5f9;margin-top:4px;">{label}</div>
          <div style="font-size:0.68rem;color:#475569;margin-top:3px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Step-by-step guide ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2 = st.columns([1.4, 1])

    with g1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:14px;padding:28px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
               color:#475569;margin-bottom:18px;">HOW TO RUN YOUR FIRST BACKTEST</div>

          <div style="display:flex;gap:14px;margin-bottom:16px;align-items:flex-start;">
            <div style="background:#1e3a5f;color:#3b82f6;font-size:0.8rem;font-weight:800;
                 width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
                 justify-content:center;flex-shrink:0;">1</div>
            <div>
              <div style="color:#f1f5f9;font-weight:600;font-size:0.85rem;">Choose a Symbol</div>
              <div style="color:#64748b;font-size:0.78rem;margin-top:2px;">
                Try <code style="background:#1e293b;padding:1px 6px;border-radius:3px;color:#94a3b8;">^NSEI</code>
                for NIFTY 50 or
                <code style="background:#1e293b;padding:1px 6px;border-radius:3px;color:#94a3b8;">^NSEBANK</code>
                for Bank NIFTY — these work best with 500pt round levels.
              </div>
            </div>
          </div>

          <div style="display:flex;gap:14px;margin-bottom:16px;align-items:flex-start;">
            <div style="background:#1e3a5f;color:#3b82f6;font-size:0.8rem;font-weight:800;
                 width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
                 justify-content:center;flex-shrink:0;">2</div>
            <div>
              <div style="color:#f1f5f9;font-weight:600;font-size:0.85rem;">Select 6 Months or 1 Year</div>
              <div style="color:#64748b;font-size:0.78rem;margin-top:2px;">
                Start with <b style="color:#f1f5f9;">1 Year</b> to get statistically meaningful results.
              </div>
            </div>
          </div>

          <div style="display:flex;gap:14px;margin-bottom:16px;align-items:flex-start;">
            <div style="background:#1e3a5f;color:#3b82f6;font-size:0.8rem;font-weight:800;
                 width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
                 justify-content:center;flex-shrink:0;">3</div>
            <div>
              <div style="color:#f1f5f9;font-weight:600;font-size:0.85rem;">Pick Intraday or Positional</div>
              <div style="color:#64748b;font-size:0.78rem;margin-top:2px;">
                <b style="color:#f1f5f9;">Intraday</b> = 500pt levels, daily bars. &nbsp;
                <b style="color:#f1f5f9;">Positional</b> = 1000pt levels, weekly bars.
              </div>
            </div>
          </div>

          <div style="display:flex;gap:14px;align-items:flex-start;">
            <div style="background:#1a3a1a;color:#10b981;font-size:0.8rem;font-weight:800;
                 width:28px;height:28px;border-radius:50%;display:flex;align-items:center;
                 justify-content:center;flex-shrink:0;">4</div>
            <div>
              <div style="color:#10b981;font-weight:700;font-size:0.85rem;">Click ▶ Run Backtest</div>
              <div style="color:#64748b;font-size:0.78rem;margin-top:2px;">
                Results appear in ~5 seconds — score ring, metrics, charts, and full trade log.
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:14px;padding:28px;height:100%;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
               color:#475569;margin-bottom:18px;">SCORE INTERPRETATION</div>

          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="background:#10b981;width:10px;height:10px;border-radius:50%;flex-shrink:0;"></div>
              <div style="background:#1a3a1a;border-radius:20px;padding:4px 14px;
                   font-size:0.78rem;font-weight:700;color:#10b981;min-width:90px;text-align:center;">85–100%</div>
              <div style="color:#64748b;font-size:0.78rem;">Excellent — strategy is working well</div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="background:#22c55e;width:10px;height:10px;border-radius:50%;flex-shrink:0;"></div>
              <div style="background:#1a3a1a;border-radius:20px;padding:4px 14px;
                   font-size:0.78rem;font-weight:700;color:#22c55e;min-width:90px;text-align:center;">70–85%</div>
              <div style="color:#64748b;font-size:0.78rem;">Strong — good risk-adjusted returns</div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="background:#f59e0b;width:10px;height:10px;border-radius:50%;flex-shrink:0;"></div>
              <div style="background:#3d2c00;border-radius:20px;padding:4px 14px;
                   font-size:0.78rem;font-weight:700;color:#f59e0b;min-width:90px;text-align:center;">55–70%</div>
              <div style="color:#64748b;font-size:0.78rem;">Moderate — needs tuning</div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="background:#f97316;width:10px;height:10px;border-radius:50%;flex-shrink:0;"></div>
              <div style="background:#3b2200;border-radius:20px;padding:4px 14px;
                   font-size:0.78rem;font-weight:700;color:#f97316;min-width:90px;text-align:center;">40–55%</div>
              <div style="color:#64748b;font-size:0.78rem;">Weak — significant issues</div>
            </div>
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="background:#ef4444;width:10px;height:10px;border-radius:50%;flex-shrink:0;"></div>
              <div style="background:#3b0f0f;border-radius:20px;padding:4px 14px;
                   font-size:0.78rem;font-weight:700;color:#ef4444;min-width:90px;text-align:center;">0–40%</div>
              <div style="color:#64748b;font-size:0.78rem;">Poor — strategy is losing money</div>
            </div>
          </div>

          <div style="margin-top:20px;padding-top:16px;border-top:1px solid #1e293b;
               font-size:0.75rem;color:#475569;line-height:1.6;">
            Score = <b style="color:#3b82f6;">30%</b> Sharpe +
            <b style="color:#10b981;">25%</b> Win Rate +
            <b style="color:#8b5cf6;">20%</b> Profit Factor +
            <b style="color:#f59e0b;">15%</b> vs B&H +
            <b style="color:#06b6d4;">10%</b> Low Drawdown
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Quick symbols bar ──────────────────────────────────────
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #1e3a5f;border-radius:12px;
                padding:14px 20px;margin-top:20px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
      <span style="font-size:0.72rem;color:#3b82f6;font-weight:700;letter-spacing:1px;">
        QUICK TEST SYMBOLS
      </span>
      <span style="color:#1e293b;">|</span>
      <code style="background:#1e293b;padding:3px 10px;border-radius:6px;color:#94a3b8;font-size:0.75rem;">^NSEI</code>
      <code style="background:#1e293b;padding:3px 10px;border-radius:6px;color:#94a3b8;font-size:0.75rem;">^NSEBANK</code>
      <code style="background:#1e293b;padding:3px 10px;border-radius:6px;color:#94a3b8;font-size:0.75rem;">RELIANCE.NS</code>
      <code style="background:#1e293b;padding:3px 10px;border-radius:6px;color:#94a3b8;font-size:0.75rem;">HDFCBANK.NS</code>
      <code style="background:#1e293b;padding:3px 10px;border-radius:6px;color:#94a3b8;font-size:0.75rem;">TCS.NS</code>
      <code style="background:#1e293b;padding:3px 10px;border-radius:6px;color:#94a3b8;font-size:0.75rem;">INFY.NS</code>
      <code style="background:#1e293b;padding:3px 10px;border-radius:6px;color:#94a3b8;font-size:0.75rem;">AAPL</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# Load & run
# ─────────────────────────────────────────────────────────────

status = st.empty()

with status.container():
    with st.spinner("Compiling strategy..."):
        try:
            StrategyCls = load_strategy(strategy_code)
        except Exception as e:
            st.error(f"**Strategy error:** {e}")
            st.stop()

config = BacktestConfig(
    symbol=symbol, start_date=str(start_dt), end_date=str(end_dt),
    initial_capital=float(capital), position_size_pct=pos_size,
    commission_pct=commission, slippage_pct=0.0005,
    allow_short=allow_short, stop_loss_pct=sl_pct, take_profit_pct=tp_pct,
    interval=bar_interval,
)

with status.container():
    with st.spinner(f"Fetching {symbol} market data..."):
        try:
            data = fetch_data(config)
        except Exception as e:
            st.error(f"**Data error:** {e}")
            st.stop()

with status.container():
    with st.spinner("Running backtest..."):
        try:
            strategy = StrategyCls(data, mode=gs_mode, custom_step=auto_step)
            signals  = strategy.generate_signals()
            engine   = Backtester(config)
            engine.data = data
            result   = engine.run(signals)
            m        = result.metrics
            score    = strategy_score(m)
            g_label, g_color, g_color2 = grade(score)
        except Exception as e:
            st.error(f"**Backtest error:** {e}")
            st.stop()

status.empty()

# Pre-compute trigger levels
step          = auto_step
freq          = "D" if gs_mode == "intraday" else "W"
period_groups = data.index.to_period(freq)
period_open   = data["Open"].groupby(period_groups).transform("first")
upper_trig    = (period_open // step + 1) * step
lower_trig    = (period_open // step)       * step


# ─────────────────────────────────────────────────────────────
# Results banner
# ─────────────────────────────────────────────────────────────

pnl        = m["final_equity"] - m["initial_capital"]
pnl_sign   = "+" if pnl >= 0 else ""
bh_delta   = m["total_return_pct"] - m["buy_and_hold_return_pct"]
bh_sign    = "+" if bh_delta >= 0 else ""
bh_color   = "#10b981" if bh_delta >= 0 else "#ef4444"
ret_color  = "#10b981" if m["total_return_pct"] >= 0 else "#ef4444"

st.markdown(f"""
<div class="top-banner" style="margin-top:12px;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
    <div>
      <div class="banner-title">{symbol} &nbsp;·&nbsp; {gs_mode.title()} Mode</div>
      <div class="banner-subtitle">
        {start_dt.strftime("%d %b %Y")} to {end_dt.strftime("%d %b %Y")} &nbsp;·&nbsp;
        {len(data):,} bars &nbsp;·&nbsp; {m['total_trades']} trades executed
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:2rem; font-weight:800; color:{ret_color};">
        {m['total_return_pct']:+.2f}%
      </div>
      <div style="font-size:0.75rem; color:{bh_color};">
        {bh_sign}{bh_delta:.2f}% vs Buy & Hold
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Score ring + Metrics
# ─────────────────────────────────────────────────────────────

col_ring, col_m1, col_m2 = st.columns([1, 1.6, 1.6])

# ── Score ring ─────────────────────────────────────────────
with col_ring:
    st.markdown(f"""
    <div class="score-wrapper">
      <div class="score-title">Strategy Score</div>
      {score_ring_svg(score, g_color, g_color2, g_label)}
      <div style="font-size:0.72rem; color:#475569; margin-top:8px; line-height:1.5;">
        Sharpe · Win Rate · Profit Factor<br>vs Buy&Hold · Drawdown
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Left metric column ─────────────────────────────────────
with col_m1:
    win_c  = "#10b981" if m["win_rate_pct"] >= 50 else "#ef4444"
    pf_c   = "#10b981" if m["profit_factor"] >= 1  else "#ef4444"
    sh_c   = "#10b981" if m["sharpe_ratio"]  >= 0  else "#ef4444"
    ann_c  = "#10b981" if m["annualized_return_pct"] >= 0 else "#ef4444"
    pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float("inf") else "∞"

    st.markdown(
        mcard("Win Rate", f'<span style="color:{win_c}">{m["win_rate_pct"]:.1f}%</span>',
              f'{m["total_trades"]} trades total', accent=win_c),
        unsafe_allow_html=True)
    st.markdown(
        mcard("Profit Factor", f'<span style="color:{pf_c}">{pf_str}</span>',
              "Gross win / Gross loss", accent=pf_c),
        unsafe_allow_html=True)
    st.markdown(
        mcard("Sharpe Ratio", f'<span style="color:{sh_c}">{m["sharpe_ratio"]:.2f}</span>',
              "Risk-adjusted return", accent=sh_c),
        unsafe_allow_html=True)

# ── Right metric column ────────────────────────────────────
with col_m2:
    dd_c  = "#ef4444"
    pnl_c = "#10b981" if pnl >= 0 else "#ef4444"

    st.markdown(
        mcard("Net P&L", f'<span style="color:{pnl_c}">{pnl_sign}{pnl:,.0f}</span>',
              f'Capital: {m["initial_capital"]:,.0f}', accent=pnl_c),
        unsafe_allow_html=True)
    st.markdown(
        mcard("Max Drawdown", f'<span style="color:{dd_c}">{m["max_drawdown_pct"]:.2f}%</span>',
              "Peak-to-trough loss", accent="#ef4444"),
        unsafe_allow_html=True)
    st.markdown(
        mcard("Ann. Return", f'<span style="color:{ann_c}">{m["annualized_return_pct"]:+.2f}%</span>',
              f'B&H: {m["buy_and_hold_return_pct"]:+.1f}%', accent=ann_c),
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Score breakdown
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="sec-title">Score Breakdown</div>', unsafe_allow_html=True)

breakdown_items = [
    ("Sharpe Ratio",       clamp((m["sharpe_ratio"]+1)/4*100, 0, 100),        "#3b82f6","#60a5fa"),
    ("Win Rate",           clamp(m["win_rate_pct"], 0, 100),                   "#10b981","#34d399"),
    ("Profit Factor",      clamp(m["profit_factor"]/3*100, 0, 100),            "#8b5cf6","#a78bfa"),
    ("vs Buy & Hold",      clamp(((m["total_return_pct"]-m["buy_and_hold_return_pct"])+50)/100*100,0,100), "#f59e0b","#fbbf24"),
    ("Low Drawdown",       clamp((1+m["max_drawdown_pct"]/20)*100, 0, 100),    "#06b6d4","#22d3ee"),
]
weights = ["30%", "25%", "20%", "15%", "10%"]

bd_cols = st.columns(2)
html_left  = ""
html_right = ""
for i, ((lbl, val, c1, c2), w) in enumerate(zip(breakdown_items, weights)):
    bar_html = breakdown_bar(f"{lbl} <span style='color:#475569;font-size:0.65rem;'>({w})</span>", val, c1, c2)
    if i < 3:
        html_left  += bar_html
    else:
        html_right += bar_html

bd_cols[0].markdown(html_left,  unsafe_allow_html=True)
bd_cols[1].markdown(html_right, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Chart tabs
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="sec-title">Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "  📈  Price & Signals  ",
    "  💹  Equity & Drawdown  ",
    "  📅  Monthly Returns  ",
    "  🔁  Trade Log  ",
])


# ── TAB 1 — Price with round levels & signals ──────────────
with tab1:
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"],
        low=data["Low"],   close=data["Close"],
        name="Price",
        increasing=dict(line_color="#10b981", fillcolor="rgba(16,185,129,0.2)"),
        decreasing=dict(line_color="#ef4444", fillcolor="rgba(239,68,68,0.2)"),
    ))

    # Trigger levels
    fig.add_trace(go.Scatter(
        x=upper_trig.index, y=upper_trig.values,
        mode="lines", name=f"Upper ({step}pt)",
        line=dict(color="#3b82f6", width=1.2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=lower_trig.index, y=lower_trig.values,
        mode="lines", name=f"Lower ({step}pt)",
        line=dict(color="#f59e0b", width=1.2, dash="dot"),
    ))

    # Trade markers
    for trades, sym_e, sym_x, col_e, col_x, lbl in [
        ([t for t in result.trades if t.direction=="long"],
         "triangle-up", "triangle-down", "#10b981", "#ef4444", "Long"),
        ([t for t in result.trades if t.direction=="short"],
         "triangle-down", "triangle-up", "#f59e0b", "#ef4444", "Short"),
    ]:
        if trades:
            fig.add_trace(go.Scatter(
                x=[t.entry_date for t in trades], y=[t.entry_price for t in trades],
                mode="markers", name=f"{lbl} Entry",
                marker=dict(symbol=sym_e, size=11, color=col_e,
                            line=dict(width=1.5, color="#f1f5f9")),
            ))
            fig.add_trace(go.Scatter(
                x=[t.exit_date for t in trades if t.exit_date],
                y=[t.exit_price for t in trades if t.exit_price],
                mode="markers", name=f"{lbl} Exit",
                marker=dict(symbol=sym_x, size=11, color=col_x,
                            line=dict(width=1.5, color="#f1f5f9")),
            ))

    fig.update_layout(**{**CHART_LAYOUT, "height": 460,
                         "xaxis_rangeslider_visible": False})
    st.plotly_chart(fig, use_container_width=True)

    # Signal strip
    sig_colors = signals.map({1: "#10b981", -1: "#ef4444"}).fillna("#1e293b").tolist()
    fig_sig = go.Figure(go.Bar(
        x=signals.index, y=[1]*len(signals),
        marker_color=sig_colors, showlegend=False,
    ))
    fig_sig.update_layout(
        height=36, margin=dict(l=8,r=8,t=0,b=0),
        paper_bgcolor="#070b14", plot_bgcolor="#070b14",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig_sig, use_container_width=True)
    st.caption("🟢 Long active &nbsp;&nbsp; 🔴 Short active &nbsp;&nbsp; ⬛ Flat")


# ── TAB 2 — Equity & Drawdown ──────────────────────────────
with tab2:
    eq       = result.equity_curve
    bh       = data["Close"] / data["Close"].iloc[0] * config.initial_capital
    roll_max = eq.cummax()
    dd       = (eq - roll_max) / roll_max * 100

    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38], vertical_spacing=0.04,
        subplot_titles=("Portfolio Value", "Drawdown"),
    )
    fig2.add_trace(go.Scatter(
        x=bh.index, y=bh, name="Buy & Hold",
        line=dict(color="#475569", width=1.5, dash="dash"),
    ), 1, 1)
    fig2.add_trace(go.Scatter(
        x=eq.index, y=eq, name="Strategy",
        line=dict(color="#3b82f6", width=2.5),
        fill="tonexty", fillcolor="rgba(59,130,246,0.06)",
    ), 1, 1)
    fig2.add_hline(y=config.initial_capital, line_color="#334155",
                   line_width=1, line_dash="dot", row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=dd.index, y=dd, name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="#ef4444", width=1.5),
    ), 2, 1)

    layout2 = {**CHART_LAYOUT, "height": 460}
    layout2.pop("xaxis", None); layout2.pop("yaxis", None)
    fig2.update_layout(**layout2)
    for r in [1, 2]:
        fig2.update_xaxes(gridcolor="#1e293b", row=r, col=1)
        fig2.update_yaxes(gridcolor="#1e293b", row=r, col=1)
    fig2.update_yaxes(ticksuffix="%", row=2, col=1)
    st.plotly_chart(fig2, use_container_width=True)

    avg_dd    = dd[dd < 0].mean()
    under_pct = (dd < -5).sum() / len(dd) * 100
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Max Drawdown",        f"{m['max_drawdown_pct']:.2f}%"),
        (c2, "Avg Drawdown",        f"{avg_dd:.2f}%" if not np.isnan(avg_dd) else "N/A"),
        (c3, "Days >5% Underwater", f"{under_pct:.1f}%"),
        (c4, "Calmar Ratio",        f"{m['calmar_ratio']:.2f}"),
    ]:
        col.markdown(f"""
        <div class="stat-box">
          <div class="stat-box-val" style="color:#f1f5f9">{val}</div>
          <div class="stat-box-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 3 — Monthly returns ────────────────────────────────
with tab3:
    eq         = result.equity_curve
    monthly    = eq.resample("ME").last().pct_change().dropna() * 100
    bar_colors = ["#10b981" if r >= 0 else "#ef4444" for r in monthly.values]

    fig3 = go.Figure(go.Bar(
        x=[d.strftime("%b '%y") for d in monthly.index],
        y=monthly.values,
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{v:+.1f}%" for v in monthly.values],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig3.add_hline(y=0, line_color="#334155", line_width=1.5)
    layout3 = {**CHART_LAYOUT, "height": 340, "showlegend": False}
    layout3.pop("hovermode", None)
    layout3.pop("xaxis", None)
    layout3.pop("yaxis", None)
    fig3.update_layout(**layout3,
        xaxis=dict(gridcolor="#1e293b", tickangle=-45, tickfont_size=10),
        yaxis=dict(gridcolor="#1e293b", ticksuffix="%"),
    )
    st.plotly_chart(fig3, use_container_width=True)

    pos = (monthly > 0).sum()
    neg = (monthly <= 0).sum()
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, color in [
        (c1, "Positive Months", str(int(pos)),           "#10b981"),
        (c2, "Negative Months", str(int(neg)),           "#ef4444"),
        (c3, "Monthly Win %",   f"{pos/len(monthly)*100:.1f}%" if len(monthly) else "N/A", "#3b82f6"),
        (c4, "Best Month",      f"{monthly.max():+.2f}%","#f59e0b"),
    ]:
        col.markdown(f"""
        <div class="stat-box">
          <div class="stat-box-val" style="color:{color}">{val}</div>
          <div class="stat-box-lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)


# ── TAB 4 — Trade log ──────────────────────────────────────
with tab4:
    trades_df = pd.DataFrame([{
        "Entry":       t.entry_date.date() if t.entry_date else None,
        "Exit":        t.exit_date.date()  if t.exit_date  else None,
        "Dir":         t.direction.upper(),
        "Entry Px":    round(t.entry_price, 2),
        "Exit Px":     round(t.exit_price, 2)  if t.exit_price else None,
        "Qty":         round(t.shares, 4),
        "P&L":         round(t.pnl, 2)         if t.pnl is not None else None,
        "P&L %":       round(t.pnl_pct * 100, 2) if t.pnl_pct is not None else None,
        "Hold (d)":    (t.exit_date - t.entry_date).days if t.exit_date else None,
        "Exit":        t.exit_reason,
    } for t in result.trades])

    if not trades_df.empty:
        winners  = trades_df[trades_df["P&L"] > 0]
        losers   = trades_df[trades_df["P&L"] <= 0]
        avg_win  = winners["P&L"].mean() if len(winners) else 0
        avg_loss = losers["P&L"].mean()  if len(losers)  else 0
        rr       = abs(avg_win / avg_loss) if avg_loss else float("inf")

        cL, cR = st.columns(2)
        with cL:
            fig4a = go.Figure(go.Histogram(
                x=trades_df["P&L"].dropna(), nbinsx=20,
                marker_color="#3b82f6",
                marker_line_color="#1d4ed8", marker_line_width=0.8,
            ))
            fig4a.add_vline(x=0, line_color="#ef4444", line_dash="dash", line_width=1.5)
            fig4a.update_layout(**{**CHART_LAYOUT, "height": 280,
                                   "showlegend": False,
                                   "title": dict(text="P&L Distribution",
                                                 font_color="#94a3b8", font_size=12)})
            st.plotly_chart(fig4a, use_container_width=True)

        with cR:
            fig4b = go.Figure(go.Pie(
                labels=["Winners", "Losers"],
                values=[len(winners), len(losers)],
                marker_colors=["#10b981", "#ef4444"],
                hole=0.60,
                textinfo="label+percent",
                textfont_size=11,
            ))
            fig4b.update_layout(
                height=280, paper_bgcolor="#070b14", plot_bgcolor="#070b14",
                font=dict(color="#94a3b8"),
                showlegend=False, margin=dict(l=0,r=0,t=36,b=0),
                title=dict(text=f"Win Rate: {m['win_rate_pct']:.1f}%",
                           font_color="#94a3b8", font_size=12),
            )
            st.plotly_chart(fig4b, use_container_width=True)

        # Stats row
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val, color in [
            (c1, "Win Rate",         f"{m['win_rate_pct']:.1f}%",  "#10b981" if m["win_rate_pct"] >= 50 else "#ef4444"),
            (c2, "Avg Winning Trade",f"{avg_win:+,.0f}",           "#10b981"),
            (c3, "Avg Losing Trade", f"{avg_loss:+,.0f}",          "#ef4444"),
            (c4, "Risk / Reward",    f"1 : {rr:.2f}",              "#3b82f6"),
        ]:
            col.markdown(f"""
            <div class="stat-box">
              <div class="stat-box-val" style="color:{color}">{val}</div>
              <div class="stat-box-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        def style_pnl(val):
            if isinstance(val, (int, float)):
                return f"color: {'#10b981' if val > 0 else '#ef4444'}; font-weight: 600"
            return ""

        st.dataframe(
            trades_df.style.map(style_pnl, subset=["P&L","P&L %"])
                           .set_properties(**{"background-color": "#111827",
                                              "color": "#94a3b8",
                                              "border-color": "#1e293b"}),
            use_container_width=True, height=340,
        )
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#475569;">
          <div style="font-size:2rem;">📭</div>
          <div style="margin-top:8px;">No trades executed in this period.</div>
          <div style="font-size:0.8rem;margin-top:4px;">Try a wider date range or adjusting the strategy.</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; margin-top:40px; padding-top:20px;
     border-top:1px solid #1e293b; font-size:0.72rem; color:#334155;">
  Strategy Backtester Pro &nbsp;·&nbsp;
  Educational purposes only &nbsp;·&nbsp;
  Not financial advice &nbsp;·&nbsp;
  Data via Yahoo Finance
</div>
""", unsafe_allow_html=True)
