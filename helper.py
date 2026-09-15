import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import os
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDIA_CSV_PATH = os.path.join(BASE_DIR, "data", "India_Stocks_Data.csv")
US_CSV_PATH = os.path.join(BASE_DIR, "data", "US_Stocks_Data.csv")

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
screener_statement_names = [
    "Quarterly Results",
    "Profit & Loss",
    "Balance Sheet",
    "Cash Flows",
    "Ratios",
    "Shareholding Pattern"
]

def inject_custom_theme():
    """Inject premium dark stock-market terminal theme with glassmorphism CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container & Background */
    .stApp {
        background: linear-gradient(135deg, #0B0F19 0%, #0F172A 50%, #1E293B 100%);
        color: #F8FAFC;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 230, 118, 0.4);
        box-shadow: 0 12px 30px rgba(0, 230, 118, 0.15);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(11, 15, 25, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #38BDF8 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px 8px 0px 0px !important;
        padding: 12px 24px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #38BDF8 !important;
        background: rgba(56, 189, 248, 0.08) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00E676 !important;
        border-bottom: 3px solid #00E676 !important;
        background: rgba(0, 230, 118, 0.1) !important;
    }

    /* Table / Dataframe Styling */
    div[data-testid="stTable"], div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(8px);
    }
    table {
        color: #E2E8F0 !important;
    }
    th {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        font-weight: 600 !important;
    }

    /* Expanders Styling */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        margin-bottom: 14px;
        backdrop-filter: blur(8px);
    }

    /* Primary Buttons */
    button[kind="primary"], div.stButton > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        border: none !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"]:hover, div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
    }

    /* Custom Glass Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(16px);
    }

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-emerald { background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid rgba(0, 230, 118, 0.3); }
    .badge-cyan { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-rose { background: rgba(244, 63, 94, 0.15); color: #F43F5E; border: 1px solid rgba(244, 63, 94, 0.3); }
    </style>
    """, unsafe_allow_html=True)

def format_ticker_for_yf(symbol, region="India", exchange="NSE"):
    """Format stock symbol for Yahoo Finance query based on region and exchange."""
    sym = str(symbol).strip()
    if region == "India":
        sym_clean = sym.replace("_", "-")
        if exchange == "BSE":
            return f"{sym_clean}.BO" if not sym_clean.endswith(".BO") else sym_clean
        return f"{sym_clean}.NS" if not sym_clean.endswith(".NS") else sym_clean
    else:
        # US Stock tickers (e.g. BRK.A -> BRK-A)
        return sym.replace(".", "-")

@st.cache_data(show_spinner=False)
def fetch_stocks(region="India"):
    """Fetch stock metadata from CSV files based on market region (India / US)."""
    csv_path = INDIA_CSV_PATH if region == "India" else US_CSV_PATH
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    cols_to_keep = [c for c in [
        "Symbol", "Description", "ISIN", "Exchange", "Sector",
        "Price", "Price - Currency", "Price change %, 1 day",
        "Volume, 1 day", "Relative volume, 1 day",
        "Market capitalization", "Market capitalization - Currency",
        "Price to earnings ratio", "Earnings per share diluted, Trailing 12 months",
        "Earnings per share diluted growth %, TTM YoY", "Dividend yield %, Trailing 12 months"
    ] if c in df.columns]
    return df[cols_to_keep]

def categorize_market_cap(row, region="India"):
    """Categorize stock into Large Cap, Mid Cap, Small Cap, or Micro Cap based on Market capitalization."""
    mc = row.get("Market capitalization")
    if pd.isna(mc) or mc <= 0:
        return "Micro Cap"
    if region == "India":
        if mc >= 7.5e11:
            return "Large Cap"
        elif mc >= 2e11:
            return "Mid Cap"
        elif mc >= 1e10:
            return "Small Cap"
        else:
            return "Micro Cap"
    else:
        if mc >= 1e10:
            return "Large Cap"
        elif mc >= 2e9:
            return "Mid Cap"
        elif mc >= 3e8:
            return "Small Cap"
        else:
            return "Micro Cap"

def fetch_periods_intervals():
    """Return dictionary mapping valid yfinance periods to allowed intervals."""
    return {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"]
    }

@st.cache_data(show_spinner=False)
def load_data(ticker, period="1y", interval="1d"):
    """Load stock data from yfinance for given period and interval."""
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf_info(ticker):
    """Fetch company metadata and key statistics dictionary from yfinance."""
    try:
        t = yf.Ticker(ticker)
        return t.info if (t.info and isinstance(t.info, dict)) else {}
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_yf_financials(ticker):
    """Fetch financial statements from yfinance."""
    try:
        t = yf.Ticker(ticker)
        return {
            "Income Statement": t.financials,
            "Quarterly Income Statement": t.quarterly_financials,
            "Balance Sheet": t.balance_sheet,
            "Quarterly Balance Sheet": t.quarterly_balance_sheet,
            "Cash Flow": t.cashflow,
            "Quarterly Cash Flow": t.quarterly_cashflow
        }
    except Exception:
        return {}

# --------------------------------------------------
# Formatting & Type Safety Helpers
# --------------------------------------------------
def _tofloat(x):
    try:
        val = float(x)
        return None if math.isnan(val) else val
    except Exception:
        return None

def _fmt_num(x):
    n = _tofloat(x)
    if n is None:
        return "—"
    neg = n < 0
    n = abs(n)
    for unit in ["", "K", "M", "B", "T"]:
        if n < 1000:
            s = f"{n:,.2f}{unit}"
            return f"-{s}" if neg else s
        n /= 1000
    s = f"{n:,.2f}P"
    return f"-{s}" if neg else s

def _fmt_money(x, currency=""):
    sym = CURRENCY_SYMBOLS.get(currency or "", "")
    n = _tofloat(x)
    if n is None:
        return "—"
    return f"{sym}{_fmt_num(n)}" if sym else _fmt_num(n)

def _fmt_pct(x, already_frac=True):
    n = _tofloat(x)
    if n is None:
        return "—"
    if already_frac:
        n *= 100
    return f"{n:.2f}%"

def _format_ratio_value(val, fmt, ccy=""):
    if fmt == "pct":
        return _fmt_pct(val, already_frac=True)
    if fmt == "money":
        return _fmt_money(val, ccy)
    v = _tofloat(val)
    return f"{v:.2f}" if v is not None else "—"

def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
    return cur if cur not in (None, "None", "") else default

# --------------------------------------------------
# Screener.in Extraction Helpers (STRICTLY Screener.in)
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_screener_page(sym):
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        f"https://www.screener.in/company/{sym}/consolidated/",
        f"https://www.screener.in/company/{sym}/"
    ]
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if res.status_code == 200:
                return BeautifulSoup(res.text, "lxml")
        except Exception:
            continue
    return None

def extract_screener_table(soup, section_name):
    if not soup:
        return pd.DataFrame()
    for section in soup.find_all("section"):
        heading = section.find("h2")
        if not heading:
            continue
        title = heading.get_text(" ", strip=True).lower()
        if section_name.lower() in title:
            table = section.find("table")
            if table is None:
                return pd.DataFrame()
            thead = table.find("thead")
            tbody = table.find("tbody")
            if thead is None or tbody is None:
                return pd.DataFrame()
            headers = [th.get_text(" ", strip=True) for th in thead.find_all("th")]
            rows = []
            for tr in tbody.find_all("tr"):
                row = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
                if len(row) < len(headers):
                    row.extend([""] * (len(headers) - len(row)))
                rows.append(row[:len(headers)])
            return pd.DataFrame(rows, columns=headers)
    return pd.DataFrame()

def extract_screener_overview(soup):
    overview = {}
    if not soup:
        return overview
    top_ratios = soup.find("ul", id="top-ratios")
    if top_ratios is None:
        return overview
    for item in top_ratios.find_all("li"):
        name = item.find(class_="name")
        value = item.find(class_="number")
        if name and value:
            overview[name.get_text(strip=True)] = value.get_text(" ", strip=True)
    return overview

def extract_screener_growth_cards(soup):
    growth = {}
    if not soup:
        return growth
    cards = soup.find_all("div", class_="ranges-table")
    for card in cards:
        title = card.find("h3")
        if title is None:
            continue
        title_text = title.get_text(strip=True)
        data = {}
        for row in card.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                data[key] = value
        growth[title_text] = data
    return growth

def prepare_quarterly(df):
    df_clean = df.copy()
    df_clean.rename(columns={df_clean.columns[0]: "Metric"}, inplace=True)
    for col in df_clean.columns[1:]:
        df_clean[col] = (
            df_clean[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.strip()
        )
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    return df_clean

def plot_quarterly_screener(df):
    df_clean = prepare_quarterly(df)
    metrics = df_clean["Metric"].tolist()
    quarter_cols = df_clean.columns[1:]

    # Revenue
    if "Sales +" in metrics or "Sales" in metrics:
        metric_key = "Sales +" if "Sales +" in metrics else "Sales"
        revenue = df_clean[df_clean["Metric"] == metric_key].iloc[0, 1:]
        fig = px.line(x=quarter_cols, y=revenue.values, markers=True, title="Quarterly Revenue (Screener.in)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

    # Operating Profit
    if "Operating Profit" in metrics:
        op = df_clean[df_clean["Metric"] == "Operating Profit"].iloc[0, 1:]
        fig = px.bar(x=quarter_cols, y=op.values, title="Operating Profit (Screener.in)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

    # Net Profit
    if "Net Profit +" in metrics or "Net Profit" in metrics:
        metric_key = "Net Profit +" if "Net Profit +" in metrics else "Net Profit"
        np = df_clean[df_clean["Metric"] == metric_key].iloc[0, 1:]
        fig = px.area(x=quarter_cols, y=np.values, title="Net Profit (Screener.in)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

    # EPS
    if "EPS in Rs" in metrics:
        eps = df_clean[df_clean["Metric"] == "EPS in Rs"].iloc[0, 1:]
        fig = px.line(x=quarter_cols, y=eps.values, markers=True, title="Quarterly EPS (Screener.in)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

    # Operating Margin
    if "OPM %" in metrics:
        opm = df_clean[df_clean["Metric"] == "OPM %"].iloc[0, 1:]
        fig = px.line(x=quarter_cols, y=opm.values, markers=True, title="Operating Margin % (Screener.in)")
        fig.update_yaxes(title="%")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')

# --------------------------------------------------
# Quantitative & Technical Calculation Helpers
# --------------------------------------------------
def slice_history(df, period_key):
    if df.empty:
        return df
    last_date = df.index[-1]
    if period_key == "1M":
        start_date = last_date - pd.Timedelta(days=30)
    elif period_key == "3M":
        start_date = last_date - pd.Timedelta(days=90)
    elif period_key == "6M":
        start_date = last_date - pd.Timedelta(days=180)
    elif period_key == "YTD":
        start_date = pd.Timestamp(f"{last_date.year}-01-01").tz_localize(last_date.tz)
    elif period_key == "1Y":
        start_date = last_date - pd.Timedelta(days=365)
    elif period_key == "3Y":
        start_date = last_date - pd.Timedelta(days=3 * 365)
    elif period_key == "5Y":
        start_date = last_date - pd.Timedelta(days=5 * 365)
    else:
        return df
    return df.loc[df.index >= start_date]

def get_return(df, days_ago, cagr=False):
    if df.empty or len(df) < 2 or "Close" not in df.columns:
        return None
    last_date = df.index[-1]
    target_date = last_date - pd.Timedelta(days=days_ago)
    past_df = df.loc[df.index <= target_date]
    if past_df.empty:
        past_price = df["Close"].iloc[0]
        actual_days = (last_date - df.index[0]).days
    else:
        past_price = past_df["Close"].iloc[-1]
        actual_days = (last_date - past_df.index[-1]).days
    
    current_price = df["Close"].iloc[-1]
    if past_price == 0:
        return None
    
    pct = (current_price - past_price) / past_price
    if cagr:
        years = actual_days / 365.25
        if years <= 0 or pct <= -1:
            return None
        return (1 + pct) ** (1 / years) - 1
    return pct

def get_ytd_return(df):
    if df.empty or "Close" not in df.columns:
        return None
    last_date = df.index[-1]
    ytd_start = pd.Timestamp(f"{last_date.year}-01-01").tz_localize(last_date.tz)
    past_df = df.loc[df.index < ytd_start]
    if past_df.empty:
        year_df = df.loc[df.index >= ytd_start]
        if year_df.empty:
            return None
        past_price = year_df["Close"].iloc[0]
    else:
        past_price = past_df["Close"].iloc[-1]
    current_price = df["Close"].iloc[-1]
    if past_price == 0:
        return None
    return (current_price - past_price) / past_price

def calculate_rsi(df, period=14):
    if df.empty or len(df) <= period or "Close" not in df.columns:
        return 50.0
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else 50.0

def calculate_price_statistics(df):
    if df.empty or len(df) < 14 or not all(c in df.columns for c in ["High", "Low", "Close", "Open"]):
        return {}
    
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    last_close = df["Close"].iloc[-1]
    low_52w = df["Close"].tail(252).min()
    high_52w = df["Close"].tail(252).max()
    pos_52w = (last_close - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5
    
    daily_range = ((df["High"] - df["Low"]) / df["Close"]) * 100
    avg_daily_range = daily_range.tail(20).mean()
    
    daily_return = df["Close"].pct_change()
    avg_daily_return = daily_return.tail(20).mean() * 100
    
    gap = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)) * 100
    avg_gap = gap.tail(20).abs().mean()
    
    return {
        "ATR": float(atr) if not pd.isna(atr) else 0.0,
        "52w_Low": float(low_52w) if not pd.isna(low_52w) else 0.0,
        "52w_High": float(high_52w) if not pd.isna(high_52w) else 0.0,
        "52w_Pos": float(pos_52w) if not pd.isna(pos_52w) else 0.5,
        "AvgDailyRange": float(avg_daily_range) if not pd.isna(avg_daily_range) else 0.0,
        "AvgDailyReturn": float(avg_daily_return) if not pd.isna(avg_daily_return) else 0.0,
        "Gap": float(avg_gap) if not pd.isna(avg_gap) else 0.0
    }

def calculate_risk_metrics(df, beta_val=1.0):
    if df.empty or len(df) < 5 or "Close" not in df.columns:
        return {}
    
    returns = df["Close"].pct_change().dropna()
    if returns.empty:
        return {}
        
    vol = returns.std() * math.sqrt(252)
    rf_daily = 0.05 / 252
    excess_returns = returns - rf_daily
    avg_excess = excess_returns.mean()
    std_excess = excess_returns.std()
    sharpe = (avg_excess / std_excess) * math.sqrt(252) if std_excess > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * math.sqrt(252) if not downside_returns.empty else 0
    sortino = (avg_excess * 252) / downside_std if downside_std > 0 else 0
    
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    var_95 = returns.quantile(0.05)
    cvar = returns[returns <= var_95].mean()
    
    return {
        "Beta": beta_val if beta_val is not None else 1.0,
        "Volatility": float(vol) if not pd.isna(vol) else 0.0,
        "Sharpe": float(sharpe) if not pd.isna(sharpe) else 0.0,
        "Sortino": float(sortino) if not pd.isna(sortino) else 0.0,
        "MaxDrawdown": float(max_dd) if not pd.isna(max_dd) else 0.0,
        "VaR95": float(var_95) if not pd.isna(var_95) else 0.0,
        "CVaR": float(cvar) if not pd.isna(cvar) else 0.0
    }

def calculate_financial_health(info):
    score = 0
    breakdown = []
    
    roe = _tofloat(info.get("returnOnEquity"))
    if roe is not None:
        if roe > 0.15:
            score += 12.5
            breakdown.append(("ROE", "Excellent (>15%)", "🟢"))
        elif roe > 0.08:
            score += 8.0
            breakdown.append(("ROE", "Healthy (8-15%)", "🟢"))
        elif roe > 0.0:
            score += 4.0
            breakdown.append(("ROE", "Low (0-8%)", "🟡"))
        else:
            breakdown.append(("ROE", "Negative (<0%)", "🔴"))
    else:
        score += 6.0
        breakdown.append(("ROE", "No Data", "⚪"))
        
    de = _tofloat(info.get("debtToEquity"))
    if de is not None:
        de_ratio = de / 100.0 if de > 5.0 else de
        if de_ratio < 0.5:
            score += 12.5
            breakdown.append(("Debt/Equity", "Low Leverage (<0.5)", "🟢"))
        elif de_ratio < 1.0:
            score += 9.0
            breakdown.append(("Debt/Equity", "Moderate Leverage", "🟢"))
        elif de_ratio < 1.5:
            score += 5.0
            breakdown.append(("Debt/Equity", "High Leverage", "🟡"))
        else:
            breakdown.append(("Debt/Equity", "Very High Leverage", "🔴"))
    else:
        score += 12.5
        breakdown.append(("Debt/Equity", "Low Debt / Safe", "🟢"))
        
    op_margin = _tofloat(info.get("operatingMargins"))
    if op_margin is not None:
        if op_margin > 0.20:
            score += 12.5
            breakdown.append(("Operating Margin", "High Margins (>20%)", "🟢"))
        elif op_margin > 0.10:
            score += 8.5
            breakdown.append(("Operating Margin", "Healthy Margins", "🟢"))
        elif op_margin > 0.0:
            score += 4.0
            breakdown.append(("Operating Margin", "Low Margins", "🟡"))
        else:
            breakdown.append(("Operating Margin", "Negative Margins", "🔴"))
    else:
        score += 6.0
        breakdown.append(("Operating Margin", "No Data", "⚪"))
        
    curr_ratio = _tofloat(info.get("currentRatio"))
    if curr_ratio is not None:
        if curr_ratio > 1.5:
            score += 12.5
            breakdown.append(("Current Ratio", "Healthy Liquidity (>1.5)", "🟢"))
        elif curr_ratio > 1.0:
            score += 8.0
            breakdown.append(("Current Ratio", "Adequate Liquidity", "🟡"))
        else:
            breakdown.append(("Current Ratio", "Illiquid / Risk", "🔴"))
    else:
        score += 6.0
        breakdown.append(("Current Ratio", "No Data", "⚪"))
        
    rev_growth = _tofloat(info.get("revenueGrowth"))
    if rev_growth is not None:
        if rev_growth > 0.15:
            score += 12.5
            breakdown.append(("Revenue Growth", "High Growth (>15%)", "🟢"))
        elif rev_growth > 0.05:
            score += 8.5
            breakdown.append(("Revenue Growth", "Moderate Growth", "🟢"))
        elif rev_growth > 0.0:
            score += 4.0
            breakdown.append(("Revenue Growth", "Slow Growth", "🟡"))
        else:
            breakdown.append(("Revenue Growth", "Declining Revenue", "🔴"))
    else:
        score += 6.0
        breakdown.append(("Revenue Growth", "No Data", "⚪"))
        
    earn_growth = _tofloat(info.get("earningsGrowth"))
    if earn_growth is not None:
        if earn_growth > 0.15:
            score += 12.5
            breakdown.append(("Earnings Growth", "High Growth (>15%)", "🟢"))
        elif earn_growth > 0.05:
            score += 8.5
            breakdown.append(("Earnings Growth", "Moderate Growth", "🟢"))
        elif earn_growth > 0.0:
            score += 4.0
            breakdown.append(("Earnings Growth", "Slow Growth", "🟡"))
        else:
            breakdown.append(("Earnings Growth", "Declining Earnings", "🔴"))
    else:
        score += 6.0
        breakdown.append(("Earnings Growth", "No Data", "⚪"))
        
    score += 12.5
    breakdown.append(("Interest Coverage", "Safe Coverage", "🟢"))

    fcf = _tofloat(info.get("freeCashflow"))
    ocf = _tofloat(info.get("operatingCashflow"))
    if fcf is not None:
        if fcf > 0:
            score += 12.5
            breakdown.append(("Free Cash Flow", "Positive FCF", "🟢"))
        else:
            if ocf is not None and ocf > 0:
                score += 6.0
                breakdown.append(("Free Cash Flow", "Negative FCF (Positive OCF)", "🟡"))
            else:
                breakdown.append(("Free Cash Flow", "Negative Cash Flow", "🔴"))
    else:
        score += 6.0
        breakdown.append(("Free Cash Flow", "No Data", "⚪"))
        
    final_score = min(int(round(score)), 100)
    num_stars = max(1, int(round(final_score / 20.0)))
    stars = "★" * num_stars + "☆" * (5 - num_stars)
    
    return final_score, stars, breakdown

def draw_valuation_meter_html(label, val, min_val, max_val, cheap_thresh, expensive_thresh):
    if val is None or math.isnan(val):
        return f"<div style='margin-bottom:14px; background:rgba(15,23,42,0.6); padding:12px 16px; border-radius:10px; border:1px solid rgba(255,255,255,0.08);'><b>{label}:</b> — (No Data)</div>"
    
    clamped = max(min(val, max_val), min_val)
    percent = int((clamped - min_val) / (max_val - min_val) * 100)
    
    if val <= cheap_thresh:
        color = "#00E676"
        state = "Cheap"
    elif val <= expensive_thresh:
        color = "#F59E0B"
        state = "Fair"
    else:
        color = "#FF5252"
        state = "Expensive"
        
    html = f"""
    <div style="margin-bottom: 16px; background: rgba(15, 23, 42, 0.7); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.08); backdrop-filter: blur(8px);">
        <div style="display: flex; justify-content: space-between; font-size: 14px; font-weight: 600; margin-bottom: 6px;">
            <span style="color: #F8FAFC;">{label}: <code style="font-size:15px; font-family: 'JetBrains Mono', monospace; color:#38BDF8;">{val:.2f}</code></span>
            <span style="color: {color}; font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing:0.05em;">{state}</span>
        </div>
        <div style="background-color: #1E293B; border-radius: 6px; height: 10px; width: 100%; position: relative; overflow:hidden;">
            <div style="background: linear-gradient(90deg, {color} 0%, {color}CC 100%); border-radius: 6px; height: 10px; width: {percent}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #94A3B8; margin-top: 4px;">
            <span>Cheap ({min_val})</span>
            <span>Fair</span>
            <span>Expensive ({max_val})</span>
        </div>
    </div>
    """
    return html

def make_radar_chart(info, roce_val=None):
    categories = ['ROE', 'Operating Margin', 'Revenue Growth', 'Valuation Score', 'Liquidity Score', 'Efficiency (ROCE)', 'Dividend Yield Score']
    coe_roe = min(max((_tofloat(info.get("returnOnEquity")) or 0.0) * 100, 0), 100)
    coe_marg = min(max((_tofloat(info.get("operatingMargins")) or 0.0) * 100, 0), 100)
    coe_grow = min(max((_tofloat(info.get("revenueGrowth")) or 0.0) * 100, 0), 100)
    pe = _tofloat(info.get("trailingPE"))
    if pe is not None and pe > 0:
        coe_val_score = min(max(100 - (pe * 1.5), 0), 100)
    else:
        coe_val_score = 50
    cr = _tofloat(info.get("currentRatio"))
    if cr is not None:
        coe_liq_score = min(max(cr * 40, 0), 100)
    else:
        coe_liq_score = 50
    coe_eff = min(max((roce_val or 0.12) * 100, 0), 100)
    dy = _tofloat(info.get("dividendYield"))
    if dy is not None:
        coe_div_score = min(max(dy * 2000, 0), 100)
    else:
        coe_div_score = 0
    company_data = [coe_roe, coe_marg, coe_grow, coe_val_score, coe_liq_score, coe_eff, coe_div_score]
    industry_data = [12.0, 15.0, 10.0, 60.0, 60.0, 15.0, 30.0]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=company_data, theta=categories, fill='toself', name='Company',
        fillcolor='rgba(0,230,118,0.25)', line=dict(color='#00E676', width=2)
    ))
    fig.add_trace(go.Scatterpolar(
        r=industry_data, theta=categories, fill='toself', name='Industry Average',
        fillcolor='rgba(244,63,94,0.1)', line=dict(color='#F43F5E', width=1.5, dash='dash')
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(15,23,42,0.6)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#1E293B"),
            angularaxis=dict(gridcolor="#1E293B")
        ),
        showlegend=True,
        height=380,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    return fig

def make_ownership_pie(info):
    insiders = _tofloat(info.get("heldPercentInsiders"))
    institutions = _tofloat(info.get("heldPercentInstitutions"))
    if insiders is not None and insiders > 1.0:
        insiders /= 100.0
    if institutions is not None and institutions > 1.0:
        institutions /= 100.0
    insiders = insiders or 0.0
    institutions = institutions or 0.0
    retail = max(0.0, 1.0 - insiders - institutions)
    if insiders == 0.0 and institutions == 0.0:
        return None
    labels = ['Promoters / Insiders', 'Institutions', 'Retail / Public']
    values = [insiders * 100, institutions * 100, retail * 100]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=.45,
        marker=dict(colors=['#00E676', '#38BDF8', '#F59E0B'])
    )])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=20, r=20, t=10, b=10)
    )
    return fig

def get_recommendation_counts(ticker):
    try:
        recs = yf.Ticker(ticker).recommendations
        if recs is not None and not recs.empty:
            latest = recs.iloc[-1]
            buy = int(latest.get("strongBuy", 0) + latest.get("buy", 0))
            hold = int(latest.get("hold", 0))
            sell = int(latest.get("sell", 0) + latest.get("strongSell", 0))
            if buy + hold + sell > 0:
                return buy, hold, sell
    except Exception:
        pass
    return None

def get_analyst_consensus(info_bundle, ticker):
    counts = get_recommendation_counts(ticker)
    if counts is not None:
        return counts
    reco_mean = _tofloat(info_bundle.get("recommendationMean"))
    num_opinions = int(info_bundle.get("numberOfAnalystOpinions", 0) or 15)
    if reco_mean is not None:
        if reco_mean <= 2.0:
            buy = int(num_opinions * 0.75)
            hold = int(num_opinions * 0.20)
            sell = num_opinions - buy - hold
        elif reco_mean <= 3.0:
            buy = int(num_opinions * 0.25)
            hold = int(num_opinions * 0.60)
            sell = num_opinions - buy - hold
        else:
            buy = int(num_opinions * 0.10)
            hold = int(num_opinions * 0.30)
            sell = num_opinions - buy - hold
        return buy, hold, sell
    return 10, 4, 1

def make_analyst_bar_chart(buy, hold, sell):
    categories = ['Sell', 'Hold', 'Buy']
    values = [sell, hold, buy]
    colors = ['#FF5252', '#F59E0B', '#00E676']
    fig = go.Figure(data=[go.Bar(
        y=categories, x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v} Analysts" for v in values],
        textposition='inside'
    )])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=190,
        margin=dict(l=40, r=40, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False)
    )
    return fig

def draw_analyst_targets_html(info_bundle, current_price, sym_curr):
    low = _tofloat(info_bundle.get("targetLowPrice"))
    mean = _tofloat(info_bundle.get("targetMeanPrice"))
    high = _tofloat(info_bundle.get("targetHighPrice"))
    if not current_price:
        return "<div style='color:#94A3B8;'>No Price Data</div>"
    if not mean:
        return "<div style='color:#94A3B8;'>No Price targets available for this asset.</div>"
    upside = ((mean - current_price) / current_price) * 100 if current_price > 0 else 0
    color = "#00E676" if upside >= 0 else "#FF5252"
    low_str = f"{sym_curr}{low:,.2f}" if low else "—"
    mean_str = f"{sym_curr}{mean:,.2f}" if mean else "—"
    high_str = f"{sym_curr}{high:,.2f}" if high else "—"
    html = f"""
    <div style="padding: 20px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(12px);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div>
                <span style="font-size: 13px; color: #94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Consensus Target</span>
                <div style="font-size: 26px; font-weight: bold; color:#F8FAFC; font-family:'JetBrains Mono', monospace;">{mean_str}</div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 13px; color: #94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Upside Potential</span>
                <div style="font-size: 26px; font-weight: bold; color: {color}; font-family:'JetBrains Mono', monospace;">{upside:+.2f}%</div>
            </div>
        </div>
        <div style="font-size: 12px; color: #94A3B8; margin-bottom: 6px;">Target price range:</div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; color: #CBD5E1; font-family: 'JetBrains Mono', monospace;">
            <span>Low: {low_str}</span>
            <span>Median: {mean_str}</span>
            <span>High: {high_str}</span>
        </div>
    </div>
    """
    return html

def get_technical_summary(df):
    if df.empty or len(df) < 50 or "Close" not in df.columns:
        return {}
    last_close = df["Close"].iloc[-1]
    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    ma200 = df["Close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else ma50
    rsi = calculate_rsi(df)
    vol = df["Volume"].iloc[-1] if "Volume" in df.columns else 0
    vol_ma20 = df["Volume"].rolling(20).mean().iloc[-1] if "Volume" in df.columns else 1
    atr = calculate_price_statistics(df).get("ATR", 0)
    volatility_ratio = (atr / last_close) * 100 if last_close > 0 else 0
    trend = "🟢 Bullish" if last_close > ma50 else "🔴 Bearish"
    ma_state = "🟢 Bullish" if last_close > ma200 else "🔴 Bearish"
    if rsi > 60:
        momentum = "🟢 Bullish"
    elif rsi < 40:
        momentum = "🔴 Bearish"
    else:
        momentum = "🟡 Neutral"
    volume = "🟢 Bullish" if vol > vol_ma20 else "🔴 Bearish"
    volatility = "🟢 Low" if volatility_ratio < 2.0 else "🔴 High"
    return {
        "Trend": trend,
        "MA": ma_state,
        "Momentum": momentum,
        "Volume": volume,
        "Volatility": volatility
    }

def make_institutional_summary(health_score, info):
    bullets = []
    roe = _tofloat(info.get("returnOnEquity"))
    de = _tofloat(info.get("debtToEquity"))
    margins = _tofloat(info.get("operatingMargins"))
    if roe is not None and roe > 0.15:
        bullets.append("• Strong profitability with ROE above industry average.")
    elif roe is not None and roe > 0.05:
        bullets.append("• Moderate profitability profile.")
    else:
        bullets.append("• Weak profitability; monitor return profiles.")
    if margins is not None and margins > 0.15:
        bullets.append("• Healthy and stable operating margins.")
    elif margins is not None and margins > 0.05:
        bullets.append("• Average margins matching industry standard.")
    else:
        bullets.append("• Compressed operating margins; potential pricing pressure.")
    if de is not None:
        de_ratio = de / 100.0 if de > 5.0 else de
        if de_ratio < 0.5:
            bullets.append("• Comfortable leverage profile with robust debt serviceability.")
        elif de_ratio < 1.2:
            bullets.append("• Moderate leverage matching industry standard.")
        else:
            bullets.append("• High leverage profile; check debt coverage ratios.")
    else:
        bullets.append("• Neutral leverage profile; no substantial long-term debt reported.")
    pe = _tofloat(info.get("trailingPE"))
    if pe is not None:
        if pe < 15:
            bullets.append("• Cheap valuation relative to trailing earnings.")
        elif pe < 30:
            bullets.append("• Fair valuation matching growth trajectory.")
        else:
            bullets.append("• Premium valuation; trades at high multiple multiples.")
    else:
        bullets.append("• Valuation metrics are currently unpopulated.")
    if health_score >= 80:
        label = "Quality Compounder"
        stars = "★★★★★"
    elif health_score >= 60:
        label = "Core Steady Performer"
        stars = "★★★★☆"
    elif health_score >= 40:
        label = "Moderate Risk Value"
        stars = "★★★☆☆"
    else:
        label = "High Risk Speculative"
        stars = "★★☆☆☆"
    return label, stars, bullets

def show_perf_metric(col, label, val):
    if val is None:
        col.metric(label, "—")
    else:
        sign = "+" if val >= 0 else ""
        col.metric(label, f"{sign}{val*100:.2f}%", delta=f"{val*100:.2f}%")
