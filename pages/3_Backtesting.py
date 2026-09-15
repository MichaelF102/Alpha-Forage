"""
AlphaForge | Strategy Backtesting
Historical backtest evaluation against Benchmark and Monte Carlo future wealth projection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helper import inject_custom_theme
from sidebar import render_sidebar, get_or_run_quant_results
from engine import QuantEngine

st.set_page_config(page_title="AlphaForge - Backtesting", page_icon="📈", layout="wide")
inject_custom_theme()

# Render Unified Sidebar
ticker, company, exchange, period, interval, region = render_sidebar()

# Fetch active quantitative factor results
results = get_or_run_quant_results()
portfolio_df = results["portfolio_ts"].copy()
rolling_df = results["rolling_df"].copy()
benchmark_name = results.get("benchmark_name", "Nifty 50" if region == "India" else "S&P 500")
curr_sym = results.get("currency_symbol", "₹" if region == "India" else "$")
rf = results.get("rf", 0.065 if region == "India" else 0.040)

st.markdown(f"# 📈 Strategy Backtesting — {region.upper()}")
st.caption(
    f"Historical backtest of the **Optimized Factor Strategy** compared against a naive **Equal Weight Portfolio** "
    f"and the **{benchmark_name} Index (Benchmark)**."
)

# Date Range & Initial Amount Controls
min_date = portfolio_df.index.min().date()
max_date = portfolio_df.index.max().date()

c_ctrl1, c_ctrl2 = st.columns([1, 2])
with c_ctrl1:
    default_inv = 100000.0 if region == "India" else 10000.0
    initial_amount = st.number_input(
        f"Initial Investment ({curr_sym})",
        min_value=100.0,
        max_value=100000000.0,
        value=default_inv,
        step=5000.0,
        format="%.0f"
    )

with c_ctrl2:
    date_range = st.date_input(
        "Select Backtest Time Window",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

# Safe date extraction
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    sd, ed = date_range[0], date_range[1]
else:
    sd, ed = min_date, max_date

sd_dt = pd.to_datetime(sd)
ed_dt = pd.to_datetime(ed)

# Slice datasets
sliced_df = portfolio_df.loc[sd_dt:ed_dt].copy()
sliced_rolling = rolling_df.loc[sd_dt:ed_dt].copy()

if len(sliced_df) < 5:
    st.warning("Please select a date range containing at least 5 trading days.")
else:
    # Recalculate Wealth Series dynamically
    sliced_df["Portfolio_Wealth"] = initial_amount * (1.0 + sliced_df["Return"]).cumprod()
    sliced_df["EqualWeight_Wealth"] = initial_amount * (1.0 + sliced_df["EqualWeight_Return"]).cumprod()
    sliced_df["Benchmark_Wealth"] = initial_amount * (1.0 + sliced_df["Benchmark_Return"]).cumprod()

    # Recalculate Period Performance
    n_days = len(sliced_df)
    years = max(n_days / 252.0, 0.05)

    def calc_period_metrics(ret_series, wealth_series):
        cagr = (wealth_series.iloc[-1] / wealth_series.iloc[0]) ** (1.0 / years) - 1.0 if len(wealth_series) > 1 else 0.0
        vol = ret_series.std() * np.sqrt(252.0)
        excess = ret_series.mean() * 252.0 - rf
        sharpe = excess / vol if vol > 0 else 0.0
        downside = ret_series[ret_series < 0]
        d_vol = downside.std() * np.sqrt(252.0)
        sortino = excess / d_vol if d_vol > 0 else 0.0
        peaks = wealth_series.cummax()
        dd = (wealth_series - peaks) / peaks
        max_dd = dd.min()
        return cagr, vol, sharpe, sortino, max_dd

    opt_stats = calc_period_metrics(sliced_df["Return"], sliced_df["Portfolio_Wealth"])
    ew_stats = calc_period_metrics(sliced_df["EqualWeight_Return"], sliced_df["EqualWeight_Wealth"])
    bm_stats = calc_period_metrics(sliced_df["Benchmark_Return"], sliced_df["Benchmark_Wealth"])

    st.subheader(f"📊 Strategy Performance Comparison ({sd} to {ed})")

    perf_summary = pd.DataFrame([
        {"Metric": "CAGR", "Portfolio": opt_stats[0], f"Benchmark ({benchmark_name})": bm_stats[0], "Equal Weight": ew_stats[0]},
        {"Metric": "Annualized Volatility", "Portfolio": opt_stats[1], f"Benchmark ({benchmark_name})": bm_stats[1], "Equal Weight": ew_stats[1]},
        {"Metric": "Sharpe Ratio", "Portfolio": opt_stats[2], f"Benchmark ({benchmark_name})": bm_stats[2], "Equal Weight": ew_stats[2]},
        {"Metric": "Sortino Ratio", "Portfolio": opt_stats[3], f"Benchmark ({benchmark_name})": bm_stats[3], "Equal Weight": ew_stats[3]},
        {"Metric": "Maximum Drawdown", "Portfolio": opt_stats[4], f"Benchmark ({benchmark_name})": bm_stats[4], "Equal Weight": ew_stats[4]}
    ])

    st.dataframe(
        perf_summary.style.format({
            "Portfolio": "{:.2%}" if perf_summary["Metric"].str.contains("CAGR|Volatility|Drawdown").any() else "{:.2f}",
            f"Benchmark ({benchmark_name})": "{:.2%}",
            "Equal Weight": "{:.2%}"
        }),
        width="stretch"
    )

    # 1. Cumulative Wealth Chart
    st.subheader(f"📈 Cumulative Growth of Investment ({curr_sym})")

    fig_wealth = go.Figure()
    fig_wealth.add_trace(go.Scatter(
        x=sliced_df.index, y=sliced_df["Portfolio_Wealth"],
        mode="lines", name="Optimized Factor Strategy",
        line=dict(color="#00E676", width=2.5)
    ))
    fig_wealth.add_trace(go.Scatter(
        x=sliced_df.index, y=sliced_df["EqualWeight_Wealth"],
        mode="lines", name="Equal Weight Portfolio",
        line=dict(color="#38BDF8", width=1.5, dash="dash")
    ))
    fig_wealth.add_trace(go.Scatter(
        x=sliced_df.index, y=sliced_df["Benchmark_Wealth"],
        mode="lines", name=f"{benchmark_name} (Benchmark)",
        line=dict(color="#F43F5E", width=1.8, dash="dot")
    ))

    fig_wealth.update_layout(
        title=f"Portfolio Wealth Trajectory vs. {benchmark_name}",
        xaxis_title="Date",
        yaxis_title=f"Portfolio Value ({curr_sym})",
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        yaxis=dict(showgrid=True, gridcolor="#21262D"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
    )
    st.plotly_chart(fig_wealth, width="stretch")

    st.divider()

    # 2. Drawdown Profile
    st.subheader("📉 Historical Drawdown Profile")

    peaks = sliced_df["Portfolio_Wealth"].cummax()
    dd_opt = (sliced_df["Portfolio_Wealth"] - peaks) / peaks

    bm_peaks = sliced_df["Benchmark_Wealth"].cummax()
    dd_bm = (sliced_df["Benchmark_Wealth"] - bm_peaks) / bm_peaks

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=sliced_df.index, y=dd_opt,
        mode="lines", name="Optimized Strategy Drawdown",
        fill="tozeroy", line=dict(color="#00E676", width=1.5),
        fillcolor="rgba(0, 230, 118, 0.2)"
    ))
    fig_dd.add_trace(go.Scatter(
        x=sliced_df.index, y=dd_bm,
        mode="lines", name=f"{benchmark_name} Drawdown",
        line=dict(color="#F43F5E", width=1.5, dash="dot")
    ))
    fig_dd.update_layout(
        title="Underwater Equity Curve (Drawdown %)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="#21262D"),
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF"
    )
    st.plotly_chart(fig_dd, width="stretch")

    st.divider()

    # 3. Rolling Analytics
    st.subheader("🔄 Rolling Risk & Return Dynamics (30-Day Window)")
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        fig_rvol = px.line(
            sliced_rolling,
            y="RollingVolatility",
            title="Rolling Annualized Volatility",
            labels={"RollingVolatility": "Volatility", "index": "Date"}
        )
        fig_rvol.update_traces(line_color="#38BDF8")
        fig_rvol.update_layout(
            plot_bgcolor="#161B22",
            paper_bgcolor="#0E1117",
            font_color="#FFFFFF",
            yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="#21262D"),
            xaxis=dict(showgrid=True, gridcolor="#21262D")
        )
        st.plotly_chart(fig_rvol, width="stretch")

    with r_col2:
        fig_rsharpe = px.line(
            sliced_rolling,
            y="RollingSharpe",
            title="Rolling Sharpe Ratio",
            labels={"RollingSharpe": "Sharpe", "index": "Date"}
        )
        fig_rsharpe.update_traces(line_color="#FFD700")
        fig_rsharpe.update_layout(
            plot_bgcolor="#161B22",
            paper_bgcolor="#0E1117",
            font_color="#FFFFFF",
            yaxis=dict(showgrid=True, gridcolor="#21262D"),
            xaxis=dict(showgrid=True, gridcolor="#21262D")
        )
        st.plotly_chart(fig_rsharpe, width="stretch")

    st.divider()

    # 4. Monte Carlo Future Projection Fan
    st.subheader("🔮 Monte Carlo Future Wealth Projection (1,000 Paths)")
    st.markdown(
        f"Simulating 1,000 future price trajectories using Geometric Brownian Motion calibrated on the active strategy's "
        f"historical drift and volatility. Illustrates probability bounds over the chosen forward horizon."
    )

    mc_years = st.radio("Forward Horizon", [1, 2, 3], index=0, horizontal=True, format_func=lambda y: f"{y} Year{'s' if y > 1 else ''}")

    # Generate Monte Carlo simulation
    quant_engine = QuantEngine(region=region)
    mc_df = quant_engine.simulate_monte_carlo(sliced_df["Return"], years=mc_years, n_sims=1000, initial_wealth=initial_amount)

    # Fan chart
    fig_mc = go.Figure()

    # 95th Percentile (Bull Case)
    fig_mc.add_trace(go.Scatter(
        x=mc_df.index, y=mc_df["P95_Bull"],
        mode="lines", name="95th Percentile (Bull Case)",
        line=dict(color="#00E676", width=2)
    ))

    # 75th Percentile
    fig_mc.add_trace(go.Scatter(
        x=mc_df.index, y=mc_df["P75"],
        mode="lines", name="75th Percentile",
        line=dict(color="#38BDF8", width=1.5, dash="dash")
    ))

    # 50th Percentile (Median Path)
    fig_mc.add_trace(go.Scatter(
        x=mc_df.index, y=mc_df["P50_Median"],
        mode="lines", name="50th Percentile (Median Expected)",
        line=dict(color="#FFD700", width=2.5)
    ))

    # 25th Percentile
    fig_mc.add_trace(go.Scatter(
        x=mc_df.index, y=mc_df["P25"],
        mode="lines", name="25th Percentile",
        line=dict(color="#F59E0B", width=1.5, dash="dash")
    ))

    # 5th Percentile (Bear Case)
    fig_mc.add_trace(go.Scatter(
        x=mc_df.index, y=mc_df["P5_Bear"],
        mode="lines", name="5th Percentile (Severe Bear Case)",
        line=dict(color="#F43F5E", width=2)
    ))

    fig_mc.update_layout(
        title=f"Forward Wealth Fan ({mc_years}-Year Projection, Starting Value {curr_sym}{initial_amount:,.0f})",
        xaxis_title="Future Date",
        yaxis_title=f"Projected Value ({curr_sym})",
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        yaxis=dict(showgrid=True, gridcolor="#21262D"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
    )
    st.plotly_chart(fig_mc, width="stretch")

    # Monte Carlo summary metrics
    end_p50 = mc_df["P50_Median"].iloc[-1]
    end_p95 = mc_df["P95_Bull"].iloc[-1]
    end_p5 = mc_df["P5_Bear"].iloc[-1]

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Median Ending Capital (50%)", f"{curr_sym}{end_p50:,.2f}", delta=f"{(end_p50 / initial_amount - 1):+.1%}")
    mc2.metric("Bull Scenario (95%)", f"{curr_sym}{end_p95:,.2f}", delta=f"{(end_p95 / initial_amount - 1):+.1%}")
    mc3.metric("Tail Risk Floor (5%)", f"{curr_sym}{end_p5:,.2f}", delta=f"{(end_p5 / initial_amount - 1):+.1%}")
