"""
AlphaForge | Institutional Risk Analytics
Side-by-side risk decomposition against benchmark, VaR/CVaR, and macroeconomic stress tests.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from helper import inject_custom_theme
from sidebar import render_sidebar, get_or_run_quant_results

st.set_page_config(page_title="AlphaForge - Risk Analytics", page_icon="🛡️", layout="wide")
inject_custom_theme()

# Render Unified Sidebar
ticker, company, exchange, period, interval, region = render_sidebar()

# Fetch active quantitative factor results
results = get_or_run_quant_results()
risk_df = results["risk_df"]
performance_df = results["performance_df"]
portfolio_df = results["portfolio_ts"]
benchmark_name = results.get("benchmark_name", "Nifty 50" if region == "India" else "S&P 500")
risk_metrics = results["risk_summary"]
curr_sym = results.get("currency_symbol", "₹" if region == "India" else "$")

st.markdown(f"# 🛡️ Institutional Risk Analytics — {region.upper()}")
st.caption(
    f"Risk decomposition and stress testing comparing the active portfolio side-by-side with the "
    f"**{benchmark_name}** benchmark."
)

# 1. Side-by-Side Comparison
st.subheader(f"📊 Portfolio vs. {benchmark_name} Metric Comparison")

comparison_data = []
for _, row in performance_df.iterrows():
    metric = row["Metric"]
    p_val = row["Portfolio"]
    b_val = row["Benchmark"]
    diff = p_val - b_val

    comparison_data.append({
        "Metric": metric,
        "Portfolio Strategy": p_val,
        f"{benchmark_name} (Benchmark)": b_val,
        "Active Spread": diff
    })

comp_df = pd.DataFrame(comparison_data)

st.dataframe(
    comp_df.style.format({
        "Portfolio Strategy": "{:.2%}",
        f"{benchmark_name} (Benchmark)": "{:.2%}",
        "Active Spread": "{:+.2%}"
    }).background_gradient(subset=["Active Spread"], cmap="coolwarm"),
    width="stretch"
)

st.divider()

# 2. Institutional Risk Metric Tiles
st.subheader("🛡️ Advanced Institutional Risk Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Tracking Error (Active Vol)",
        value=f"{risk_metrics.get('TrackingError', 0):.2%}",
        help="Annualized standard deviation of active returns (Portfolio - Benchmark)."
    )
    st.metric(
        label="Downside Deviation",
        value=f"{risk_metrics.get('DownsideDeviation', 0):.2%}",
        help="Standard deviation of negative daily returns only."
    )

with col2:
    st.metric(
        label="Information Ratio",
        value=f"{risk_metrics.get('InformationRatio', 0):.2f}",
        help="Active Return divided by Tracking Error. Measures consistency of excess return generation."
    )
    st.metric(
        label="Calmar Ratio",
        value=f"{risk_metrics.get('CalmarRatio', 0):.2f}",
        help="CAGR divided by Maximum Drawdown."
    )

with col3:
    st.metric(
        label="Treynor Ratio",
        value=f"{risk_metrics.get('TreynorRatio', 0):.2f}",
        help="Excess CAGR divided by Beta. Return generated per unit of systematic market risk."
    )
    st.metric(
        label="Omega Ratio",
        value=f"{risk_metrics.get('OmegaRatio', 0):.2f}",
        help="Ratio of upside gains to downside losses."
    )

with col4:
    st.metric(
        label="Historical VaR (95% 1-Day)",
        value=f"{risk_metrics.get('VaR', 0):.2%}",
        help="Maximum expected 1-day percentage loss with 95% statistical confidence."
    )
    st.metric(
        label="CVaR (Expected Shortfall)",
        value=f"{risk_metrics.get('CVaR', 0):.2%}",
        help="Expected average loss on days exceeding the 95% VaR threshold."
    )

st.divider()

# 3. Daily Returns Distribution & Value at Risk
st.subheader("📊 Return Distribution & Tail Risk (VaR Threshold)")

port_returns = portfolio_df["Return"]
var_95 = risk_metrics.get("VaR", 0.015)

fig_hist = px.histogram(
    port_returns,
    nbins=60,
    title="Daily Portfolio Return Distribution vs. 95% Historical VaR",
    labels={"value": "Daily Return"},
    color_discrete_sequence=["#38BDF8"]
)
fig_hist.add_vline(
    x=-var_95,
    line_dash="dash",
    line_color="#F43F5E",
    annotation_text=f"VaR (95%): -{var_95:.2%}",
    annotation_position="top left"
)
fig_hist.update_layout(
    plot_bgcolor="#161B22",
    paper_bgcolor="#0E1117",
    font_color="#FFFFFF",
    xaxis=dict(tickformat=".1%", showgrid=True, gridcolor="#21262D"),
    yaxis=dict(showgrid=True, gridcolor="#21262D")
)
st.plotly_chart(fig_hist, width="stretch")

st.divider()

# 4. Macro Stress Testing Scenarios
st.subheader("🌪️ Macroeconomic Stress Testing Scenarios")
st.markdown("Hypothetical impact on current portfolio capitalization under severe market stress conditions:")

final_equity = portfolio_df["Portfolio"].iloc[-1]

stress_scenarios = [
    {
        "Scenario": "Global Market Crash (-30%)",
        "Assumed Market Shock": "-30.0%",
        "Estimated Equity Impact": f"{curr_sym}{final_equity * 0.70:,.2f}",
        "Net Drawdown": "-30.0%"
    },
    {
        "Scenario": "Central Bank Rate Shock (-10%)",
        "Assumed Market Shock": "-10.0%",
        "Estimated Equity Impact": f"{curr_sym}{final_equity * 0.90:,.2f}",
        "Net Drawdown": "-10.0%"
    },
    {
        "Scenario": "Macro Recessionary Shock (-20%)",
        "Assumed Market Shock": "-20.0%",
        "Estimated Equity Impact": f"{curr_sym}{final_equity * 0.80:,.2f}",
        "Net Drawdown": "-20.0%"
    }
]

st.dataframe(pd.DataFrame(stress_scenarios), width="stretch", hide_index=True)
