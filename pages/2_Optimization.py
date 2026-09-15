"""
AlphaForge | Portfolio Construction & Optimization
Multi-Model Optimization: Max Sharpe, Minimum Volatility, and Hierarchical Risk Parity (HRP).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from helper import inject_custom_theme
from sidebar import render_sidebar, get_or_run_quant_results

st.set_page_config(page_title="AlphaForge - Portfolio Optimization", page_icon="⚖️", layout="wide")
inject_custom_theme()

# Render Unified Sidebar
ticker, company, exchange, period, interval, region = render_sidebar()

# Fetch active quantitative factor results
results = get_or_run_quant_results()
weights_df = results["weights_df"]
frontier_df = results["frontier_df"]
factors_df = results["factors_df"]
rf = results.get("rf", 0.065 if region == "India" else 0.040)
currency_symbol = results.get("currency_symbol", "₹" if region == "India" else "$")
opt_model = results.get("optimization_model", "Sharpe")

st.markdown(f"# ⚖️ Portfolio Construction & Optimization — {region.upper()}")
st.caption(f"Asset allocation engine running **{opt_model.upper()}** methodology under institutional bounds.")

# Active Model Callout
model_descriptions = {
    "Sharpe": {
        "title": "🎯 Maximum Sharpe Ratio (Classical Mean-Variance)",
        "formula": r"\max_{\mathbf{w}} \frac{\mathbf{w}^{T}\boldsymbol{\mu} - r_f}{\sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}}",
        "desc": "Optimizes risk-adjusted excess returns using factor-implied expected alpha and quadratic covariance."
    },
    "MinVol": {
        "title": "🛡️ Minimum Volatility (Capital Preservation)",
        "formula": r"\min_{\mathbf{w}} \mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}",
        "desc": "Finds the global minimum variance portfolio on the efficient frontier, prioritizing downside safety."
    },
    "HRP": {
        "title": "🌳 Hierarchical Risk Parity (HRP - Machine Learning Clustering)",
        "formula": r"\mathbf{w}_{\mathrm{HRP}} = \text{TreeClustering}(\boldsymbol{\Sigma}) \times \text{RecursiveBisection}()",
        "desc": "Uses hierarchical tree clustering to group correlated assets and allocate risk recursively, avoiding matrix inversion instability."
    }
}

curr_desc = model_descriptions.get(opt_model, model_descriptions["Sharpe"])

with st.container(border=True):
    st.markdown(f"### {curr_desc['title']}")
    st.latex(curr_desc["formula"])
    st.write(curr_desc["desc"])
    st.caption("Change the Optimization Model at any time from the sidebar controls on the left.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🔮 Factor-Model Derived Expected Returns")
    st.markdown(
        r"$\mu_i = r_f + \beta_i \cdot \mathrm{MRP} + (\mathrm{Composite\ Score}_i - 0.5) \cdot 8\%$"
    )

    selected_tickers = results["selected_tickers"]
    selected_factors = factors_df[factors_df["Ticker"].isin(selected_tickers)].copy()
    
    selected_factors["Expected Return"] = rf + (0.075 if region == "India" else 0.060) + (selected_factors["Composite"] - 0.5) * 0.08
    selected_factors = selected_factors.sort_values(by="Expected Return", ascending=False)

    st.dataframe(
        selected_factors[["Ticker", "Company", "Composite", "Expected Return"]].style.format({
            "Composite": "{:.2f}",
            "Expected Return": "{:.2%}"
        }).background_gradient(subset=["Expected Return"], cmap="viridis"),
        width="stretch",
        height=350
    )

with col_right:
    st.subheader(f"⚖️ Optimal Allocation ({opt_model})")
    st.markdown(f"Optimized weights assigned across the selected portfolio assets:")

    opt_weights = weights_df[weights_df["Selected"]].sort_values(by="Weight", ascending=False)
    st.dataframe(
        opt_weights[["Ticker", "Company", "Sector", "Weight"]].style.format({
            "Weight": "{:.2%}"
        }).bar(subset=["Weight"], color="#00E676"),
        width="stretch",
        height=350
    )

st.divider()

# Allocation Donut Chart
col_pie, col_bar = st.columns([1, 1])

with col_pie:
    st.subheader("🍩 Asset Allocation Breakdown")
    fig_pie = px.pie(
        opt_weights,
        values="Weight",
        names="Ticker",
        title="Asset Weight Distribution",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Tealgrn_r
    )
    fig_pie.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF"
    )
    st.plotly_chart(fig_pie, width="stretch")

with col_bar:
    st.subheader("🏢 Sector Exposure Breakdown")
    sector_weights = opt_weights.groupby("Sector")["Weight"].sum().reset_index().sort_values(by="Weight", ascending=True)
    fig_sec = px.bar(
        sector_weights,
        x="Weight",
        y="Sector",
        orientation="h",
        title="Portfolio Sector Concentration",
        text_auto=".1%",
        color="Weight",
        color_continuous_scale="Teal"
    )
    fig_sec.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#21262D", tickformat=".0%"),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig_sec, width="stretch")

st.divider()

# Efficient Frontier Chart
st.subheader("📈 Modern Portfolio Theory (MPT) Efficient Frontier Space")
st.markdown("Below is the simulated portfolio opportunity set plotted against the calculated Efficient Frontier. The **Active Factor Portfolio** represents your selected optimization point.")

sim_points = frontier_df[frontier_df["Type"] == "Simulated"]
max_sharpe_list = frontier_df[frontier_df["Type"] == "Max Sharpe"]
min_vol_list = frontier_df[frontier_df["Type"] == "Min Vol"]

fig_frontier = go.Figure()

if not sim_points.empty:
    fig_frontier.add_trace(go.Scatter(
        x=sim_points["Risk"],
        y=sim_points["Return"],
        mode="markers",
        name="Simulated Portfolios",
        marker=dict(
            size=5,
            color=(sim_points["Return"] - rf) / sim_points["Risk"],
            colorscale="Teal_r",
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        opacity=0.6
    ))

if not min_vol_list.empty:
    min_vol = min_vol_list.iloc[0]
    fig_frontier.add_trace(go.Scatter(
        x=[min_vol["Risk"]],
        y=[min_vol["Return"]],
        mode="markers+text",
        text=["Min Vol"],
        textposition="top center",
        name="Min Volatility Point",
        marker=dict(color="#00FFCC", size=14, symbol="circle")
    ))

if not max_sharpe_list.empty:
    max_sharpe = max_sharpe_list.iloc[0]
    fig_frontier.add_trace(go.Scatter(
        x=[max_sharpe["Risk"]],
        y=[max_sharpe["Return"]],
        mode="markers+text",
        text=["Max Sharpe"],
        textposition="top center",
        name="Max Sharpe Point",
        marker=dict(color="#FFD700", size=18, symbol="star")
    ))

fig_frontier.update_layout(
    title=f"Efficient Frontier Space — {region} Equity Universe",
    xaxis_title="Annualized Volatility (Risk)",
    yaxis_title="Expected Return",
    plot_bgcolor="#161B22",
    paper_bgcolor="#0E1117",
    font_color="#FFFFFF",
    xaxis=dict(showgrid=True, gridcolor="#21262D", tickformat=".1%"),
    yaxis=dict(showgrid=True, gridcolor="#21262D", tickformat=".1%"),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
)

st.plotly_chart(fig_frontier, width="stretch")
