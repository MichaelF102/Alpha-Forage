"""
AlphaForge | Factor Research & Universe Selection
Multi-factor rankings and individual security fundamental & analyst intelligence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from helper import (
    inject_custom_theme,
    load_data,
    fetch_yf_info,
    calculate_price_statistics,
    calculate_financial_health,
    draw_valuation_meter_html,
    make_radar_chart,
    get_technical_summary,
    make_ownership_pie,
    get_analyst_consensus,
    make_analyst_bar_chart,
    draw_analyst_targets_html
)
from sidebar import render_sidebar, get_or_run_quant_results

st.set_page_config(page_title="AlphaForge - Factor Research", page_icon="🔬", layout="wide")
inject_custom_theme()

# Render Unified Sidebar
ticker, company, exchange, period, interval, region = render_sidebar()

# Fetch active quantitative factor results
results = get_or_run_quant_results()
factors_df = results["factors_df"]
factor_weights = results.get("factor_weights", {"Momentum": 0.25, "Volatility": 0.25, "Value": 0.25, "Quality": 0.25})

st.markdown(f"# 🔬 Factor Research & Stock Analysis — {region.upper()}")
st.caption("Quantitative multi-factor ranking, systematic factor construction, and individual security deep-dive analytics.")

# Two Comprehensive Tabs
tab1, tab2 = st.tabs(["📊 Multi-Factor Universe Rankings", "🔍 Single Stock Deep Dive"])

with tab1:
    st.subheader("💡 Factor Definitions & Active Tactical Tilts")

    # Display Active Factor Weights Badges
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("Momentum Tilt", f"{factor_weights.get('Momentum', 0.25):.0%}", help="12-1 Month Price Return")
    b_col2.metric("Low Volatility Tilt", f"{factor_weights.get('Volatility', 0.25):.0%}", help="Inverse Annualized Volatility")
    b_col3.metric("Fundamental Value Tilt", f"{factor_weights.get('Value', 0.25):.0%}", help="Earnings Yield + Dividend Yield")
    b_col4.metric("Quality / Growth Tilt", f"{factor_weights.get('Quality', 0.25):.0%}", help="TTM YoY Diluted EPS Growth")

    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Momentum Factor")
        st.write("Captures the tendency of assets with high recent returns to continue outperforming.")
        st.latex(r"\text{12-1 Month Momentum}=\frac{P_{t-21}}{P_{t-252}}-1")
        st.caption("Uses the most recent 12 months excluding the latest trading month to avoid short-term reversal effects.")

        st.markdown("---")

        st.markdown("### 🛡️ Low Volatility Factor")
        st.write("Captures the low-volatility anomaly where less volatile stocks deliver superior risk-adjusted returns.")
        st.latex(r"\sigma_{\mathrm{ann}}=\mathrm{std}(r_{\mathrm{daily}})\times\sqrt{252}")
        st.caption("Ranked in reverse so securities with lower volatility receive higher factor ranks.")

    with col2:
        st.markdown("### 💎 Fundamental Value Factor")
        st.write("Identifies undervalued companies using real fundamental valuation metrics from dataset.")
        st.latex(r"\text{Value Score}=0.8 \cdot \frac{1}{\mathrm{P/E}} + 0.2 \cdot \mathrm{Dividend\ Yield}")
        st.caption("Higher earnings yield and dividend yield indicate attractive valuation.")

        st.markdown("---")

        st.markdown("### 🏆 Quality Factor")
        st.write("Selects companies demonstrating operational efficiency and robust fundamental earnings growth.")
        st.latex(r"\text{Quality Score}=\mathrm{EPS\ Growth\ (TTM\ YoY)}")
        st.caption("Real Trailing-Twelve-Month Diluted EPS YoY Growth rate from the stock dataset.")

    st.divider()

    # 2. Rankings and Selection Universe
    st.subheader(f"📊 {region} Universe Factor Rankings")
    st.markdown(
        f"The universe of **{len(factors_df)} securities** is ranked on each factor from 0.0 to 1.0 (percentile rank), "
        f"combined into an institutional **Composite Factor Score** using active tilts, and the **Top {len(results['selected_tickers'])} stocks** are selected for portfolio construction."
    )

    display_df = factors_df.copy()
    display_df["Selected?"] = display_df["Selected"].apply(lambda x: "✅ Selected" if x else "❌ Rejected")

    cols_to_show = [
        "Rank", "Ticker", "Company", "Sector", "Momentum", "Volatility", "Value_PE", "EPS_Growth", "Dividend_Yield",
        "Momentum_Rank", "Volatility_Rank", "Value_Rank", "Quality_Rank", "Composite", "Selected?"
    ]
    avail_cols = [c for c in cols_to_show if c in display_df.columns]

    rename_dict = {
        "Value_PE": "P/E Ratio",
        "EPS_Growth": "EPS Growth (YoY)",
        "Dividend_Yield": "Div Yield",
        "Volatility": "Annual Vol",
        "Momentum_Rank": "Mom Pct",
        "Volatility_Rank": "Low Vol Pct",
        "Value_Rank": "Value Pct",
        "Quality_Rank": "Quality Pct",
        "Composite": "Composite Score"
    }

    styled_df = display_df[avail_cols].rename(columns=rename_dict)

    format_dict = {
        "Momentum": "{:.2%}",
        "Annual Vol": "{:.2%}",
        "P/E Ratio": "{:.1f}",
        "EPS Growth (YoY)": "{:.2%}",
        "Div Yield": "{:.2%}",
        "Mom Pct": "{:.2f}",
        "Low Vol Pct": "{:.2f}",
        "Value Pct": "{:.2f}",
        "Quality Pct": "{:.2f}",
        "Composite Score": "{:.2f}"
    }
    active_format = {k: v for k, v in format_dict.items() if k in styled_df.columns}

    st.dataframe(
        styled_df.style.format(active_format).background_gradient(subset=["Composite Score"], cmap="coolwarm"),
        width="stretch",
        height=450
    )

    st.divider()

    # 3. Sector Distribution
    st.subheader("🏢 Selected Universe Sector Allocation")
    selected_only = factors_df[factors_df["Selected"]]
    sector_counts = selected_only["Sector"].value_counts().reset_index()
    sector_counts.columns = ["Sector", "Count"]

    fig = px.bar(
        sector_counts,
        x="Count",
        y="Sector",
        orientation="h",
        title=f"Selected Portfolio Asset Count by Sector ({region})",
        color="Count",
        color_continuous_scale="Teal"
    )
    fig.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader(f"🔍 Security Intelligence: {company} ({ticker})")
    st.caption(f"Fundamental diagnostics, analyst consensus, target price gauges, and technical indicators for **{ticker}**.")

    # Load stock price history
    with st.spinner(f"Loading intelligence for {ticker}..."):
        stock_df = load_data(ticker, period=period, interval="1d")
        info = fetch_yf_info(ticker)

    if stock_df.empty:
        st.warning(f"Historical price data for {ticker} could not be retrieved from Yahoo Finance.")
    else:
        # 1. Top Stock Metrics
        p_stats = calculate_price_statistics(stock_df)
        curr_sym = results.get("currency_symbol", "$")

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("Current Price", f"{curr_sym}{p_stats.get('current_price', 0):,.2f}", delta=f"{p_stats.get('daily_change_pct', 0):+.2%}")
        sm2.metric("Period High", f"{curr_sym}{p_stats.get('high_52w', 0):,.2f}")
        sm3.metric("Period Low", f"{curr_sym}{p_stats.get('low_52w', 0):,.2f}")
        sm4.metric("Avg Volume", f"{p_stats.get('avg_volume', 0):,.0f}")
        sm5.metric("Volatility (Ann)", f"{p_stats.get('annualized_volatility', 0):.2%}")

        st.markdown("")

        # 2. Interactive Price Chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=stock_df.index, y=stock_df["Close"],
            mode="lines", name="Close Price",
            line=dict(color="#00E676", width=2)
        ))
        if len(stock_df) >= 50:
            stock_df["SMA_50"] = stock_df["Close"].rolling(50).mean()
            fig_price.add_trace(go.Scatter(
                x=stock_df.index, y=stock_df["SMA_50"],
                mode="lines", name="50-Day SMA",
                line=dict(color="#38BDF8", width=1.5, dash="dash")
            ))
        if len(stock_df) >= 200:
            stock_df["SMA_200"] = stock_df["Close"].rolling(200).mean()
            fig_price.add_trace(go.Scatter(
                x=stock_df.index, y=stock_df["SMA_200"],
                mode="lines", name="200-Day SMA",
                line=dict(color="#F59E0B", width=1.5, dash="dot")
            ))

        fig_price.update_layout(
            title=f"{company} ({ticker}) Price History",
            xaxis_title="Date",
            yaxis_title=f"Price ({curr_sym})",
            plot_bgcolor="#161B22",
            paper_bgcolor="#0E1117",
            font_color="#FFFFFF",
            xaxis=dict(showgrid=True, gridcolor="#21262D"),
            yaxis=dict(showgrid=True, gridcolor="#21262D"),
            height=380
        )
        st.plotly_chart(fig_price, width="stretch")

        # 3. Analyst Consensus & Target Price Section
        st.markdown("#### 🎯 Institutional Analyst Consensus & Target Prices")
        c_tgt1, c_tgt2 = st.columns([1, 1])

        with c_tgt1:
            st.markdown("##### 12-Month Target Price Forecast")
            cur_price = p_stats.get("current_price", 100.0)
            target_html = draw_analyst_targets_html(info, cur_price, curr_sym)
            st.html(target_html)

        with c_tgt2:
            st.markdown("##### Broker Recommendation Distribution")
            buy, hold, sell = get_analyst_consensus(info, ticker)
            bar_fig = make_analyst_bar_chart(buy, hold, sell)
            st.plotly_chart(bar_fig, width="stretch")

        st.divider()

        # 4. Financial Health & Radar Analytics + Ownership
        col_rad, col_val, col_own = st.columns([1, 1, 1])

        with col_rad:
            st.markdown("#### 🛡️ Health Radar")
            radar_fig = make_radar_chart(info)
            radar_fig.update_layout(height=300)
            st.plotly_chart(radar_fig, width="stretch")

            health_score, _ = calculate_financial_health(info)
            st.markdown(f"**Financial Health Score**: `{health_score:.1f} / 100`")

        with col_val:
            st.markdown("#### ⚖️ Valuation Gauges")
            pe_val = info.get("trailingPE") or info.get("forwardPE")
            if pe_val and pe_val > 0:
                val_html = draw_valuation_meter_html("P/E Ratio", pe_val, 5, 60, 18, 35)
                st.html(val_html)

            pb_val = info.get("priceToBook")
            if pb_val and pb_val > 0:
                val_pb = draw_valuation_meter_html("Price / Book", pb_val, 0.5, 15, 2.0, 6.0)
                st.html(val_pb)

        with col_own:
            st.markdown("#### 🥧 Institutional Ownership")
            pie_own = make_ownership_pie(info)
            st.plotly_chart(pie_own, width="stretch")

        # 5. Technical Summary
        st.markdown("#### 📊 Technical Indicator Summary")
        tech_summary = get_technical_summary(stock_df)
        t_cols = st.columns(4)
        t_cols[0].metric("14-Day RSI", f"{tech_summary.get('rsi', 0):.1f}", delta=tech_summary.get("rsi_signal", "Neutral"))
        t_cols[1].metric("20-Day SMA", f"{curr_sym}{tech_summary.get('sma20', 0):,.2f}")
        t_cols[2].metric("50-Day SMA", f"{curr_sym}{tech_summary.get('sma50', 0):,.2f}")
        t_cols[3].metric("200-Day SMA", f"{curr_sym}{tech_summary.get('sma200', 0):,.2f}")
