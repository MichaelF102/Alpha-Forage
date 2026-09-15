"""
AlphaForge | Institutional Multi-Factor Equity Research & Portfolio Construction Platform
Main Application Landing Page
"""

import streamlit as st
import pandas as pd
import numpy as np

from helper import inject_custom_theme
from sidebar import render_sidebar, get_or_run_quant_results

def main():
    st.set_page_config(
        page_title="AlphaForge | Multi-Factor Equity Research",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject terminal theme
    inject_custom_theme()

    # Render Unified Sidebar
    ticker, company, exchange, period, interval, region = render_sidebar()

    # Get or run results
    results = get_or_run_quant_results()

    # --------------------------------------------------
    # Header
    # --------------------------------------------------
    st.title("⚡ AlphaForge")
    st.subheader(f"Institutional Multi-Factor Equity Research & Portfolio Platform — {region.upper()}")

    st.write(
        f"""
        AlphaForge provides an end-to-end quantitative investment workflow for factor research,
        portfolio optimization, risk analytics, and performance attribution across **{region}**
        and global equities. Powered by real fundamental factor scoring and constrained Mean-Variance optimization.
        """
    )

    st.divider()

    # --------------------------------------------------
    # Active Universe Snapshot Metrics
    # --------------------------------------------------
    st.markdown("### 🌐 Active Universe Overview")

    col1, col2, col3, col4 = st.columns(4)

    curr_sym = results.get("currency_symbol", "₹" if region == "India" else "$")
    benchmark_name = results.get("benchmark_name", "Nifty 50" if region == "India" else "S&P 500")
    uni_count = len(results["factors_df"]) if "factors_df" in results else 30
    selected_count = len(results["selected_tickers"]) if "selected_tickers" in results else 12

    col1.metric("Selected Region", f"{region} ({exchange})", delta=f"{curr_sym} Base")
    col2.metric("Market Benchmark", benchmark_name, delta=results.get("benchmark_ticker", ""))
    col3.metric("Factor Universe", f"{uni_count} Stocks", delta=f"{selected_count} Selected")

    top_stock = results["factors_df"].iloc[0]["Company"] if "factors_df" in results and not results["factors_df"].empty else "N/A"
    top_ticker = results["factors_df"].iloc[0]["Ticker"] if "factors_df" in results and not results["factors_df"].empty else ""
    col4.metric("Top Factor Alpha", top_ticker, delta=top_stock[:18])

    st.markdown("")

    # --------------------------------------------------
    # Workflow Navigation Cards
    # --------------------------------------------------
    st.markdown("### 🧭 Quantitative Investment Workflow")

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("#### 🔬 1. Factor Research & Universe Selection")
            st.write(
                "Rank universe on **Momentum**, **Low Volatility**, **Fundamental Value (P/E & Dividend Yield)**, "
                "and **Quality (YoY EPS Growth)**. Inspect individual stock health, valuation gauges, and technical summary."
            )
            st.page_link("pages/1_Research.py", label="Open Factor Research →", icon="🔬")

        with st.container(border=True):
            st.markdown("#### ⚖️ 2. Portfolio Optimization")
            st.write(
                "Derive factor-implied expected returns and construct optimal weights using constrained Mean-Variance Optimization "
                "with maximum ticker and sector exposure constraints, and view the Efficient Frontier."
            )
            st.page_link("pages/2_Optimization.py", label="Open Optimization →", icon="⚖️")

        with st.container(border=True):
            st.markdown("#### 📈 3. Strategy Backtesting")
            st.write(
                f"Evaluate historical performance of the Optimized Strategy against an Equal Weight portfolio and the "
                f"**{benchmark_name}** index. Analyze cumulative wealth, drawdown series, and monthly return heatmaps."
            )
            st.page_link("pages/3_Backtesting.py", label="Open Backtesting →", icon="📈")

    with c2:
        with st.container(border=True):
            st.markdown("#### 🛡️ 4. Institutional Risk Analytics")
            st.write(
                "Measure portfolio risk via tracking error, Information Ratio, Treynor Ratio, Calmar Ratio, Omega Ratio, "
                "Historical & Parametric VaR (95%), Conditional VaR (Expected Shortfall), and macroeconomic stress testing."
            )
            st.page_link("pages/4_Risk.py", label="Open Risk Analytics →", icon="🛡️")

        with st.container(border=True):
            st.markdown("#### 🧩 5. Performance & Factor Attribution")
            st.write(
                "Decompose active returns into systematic factor tilts using Fama-French style long-short factor mimicking portfolios "
                "(Momentum, Volatility, Value, Quality) and analyze cumulative alpha generation."
            )
            st.page_link("pages/5_Attribution.py", label="Open Attribution →", icon="🧩")

        with st.container(border=True):
            st.markdown("#### 🔍 Single Stock Deep Dive")
            st.write(
                f"Currently inspecting **{st.session_state.get('symbol', 'RELIANCE')}** ({st.session_state.get('company', '')}). "
                f"Head to the **Research** page to view comprehensive financial health scores, valuation meters, and technical indicators."
            )
            st.page_link("pages/1_Research.py", label="Inspect Selected Stock →", icon="🔍")

    st.divider()

    # --------------------------------------------------
    # Quick Snapshot Table
    # --------------------------------------------------
    if "performance_df" in results and "weights_df" in results:
        st.markdown("### 📊 Active Strategy Performance Snapshot")
        col_snap1, col_snap2 = st.columns([1, 1])

        with col_snap1:
            st.markdown("##### Comparative Performance")
            st.dataframe(
                results["performance_df"].style.format({
                    "Portfolio": "{:.2%}",
                    "Benchmark": "{:.2%}",
                    "EqualWeight": "{:.2%}"
                }),
                width="stretch"
            )

        with col_snap2:
            st.markdown("##### Top Portfolio Allocations")
            top_alloc = results["weights_df"][results["weights_df"]["Selected"]].sort_values(by="Weight", ascending=False).head(5)
            st.dataframe(
                top_alloc[["Ticker", "Company", "Sector", "Weight"]].style.format({
                    "Weight": "{:.2%}"
                }),
                width="stretch"
            )

    st.caption("Use the sidebar on the left to change markets (India / US), adjust filters, or inspect individual stocks.")

if __name__ == "__main__":
    main()