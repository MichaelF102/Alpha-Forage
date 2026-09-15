"""
AlphaForge Sidebar Module
Manages unified data controls: market region (India / US), exchange, market cap filter,
universe mode (Automatic / Custom Basket), tactical factor tilts, optimization models,
and single stock deep-dive inspection.
"""

import os
import streamlit as st
import pandas as pd
from helper import fetch_stocks, categorize_market_cap, format_ticker_for_yf, fetch_periods_intervals
from engine import QuantEngine

def get_or_run_quant_results(force_rerun=False):
    """Fetch current quantitative pipeline results from session_state or disk, or run if absent."""
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    factors_file = os.path.join(reports_dir, "factors.csv")

    # If results exist in session_state and not force_rerun, return them
    if not force_rerun and "quant_results" in st.session_state:
        return st.session_state["quant_results"]

    # Gather session parameters
    region = st.session_state.get("region", "India")
    exchange = st.session_state.get("exchange", "NSE" if region == "India" else "NASDAQ")
    mcap = st.session_state.get("market_cap_filter", "All")
    uni_size = st.session_state.get("universe_size", 30)
    period = st.session_state.get("period", "2y")
    custom_tickers = st.session_state.get("custom_basket_symbols", None)
    uni_mode = st.session_state.get("universe_mode", "Automatic (Top Market Cap)")
    opt_model = st.session_state.get("opt_model_key", "Sharpe")
    factor_weights = st.session_state.get("factor_weights", {"Momentum": 0.25, "Volatility": 0.25, "Value": 0.25, "Quality": 0.25})

    active_custom = custom_tickers if "Custom Basket" in uni_mode and custom_tickers and len(custom_tickers) >= 2 else None

    with st.spinner(f"Computing quantitative factor model for {region}..."):
        engine = QuantEngine(
            region=region,
            exchange=exchange,
            market_cap_filter=mcap,
            universe_size=uni_size,
            period=period,
            custom_tickers=active_custom,
            factor_weights=factor_weights,
            optimization_model=opt_model
        )
        results = engine.run(export_files=True, reports_dir=reports_dir)
        st.session_state["quant_results"] = results
        return results

def render_sidebar():
    """
    Render unified sidebar for AlphaForge:
    1. Market Region (India / US)
    2. Universe Selection Mode (Auto Top Market Cap vs Custom Basket)
    3. Optimization Strategy Selector (Sharpe, MinVol, HRP)
    4. Tactical Factor Tilt Sliders & Presets
    5. Action button to trigger multi-factor pipeline
    6. Single Stock Inspection / Deep-Dive selector
    """
    st.sidebar.markdown("## ⚡ AlphaForge Controls")

    # 1. Market Region (India / US)
    curr_region = st.session_state.get("region", "India")
    region_idx = 0 if curr_region == "India" else 1
    region = st.sidebar.radio(
        "Market Region",
        ["India", "US"],
        index=region_idx,
        horizontal=True,
        help="Select equity market universe from local CSV datasets"
    )
    prev_region = st.session_state.get("region")
    region_changed = (prev_region is not None and prev_region != region)
    st.session_state["region"] = region

    # Fetch stock metadata from CSV for selected region
    stocks_df = fetch_stocks(region)
    if stocks_df.empty:
        st.sidebar.error(f"No stock data found in CSV for '{region}'.")
        st.stop()

    # Format Option Label
    stocks_df["Option_Label"] = stocks_df.apply(
        lambda r: f"{r['Symbol']} - {r['Description']}" if pd.notna(r.get('Description')) and str(r['Symbol']).strip() != str(r['Description']).strip() else str(r['Symbol']),
        axis=1
    )

    # 2. Universe Mode: Automatic vs Custom Basket
    uni_modes = ["Automatic (Top Market Cap)", "Custom Basket (Select Stocks)"]
    curr_mode = st.session_state.get("universe_mode", uni_modes[0])
    mode_idx = uni_modes.index(curr_mode) if curr_mode in uni_modes else 0
    universe_mode = st.sidebar.radio(
        "Universe Mode",
        uni_modes,
        index=mode_idx,
        help="Choose automatic selection of largest market cap stocks or hand-pick a custom equity basket"
    )
    st.session_state["universe_mode"] = universe_mode

    exchange = "All"
    market_cap_filter = "All"
    universe_size = 30
    custom_basket_symbols = None

    if "Automatic" in universe_mode:
        # Exchange Selection
        raw_exchanges = list(stocks_df["Exchange"].dropna().unique())
        preferred_order = ["NSE", "BSE"] if region == "India" else ["NASDAQ", "NYSE", "All"]
        available_exchanges = [e for e in preferred_order if e in raw_exchanges] + [e for e in raw_exchanges if e not in preferred_order]
        if "All" not in available_exchanges and len(available_exchanges) > 1:
            available_exchanges.append("All")

        default_ex = "NSE" if region == "India" else "NASDAQ"
        curr_exchange = st.session_state.get("exchange", default_ex if default_ex in available_exchanges else available_exchanges[0])
        ex_idx = available_exchanges.index(curr_exchange) if curr_exchange in available_exchanges else 0

        exchange = st.sidebar.selectbox("Exchange", available_exchanges, index=ex_idx)
        st.session_state["exchange"] = exchange

        # Filter stock options by selected Exchange
        if exchange != "All":
            filtered_stocks = stocks_df[stocks_df["Exchange"] == exchange].copy()
        else:
            filtered_stocks = stocks_df.copy()

        # Market Cap Filter
        mcap_options = ["All", "Large Cap", "Mid Cap", "Small Cap", "Micro Cap"]
        curr_mcap = st.session_state.get("market_cap_filter", "All")
        mcap_idx = mcap_options.index(curr_mcap) if curr_mcap in mcap_options else 0

        market_cap_filter = st.sidebar.selectbox("Market Cap Category", mcap_options, index=mcap_idx)
        st.session_state["market_cap_filter"] = market_cap_filter

        filtered_stocks["Market_Cap_Category"] = filtered_stocks.apply(
            lambda r: categorize_market_cap(r, region), axis=1
        )
        if market_cap_filter != "All":
            mcap_filtered = filtered_stocks[filtered_stocks["Market_Cap_Category"] == market_cap_filter]
            if not mcap_filtered.empty:
                filtered_stocks = mcap_filtered

        # Universe Size
        uni_size_options = [20, 30, 40, 50]
        curr_uni_size = st.session_state.get("universe_size", 30)
        uni_idx = uni_size_options.index(curr_uni_size) if curr_uni_size in uni_size_options else 1
        universe_size = st.sidebar.selectbox("Universe Size (Top Liquid)", uni_size_options, index=uni_idx)
        st.session_state["universe_size"] = universe_size

        st.sidebar.markdown(f"📊 **Available Stocks**: `{len(filtered_stocks):,}`")

    else:
        # Custom Basket Multi-Select
        filtered_stocks = stocks_df.copy()
        all_symbols = list(stocks_df["Symbol"].dropna().unique())
        default_basket = all_symbols[:8] if region == "India" else ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD"]
        default_basket = [s for s in default_basket if s in all_symbols]

        selected_symbols = st.sidebar.multiselect(
            "Select Custom Equities (Min 3)",
            options=all_symbols,
            default=default_basket,
            help="Select the specific stocks you want to optimize in your custom portfolio"
        )
        custom_basket_symbols = selected_symbols
        st.session_state["custom_basket_symbols"] = custom_basket_symbols
        st.sidebar.markdown(f"💼 **Custom Basket Count**: `{len(selected_symbols)}` securities")

    # 3. Optimization Model Selector
    opt_models = {
        "Max Sharpe Ratio (Mean-Variance)": "Sharpe",
        "Minimum Volatility (Capital Defense)": "MinVol",
        "Hierarchical Risk Parity (HRP)": "HRP"
    }
    opt_names = list(opt_models.keys())
    curr_opt_name = st.session_state.get("opt_model_name", opt_names[0])
    opt_idx = opt_names.index(curr_opt_name) if curr_opt_name in opt_names else 0

    selected_opt_name = st.sidebar.selectbox(
        "Optimization Model",
        opt_names,
        index=opt_idx,
        help="Select portfolio construction algorithm"
    )
    st.session_state["opt_model_name"] = selected_opt_name
    st.session_state["opt_model_key"] = opt_models[selected_opt_name]

    # 4. Tactical Factor Tilting Controls (Expander)
    with st.sidebar.expander("🎛️ Tactical Factor Tilts", expanded=False):
        tilt_presets = {
            "Balanced (25/25/25/25)": (25, 25, 25, 25),
            "Aggressive Momentum": (50, 15, 15, 20),
            "Deep Value & Yield": (10, 15, 50, 25),
            "Quality Growth": (25, 10, 15, 50),
            "Low Vol Defense": (10, 50, 15, 25),
            "Custom Sliders": None
        }
        preset_names = list(tilt_presets.keys())
        curr_preset = st.session_state.get("tilt_preset", preset_names[0])
        pre_idx = preset_names.index(curr_preset) if curr_preset in preset_names else 0

        selected_preset = st.selectbox("Factor Preset", preset_names, index=pre_idx)
        st.session_state["tilt_preset"] = selected_preset

        if tilt_presets[selected_preset] is not None:
            w_mom, w_vol, w_val, w_qual = tilt_presets[selected_preset]
        else:
            w_mom = st.slider("Momentum Weight (%)", 0, 100, 25, 5)
            w_vol = st.slider("Low Volatility Weight (%)", 0, 100, 25, 5)
            w_val = st.slider("Fundamental Value (%)", 0, 100, 25, 5)
            w_qual = st.slider("Quality (EPS Growth) (%)", 0, 100, 25, 5)

        total_weight = w_mom + w_vol + w_val + w_qual
        st.caption(f"Weights: {w_mom}% Mom | {w_vol}% Vol | {w_val}% Val | {w_qual}% Qual")
        st.session_state["factor_weights"] = {
            "Momentum": w_mom / max(total_weight, 1),
            "Volatility": w_vol / max(total_weight, 1),
            "Value": w_val / max(total_weight, 1),
            "Quality": w_qual / max(total_weight, 1)
        }

    # Lookback Period
    period_options = ["1y", "2y", "3y", "5y"]
    curr_period = st.session_state.get("period", "2y")
    p_idx = period_options.index(curr_period) if curr_period in period_options else 1
    period = st.sidebar.selectbox("Lookback Period", period_options, index=p_idx)
    st.session_state["period"] = period

    # Run Pipeline Button
    st.sidebar.markdown("")
    rerun_requested = st.sidebar.button("🚀 Run / Refresh Factor Engine", type="primary", width="stretch")

    if rerun_requested or region_changed:
        get_or_run_quant_results(force_rerun=True)
        st.sidebar.success("Quantitative factor model updated!")

    st.sidebar.divider()

    # 5. Single Stock Deep Dive Selector
    st.sidebar.markdown("### 🔍 Single Stock Deep Dive")
    sorted_options = list(filtered_stocks["Option_Label"].dropna().sort_values().unique())
    curr_option = st.session_state.get("stock_option", sorted_options[0] if sorted_options else "")
    opt_idx = sorted_options.index(curr_option) if curr_option in sorted_options else 0

    selected_option = st.sidebar.selectbox(
        "Select Stock for Deep Dive",
        sorted_options,
        index=opt_idx,
        help="Search and inspect individual stock fundamentals, valuation meters, and technical health"
    )
    st.session_state["stock_option"] = selected_option

    selected_row = filtered_stocks.loc[filtered_stocks["Option_Label"] == selected_option].iloc[0]
    symbol = str(selected_row["Symbol"]).strip()
    company = str(selected_row.get("Description", symbol)).strip()
    sector = str(selected_row.get("Sector", "General")).strip()
    price = selected_row.get("Price")

    ticker = format_ticker_for_yf(symbol, region, selected_row.get("Exchange", exchange))
    st.sidebar.text_input("Yahoo Finance Ticker", value=ticker, disabled=True)

    curr_sym = "₹" if region == "India" else "$"
    price_str = f"{curr_sym}{price:,.2f}" if pd.notna(price) else "N/A"
    st.sidebar.caption(f"🏢 **Sector**: {sector} | 💵 **Price**: {price_str}")

    st.session_state["ticker"] = ticker
    st.session_state["selected_ticker"] = ticker
    st.session_state["company"] = company
    st.session_state["symbol"] = symbol
    st.session_state["sector"] = sector
    st.session_state["currency"] = "INR" if region == "India" else "USD"
    st.session_state["currency_symbol"] = curr_sym

    st.sidebar.markdown("---")
    st.sidebar.caption("⚡ **AlphaForge** Quantitative Research")

    return ticker, company, exchange, period, "1d", region
