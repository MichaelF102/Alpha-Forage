import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="AlphaForge - Backtesting", page_icon="📈", layout="wide")

st.markdown("""
# 📈 Strategy Backtesting
This page displays the historical backtest of the **Optimized Factor Portfolio** compared against a naive **Equal Weighted Portfolio** and the **Nifty 50 Index (Benchmark)**. You can customize the initial investment amount and date range in the sidebar.
""")

reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
portfolio_path = os.path.join(reports_dir, "portfolio.csv")
performance_path = os.path.join(reports_dir, "performance.csv")
rolling_path = os.path.join(reports_dir, "rolling.csv")

if not (os.path.exists(portfolio_path) and os.path.exists(performance_path) and os.path.exists(rolling_path)):
    st.error("Please run the backend engine calculations first (`python main.py`).")
else:
    portfolio_df = pd.read_csv(portfolio_path, index_col=0)
    performance_df = pd.read_csv(performance_path)
    rolling_df = pd.read_csv(rolling_path, index_col=0)
    
    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    rolling_df.index = pd.to_datetime(rolling_df.index)
    
    # 0. Sidebar Controls
    st.sidebar.header("⚙️ Backtest Settings")
    
    initial_amount = st.sidebar.number_input(
        "Initial Investment (INR)",
        min_value=100.0,
        max_value=100000000.0,
        value=10000.0,
        step=5000.0,
        format="%.0f"
    )
    
    min_date = portfolio_df.index.min().date()
    max_date = portfolio_df.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Select Backtest Period",
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
    sliced_df = portfolio_df.loc[sd_dt:ed_dt]
    sliced_rolling = rolling_df.loc[sd_dt:ed_dt]
    
    if len(sliced_df) < 5:
        st.warning("Please select a date range containing at least 5 trading days.")
    else:
        sliced_df = sliced_df.copy()
        
        # Calculate wealth indices dynamically from daily returns
        # WealthIndex_t = Initial_Amount * Cumprod(1 + Return_t)
        sliced_df['Portfolio_Wealth'] = initial_amount * (1 + sliced_df['Return']).cumprod()
        sliced_df['EqualWeight_Wealth'] = initial_amount * (1 + sliced_df['EqualWeight_Return']).cumprod()
        sliced_df['Benchmark_Wealth'] = initial_amount * (1 + sliced_df['Benchmark_Return']).cumprod()
        
        # 1. Recalculate Performance Summary
        st.subheader("📊 Performance Summary Comparison ")
        
        n_days = len(sliced_df)
        years = n_days / 252.0
        rf_rate = 0.04
        
        def calculate_sliced_stats(ret_series, cum_series):
            # CAGR
            cagr = (cum_series.iloc[-1] / cum_series.iloc[0]) ** (1.0 / years) - 1.0 if len(cum_series) > 1 else 0.0
            # Volatility
            vol = ret_series.std() * np.sqrt(252)
            # Sharpe
            excess_ret = ret_series.mean() * 252 - rf_rate
            sharpe = excess_ret / vol if vol != 0 else 0.0
            # Sortino
            downside_ret = ret_series[ret_series < 0]
            downside_vol = downside_ret.std() * np.sqrt(252)
            sortino = excess_ret / downside_vol if downside_vol != 0 else 0.0
            # Max Drawdown
            peaks = cum_series.cummax()
            dd = (cum_series - peaks) / peaks
            max_dd = dd.min()
            return cagr, vol, sharpe, sortino, max_dd
            
        cagr_p, vol_p, sharpe_p, sortino_p, dd_p = calculate_sliced_stats(sliced_df['Return'], sliced_df['Portfolio_Wealth'])
        cagr_ew, vol_ew, sharpe_ew, sortino_ew, dd_ew = calculate_sliced_stats(sliced_df['EqualWeight_Return'], sliced_df['EqualWeight_Wealth'])
        cagr_bm, vol_bm, sharpe_bm, sortino_bm, dd_bm = calculate_sliced_stats(sliced_df['Benchmark_Return'], sliced_df['Benchmark_Wealth'])
        
        recalc_perf_df = pd.DataFrame([
            {'Metric': 'CAGR', 'Portfolio': cagr_p, 'Benchmark': cagr_bm, 'EqualWeight': cagr_ew},
            {'Metric': 'Volatility', 'Portfolio': vol_p, 'Benchmark': vol_bm, 'EqualWeight': vol_ew},
            {'Metric': 'Sharpe Ratio', 'Portfolio': sharpe_p, 'Benchmark': sharpe_bm, 'EqualWeight': sharpe_ew},
            {'Metric': 'Sortino Ratio', 'Portfolio': sortino_p, 'Benchmark': sortino_bm, 'EqualWeight': sortino_ew},
            {'Metric': 'Max Drawdown', 'Portfolio': dd_p, 'Benchmark': dd_bm, 'EqualWeight': dd_ew}
        ])
        
        st.dataframe(
            recalc_perf_df.style.format({
                'Portfolio': '{:.2%}',
                'Benchmark': '{:.2%}',
                'EqualWeight': '{:.2%}'
            }),
            use_container_width=True
        )
        
        st.divider()
        
        # 2. Cumulative Growth Chart
        st.subheader(f"💵 Cumulative Wealth Growth (₹{initial_amount:,.0f} Initial Investment)")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sliced_df.index, y=sliced_df['Portfolio_Wealth'],
            mode='lines', name='Optimized Factor Portfolio',
            line=dict(color='#00C8FF', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=sliced_df.index, y=sliced_df['EqualWeight_Wealth'],
            mode='lines', name='Equal Weighted Portfolio',
            line=dict(color='#A8B3C4', width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=sliced_df.index, y=sliced_df['Benchmark_Wealth'],
            mode='lines', name='Nifty 50 Index (Benchmark)',
            line=dict(color='#FF5E62', width=2)
        ))
        
        fig.update_layout(
            plot_bgcolor="#161B22",
            paper_bgcolor="#0E1117",
            font_color="#FFFFFF",
            xaxis=dict(showgrid=True, gridcolor="#21262D"),
            yaxis=dict(showgrid=True, gridcolor="#21262D", tickprefix="₹"),
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # 3. Monthly Returns Heatmap
        st.subheader("🗓️ Compounded Monthly Returns Heatmap (Optimized Portfolio)")
        
        monthly_ret = sliced_df['Return'].groupby([sliced_df.index.year, sliced_df.index.month]).apply(lambda x: (1 + x).prod() - 1.0)
        monthly_ret = monthly_ret.unstack(level=-1)
        # Handle cases where not all months are selected
        all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_ret.columns = [all_months[m-1] for m in monthly_ret.columns]
        
        fig_heatmap = px.imshow(
            monthly_ret * 100.0,
            labels=dict(x="Month", y="Year", color="Return (%)"),
            x=monthly_ret.columns,
            y=monthly_ret.index,
            color_continuous_scale="RdYlGn",
            text_auto=".1f"
        )
        fig_heatmap.update_layout(
            plot_bgcolor="#161B22",
            paper_bgcolor="#0E1117",
            font_color="#FFFFFF"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.divider()
        
        # 4. Rolling Risk Stats
        st.subheader("🛡️ 30-Day Rolling Risk Analytics")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_rvol = px.line(
                sliced_rolling, x=sliced_rolling.index, y='RollingVolatility',
                title="30-Day Rolling Annualized Volatility",
                color_discrete_sequence=['#00C8FF']
            )
            fig_rvol.update_layout(
                plot_bgcolor="#161B22",
                paper_bgcolor="#0E1117",
                font_color="#FFFFFF",
                xaxis=dict(showgrid=True, gridcolor="#21262D"),
                yaxis=dict(showgrid=True, gridcolor="#21262D", tickformat=".1%")
            )
            st.plotly_chart(fig_rvol, use_container_width=True)
            
        with col_r2:
            fig_rsharpe = px.line(
                sliced_rolling, x=sliced_rolling.index, y='RollingSharpe',
                title="30-Day Rolling Sharpe Ratio",
                color_discrete_sequence=['#FFD700']
            )
            fig_rsharpe.update_layout(
                plot_bgcolor="#161B22",
                paper_bgcolor="#0E1117",
                font_color="#FFFFFF",
                xaxis=dict(showgrid=True, gridcolor="#21262D"),
                yaxis=dict(showgrid=True, gridcolor="#21262D")
            )
            st.plotly_chart(fig_rsharpe, use_container_width=True)
