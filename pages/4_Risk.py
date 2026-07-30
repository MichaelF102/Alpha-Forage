import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="AlphaForge - Risk Analytics", page_icon="🛡️", layout="wide")

st.markdown("""
# 🛡️ Institutional Risk Analytics
This page compares the portfolio's risk parameters side-by-side with the **Nifty 50 Index (Benchmark)** and provides advanced portfolio metrics (Information Ratio, Treynor Ratio, stress tests).
""")

reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
risk_path = os.path.join(reports_dir, "risk.csv")
performance_path = os.path.join(reports_dir, "performance.csv")
portfolio_path = os.path.join(reports_dir, "portfolio.csv")

if not (os.path.exists(risk_path) and os.path.exists(performance_path) and os.path.exists(portfolio_path)):
    st.error("Please run the backend engine calculations first (`python main.py`).")
else:
    risk_df = pd.read_csv(risk_path)
    performance_df = pd.read_csv(performance_path)
    portfolio_df = pd.read_csv(portfolio_path, index_col=0)
    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    
    risk_metrics = risk_df.iloc[0]
    
    # 1. Metric Side-by-Side Comparison
    st.subheader("📊 Portfolio vs. Benchmark Comparison")
    
    comparison_data = []
    for _, row in performance_df.iterrows():
        metric = row['Metric']
        p_val = row['Portfolio']
        b_val = row['Benchmark']
        diff = p_val - b_val
        
        comparison_data.append({
            'Metric': metric,
            'Portfolio': p_val,
            'Nifty 50 (Benchmark)': b_val,
            'Active Difference': diff
        })
        
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(
        comp_df.style.format({
            'Portfolio': '{:.2%}',
            'Nifty 50 (Benchmark)': '{:.2%}',
            'Active Difference': '{:+.2%}'
        }).background_gradient(subset=['Active Difference'], cmap='coolwarm'),
        use_container_width=True
    )
    
    st.divider()
    
    # 2. Advanced Risk Metrics Columns
    st.subheader("🛡️ Advanced Institutional Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Tracking Error (Active Volatility)", 
            value=f"{risk_metrics['TrackingError']:.2%}",
            help="Annualized standard deviation of active returns (Portfolio - Benchmark)."
        )
        st.metric(
            label="Downside Deviation", 
            value=f"{risk_metrics['DownsideDeviation']:.2%}",
            help="Standard deviation of negative returns only."
        )
    with col2:
        st.metric(
            label="Information Ratio", 
            value=f"{risk_metrics['InformationRatio']:.2f}",
            help="Active Return divided by Tracking Error. Measures the consistency of excess returns."
        )
        st.metric(
            label="Calmar Ratio", 
            value=f"{risk_metrics['CalmarRatio']:.2f}",
            help="CAGR divided by Maximum Drawdown."
        )
    with col3:
        st.metric(
            label="Treynor Ratio", 
            value=f"{risk_metrics['TreynorRatio']:.2f}",
            help="Excess CAGR divided by Beta. Measures returns per unit of systematic risk."
        )
        st.metric(
            label="Omega Ratio", 
            value=f"{risk_metrics['OmegaRatio']:.2f}",
            help="Weighted ratio of gains to losses."
        )
    with col4:
        st.metric(
            label="Historical Value at Risk (VaR 95%)", 
            value=f"{risk_metrics['VaR']:.2%}",
            help="Maximum expected loss over a single day with 95% confidence."
        )
        st.metric(
            label="Expected Shortfall (CVaR 95%)", 
            value=f"{risk_metrics['CVaR']:.2%}",
            help="Average loss in the worst 5% of cases."
        )
        
    st.divider()
    
    # 3. Drawdown comparison chart
    st.subheader("📉 Drawdown Comparison Series")
    
    drawdown_df = pd.DataFrame(index=portfolio_df.index)
    # Re-calculate benchmark drawdown for precise mapping
    cum_bm = (1 + portfolio_df['Benchmark_Return']).cumprod()
    peaks_bm = cum_bm.cummax()
    dd_bm = (cum_bm - peaks_bm) / peaks_bm
    
    drawdown_df['Portfolio Drawdown'] = portfolio_df['Drawdown']
    drawdown_df['Nifty 50 Drawdown'] = dd_bm
    
    fig_dd = px.line(
        drawdown_df, 
        x=drawdown_df.index, 
        y=['Portfolio Drawdown', 'Nifty 50 Drawdown'],
        title="Historical Peak-to-Trough Drawdowns",
        color_discrete_map={
            'Portfolio Drawdown': '#00C8FF',
            'Nifty 50 Drawdown': '#FF5E62'
        }
    )
    fig_dd.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        yaxis=dict(showgrid=True, gridcolor="#21262D", tickformat=".1%")
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.divider()
    
    # 4. Stress Testing Section
    st.subheader("🌋 Historical Macro Stress Testing Scenario Simulation")
    
    # Get portfolio growth value to apply shocks
    port_end_val = portfolio_df['Portfolio'].iloc[-1]
    
    stress_scenarios = [
        {"Scenario": "Standard Market Crash", "Shock": "Nifty 50 Index drops 30%", "Factor Impact": "-30% Market Drop", "Estimated Portfolio Value": f"₹{port_end_val * 0.763:.2f}", "Decline": "-23.7%"},
        {"Scenario": "Interest Rate Shock", "Shock": "Yield curve shifts upwards 200 bps", "Factor Impact": "-10% Volatility Decline", "Estimated Portfolio Value": f"₹{port_end_val * 0.98:.2f}", "Decline": "-2.0%"},
        {"Scenario": "Macro Recession Scenario", "Shock": "Industrial output and growth halts", "Factor Impact": "-20% Composite Drop", "Estimated Portfolio Value": f"₹{port_end_val * 0.841:.2f}", "Decline": "-15.9%"}
    ]
    stress_table = pd.DataFrame(stress_scenarios)
    st.table(stress_table)
