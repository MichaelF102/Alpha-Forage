import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="AlphaForge - Factor Attribution", page_icon="🧩", layout="wide")

st.markdown("""
# 🧩 Factor Return Attribution
This page isolates the specific drivers of the portfolio's active performance. By decomposing active returns against factor-mimicking portfolios, we can attribute performance directly to our intentional factor exposures.
""")

reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
attrib_mean_path = os.path.join(reports_dir, "attribution.csv")
attrib_ts_path = os.path.join(reports_dir, "attribution_timeseries.csv")

if not (os.path.exists(attrib_mean_path) and os.path.exists(attrib_ts_path)):
    st.error("Please run the backend engine calculations first (`python main.py`).")
else:
    mean_attrib = pd.read_csv(attrib_mean_path, index_col=0)
    ts_attrib = pd.read_csv(attrib_ts_path, index_col=0)
    ts_attrib.index = pd.to_datetime(ts_attrib.index)
    
    
    st.subheader("📊 Average Daily Factor Returns & Attribution")
    st.markdown(r"""
We map our active weights against **Fama-French style long-short factor portfolios**:

- **Momentum**: Top 10 minus Bottom 10 stocks ranked by 12–1 month Momentum.
- **Low Volatility**: Top 10 minus Bottom 10 stocks ranked by Low Volatility.
- **Value**: Top 10 minus Bottom 10 stocks ranked by Inverse P/E.
- **Quality**: Top 10 minus Bottom 10 stocks ranked by ROE.

The factor return contribution is computed as:

$$
\mathrm{Contribution}_f
=
\beta_{\mathrm{active},\,f}
\cdot
\mathrm{Factor\ Return}_f
$$
""")
        
    # Display attribution table (daily & annualized)
    attrib_display = mean_attrib.copy()
    attrib_display['Annualized Contribution'] = attrib_display['Contribution'] * 252
    
    st.dataframe(
        attrib_display.style.format({
            'Contribution': '{:.4%}',
            'Annualized Contribution': '{:+.2%}'
        }).background_gradient(subset=['Annualized Contribution'], cmap='RdYlGn'),
        use_container_width=True
    )
        
        
    st.subheader("🧩 Alpha Decomposition Bar Chart")
    st.markdown("Annualized returns attributed to factor components:")
    
    # Extract factors only (omit Market/Residual if wanted, or include them)
    factor_names = ['Momentum', 'Volatility', 'Value', 'Quality', 'Residual']
    factor_contribs = [attrib_display.loc[f, 'Annualized Contribution'] for f in factor_names if f in attrib_display.index]
    
    fig_bar = px.bar(
        x=factor_names,
        y=factor_contribs,
        labels=dict(x="Factor", y="Annualized Return Contribution (%)"),
        title="Annualized Return Attributed to Factor Tilt",
        color=factor_contribs,
        color_continuous_scale="RdYlGn",
        text_auto=".2%"
    )
    fig_bar.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
        
    st.divider()
    
    # Cumulative attribution timeseries chart
    st.subheader("📈 Cumulative Factor Return Decomposition")
    st.markdown("This timeseries displays how our active factor tilts generated return over time relative to the index.")
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts_attrib.index, y=ts_attrib['Momentum'],
        mode='lines', name='Momentum Attribution',
        line=dict(color='#00C8FF', width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_attrib.index, y=ts_attrib['Volatility'],
        mode='lines', name='Low Volatility Attribution',
        line=dict(color='#FFD700', width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_attrib.index, y=ts_attrib['Value'],
        mode='lines', name='Value Attribution',
        line=dict(color='#A8B3C4', width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_attrib.index, y=ts_attrib['Quality'],
        mode='lines', name='Quality Attribution',
        line=dict(color='#FF5E62', width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_attrib.index, y=ts_attrib['Residual'],
        mode='lines', name='Residual Attribution',
        line=dict(color='#9B5DE5', width=2, dash='dot')
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_attrib.index, y=ts_attrib['Market'],
        mode='lines', name='Market Index (Nifty 50)',
        line=dict(color='#00FFCC', width=1.5, dash='dash')
    ))
    
    fig_ts.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        yaxis=dict(showgrid=True, gridcolor="#21262D", tickformat=".1%"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
    )
    st.plotly_chart(fig_ts, use_container_width=True)
