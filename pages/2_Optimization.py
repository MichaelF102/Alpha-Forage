import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="AlphaForge - Portfolio Optimization", page_icon="⚖️", layout="wide")

rf = 0.04

st.markdown("""
# ⚖️ Portfolio Construction & Optimization
This page details how factor-derived expected returns are combined with historical asset covariance to construct the optimal portfolio weights using a constrained Mean-Variance Optimizer.
""")

reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
er_path = os.path.join(reports_dir, "expected_returns.csv")
weights_path = os.path.join(reports_dir, "weights.csv")
frontier_path = os.path.join(reports_dir, "frontier.csv")

if not (os.path.exists(er_path) and os.path.exists(weights_path) and os.path.exists(frontier_path)):
    st.error("Please run the backend engine calculations first (`python main.py`).")
else:
    er_df = pd.read_csv(er_path)
    weights_df = pd.read_csv(weights_path)
    frontier_df = pd.read_csv(frontier_path)
    
    st.subheader("📝 Optimization Setup & Constraints")
    st.markdown(r"""
    To match institutional risk management practices, we avoid simple equal weighting or unconstrained historical estimations. We use the **Sequential Least Squares Programming (SLSQP)** algorithm to solve:

$$
\max_{\mathbf{w}}
\quad
\frac{\mathbf{w}^{T}\boldsymbol{\mu} - r_f}
{\sqrt{\mathbf{w}^{T}\boldsymbol{\Sigma}\mathbf{w}}}
$$
    

**Subject to the following institutional constraints:**

- **Fully Invested**:
  $$
  \sum_{i=1}^{N} w_i = 100\%
  $$

- **Single Ticker Limits**:
  $$
  1\% \leq w_i \leq 10\%
  $$

- **Sector Limits**:
  $$
  \sum_{i \in \mathrm{Sector}} w_i \leq 30\%
  $$
        """)
        
    st.subheader("🔮 Expected Returns (Factor-Model Derived)")
    st.markdown(
        r"Rather than using historical averages (which are backward-looking and noisy), expected returns are calculated using a factor-pricing model: "
        
        r"$\mu_i = r_f + \beta_i \cdot \mathrm{MRP} + (\mathrm{Composite\ Score}_i - 0.5) \cdot 8\%$"
    )   
    
    # Display Expected Returns for selected stocks
    selected_tickers = weights_df[weights_df['Selected']]['Ticker'].tolist()
    er_selected = er_df[er_df['Ticker'].isin(selected_tickers)].sort_values(by='ExpectedReturn', ascending=False)
    
    st.dataframe(
        er_selected.style.format({
            'Composite': '{:.2f}',
            'ExpectedReturn': '{:.2%}'
        }).background_gradient(subset=['ExpectedReturn'], cmap='viridis'),
        use_container_width=True,
        height=260
    )
        
    
    st.subheader("⚖️ Optimal Portfolio Weights")
    st.markdown("Optimized weights for the selected 15 assets (the remaining 33 universe assets are assigned 0% weight):")
    
    opt_weights = weights_df[weights_df['Selected']].sort_values(by='Weight', ascending=False)
    st.dataframe(
        opt_weights.style.format({
            'Weight': '{:.2%}'
        }).bar(subset=['Weight'], color='#00C8FF'),
        use_container_width=True,
        height=430
    )
        
    st.divider()
    
    # Efficient Frontier Chart
    st.subheader("📈 Efficient Frontier Space (Mean-Variance)")
    st.markdown("Below is the simulated portfolio opportunity set plotted against the calculated Efficient Frontier. The **Optimized Portfolio** represents the Maximum Sharpe Ratio point under the active constraints.")
    
    # Separate types
    sim_points = frontier_df[frontier_df['Type'] == 'Simulated']
    max_sharpe = frontier_df[frontier_df['Type'] == 'Max Sharpe'].iloc[0]
    min_vol = frontier_df[frontier_df['Type'] == 'Min Vol'].iloc[0]
    
    fig = go.Figure()
    
    # Scatter plot of random portfolios
    fig.add_trace(go.Scatter(
        x=sim_points['Risk'],
        y=sim_points['Return'],
        mode='markers',
        name='Simulated Portfolios',
        marker=dict(
            size=5,
            color=(sim_points['Return'] - rf) / sim_points['Risk'],
            colorscale='Teal_r',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        opacity=0.6
    ))
    
    # Highlight Minimum Volatility
    fig.add_trace(go.Scatter(
        x=[min_vol['Risk']],
        y=[min_vol['Return']],
        mode='markers',
        name='Min Volatility Portfolio',
        marker=dict(color='#00FFCC', size=15, symbol='circle')
    ))
    
    # Highlight Max Sharpe (Our Portfolio)
    fig.add_trace(go.Scatter(
        x=[max_sharpe['Risk']],
        y=[max_sharpe['Return']],
        mode='markers',
        name='Optimized Factor Portfolio (Max Sharpe)',
        marker=dict(color='#FFD700', size=18, symbol='star')
    ))
    
    fig.update_layout(
        title="Modern Portfolio Theory (MPT) Efficient Frontier",
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Expected Return",
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        yaxis=dict(showgrid=True, gridcolor="#21262D", tickformat=".1%"),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
