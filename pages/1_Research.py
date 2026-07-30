import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="AlphaForge - Factor Research", page_icon="🔬", layout="wide")

st.markdown("""
# 🔬 Factor Research & Universe Selection
This page details the construction of our investment factors, their respective formulas, and the final ranking of the universe to determine selection status.
""")

# Load factors
reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
factors_path = os.path.join(reports_dir, "factors.csv")

if not os.path.exists(factors_path):
    st.error("Please run the backend engine calculations first (`python main.py`).")
else:
    factors_df = pd.read_csv(factors_path)
    
    # 1. Explain Factor Construction
    st.subheader("💡 Factor Definitions & Methodology")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Momentum Factor")
        st.write(
            "Captures the tendency of assets with high recent returns to continue outperforming."
        )
        st.latex(r"\text{12-1 Month Momentum}=\frac{P_{t-21}}{P_{t-252}}-1")
        st.caption("Uses the most recent 12 months excluding the latest trading month to avoid short-term reversal effects.")

        st.markdown("---")

        st.markdown("### 🛡️ Low Volatility Factor")
        st.write(
            "Captures the low-volatility anomaly where less volatile stocks often deliver superior risk-adjusted returns."
        )
        st.latex(r"\sigma_{\mathrm{ann}}=\mathrm{std}(r_{\mathrm{daily}})\times\sqrt{252}")
        st.caption("Stocks are ranked in reverse so lower volatility receives higher scores.")

    with col2:
        st.markdown("### 💎 Value Factor")
        st.write(
            "Identifies undervalued companies using fundamental valuation metrics."
        )
        st.latex(r"\text{Value Score}=\frac{1}{\mathrm{P/E}}")
        st.caption("Higher earnings yield implies a cheaper valuation.")

        st.markdown("---")

        st.markdown("### 🏆 Quality Factor")
        st.write(
            "Selects companies with strong profitability and operational efficiency."
        )
        st.latex(r"\text{Quality Score}=\mathrm{ROE}")
        st.caption("Return on Equity (ROE) is used as the quality metric.")
        
    st.divider()
    
    # 2. Rankings and Selection Universe
    st.subheader("📊 Universe Rankings & Selection Status")
    st.markdown("We rank the entire universe on each factor from 0.0 to 1.0 (percentile rank), compute the **Composite Factor Score**, and select the **Top 15 stocks**.")
    
    # Format selection column with icons
    display_df = factors_df.copy()
    display_df['Selected?'] = display_df['Selected'].apply(lambda x: "✅ Selected" if x else "❌ Rejected")
    
    # Select columns to show
    cols_to_show = [
        'Rank', 'Ticker', 'Sector', 'Momentum', 'Volatility', 'Value_PE', 'Quality_ROE', 
        'Momentum_Rank', 'Volatility_Rank', 'Value_Rank', 'Quality_Rank', 'Composite', 'Selected?'
    ]
    
    # Rename columns for presentation
    rename_dict = {
        'Value_PE': 'Price/Earnings (P/E)',
        'Quality_ROE': 'Return on Equity (ROE)',
        'Volatility': 'Annualized Vol',
        'Momentum_Rank': 'Momentum Pct',
        'Volatility_Rank': 'Low Vol Pct',
        'Value_Rank': 'Value Pct',
        'Quality_Rank': 'Quality Pct',
        'Composite': 'Composite Score'
    }
    
    styled_df = display_df[cols_to_show].rename(columns=rename_dict)
    
    # Display table
    st.dataframe(
        styled_df.style.format({
            'Momentum': '{:.2%}',
            'Annualized Vol': '{:.2%}',
            'Return on Equity (ROE)': '{:.2%}',
            'Price/Earnings (P/E)': '{:.1f}',
            'Momentum Pct': '{:.2f}',
            'Low Vol Pct': '{:.2f}',
            'Value Pct': '{:.2f}',
            'Quality Pct': '{:.2f}',
            'Composite Score': '{:.2f}'
        }).background_gradient(subset=['Composite Score'], cmap='coolwarm'),
        use_container_width=True,
        height=450
    )
    
    st.divider()
    
    # 3. Sector Distribution of Selected
    st.subheader("🏢 Selected Universe Sector Exposures")
    selected_only = factors_df[factors_df['Selected']]
    sector_counts = selected_only['Sector'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'Count']
    
    fig = px.bar(
        sector_counts, 
        x='Count', 
        y='Sector', 
        orientation='h',
        title="Number of Selected Stocks by Sector",
        color='Count',
        color_continuous_scale='Teal'
    )
    fig.update_layout(
        plot_bgcolor="#161B22",
        paper_bgcolor="#0E1117",
        font_color="#FFFFFF",
        xaxis=dict(showgrid=True, gridcolor="#21262D"),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)
