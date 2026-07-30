import streamlit as st

st.set_page_config(
    page_title="AlphaForge | Multi-Factor Equity Research",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("⚡ AlphaForge")
st.subheader("Institutional Multi-Factor Equity Research & Portfolio Construction Platform")

st.write(
    """
    AlphaForge provides an end-to-end quantitative investment workflow for
    factor research, portfolio optimization, risk analytics, performance
    attribution, and historical strategy evaluation.
    """
)

st.divider()

# --------------------------------------------------
# Workflow Overview
# --------------------------------------------------
st.markdown("## Investment Workflow")

col1, col2 = st.columns(2)

with col1:

    with st.container(border=True):
        st.markdown("### 🔬 Research")
        st.write(
            """
            Explore the investment universe, construct factor scores,
            analyze Momentum, Value, Quality, and Low Volatility signals,
            and identify the highest-ranked securities.
            """
        )

    with st.container(border=True):
        st.markdown("### ⚖️ Portfolio Optimization")
        st.write(
            """
            Estimate expected returns, generate the covariance matrix,
            and construct an optimal portfolio using constrained
            Mean-Variance Optimization.
            """
        )

    with st.container(border=True):
        st.markdown("### 📈 Backtesting")
        st.write(
            """
            Evaluate historical performance of the optimized strategy
            against Equal Weight and Benchmark portfolios using rolling
            portfolio simulations.
            """
        )

with col2:

    with st.container(border=True):
        st.markdown("### 🛡️ Risk Analytics")
        st.write(
            """
            Measure portfolio risk through volatility decomposition,
            drawdown analysis, VaR, CVaR, tracking error,
            Information Ratio, Treynor Ratio, Omega Ratio,
            Calmar Ratio, and stress testing.
            """
        )

    with st.container(border=True):
        st.markdown("### 🧩 Performance Attribution")
        st.write(
            """
            Decompose active returns into systematic factor exposures
            using factor-mimicking portfolios and performance attribution
            methodologies.
            """
        )

    with st.container(border=True):
        st.markdown("### 📊 Institutional Reporting")
        st.write(
            """
            Generate publication-quality visualizations,
            portfolio statistics, and investment reports suitable
            for institutional research workflows.
            """
        )

st.divider()

# --------------------------------------------------
# Platform Highlights
# --------------------------------------------------
st.markdown("## Platform Highlights")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Factors", "4")
m2.metric("Optimization", "Mean-Variance")
m3.metric("Risk Metrics", "15+")
m4.metric("Backtesting", "Historical")

st.info(
    "👈 Select a module from the sidebar to begin the quantitative investment workflow."
)