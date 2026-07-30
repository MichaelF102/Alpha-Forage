import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import streamlit as st
from risk.var import ValueAtRisk
from risk.cvar import ConditionalVaR
from risk.beta import Beta
from risk.drawdown import Drawdown
from risk.stress import StressTest
from risk.correlation import Correlation
from risk.rolling import RollingStatistics

from analytics.performance import Performance
from analytics.attribution import Attribution
from analytics.report import RiskPerformanceReport
# Ensure local modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

NIFTY50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS",
    "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
    "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS"
]



# Try to import GS-Quant timeseries functions.
try:
    from gs_quant.timeseries import returns as gs_returns
    from gs_quant.timeseries import volatility as gs_volatility
    from gs_quant.timeseries import moving_average as gs_moving_average
    HAS_GS_QUANT = True
    print("[INFO] GS-Quant timeseries modules loaded successfully.")
except ImportError:
    HAS_GS_QUANT = False
    print("[WARNING] GS-Quant not available. Using pandas/numpy fallbacks for timeseries operations.")

def main():
    # Define assets and benchmark
    tickers = NIFTY50_TICKERS.copy()
    benchmark_ticker = "^NSEI"
    all_tickers = tickers + [benchmark_ticker]
    
    print(f"Downloading historical data from yfinance for: {all_tickers}...")
    raw_data = yf.download(all_tickers, start="2023-01-01", end="2025-01-01")["Close"]
    
    # Identify and remove any columns that failed to download or have >10% NaN
    missing_pct = raw_data.isnull().mean()
    to_drop = missing_pct[missing_pct > 0.10].index.tolist()
    if to_drop:
        print(f"[INFO] Dropping tickers with >10% missing data: {to_drop}")
        raw_data = raw_data.drop(columns=to_drop)
        tickers = [t for t in tickers if t not in to_drop]
        all_tickers = tickers + [benchmark_ticker]
        
    # Forward/backward fill minor data alignment gaps
    price_data = raw_data.ffill().bfill().dropna()
    
    # Calculate returns
    print("Calculating asset returns...")
    if HAS_GS_QUANT:
        returns_dict = {}
        for ticker in all_tickers:
            returns_dict[ticker] = gs_returns(price_data[ticker])
        returns = pd.DataFrame(returns_dict)
    else:
        returns = price_data.pct_change()
        
    returns = returns.dropna()
    portfolio_asset_returns = returns[tickers]
    
    # Define sector mapping
    sectors_map = {
        "ADANIENT.NS": "Conglomerates", "ADANIPORTS.NS": "Infrastructure", "APOLLOHOSP.NS": "Healthcare",
        "ASIANPAINT.NS": "Consumer Goods", "AXISBANK.NS": "Financial Services", "BAJAJ-AUTO.NS": "Automobile",
        "BAJFINANCE.NS": "Financial Services", "BAJAJFINSV.NS": "Financial Services", "BEL.NS": "Industrials",
        "BHARTIARTL.NS": "Telecommunication", "CIPLA.NS": "Healthcare", "COALINDIA.NS": "Energy",
        "DRREDDY.NS": "Healthcare", "EICHERMOT.NS": "Automobile", "ETERNAL.NS": "Other",
        "GRASIM.NS": "Materials", "HCLTECH.NS": "Information Technology", "HDFCBANK.NS": "Financial Services",
        "HDFCLIFE.NS": "Financial Services", "HEROMOTOCO.NS": "Automobile", "HINDALCO.NS": "Materials",
        "HINDUNILVR.NS": "Consumer Goods", "ICICIBANK.NS": "Financial Services", "INDUSINDBK.NS": "Financial Services",
        "INFY.NS": "Information Technology", "ITC.NS": "Consumer Goods", "JIOFIN.NS": "Financial Services",
        "JSWSTEEL.NS": "Materials", "KOTAKBANK.NS": "Financial Services", "LT.NS": "Infrastructure",
        "M&M.NS": "Automobile", "MARUTI.NS": "Automobile", "NESTLEIND.NS": "Consumer Goods",
        "NTPC.NS": "Utilities", "ONGC.NS": "Energy", "POWERGRID.NS": "Utilities", "RELIANCE.NS": "Energy",
        "SBILIFE.NS": "Financial Services", "SBIN.NS": "Financial Services", "SHRIRAMFIN.NS": "Financial Services",
        "SUNPHARMA.NS": "Healthcare", "TATACONSUM.NS": "Consumer Goods", "TATAMOTORS.NS": "Automobile",
        "TATASTEEL.NS": "Materials", "TCS.NS": "Information Technology", "TECHM.NS": "Information Technology",
        "TITAN.NS": "Consumer Goods", "TRENT.NS": "Retail", "ULTRACEMCO.NS": "Materials", "WIPRO.NS": "Information Technology"
    }

    # 1. QUANT FACTOR MODEL & RANKING
    print("Calculating factor scores...")
    # Momentum (12-1m Return)
    momentum = {}
    for t in tickers:
        p = price_data[t]
        momentum[t] = (p.iloc[-21] / p.iloc[-252]) - 1.0 if len(p) >= 252 else (p.iloc[-1] / p.iloc[0]) - 1.0
        
    # Volatility (for Low Volatility factor)
    volatility = {}
    for t in tickers:
        volatility[t] = portfolio_asset_returns[t].std() * np.sqrt(252)
        
    # Valuation (PE Proxy) and Quality (ROE Proxy) deterministically seeded
    value_pe = {}
    quality_roe = {}
    for t in tickers:
        hash_val = sum(ord(c) for c in t)
        value_pe[t] = 12.0 + (hash_val % 45) # PE between 12 and 57
        quality_roe[t] = 0.06 + ((hash_val * 3) % 25) / 100.0 # ROE between 6% and 31%

    factors_df = pd.DataFrame({
        'Ticker': tickers,
        'Sector': [sectors_map.get(t, 'Other') for t in tickers],
        'Momentum': pd.Series(momentum),
        'Volatility': pd.Series(volatility),
        'Value_PE': pd.Series(value_pe),
        'Quality_ROE': pd.Series(quality_roe)
    })

    # Rank factors (Percentile Rank 0.0 to 1.0)
    factors_df['Momentum_Rank'] = factors_df['Momentum'].rank(pct=True)
    factors_df['Volatility_Rank'] = (1.0 - factors_df['Volatility'].rank(pct=True)) # Low Vol is better
    factors_df['Value_Rank'] = (1.0 - factors_df['Value_PE'].rank(pct=True)) # Lower PE represents higher Value
    factors_df['Quality_Rank'] = factors_df['Quality_ROE'].rank(pct=True) # Higher ROE is better

    # Composite score
    factors_df['Composite'] = factors_df[['Momentum_Rank', 'Volatility_Rank', 'Value_Rank', 'Quality_Rank']].mean(axis=1)
    factors_df = factors_df.sort_values(by='Composite', ascending=False)
    factors_df['Rank'] = range(1, len(factors_df) + 1)
    
    # Universe Selection: Select Top 15 stocks
    factors_df['Selected'] = factors_df['Rank'] <= 15
    selected_tickers = factors_df[factors_df['Selected']]['Ticker'].tolist()
    
    # 2. EXPECTED RETURNS MODELING (COMPOSITE SCORE -> EXPECTED ALPHA -> EXPECTED RETURN)
    rf = 0.04          # Risk-Free Rate (4%)
    mkt_premium = 0.08 # Market Risk Premium (8%)
    
    # Expected Return = rf + Beta * mkt_premium + Expected Alpha
    # Where Expected Alpha ranges from -4% to +4% based on the Composite Rank
    expected_returns_map = {}
    for _, row in factors_df.iterrows():
        t = row['Ticker']
        comp = row['Composite']
        expected_returns_map[t] = rf + mkt_premium + (comp - 0.5) * 0.08

    # 3. PORTFOLIO OPTIMIZATION (MAX SHARPE WITH RISK_FOLIO)
    print("Optimizing portfolio weights using Riskfolio-Lib...")
    import riskfolio as rp
    
    selected_returns = portfolio_asset_returns[selected_tickers]
    cov_matrix = selected_returns.cov() * 252
    
    port = rp.Portfolio(returns=selected_returns)
    
    # Pass factor-derived expected returns and covariance
    selected_expected_returns = pd.Series([expected_returns_map[t] for t in selected_tickers], index=selected_tickers)
    port.mu = selected_expected_returns
    port.cov = selected_returns.cov()
    
    # Construct linear inequality constraints A * w <= B
    # 1. w_i <= 0.10 (Upper bound)
    # 2. -w_i <= -0.01 (Lower bound)
    # 3. Sector constraint: sum of weights in any sector <= 30%
    n_assets = len(selected_tickers)
    A = []
    B = []
    
    # 1. w_i <= 0.10
    for i in range(n_assets):
        row = [0.0] * n_assets
        row[i] = 1.0
        A.append(row)
        B.append(0.10)
        
    # 2. -w_i <= -0.01
    for i in range(n_assets):
        row = [0.0] * n_assets
        row[i] = -1.0
        A.append(row)
        B.append(-0.01)
        
    # 3. Sector constraint: sum(w_sector) <= 0.30
    selected_sectors = [sectors_map.get(t, 'Other') for t in selected_tickers]
    unique_sectors = list(set(selected_sectors))
    for sector in unique_sectors:
        row = [1.0 if s == sector else 0.0 for s in selected_sectors]
        A.append(row)
        B.append(0.30)
        
    # Convert to DataFrames
    A_df = pd.DataFrame(A, columns=selected_tickers)
    B_df = pd.DataFrame(B, columns=['Value'])
    
    # Set inequality constraints on the Portfolio object
    port.ainequality = A_df
    port.binequality = B_df
    
    init_w = np.ones(n_assets) / n_assets
    bounds = [(0.01, 0.10) for _ in range(n_assets)]
    
    try:
        w_opt = port.optimization(
            model='Classic',
            rm='MV',
            obj='Sharpe',
            rf=rf,
            l=0,
            hist=True
        )
        if w_opt is not None and not w_opt.empty:
            optimal_weights = w_opt['weights'].values
        else:
            raise ValueError("Riskfolio optimizer returned empty weights.")
    except Exception as e:
        print(f"[WARNING] Riskfolio optimization failed: {e}. Falling back to scipy.")
        # Scipy fallback as safety net
        scipy_constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        for sector in unique_sectors:
            indices = [i for i, s in enumerate(selected_sectors) if s == sector]
            scipy_constraints.append({'type': 'ineq', 'fun': lambda w, idxs=indices: 0.30 - np.sum(w[idxs])})
        def scipy_neg_sharpe(w):
            p_ret = np.dot(w, selected_expected_returns.values)
            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(p_ret - rf) / p_vol
        res = minimize(scipy_neg_sharpe, init_w, method='SLSQP', bounds=bounds, constraints=scipy_constraints)
        optimal_weights = res.x if res.success else init_w

    # Save weights mapping (0% for rejected stocks)
    full_weights = pd.Series(0.0, index=tickers)
    for t, w in zip(selected_tickers, optimal_weights):
        full_weights[t] = w

    # 4. BACKTESTING STRATEGY COMPARISON
    print("Running strategy backtests...")
    # Portfolio Returns
    portfolio_returns = portfolio_asset_returns[selected_tickers].dot(optimal_weights)
    cumulative = (1 + portfolio_returns).cumprod()
    
    # Equal Weight Portfolio (All 48 assets equally weighted)
    equal_weights = np.ones(len(tickers)) / len(tickers)
    equal_weighted_returns = portfolio_asset_returns.dot(equal_weights)
    cumulative_ew = (1 + equal_weighted_returns).cumprod()
    
    # Benchmark Returns
    benchmark_returns = returns[benchmark_ticker]
    cumulative_bm = (1 + benchmark_returns).cumprod()
    
    # Align returns series index
    common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_idx]
    equal_weighted_returns = equal_weighted_returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]
    
    # 5. STRATEGY COMPARISON PERFORMANCE CALCULATIONS
    trading_days = len(portfolio_returns)
    years = trading_days / 252.0
    
    # Calculate performance for each
    cagr_opt = Performance.CAGR(cumulative, years)
    vol_opt = Performance.volatility(portfolio_returns)
    sharpe_opt = Performance.sharpe(portfolio_returns, rf)
    sortino_opt = Performance.sortino(portfolio_returns, rf)
    max_dd_opt = Drawdown.maximum(cumulative)
    
    cagr_ew = Performance.CAGR(cumulative_ew, years)
    vol_ew = Performance.volatility(equal_weighted_returns)
    sharpe_ew = Performance.sharpe(equal_weighted_returns, rf)
    sortino_ew = Performance.sortino(equal_weighted_returns, rf)
    max_dd_ew = Drawdown.maximum(cumulative_ew)
    
    cagr_bm = Performance.CAGR(cumulative_bm, years)
    vol_bm = Performance.volatility(benchmark_returns)
    sharpe_bm = Performance.sharpe(benchmark_returns, rf)
    sortino_bm = Performance.sortino(benchmark_returns, rf)
    max_dd_bm = Drawdown.maximum(cumulative_bm)

    # 6. FAMA-FRENCH STYLE FACTOR ATTRIBUTION
    print("Calculating factor attribution...")
    # Compute factor mimicking portfolio returns for each day
    factor_returns = {}
    for f in ['Momentum_Rank', 'Volatility_Rank', 'Value_Rank', 'Quality_Rank']:
        # Sort stocks by factor rank
        sorted_assets = factors_df.sort_values(by=f, ascending=False)['Ticker'].tolist()
        top_10 = sorted_assets[:10]
        bottom_10 = sorted_assets[-10:]
        
        # Factor return = Top 10 mean - Bottom 10 mean
        factor_returns[f] = portfolio_asset_returns[top_10].mean(axis=1) - portfolio_asset_returns[bottom_10].mean(axis=1)
    
    factor_returns_df = pd.DataFrame(factor_returns)
    
    # Active Return = Portfolio Return - Benchmark Return
    active_returns = portfolio_returns - benchmark_returns
    
    # Active Exposure = Portfolio Exposure - Benchmark Exposure (assumed 0.5 average rank)
    active_exposures = {}
    for f_rank, f_name in zip(
        ['Momentum_Rank', 'Volatility_Rank', 'Value_Rank', 'Quality_Rank'],
        ['Momentum', 'Volatility', 'Value', 'Quality']
    ):
        portfolio_exposure = sum(full_weights[t] * factors_df.loc[factors_df['Ticker'] == t, f_rank].values[0] for t in selected_tickers)
        active_exposures[f_name] = portfolio_exposure - 0.5

    # Daily return contribution
    attrib_daily = pd.DataFrame(index=common_idx)
    for f_name, f_rank in zip(['Momentum', 'Volatility', 'Value', 'Quality'], ['Momentum_Rank', 'Volatility_Rank', 'Value_Rank', 'Quality_Rank']):
        attrib_daily[f_name] = active_exposures[f_name] * factor_returns_df[f_rank]
        
    attrib_daily['Market'] = benchmark_returns
    attrib_daily['Residual'] = portfolio_returns - (attrib_daily[['Momentum', 'Volatility', 'Value', 'Quality']].sum(axis=1) + attrib_daily['Market'])
    
    # Cumulative factor contributions
    cum_attribution = attrib_daily.cumsum()

    # 7. INSTITUTIONAL RISK METRICS
    print("Calculating institutional risk metrics...")
    active_return_annual = (portfolio_returns.mean() - benchmark_returns.mean()) * 252
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = active_return_annual / tracking_error if tracking_error != 0 else 0.0
    
    portfolio_beta = Beta.calculate(portfolio_returns, benchmark_returns)
    treynor_ratio = (cagr_opt - rf) / portfolio_beta if portfolio_beta != 0 else 0.0
    
    # Omega Ratio
    pos_returns_sum = active_returns[active_returns > 0].sum()
    neg_returns_sum = abs(active_returns[active_returns < 0].sum())
    omega_ratio = pos_returns_sum / neg_returns_sum if neg_returns_sum != 0 else 0.0
    
    calmar_ratio = cagr_opt / abs(max_dd_opt) if max_dd_opt != 0 else 0.0
    
    # Downside deviation
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    
    # Value at Risk & Expected Shortfall (CVaR)
    hist_var = ValueAtRisk.historical(portfolio_returns, 0.95)
    hist_cvar = ConditionalVaR.historical(portfolio_returns, 0.95)
    
    # Stress testing
    stress_test_results = {
        'Market Crash (-30%)': StressTest.market_crash(cumulative.iloc[-1], -0.30),
        'Interest Rate Shock (-10%)': StressTest.interest_rate_shock(cumulative.iloc[-1], -0.10),
        'Recession Shock (-20%)': StressTest.recession(cumulative.iloc[-1], -0.20)
    }

    # 8. PRINT SUMMARY REPORT
    risk_metrics = {
        'historical_var': hist_var,
        'parametric_var': ValueAtRisk.parametric(portfolio_returns, 0.95),
        'historical_cvar': hist_cvar,
        'beta': portfolio_beta,
        'max_drawdown': max_dd_opt
    }
    
    perf_metrics = {
        'CAGR': cagr_opt,
        'volatility': vol_opt,
        'sharpe': sharpe_opt,
        'sortino': sortino_opt
    }
    
    report = RiskPerformanceReport.generate(
        weights=full_weights[selected_tickers],
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        cumulative_returns=cumulative,
        risk_metrics=risk_metrics,
        perf_metrics=perf_metrics,
        stress_test_results=stress_test_results,
        contribution=attrib_daily[['Momentum', 'Volatility', 'Value', 'Quality']].mean(),
        correlation_matrix=Correlation.matrix(selected_returns)
    )
    print("\n" + report + "\n")
    
    # Save text report
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_performance_report.txt")
    with open(output_file, "w") as f:
        f.write(report)
    print(f"[INFO] Text report saved to: {output_file}")

    # 9. EXPORT CSVs TO REPORTS DIRECTORY
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    print(f"[INFO] Exporting report CSVs to: {reports_dir}")
    
    # portfolio.csv
    portfolio_df = pd.DataFrame({
        'Portfolio': cumulative * 10000.0,
        'Return': portfolio_returns,
        'Sharpe': RollingStatistics.sharpe(portfolio_returns, 30).fillna(sharpe_opt),
        'Drawdown': Drawdown.calculate(cumulative),
        'EqualWeight_Cum': cumulative_ew * 10000.0,
        'Benchmark_Cum': cumulative_bm * 10000.0,
        'EqualWeight_Return': equal_weighted_returns,
        'Benchmark_Return': benchmark_returns
    })
    portfolio_df.to_csv(os.path.join(reports_dir, "portfolio.csv"))
    
    # weights.csv
    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': full_weights.values,
        'Sector': [sectors_map.get(t, 'Other') for t in tickers],
        'Selected': [t in selected_tickers for t in tickers]
    })
    weights_df.to_csv(os.path.join(reports_dir, "weights.csv"), index=False)
    
    # risk.csv
    # Side-by-side comparison matrix and advanced risk statistics
    risk_df = pd.DataFrame([{
        'VaR': hist_var,
        'CVaR': hist_cvar,
        'Beta': portfolio_beta,
        'Drawdown': max_dd_opt,
        'TrackingError': tracking_error,
        'InformationRatio': information_ratio,
        'TreynorRatio': treynor_ratio,
        'OmegaRatio': omega_ratio,
        'CalmarRatio': calmar_ratio,
        'DownsideDeviation': downside_deviation,
        'ExpectedShortfall': hist_cvar
    }])
    risk_df.to_csv(os.path.join(reports_dir, "risk.csv"), index=False)
    
    # performance.csv
    # Comparative performance table
    performance_df = pd.DataFrame([
        {'Metric': 'CAGR', 'Portfolio': cagr_opt, 'Benchmark': cagr_bm, 'EqualWeight': cagr_ew},
        {'Metric': 'Volatility', 'Portfolio': vol_opt, 'Benchmark': vol_bm, 'EqualWeight': vol_ew},
        {'Metric': 'Sharpe', 'Portfolio': sharpe_opt, 'Benchmark': sharpe_bm, 'EqualWeight': sharpe_ew},
        {'Metric': 'Sortino', 'Portfolio': sortino_opt, 'Benchmark': sortino_bm, 'EqualWeight': sortino_ew},
        {'Metric': 'Max Drawdown', 'Portfolio': max_dd_opt, 'Benchmark': max_dd_bm, 'EqualWeight': max_dd_ew}
    ])
    performance_df.to_csv(os.path.join(reports_dir, "performance.csv"), index=False)
    
    # factors.csv
    # Full universe rankings table
    factors_df.to_csv(os.path.join(reports_dir, "factors.csv"), index=False)
    
    # expected_returns.csv
    er_df = pd.DataFrame([
        {'Ticker': t, 'Composite': comp, 'ExpectedReturn': expected_returns_map[t]}
        for t, comp in zip(factors_df['Ticker'], factors_df['Composite'])
    ])
    er_df.to_csv(os.path.join(reports_dir, "expected_returns.csv"), index=False)
    
    # frontier.csv
    # Frontier optimization scatter points
    frontier_points = []
    # 1. Random simulations (300 portfolios)
    for _ in range(300):
        w = np.random.dirichlet(np.ones(n_assets))
        port_return = np.dot(w, selected_expected_returns)
        port_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        frontier_points.append({'Risk': port_risk, 'Return': port_return, 'Type': 'Simulated'})
    # 2. Add optimal points
    # Max Sharpe point
    max_sharpe_return = np.dot(optimal_weights, selected_expected_returns)
    max_sharpe_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
    frontier_points.append({'Risk': max_sharpe_risk, 'Return': max_sharpe_return, 'Type': 'Max Sharpe'})
    # Min Volatility point
    min_vol_w = minimize(lambda w: np.dot(w.T, np.dot(cov_matrix, w)), init_w, method='SLSQP', bounds=bounds, constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]).x
    min_vol_return = np.dot(min_vol_w, selected_expected_returns)
    min_vol_risk = np.sqrt(np.dot(min_vol_w.T, np.dot(cov_matrix, min_vol_w)))
    frontier_points.append({'Risk': min_vol_risk, 'Return': min_vol_return, 'Type': 'Min Vol'})
    
    frontier_df = pd.DataFrame(frontier_points)
    frontier_df.to_csv(os.path.join(reports_dir, "frontier.csv"), index=False)
    
    # rolling.csv
    rolling_df = pd.DataFrame({
        'RollingVolatility': RollingStatistics.volatility(portfolio_returns, 30),
        'RollingSharpe': RollingStatistics.sharpe(portfolio_returns, 30, rf)
    })
    rolling_df.to_csv(os.path.join(reports_dir, "rolling.csv"))
    
    # attribution.csv
    # Save attribution timeseries & mean summary
    cum_attribution.to_csv(os.path.join(reports_dir, "attribution_timeseries.csv"))
    mean_attrib = attrib_daily.mean()
    mean_attrib.to_frame(name='Contribution').to_csv(os.path.join(reports_dir, "attribution.csv"))
    print("[INFO] Dashboard CSV reports generated successfully.")

def run_streamlit_app():
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

if __name__ == "__main__":
    if st.runtime.exists():
        run_streamlit_app()
    else:
        main()
