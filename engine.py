"""
AlphaForge Quantitative Engine
Modular multi-factor equity research, portfolio optimization, historical backtesting,
and institutional risk analytics supporting both India and US universes.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

from helper import fetch_stocks, format_ticker_for_yf, categorize_market_cap
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

# Optional GS-Quant timeseries
try:
    from gs_quant.timeseries import returns as gs_returns
    HAS_GS_QUANT = True
except ImportError:
    HAS_GS_QUANT = False

try:
    import riskfolio as rp
    HAS_RISKFOLIO = True
except ImportError:
    HAS_RISKFOLIO = False


class QuantEngine:
    """
    Institutional Multi-Factor Equity Research & Portfolio Optimization Engine
    Supports Tactical Factor Tilting, Custom Stock Baskets, Multi-Model Optimization (Sharpe, MinVol, HRP),
    and Monte Carlo Future Simulations.
    """
    def __init__(
        self,
        region="India",
        exchange="NSE",
        market_cap_filter="All",
        universe_size=30,
        period="2y",
        top_k_select=12,
        custom_tickers=None,
        factor_weights=None,
        optimization_model="Sharpe",
        rf=None,
        mkt_premium=None
    ):
        self.region = region
        self.exchange = exchange
        self.market_cap_filter = market_cap_filter
        self.universe_size = int(universe_size)
        self.period = period
        self.top_k_select = int(top_k_select)
        self.custom_tickers = custom_tickers
        self.optimization_model = optimization_model  # 'Sharpe', 'MinVol', 'HRP'

        # Factor weights (Momentum, Volatility, Value, Quality)
        if factor_weights is None:
            self.factor_weights = {"Momentum": 0.25, "Volatility": 0.25, "Value": 0.25, "Quality": 0.25}
        else:
            total_w = sum(factor_weights.values()) if sum(factor_weights.values()) > 0 else 1.0
            self.factor_weights = {k: v / total_w for k, v in factor_weights.items()}

        # Region-specific defaults
        if self.region == "India":
            self.currency = "INR"
            self.currency_symbol = "₹"
            self.benchmark_ticker = "^BSESN" if exchange == "BSE" else "^NSEI"
            self.benchmark_name = "BSE Sensex" if exchange == "BSE" else "Nifty 50"
            self.rf = 0.065 if rf is None else rf
            self.mkt_premium = 0.075 if mkt_premium is None else mkt_premium
        else:
            self.currency = "USD"
            self.currency_symbol = "$"
            self.benchmark_ticker = "^GSPC"
            self.benchmark_name = "S&P 500"
            self.rf = 0.040 if rf is None else rf
            self.mkt_premium = 0.060 if mkt_premium is None else mkt_premium

    def build_universe(self):
        """Filter CSV dataset by exchange, market cap or custom basket, select securities."""
        stocks_df = fetch_stocks(self.region)
        if stocks_df.empty:
            raise ValueError(f"No stock data found for region '{self.region}'.")

        df = stocks_df.copy()

        # Handle Custom Basket mode if specified
        if self.custom_tickers and len(self.custom_tickers) >= 2:
            custom_syms = [str(s).strip().upper() for s in self.custom_tickers]
            matched = df[df["Symbol"].str.upper().isin(custom_syms)].copy()
            if not matched.empty:
                df = matched
            # Update top_k_select to not exceed custom basket length
            self.top_k_select = min(self.top_k_select, len(df))
        else:
            # Filter by Exchange
            if self.exchange and self.exchange != "All":
                if "Exchange" in df.columns:
                    exchange_df = df[df["Exchange"].str.upper() == self.exchange.upper()]
                    if not exchange_df.empty:
                        df = exchange_df

            # Filter by Market Cap Category
            if self.market_cap_filter and self.market_cap_filter != "All":
                df["Market_Cap_Category"] = df.apply(
                    lambda r: categorize_market_cap(r, self.region), axis=1
                )
                filtered = df[df["Market_Cap_Category"] == self.market_cap_filter]
                if not filtered.empty:
                    df = filtered

            # Ensure Market capitalization numeric and sort
            if "Market capitalization" in df.columns:
                df["Market capitalization"] = pd.to_numeric(df["Market capitalization"], errors="coerce").fillna(0)
                df = df.sort_values(by="Market capitalization", ascending=False)

            df = df.head(self.universe_size).copy()

        # Drop duplicates by Symbol
        df = df.drop_duplicates(subset=["Symbol"]).copy()
        if df.empty:
            raise ValueError("No stocks matched the specified filter criteria.")

        # Assign Yahoo Ticker
        df["YF_Ticker"] = df.apply(
            lambda r: format_ticker_for_yf(
                r["Symbol"],
                self.region,
                r.get("Exchange", self.exchange)
            ),
            axis=1
        )

        return df

    def fetch_market_data(self, universe_df):
        """Fetch historical price data from yfinance for universe + benchmark."""
        tickers = universe_df["YF_Ticker"].tolist()
        all_tickers = list(dict.fromkeys(tickers + [self.benchmark_ticker]))

        raw_data = yf.download(
            all_tickers,
            period=self.period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if raw_data is None or raw_data.empty:
            raise ValueError("Failed to download historical prices from yfinance.")

        # Extract Close prices
        if isinstance(raw_data.columns, pd.MultiIndex):
            if "Close" in raw_data.columns.levels[0]:
                prices = raw_data["Close"]
            else:
                prices = raw_data.xs("Close", axis=1, level=0)
        else:
            prices = raw_data

        # Check benchmark availability
        if self.benchmark_ticker not in prices.columns or prices[self.benchmark_ticker].dropna().empty:
            fallback_bm = "^NSEI" if self.region == "India" else "^GSPC"
            if fallback_bm != self.benchmark_ticker and fallback_bm in prices.columns:
                self.benchmark_ticker = fallback_bm

        # Drop columns with > 25% NaNs
        missing_pct = prices.isnull().mean()
        valid_cols = missing_pct[missing_pct <= 0.25].index.tolist()

        if self.benchmark_ticker not in valid_cols:
            bm_df = yf.download(self.benchmark_ticker, period=self.period, interval="1d", auto_adjust=True, progress=False)
            if isinstance(bm_df.columns, pd.MultiIndex):
                bm_close = bm_df["Close"]
            else:
                bm_close = bm_df
            prices[self.benchmark_ticker] = bm_close
            valid_cols.append(self.benchmark_ticker)

        prices = prices[valid_cols].ffill().bfill().dropna()

        # Filter valid universe tickers present in price data
        valid_tickers = [t for t in tickers if t in prices.columns and t != self.benchmark_ticker]
        if len(valid_tickers) < 2:
            raise ValueError(f"Insufficient active stock data: only {len(valid_tickers)} valid stocks available.")

        return prices, valid_tickers

    def compute_factors(self, prices, universe_df, valid_tickers):
        """
        Compute real multi-factor model rankings using tactical factor weights:
        1. Momentum (12-1m Price Return)
        2. Low Volatility (Inverse Annualized Volatility)
        3. Value (Inverse P/E + Dividend Yield)
        4. Quality (YoY EPS Growth)
        """
        returns = prices.pct_change().dropna()
        asset_returns = returns[valid_tickers]

        # 1. Momentum (12-1m return)
        momentum = {}
        for t in valid_tickers:
            p = prices[t]
            if len(p) >= 252:
                momentum[t] = (p.iloc[-21] / p.iloc[-252]) - 1.0
            elif len(p) > 21:
                momentum[t] = (p.iloc[-1] / p.iloc[0]) - 1.0
            else:
                momentum[t] = 0.0

        # 2. Low Volatility
        volatility = {}
        for t in valid_tickers:
            volatility[t] = asset_returns[t].std() * np.sqrt(252)

        # Merge metadata for fundamentals
        uni_map = universe_df.set_index("YF_Ticker")

        pe_map = {}
        div_yield_map = {}
        eps_growth_map = {}
        sector_map = {}
        name_map = {}

        for t in valid_tickers:
            if t in uni_map.index:
                row = uni_map.loc[t]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                # P/E Ratio
                pe_raw = pd.to_numeric(row.get("Price to earnings ratio"), errors="coerce")
                pe_map[t] = pe_raw if pd.notna(pe_raw) and pe_raw > 0 else np.nan

                # Dividend Yield
                dy_raw = pd.to_numeric(row.get("Dividend yield %, Trailing 12 months"), errors="coerce")
                div_yield_map[t] = (dy_raw / 100.0) if pd.notna(dy_raw) and dy_raw >= 0 else 0.0

                # EPS Growth
                eg_raw = pd.to_numeric(row.get("Earnings per share diluted growth %, TTM YoY"), errors="coerce")
                eps_growth_map[t] = (eg_raw / 100.0) if pd.notna(eg_raw) else 0.0

                # Sector & Name
                sector_map[t] = str(row.get("Sector", "General")).strip() or "General"
                name_map[t] = str(row.get("Description", t)).strip()
            else:
                pe_map[t] = np.nan
                div_yield_map[t] = 0.0
                eps_growth_map[t] = 0.0
                sector_map[t] = "General"
                name_map[t] = t

        # Impute missing P/E with median
        pe_series = pd.Series(pe_map)
        median_pe = pe_series.median() if not pe_series.dropna().empty else 20.0
        pe_filled = pe_series.fillna(median_pe).clip(lower=2.0, upper=150.0)

        # Construct factors dataframe
        factors_df = pd.DataFrame({
            "Ticker": valid_tickers,
            "Company": [name_map.get(t, t) for t in valid_tickers],
            "Sector": [sector_map.get(t, "General") for t in valid_tickers],
            "Momentum": pd.Series(momentum),
            "Volatility": pd.Series(volatility),
            "Value_PE": pe_filled,
            "Dividend_Yield": pd.Series(div_yield_map),
            "EPS_Growth": pd.Series(eps_growth_map).clip(lower=-0.60, upper=1.50)
        })

        # Calculate Percentile Ranks (0.0 to 1.0)
        factors_df["Momentum_Rank"] = factors_df["Momentum"].rank(pct=True)
        factors_df["Volatility_Rank"] = (1.0 - factors_df["Volatility"].rank(pct=True))

        pe_rank = (1.0 - factors_df["Value_PE"].rank(pct=True))
        dy_rank = factors_df["Dividend_Yield"].rank(pct=True)
        factors_df["Value_Rank"] = 0.8 * pe_rank + 0.2 * dy_rank
        factors_df["Quality_Rank"] = factors_df["EPS_Growth"].rank(pct=True)

        # Tactical Factor Tilt weighting
        wm = self.factor_weights.get("Momentum", 0.25)
        wv = self.factor_weights.get("Volatility", 0.25)
        wval = self.factor_weights.get("Value", 0.25)
        wq = self.factor_weights.get("Quality", 0.25)

        factors_df["Composite"] = (
            wm * factors_df["Momentum_Rank"] +
            wv * factors_df["Volatility_Rank"] +
            wval * factors_df["Value_Rank"] +
            wq * factors_df["Quality_Rank"]
        )

        factors_df = factors_df.sort_values(by="Composite", ascending=False).reset_index(drop=True)
        factors_df["Rank"] = range(1, len(factors_df) + 1)

        # Selection: Top K
        k = min(self.top_k_select, len(factors_df))
        factors_df["Selected"] = factors_df["Rank"] <= k

        return factors_df, asset_returns, returns[self.benchmark_ticker], sector_map

    def optimize_portfolio(self, factors_df, asset_returns, sector_map):
        """
        Multi-Model Portfolio Optimization:
        - 'Sharpe': Maximize Sharpe ratio under bounds and sector limits
        - 'MinVol': Minimum Volatility (Capital preservation)
        - 'HRP': Hierarchical Risk Parity (Tree clustering of covariance)
        """
        selected_tickers = factors_df[factors_df["Selected"]]["Ticker"].tolist()
        n_assets = len(selected_tickers)

        if n_assets < 2:
            raise ValueError("At least 2 assets must be selected for portfolio optimization.")

        selected_returns = asset_returns[selected_tickers]
        cov_matrix = selected_returns.cov() * 252

        # Implied Expected Returns
        expected_returns = {}
        for _, row in factors_df[factors_df["Selected"]].iterrows():
            t = row["Ticker"]
            comp = row["Composite"]
            expected_returns[t] = self.rf + self.mkt_premium + (comp - 0.5) * 0.08

        mu = pd.Series(expected_returns, index=selected_tickers)

        max_single_weight = max(0.15, 1.5 / n_assets)
        min_single_weight = 0.01

        selected_sectors = [sector_map.get(t, "General") for t in selected_tickers]
        unique_sectors = list(set(selected_sectors))

        optimal_weights = None

        # MODEL 1: Hierarchical Risk Parity (HRP)
        if self.optimization_model == "HRP" and HAS_RISKFOLIO:
            try:
                hrp_port = rp.HCPortfolio(returns=selected_returns)
                w_hrp = hrp_port.optimization(model="HRP", codependence="pearson", rm="MV", rf=self.rf, linkage="single")
                if w_hrp is not None and not w_hrp.empty:
                    optimal_weights = w_hrp["weights"].values
            except Exception:
                # Fallback: Inverse Volatility weighting
                inv_vol = 1.0 / selected_returns.std()
                optimal_weights = (inv_vol / inv_vol.sum()).values

        # MODEL 2: Minimum Volatility
        elif self.optimization_model == "MinVol":
            try:
                if HAS_RISKFOLIO:
                    port = rp.Portfolio(returns=selected_returns)
                    port.mu = mu
                    port.cov = selected_returns.cov()
                    w_opt = port.optimization(model="Classic", rm="MV", obj="MinRisk", rf=self.rf, hist=True)
                    if w_opt is not None and not w_opt.empty:
                        optimal_weights = w_opt["weights"].values
            except Exception:
                pass

            if optimal_weights is None:
                def vol_obj(w):
                    return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                init_w = np.ones(n_assets) / n_assets
                bounds = [(min_single_weight, max_single_weight) for _ in range(n_assets)]
                res = minimize(vol_obj, init_w, method="SLSQP", bounds=bounds, constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}])
                optimal_weights = res.x if res.success else init_w

        # MODEL 3: Max Sharpe (Default)
        else:
            if HAS_RISKFOLIO:
                try:
                    port = rp.Portfolio(returns=selected_returns)
                    port.mu = mu
                    port.cov = selected_returns.cov()

                    A, B = [], []
                    for i in range(n_assets):
                        row = [0.0] * n_assets
                        row[i] = 1.0
                        A.append(row)
                        B.append(max_single_weight)

                    for i in range(n_assets):
                        row = [0.0] * n_assets
                        row[i] = -1.0
                        A.append(row)
                        B.append(-min_single_weight)

                    for s in unique_sectors:
                        s_count = sum(1 for t in selected_tickers if sector_map.get(t, "General") == s)
                        s_limit = max(0.35, min(0.85, (s_count / n_assets) + 0.15))
                        row = [1.0 if sector_map.get(t, "General") == s else 0.0 for t in selected_tickers]
                        A.append(row)
                        B.append(s_limit)

                    port.ainequality = pd.DataFrame(A, columns=selected_tickers)
                    port.binequality = pd.DataFrame(B, columns=["Value"])

                    w_opt = port.optimization(model="Classic", rm="MV", obj="Sharpe", rf=self.rf, l=0, hist=True)
                    if w_opt is not None and not w_opt.empty:
                        optimal_weights = w_opt["weights"].values
                except Exception:
                    pass

            if optimal_weights is None:
                init_w = np.ones(n_assets) / n_assets
                bounds = [(min_single_weight, max_single_weight) for _ in range(n_assets)]
                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

                for s in unique_sectors:
                    s_count = sum(1 for t in selected_tickers if sector_map.get(t, "General") == s)
                    s_limit = max(0.35, min(0.85, (s_count / n_assets) + 0.15))
                    idxs = [i for i, t in enumerate(selected_tickers) if sector_map.get(t, "General") == s]
                    constraints.append({"type": "ineq", "fun": lambda w, idxs=idxs, lim=s_limit: lim - np.sum(w[idxs])})

                def neg_sharpe(w):
                    p_ret = np.dot(w, mu.values)
                    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                    return -(p_ret - self.rf) / p_vol if p_vol > 0 else 0.0

                res = minimize(neg_sharpe, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
                optimal_weights = res.x if res.success else init_w

        # Normalize to exactly 1.0
        optimal_weights = np.maximum(0.0, optimal_weights)
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        # Full weights across entire universe
        all_tickers = factors_df["Ticker"].tolist()
        full_weights = pd.Series(0.0, index=all_tickers)
        for t, w in zip(selected_tickers, optimal_weights):
            full_weights[t] = float(w)

        # Generate Efficient Frontier points
        frontier_points = []
        for _ in range(250):
            rw = np.random.dirichlet(np.ones(n_assets))
            r_ret = np.dot(rw, mu.values)
            r_risk = np.sqrt(np.dot(rw.T, np.dot(cov_matrix, rw)))
            frontier_points.append({"Risk": r_risk, "Return": r_ret, "Type": "Simulated"})

        # Max Sharpe Point
        ms_ret = np.dot(optimal_weights, mu.values)
        ms_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        frontier_points.append({"Risk": ms_risk, "Return": ms_ret, "Type": "Max Sharpe"})

        # Min Vol Point
        try:
            min_vol_res = minimize(
                lambda w: np.dot(w.T, np.dot(cov_matrix, w)),
                np.ones(n_assets) / n_assets,
                method="SLSQP",
                bounds=[(min_single_weight, max_single_weight) for _ in range(n_assets)],
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            )
            min_vol_w = min_vol_res.x if min_vol_res.success else optimal_weights
        except Exception:
            min_vol_w = optimal_weights

        mv_ret = np.dot(min_vol_w, mu.values)
        mv_risk = np.sqrt(np.dot(min_vol_w.T, np.dot(cov_matrix, min_vol_w)))
        frontier_points.append({"Risk": mv_risk, "Return": mv_ret, "Type": "Min Vol"})

        frontier_df = pd.DataFrame(frontier_points)

        return selected_tickers, optimal_weights, full_weights, mu, cov_matrix, frontier_df

    def simulate_monte_carlo(self, opt_returns, years=1, n_sims=1000, initial_wealth=100000.0):
        """
        Monte Carlo Future Wealth Projection using Geometric Brownian Motion.
        Generates 5th, 25th, 50th, 75th, and 95th percentile confidence trajectories.
        """
        n_days = int(years * 252)
        mu_daily = opt_returns.mean()
        sigma_daily = opt_returns.std()

        # Simulated return matrix: shape (n_sims, n_days)
        sim_rets = np.random.normal(mu_daily, sigma_daily, (n_sims, n_days))
        sim_paths = initial_wealth * np.cumprod(1.0 + sim_rets, axis=1)

        # Dates into future
        future_dates = pd.date_range(start=pd.Timestamp.today(), periods=n_days, freq="B")

        p5 = np.percentile(sim_paths, 5, axis=0)
        p25 = np.percentile(sim_paths, 25, axis=0)
        p50 = np.percentile(sim_paths, 50, axis=0)
        p75 = np.percentile(sim_paths, 75, axis=0)
        p95 = np.percentile(sim_paths, 95, axis=0)

        mc_df = pd.DataFrame({
            "P5_Bear": p5,
            "P25": p25,
            "P50_Median": p50,
            "P75": p75,
            "P95_Bull": p95
        }, index=future_dates)

        return mc_df

    def run_backtest_and_analytics(
        self,
        factors_df,
        asset_returns,
        benchmark_returns,
        selected_tickers,
        optimal_weights,
        full_weights,
        sector_map
    ):
        """
        Execute strategy backtest, comparative metrics, factor attribution,
        and institutional risk analytics.
        """
        common_idx = asset_returns.index.intersection(benchmark_returns.index)
        asset_ret = asset_returns.loc[common_idx]
        bm_ret = benchmark_returns.loc[common_idx]

        # Strategy Returns
        opt_ret = asset_ret[selected_tickers].dot(optimal_weights)
        opt_cum = (1.0 + opt_ret).cumprod()

        # Equal Weight Portfolio
        ew_weights = np.ones(len(asset_ret.columns)) / len(asset_ret.columns)
        ew_ret = asset_ret.dot(ew_weights)
        ew_cum = (1.0 + ew_ret).cumprod()

        # Benchmark Returns
        bm_cum = (1.0 + bm_ret).cumprod()

        trading_days = len(opt_ret)
        years = max(trading_days / 252.0, 0.1)

        perf_data = [
            {
                "Metric": "CAGR",
                "Portfolio": Performance.CAGR(opt_cum, years),
                "Benchmark": Performance.CAGR(bm_cum, years),
                "EqualWeight": Performance.CAGR(ew_cum, years)
            },
            {
                "Metric": "Volatility",
                "Portfolio": Performance.volatility(opt_ret),
                "Benchmark": Performance.volatility(bm_ret),
                "EqualWeight": Performance.volatility(ew_ret)
            },
            {
                "Metric": "Sharpe",
                "Portfolio": Performance.sharpe(opt_ret, self.rf),
                "Benchmark": Performance.sharpe(bm_ret, self.rf),
                "EqualWeight": Performance.sharpe(ew_ret, self.rf)
            },
            {
                "Metric": "Sortino",
                "Portfolio": Performance.sortino(opt_ret, self.rf),
                "Benchmark": Performance.sortino(bm_ret, self.rf),
                "EqualWeight": Performance.sortino(ew_ret, self.rf)
            },
            {
                "Metric": "Max Drawdown",
                "Portfolio": Drawdown.maximum(opt_cum),
                "Benchmark": Drawdown.maximum(bm_cum),
                "EqualWeight": Drawdown.maximum(ew_cum)
            }
        ]
        performance_df = pd.DataFrame(perf_data)

        # Institutional Risk Metrics
        active_returns = opt_ret - bm_ret
        active_ret_annual = (opt_ret.mean() - bm_ret.mean()) * 252.0
        tracking_error = active_returns.std() * np.sqrt(252.0)
        information_ratio = (active_ret_annual / tracking_error) if tracking_error > 0 else 0.0

        portfolio_beta = Beta.calculate(opt_ret, bm_ret)
        cagr_opt = perf_data[0]["Portfolio"]
        treynor_ratio = ((cagr_opt - self.rf) / portfolio_beta) if portfolio_beta != 0 else 0.0

        pos_sum = active_returns[active_returns > 0].sum()
        neg_sum = abs(active_returns[active_returns < 0].sum())
        omega_ratio = (pos_sum / neg_sum) if neg_sum > 0 else 1.0

        max_dd_opt = perf_data[4]["Portfolio"]
        calmar_ratio = (cagr_opt / abs(max_dd_opt)) if max_dd_opt != 0 else 0.0

        downside_ret = opt_ret[opt_ret < 0]
        downside_dev = downside_ret.std() * np.sqrt(252.0)

        hist_var = ValueAtRisk.historical(opt_ret, 0.95)
        hist_cvar = ConditionalVaR.historical(opt_ret, 0.95)
        param_var = ValueAtRisk.parametric(opt_ret, 0.95)

        risk_summary = {
            "VaR": hist_var,
            "ParametricVaR": param_var,
            "CVaR": hist_cvar,
            "Beta": portfolio_beta,
            "Drawdown": max_dd_opt,
            "TrackingError": tracking_error,
            "InformationRatio": information_ratio,
            "TreynorRatio": treynor_ratio,
            "OmegaRatio": omega_ratio,
            "CalmarRatio": calmar_ratio,
            "DownsideDeviation": downside_dev,
            "ExpectedShortfall": hist_cvar
        }
        risk_df = pd.DataFrame([risk_summary])

        # Stress testing
        final_wealth = opt_cum.iloc[-1] * 10000.0
        stress_tests = {
            "Market Crash (-30%)": StressTest.market_crash(final_wealth, -0.30),
            "Interest Rate Shock (-10%)": StressTest.interest_rate_shock(final_wealth, -0.10),
            "Recession Shock (-20%)": StressTest.recession(final_wealth, -0.20)
        }

        # Factor Attribution
        factor_returns = {}
        factor_ranks = ["Momentum_Rank", "Volatility_Rank", "Value_Rank", "Quality_Rank"]
        f_names = ["Momentum", "Volatility", "Value", "Quality"]

        top_slice = max(2, len(factors_df) // 4)
        for f_rank, f_name in zip(factor_ranks, f_names):
            sorted_t = factors_df.sort_values(by=f_rank, ascending=False)["Ticker"].tolist()
            top_t = sorted_t[:top_slice]
            bot_t = sorted_t[-top_slice:]
            factor_returns[f_name] = asset_ret[top_t].mean(axis=1) - asset_ret[bot_t].mean(axis=1)

        factor_returns_df = pd.DataFrame(factor_returns)

        active_exposures = {}
        for f_rank, f_name in zip(factor_ranks, f_names):
            p_exp = sum(
                full_weights[t] * factors_df.loc[factors_df["Ticker"] == t, f_rank].values[0]
                for t in selected_tickers
            )
            active_exposures[f_name] = p_exp - 0.5

        attrib_daily = pd.DataFrame(index=common_idx)
        for f_name in f_names:
            attrib_daily[f_name] = active_exposures[f_name] * factor_returns_df[f_name]

        attrib_daily["Market"] = bm_ret
        attrib_daily["Residual"] = opt_ret - (attrib_daily[f_names].sum(axis=1) + attrib_daily["Market"])
        cum_attribution = attrib_daily.cumsum()

        mean_attrib = attrib_daily.mean()
        attrib_summary = pd.DataFrame({
            "Contribution": mean_attrib,
            "Annualized Contribution": mean_attrib * 252.0
        })

        portfolio_ts = pd.DataFrame({
            "Portfolio": opt_cum * 10000.0,
            "Return": opt_ret,
            "Sharpe": RollingStatistics.sharpe(opt_ret, 30, self.rf).fillna(perf_data[2]["Portfolio"]),
            "Drawdown": Drawdown.calculate(opt_cum),
            "EqualWeight_Cum": ew_cum * 10000.0,
            "Benchmark_Cum": bm_cum * 10000.0,
            "EqualWeight_Return": ew_ret,
            "Benchmark_Return": bm_ret
        }, index=common_idx)

        rolling_df = pd.DataFrame({
            "RollingVolatility": RollingStatistics.volatility(opt_ret, 30),
            "RollingSharpe": RollingStatistics.sharpe(opt_ret, 30, self.rf)
        }, index=common_idx)

        weights_df = pd.DataFrame({
            "Ticker": factors_df["Ticker"],
            "Company": factors_df["Company"],
            "Weight": [full_weights.get(t, 0.0) for t in factors_df["Ticker"]],
            "Sector": factors_df["Sector"],
            "Selected": factors_df["Selected"]
        })

        # Generate default 1-year Monte Carlo projection
        mc_df = self.simulate_monte_carlo(opt_ret, years=1, n_sims=1000, initial_wealth=10000.0)

        return {
            "performance_df": performance_df,
            "risk_df": risk_df,
            "risk_summary": risk_summary,
            "stress_tests": stress_tests,
            "portfolio_ts": portfolio_ts,
            "rolling_df": rolling_df,
            "weights_df": weights_df,
            "attrib_daily": attrib_daily,
            "cum_attribution": cum_attribution,
            "attrib_summary": attrib_summary,
            "mc_df": mc_df
        }

    def export_reports(self, factors_df, weights_df, analytics, frontier_df, expected_returns_map, reports_dir=None):
        """Export calculated matrices to CSVs in reports/ directory."""
        if reports_dir is None:
            reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)

        factors_df.to_csv(os.path.join(reports_dir, "factors.csv"), index=False)
        weights_df.to_csv(os.path.join(reports_dir, "weights.csv"), index=False)
        analytics["portfolio_ts"].to_csv(os.path.join(reports_dir, "portfolio.csv"))
        analytics["performance_df"].to_csv(os.path.join(reports_dir, "performance.csv"), index=False)
        analytics["risk_df"].to_csv(os.path.join(reports_dir, "risk.csv"), index=False)
        analytics["rolling_df"].to_csv(os.path.join(reports_dir, "rolling.csv"))
        frontier_df.to_csv(os.path.join(reports_dir, "frontier.csv"), index=False)
        analytics["cum_attribution"].to_csv(os.path.join(reports_dir, "attribution_timeseries.csv"))
        analytics["attrib_summary"][["Contribution"]].to_csv(os.path.join(reports_dir, "attribution.csv"))

        er_df = pd.DataFrame([
            {"Ticker": t, "Composite": factors_df.loc[factors_df["Ticker"] == t, "Composite"].values[0], "ExpectedReturn": er}
            for t, er in expected_returns_map.items()
        ])
        er_df.to_csv(os.path.join(reports_dir, "expected_returns.csv"), index=False)

    def run(self, export_files=True, reports_dir=None):
        """Execute end-to-end quantitative multi-factor pipeline."""
        universe_df = self.build_universe()
        prices, valid_tickers = self.fetch_market_data(universe_df)
        factors_df, asset_returns, benchmark_returns, sector_map = self.compute_factors(
            prices, universe_df, valid_tickers
        )
        selected_tickers, optimal_weights, full_weights, mu, cov_matrix, frontier_df = self.optimize_portfolio(
            factors_df, asset_returns, sector_map
        )
        analytics = self.run_backtest_and_analytics(
            factors_df,
            asset_returns,
            benchmark_returns,
            selected_tickers,
            optimal_weights,
            full_weights,
            sector_map
        )

        if export_files:
            self.export_reports(factors_df, analytics["weights_df"], analytics, frontier_df, mu.to_dict(), reports_dir)

        return {
            "region": self.region,
            "exchange": self.exchange,
            "market_cap_filter": self.market_cap_filter,
            "optimization_model": self.optimization_model,
            "factor_weights": self.factor_weights,
            "currency": self.currency,
            "currency_symbol": self.currency_symbol,
            "benchmark_name": self.benchmark_name,
            "benchmark_ticker": self.benchmark_ticker,
            "rf": self.rf,
            "universe_df": universe_df,
            "prices": prices,
            "factors_df": factors_df,
            "selected_tickers": selected_tickers,
            "optimal_weights": optimal_weights,
            "full_weights": full_weights,
            "mu": mu,
            "cov_matrix": cov_matrix,
            "frontier_df": frontier_df,
            **analytics
        }
