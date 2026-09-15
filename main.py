"""
AlphaForge Institutional Multi-Factor Equity Research & Portfolio Construction Platform
Main CLI entrypoint and backend calculation engine.
"""

import os
import sys
import argparse
import streamlit as st
import pandas as pd
import numpy as np

from engine import QuantEngine
from analytics.report import RiskPerformanceReport
from risk.correlation import Correlation

def parse_arguments():
    parser = argparse.ArgumentParser(description="AlphaForge Quantitative Engine CLI")
    parser.add_argument("--region", type=str, default="India", choices=["India", "US"], help="Market region universe")
    parser.add_argument("--exchange", type=str, default=None, help="Exchange filter (e.g. NSE, BSE, NASDAQ, NYSE)")
    parser.add_argument("--universe-size", type=int, default=30, help="Universe size of top liquid stocks")
    parser.add_argument("--top-k", type=int, default=12, help="Number of selected assets in optimal portfolio")
    parser.add_argument("--period", type=str, default="2y", help="Historical price lookback period")
    return parser.parse_args()

def main():
    args = parse_arguments()

    exchange = args.exchange
    if exchange is None:
        exchange = "NSE" if args.region == "India" else "NASDAQ"

    print("=" * 70)
    print("⚡ ALPHAFORGE INSTITUTIONAL QUANTITATIVE ENGINE")
    print("=" * 70)
    print(f"Region:        {args.region}")
    print(f"Exchange:      {exchange}")
    print(f"Universe Size: {args.universe_size} securities")
    print(f"Selected Top:  {args.top_k} securities")
    print(f"Lookback:      {args.period}")
    print("-" * 70)

    engine = QuantEngine(
        region=args.region,
        exchange=exchange,
        universe_size=args.universe_size,
        top_k_select=args.top_k,
        period=args.period
    )

    print("[INFO] Building universe from dataset...")
    print("[INFO] Downloading market price history from yfinance...")
    print("[INFO] Constructing multi-factor rankings (Momentum, Low Vol, Value, Quality)...")
    print("[INFO] Running constrained Mean-Variance portfolio optimization...")
    print("[INFO] Executing strategy backtest, benchmark attribution, and risk decomposition...")

    results = engine.run(export_files=True)

    factors_df = results["factors_df"]
    selected_tickers = results["selected_tickers"]
    weights_df = results["weights_df"]
    portfolio_ts = results["portfolio_ts"]
    risk_summary = results["risk_summary"]
    perf_df = results["performance_df"].set_index("Metric")

    risk_metrics = {
        "historical_var": risk_summary.get("VaR", 0.0),
        "parametric_var": risk_summary.get("ParametricVaR", 0.0),
        "historical_cvar": risk_summary.get("CVaR", 0.0),
        "beta": risk_summary.get("Beta", 1.0),
        "max_drawdown": risk_summary.get("Drawdown", 0.0)
    }

    perf_metrics = {
        "CAGR": perf_df.loc["CAGR", "Portfolio"],
        "volatility": perf_df.loc["Volatility", "Portfolio"],
        "sharpe": perf_df.loc["Sharpe", "Portfolio"],
        "sortino": perf_df.loc["Sortino", "Portfolio"]
    }

    report = RiskPerformanceReport.generate(
        weights=weights_df.set_index("Ticker").loc[selected_tickers, "Weight"],
        portfolio_returns=portfolio_ts["Return"],
        benchmark_returns=portfolio_ts["Benchmark_Return"],
        cumulative_returns=portfolio_ts["Portfolio"] / 10000.0,
        risk_metrics=risk_metrics,
        perf_metrics=perf_metrics,
        stress_test_results=results["stress_tests"],
        contribution=results["attrib_summary"]["Contribution"],
        correlation_matrix=Correlation.matrix(results["prices"][selected_tickers].pct_change().dropna())
    )

    print("\n" + report + "\n")

    # Save summary report text file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_performance_report.txt")
    with open(output_file, "w") as f:
        f.write(report)
    print(f"[INFO] Risk and performance report successfully written to: {output_file}")
    print("[INFO] All reports CSVs exported to reports/ directory.")

def run_streamlit_app():
    # Defer to app.py
    import app
    app.main()

if __name__ == "__main__":
    if st.runtime.exists():
        run_streamlit_app()
    else:
        main()
