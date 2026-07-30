import pandas as pd


class RiskPerformanceReport:

    @staticmethod
    def generate(

        weights,

        portfolio_returns,

        benchmark_returns,

        cumulative_returns,

        risk_metrics,

        perf_metrics,

        stress_test_results,

        contribution,

        correlation_matrix

    ):

        """

        Generates a comprehensive Risk & Performance Report.

        """

        report_lines = []

        report_lines.append("=" * 60)

        report_lines.append(

            "          PORTFOLIO RISK & PERFORMANCE REPORT"

        )

        report_lines.append("=" * 60)

        report_lines.append("")

        report_lines.append(

            "--- PORTFOLIO COMPOSITION ---"

        )

        for asset, weight in weights.items():

            report_lines.append(

                f"  {asset}: {weight * 100:.2f}%"

            )

        report_lines.append("")

        report_lines.append(

            "--- PERFORMANCE METRICS ---"

        )

        report_lines.append(

            f"  CAGR:                  {perf_metrics.get('CAGR', 0) * 100:.2f}%"

        )

        report_lines.append(

            f"  Annualized Volatility: {perf_metrics.get('volatility', 0) * 100:.2f}%"

        )

        report_lines.append(

            f"  Sharpe Ratio (Rf=4%):  {perf_metrics.get('sharpe', 0):.4f}"

        )

        report_lines.append(

            f"  Sortino Ratio (Rf=4%): {perf_metrics.get('sortino', 0):.4f}"

        )

        report_lines.append("")

        report_lines.append(

            "--- RISK METRICS ---"

        )

        report_lines.append(

            f"  Historical VaR (95%):  {risk_metrics.get('historical_var', 0) * 100:.2f}%"

        )

        report_lines.append(

            f"  Parametric VaR (95%):  {risk_metrics.get('parametric_var', 0) * 100:.2f}%"

        )

        report_lines.append(

            f"  Historical CVaR (95%): {risk_metrics.get('historical_cvar', 0) * 100:.2f}%"

        )

        report_lines.append(

            f"  Portfolio Beta:        {risk_metrics.get('beta', 0):.4f}"

        )

        report_lines.append(

            f"  Maximum Drawdown:      {risk_metrics.get('max_drawdown', 0) * 100:.2f}%"

        )

        report_lines.append("")

        report_lines.append(

            "--- STRESS TESTING SCENARIOS ---"

        )

        for scenario, val in stress_test_results.items():

            report_lines.append(

                f"  {scenario:<22}: {val:.4f} (based on portfolio cumulative return)"

            )

        report_lines.append("")

        report_lines.append(

            "--- PERFORMANCE ATTRIBUTION ---"

        )

        for asset, contrib in contribution.items():

            report_lines.append(

                f"  {asset:<10} Mean Daily Return Contribution: {contrib * 100:.4f}% ({contrib * 252 * 100:.2f}% annualized)"

            )

        report_lines.append("")

        report_lines.append(

            "--- CORRELATION MATRIX ---"

        )

        report_lines.append(

            correlation_matrix.round(4).to_string()

        )

        report_lines.append("")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)
