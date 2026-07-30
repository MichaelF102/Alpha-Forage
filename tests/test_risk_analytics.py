import unittest
import numpy as np
import pandas as pd
import sys
import os

# Include parent directory in python path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk.var import ValueAtRisk
from risk.cvar import ConditionalVaR
from risk.beta import Beta
from risk.drawdown import Drawdown
from risk.stress import StressTest
from risk.correlation import Correlation
from risk.rolling import RollingStatistics

from analytics.performance import Performance
from analytics.attribution import Attribution

class TestRiskAnalytics(unittest.TestCase):
    def setUp(self):
        # Create some predictable synthetic returns data
        # Let's say 10 days of returns
        self.returns_data = pd.Series([0.01, -0.02, 0.015, -0.01, 0.025, -0.03, 0.005, -0.015, 0.02, -0.005])
        
        # Benchmark returns (highly correlated for testing beta)
        self.benchmark_data = pd.Series([0.008, -0.015, 0.012, -0.008, 0.02, -0.025, 0.004, -0.012, 0.016, -0.004])
        
        # Cumulative returns series starting from 1.0
        self.cumulative_returns = pd.Series([1.0, 1.01, 0.99, 1.02, 1.01, 1.03, 1.00, 1.04, 1.02, 1.05])
        
    def test_var_historical(self):
        # Returns sorted: [-0.03, -0.02, -0.015, -0.01, -0.005, 0.005, 0.01, 0.015, 0.02, 0.025]
        # 90% confidence on 10 returns means 10% percentile (index 0.9)
        var_val = ValueAtRisk.historical(self.returns_data, confidence=0.90)
        self.assertGreater(var_val, 0.0)
        self.assertAlmostEqual(var_val, 0.021, places=4)
        
    def test_var_parametric(self):
        var_val = ValueAtRisk.parametric(self.returns_data, confidence=0.95)
        # Should be mu + z * sigma
        mu = self.returns_data.mean()
        sigma = self.returns_data.std()
        from scipy.stats import norm
        z = norm.ppf(1 - 0.95)
        expected = abs(mu + z * sigma)
        self.assertAlmostEqual(var_val, expected, places=6)
        
    def test_cvar_historical(self):
        # 90% confidence, threshold is at 10th percentile
        cvar_val = ConditionalVaR.historical(self.returns_data, confidence=0.90)
        self.assertGreater(cvar_val, 0.0)
        
    def test_beta(self):
        beta_val = Beta.calculate(self.returns_data, self.benchmark_data)
        self.assertGreater(beta_val, 0.9)  # They are highly correlated and similar scale
        
    def test_drawdown(self):
        dd_series = Drawdown.calculate(self.cumulative_returns)
        # Check that drawdown is <= 0
        self.assertTrue((dd_series <= 0).all())
        # Maximum drawdown should match min
        max_dd = Drawdown.maximum(self.cumulative_returns)
        self.assertEqual(max_dd, dd_series.min())
        
    def test_stress_test(self):
        val = 100.0
        self.assertAlmostEqual(StressTest.market_crash(val), 70.0)
        self.assertAlmostEqual(StressTest.interest_rate_shock(val), 90.0)
        self.assertAlmostEqual(StressTest.recession(val), 80.0)
        
    def test_correlation(self):
        df_returns = pd.DataFrame({
            'asset1': self.returns_data,
            'asset2': self.benchmark_data
        })
        corr_matrix = Correlation.matrix(df_returns)
        self.assertEqual(corr_matrix.shape, (2, 2))
        self.assertAlmostEqual(corr_matrix.loc['asset1', 'asset1'], 1.0)
        self.assertGreater(corr_matrix.loc['asset1', 'asset2'], 0.9)
        
    def test_performance_cagr(self):
        # CAGR calculation
        cagr = Performance.CAGR(self.cumulative_returns, years=1.0)
        self.assertAlmostEqual(cagr, 0.05) # (1.05 ** (1/1)) - 1 = 0.05
        
    def test_performance_sharpe(self):
        # Sharpe ratio
        sharpe = Performance.sharpe(self.returns_data, rf=0.0)
        expected = (self.returns_data.mean() * 252) / (self.returns_data.std() * np.sqrt(252))
        self.assertAlmostEqual(sharpe, expected)
        
    def test_performance_sortino(self):
        # Sortino ratio
        sortino = Performance.sortino(self.returns_data, rf=0.0)
        downside = self.returns_data[self.returns_data < 0]
        expected = (self.returns_data.mean() * 252) / (downside.std() * np.sqrt(252))
        self.assertAlmostEqual(sortino, expected)
        
    def test_attribution(self):
        weights = pd.Series([0.5, 0.5], index=['asset1', 'asset2'])
        asset_returns = pd.DataFrame({
            'asset1': [0.01, 0.02, 0.03],
            'asset2': [-0.01, -0.02, -0.03]
        })
        contrib = Attribution.contribution(weights, asset_returns)
        self.assertAlmostEqual(contrib['asset1'], 0.5 * 0.02)
        self.assertAlmostEqual(contrib['asset2'], 0.5 * -0.02)

if __name__ == '__main__':
    unittest.main()
