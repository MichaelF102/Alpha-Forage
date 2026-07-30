import numpy as np


class ValueAtRisk:

    @staticmethod
    def historical(returns, confidence=0.95):

        percentile = (1 - confidence) * 100

        var = np.percentile(
            returns,
            percentile
        )

        return abs(var)

    @staticmethod
    def parametric(returns, confidence=0.95):

        from scipy.stats import norm

        mu = returns.mean()

        sigma = returns.std()

        z = norm.ppf(1 - confidence)

        var = mu + z * sigma

        return abs(var)
