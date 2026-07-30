import numpy as np


class Beta:

    @staticmethod
    def calculate(

        portfolio_returns,

        benchmark_returns

    ):

        covariance = np.cov(

            portfolio_returns,

            benchmark_returns

        )[0][1]

        variance = np.var(

            benchmark_returns

        )

        return covariance / variance
