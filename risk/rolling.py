import numpy as np


class RollingStatistics:

    @staticmethod
    def volatility(

        returns,

        window=30

    ):

        return (

            returns

            .rolling(window)

            .std()

            * np.sqrt(252)

        )

    @staticmethod
    def sharpe(

        returns,

        window=30,

        rf=0.04

    ):

        mean = (

            returns

            .rolling(window)

            .mean()

            * 252

        )

        std = (

            returns

            .rolling(window)

            .std()

            * np.sqrt(252)

        )

        return (

            mean - rf

        ) / std
