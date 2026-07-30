import numpy as np


class Performance:

    @staticmethod
    def CAGR(

        cumulative,

        years

    ):

        return (

            cumulative.iloc[-1]

            ** (1 / years)

            - 1

        )

    @staticmethod
    def volatility(

        returns

    ):

        return (

            returns.std()

            * np.sqrt(252)

        )

    @staticmethod
    def sharpe(

        returns,

        rf=0.04

    ):

        excess = (

            returns.mean()

            * 252

            - rf

        )

        return excess / (

            returns.std()

            * np.sqrt(252)

        )

    @staticmethod
    def sortino(

        returns,

        rf=0.04

    ):

        downside = returns[

            returns < 0

        ]

        downside_std = (

            downside.std()

            * np.sqrt(252)

        )

        excess = (

            returns.mean()

            * 252

            - rf

        )

        return excess / downside_std
