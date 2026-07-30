class StressTest:

    @staticmethod
    def market_crash(

        portfolio_value,

        crash=-0.30

    ):

        stressed = portfolio_value * (

            1 + crash

        )

        return stressed

    @staticmethod
    def interest_rate_shock(

        portfolio_value,

        shock=-0.10

    ):

        return portfolio_value * (

            1 + shock

        )

    @staticmethod
    def recession(

        portfolio_value,

        shock=-0.20

    ):

        return portfolio_value * (

            1 + shock

        )
