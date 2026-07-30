import pandas as pd


class Attribution:

    @staticmethod
    def contribution(

        weights,

        asset_returns

    ):

        contrib = {}

        for asset in weights.index:

            contrib[asset] = (

                weights[asset]

                *

                asset_returns[asset].mean()

            )

        return pd.Series(

            contrib

        ).sort_values(

            ascending=False

        )
