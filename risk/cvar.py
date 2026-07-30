import numpy as np


class ConditionalVaR:

    @staticmethod
    def historical(returns, confidence=0.95):

        threshold = np.percentile(

            returns,

            (1 - confidence) * 100

        )

        losses = returns[returns <= threshold]

        return abs(losses.mean())
