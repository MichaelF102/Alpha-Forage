import pandas as pd


class Correlation:

    @staticmethod
    def matrix(returns):

        return returns.corr()
