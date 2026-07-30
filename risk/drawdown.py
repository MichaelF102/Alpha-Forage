class Drawdown:

    @staticmethod
    def calculate(cumulative):

        running_max = cumulative.cummax()

        drawdown = (

            cumulative

            - running_max

        ) / running_max

        return drawdown

    @staticmethod
    def maximum(cumulative):

        return Drawdown.calculate(

            cumulative

        ).min()
