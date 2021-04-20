import pandas as pd


class Pocket(pd.Series):
    # def __init__(self, row: pd.Series):
    #     super().__init(row)

    @property
    def center(self):
        return self['x', 'y']

    @property
    def radius(self):
        return self['r']

    @property
    def x(self):
        return self['x']

    @property
    def y(self):
        return self['y']

    @property
    def r(self):
        return self.radius
