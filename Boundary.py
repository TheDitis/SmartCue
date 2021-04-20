import pandas as pd


class Boundary(pd.Series):
    def __init__(self, row: pd.Series):
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        super().__init__(row)

    @property
    def side(self):
        return self['side']

    @property
    def type(self):
        return self['type']

    @property
    def line(self):
        return self['x1':'y2']

    @property
    def pt1(self):
        return self['x1':'y1']

    @property
    def pt2(self):
        return self['x2':'y2']