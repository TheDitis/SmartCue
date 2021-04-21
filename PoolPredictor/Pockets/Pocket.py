import pandas as pd


class Pocket(pd.Series):
    """
    Wraps a row in the PocketSet df, providing more convenient and
    clear access to location and size of the pocket
    """
    @property
    def center(self):
        return self[['x', 'y']].astype(int)

    @property
    def radius(self):
        return self['r'].astype(float)

    @property
    def x(self):
        return self['x'].astype(int)

    @property
    def y(self):
        return self['y'].astype(int)

    @property
    def r(self):
        return self.radius
