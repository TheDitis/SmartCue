import pandas as pd
from typing import Union
from PoolPredictor.utils import distance, Point


@pd.api.extensions.register_series_accessor("Boundary")
class Boundary:
    """
    Represents one boundary line of the table (ie. left bumper, top
    table, etc). Wraps pd.Series and is used within BoundaryGroup
    """
    def __init__(self, row: Union[pd.Series, pd.DataFrame]):
        """
        Wrapper for Series constructor that allows conversion of df
        which keeps things concise elsewhere
        """
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        self._obj = row

        # super().__init__(row)

    @property
    def side(self) -> str:
        return self._obj['side']

    @property
    def type(self) -> str:
        return self._obj['type']

    @property
    def line(self) -> pd.Series:
        return self._obj['x1':'y2']

    @property
    def pt1(self) -> Point:
        return Point(self._obj['x1'], self._obj['y1'])

    @property
    def pt2(self) -> Point:
        return Point(self._obj['x2'], self._obj['y2'])

    @property
    def length(self) -> Union[int, float]:
        return distance(self._obj.pt1, self._obj.pt2)


# @pd.api.extensions.register_series_accessor("Boundary")
class Boundary_old(pd.Series):
    """
    Represents one boundary line of the table (ie. left bumper, top
    table, etc). Wraps pd.Series and is used within BoundaryGroup
    """

    def __init__(self, row: Union[pd.Series, pd.DataFrame]):
        """
        Wrapper for Series constructor that allows conversion of df
        which keeps things concise elsewhere
        """
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        # self._obj = row

        super().__init__(row)

    @property
    def side(self) -> str:
        return self['side']

    @property
    def type(self) -> str:
        return self['type']

    @property
    def line(self) -> pd.Series:
        return self['x1':'y2']

    @property
    def pt1(self) -> Point:
        return Point(self['x1'], self['y1'])

    @property
    def pt2(self) -> Point:
        return Point(self['x2'], self['y2'])

    @property
    def length(self) -> Union[int, float]:
        return distance(self.pt1, self.pt2)
