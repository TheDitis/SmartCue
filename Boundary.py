import pandas as pd
from typing import Union
from utils import distance, Point


class Boundary(pd.Series):
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
