import pandas as pd
import numpy as np
from typing import Union


class Point:
    def __init__(
            self,
            x: Union[int, float, tuple, list, np.array, pd.Series],
            y: Union[int, float, tuple, None] = None
    ):
        if x is not None and y is not None:
            points = [x, y]
        else:
            points = x
        self._point_locs = {'x': 0, 'y': 1}
        self._n = 0
        self._pts = points

    def __getitem__(self, item: Union[int, str]) -> Union[int, float]:
        if item in ['x', 'X', 0]:
            return self.x
        elif item in ['y', 'Y', 1]:
            return self.y
        else:
            raise KeyError

    # implementing iter so that unpacking works
    def __iter__(self):
        self._n = 0
        return self

    def __next__(self) -> Union[int, float]:
        if self._n > 1:
            raise StopIteration
        else:
            v = self._pts[self._n]
            self._n += 1
            return v

    @property
    def x(self) -> Union[int, float]:
        return self._pts[0]

    @property
    def y(self) -> Union[int, float]:
        return self._pts[1]

    @property
    def as_list(self):
        return [self.x, self.y]
