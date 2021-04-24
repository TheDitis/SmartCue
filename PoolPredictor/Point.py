import pandas as pd
import numpy as np
from typing import Union, Tuple, Iterable
from numbers import Number
from collections import namedtuple
from multipledispatch import dispatch


class Point(namedtuple("Point", ('x', 'y'))):
    """
    Class representing 2D point on the frame. Wrapper over named tuple

    Attributes:
        x: returns x value (first item)
        y: returns y value (second item)
    """

    # to keep memory utilization low:
    __slots__ = ()

    def __new__(
            cls,
            x: Union[Number, np.number, Iterable[int]],
            y: Union[Number, np.number, None] = None
    ):
        if isinstance(x, Iterable):
            points = (x[0], x[1])
        elif isinstance(x, Number) and isinstance(y, Number):
            points = (x, y)
        else:
            raise Exception(
                "Point must be initialized with a single iterator with at "
                "least 2 values, or 2 scalar values"
            )
        return tuple.__new__(cls, points)

    def __getitem__(self, item: Union[int, str]) -> Union[int, float]:
        if item in ['x', 'X']:
            return super().__getitem__(0)
        elif item in ['y', 'Y']:
            return super().__getitem__(1)
        else:
            super().__getitem__(item)

    @dispatch((int, float, np.number), (int, float, np.number))
    def translated(self, x: Number = 0, y: Number = 0):
        new_x = max(self.x + int(x), 0)
        new_y = max(self.y + int(y), 0)
        return self.__class__((new_x, new_y))

    @dispatch(tuple)
    def translated(self, amounts: Union['Point', tuple]):
        if isinstance(amounts, self.__class__):
            x, y = amounts.x, amounts.y
        else:
            x, y = amounts[0], amounts[1]

        return self.translated(x, y)

    @property
    def tup(self):
        return self.x, self.y