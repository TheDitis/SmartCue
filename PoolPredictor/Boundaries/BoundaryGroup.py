import numpy as np
import pandas as pd
from PoolPredictor.Boundaries.Boundary import Boundary
from PoolPredictor.Point import Point
from PoolPredictor.Boundaries.Box import Box
from typing import Union, Tuple


@pd.api.extensions.register_dataframe_accessor("BoundaryGroup")
class BoundaryGroup:
    """
    Represents any group of table boundaries, whether all boundaries
    on a given side of the table or all boundaries of the same type.
    Wraps pandas Dataframe, adding getters that wrap output in classes
    for convenience (BoundaryGroup, Boundary, or Box depending on the
    getter and the size of the result). Getters return a Boundary if
    there is only one row that matches the query, a BoundaryGroup if
    there are multiple rows matching the query, or a Box if corners
    are requested on an instance that has exactly 4 lines.
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def top(self):
        return self._get_by_side('t')

    @property
    def bottom(self):
        return self._get_by_side('b')

    @property
    def left(self):
        return self._get_by_side('l')

    @property
    def right(self):
        return self._get_by_side('r')

    @property
    def t(self):
        return self.top

    @property
    def b(self):
        return self.bottom

    @property
    def l(self):
        return self.left

    @property
    def r(self):
        return self.right

    @property
    def bumper(self):
        return self._get_by_type('bumper')

    @property
    def pocket(self):
        return self._get_by_type('pocket')

    @property
    def table(self):
        return self._get_by_type('table')

    @property
    def top_left(self):
        return self._get_corner("tl")

    @property
    def top_right(self):
        return self._get_corner("tr")

    @property
    def bottom_left(self):
        return self._get_corner("bl")

    @property
    def bottom_right(self):
        return self._get_corner("br")

    @property
    def tl(self):
        return self.top_left

    @property
    def tr(self):
        return self.top_right

    @property
    def bl(self):
        return self.bottom_left

    @property
    def br(self):
        return self.bottom_right

    @property
    def size_bounding(self) -> Union[Tuple[int, int], None]:
        corners = self.corners
        if corners is not None:
            return corners.size_bounding

    @property
    def size(self) -> Union[Tuple[int, int], None]:
        corners = self.corners
        if corners is not None:
            return corners.size

    @property
    def rect(self) -> Box:
        corners = self.corners
        if corners is not None:
            return corners.bounding_rect

    @property
    def corners(self) -> Union[pd.DataFrame, None]:
        if len(self._df) == 4:
            return self._df.Box._df

    def crop(self, frame: np.ndarray) -> np.ndarray:
        corners = self.corners
        if corners is not None:
            return corners.Box.crop(frame)

    def _get_by_side(
            self,
            side: str
    ) -> Union[pd.Series, pd.DataFrame, None]:
        """
        Get boundary/boundaries on a given side of the table
        Args:
            side: character representing side of the table ('t', 'b',
                'l', or 'r').

        Returns:
            Boundary instance if there is only one boundary matching,
            otherwise another BoundaryGroup containing the resulting
            subset, or None if an invalid side is passed
        """
        if side not in ['t', 'b', 'l', 'r']:
            return None
        grp = self._df[self._df["side"] == side]
        return grp
        # if len(grp) == 1:
        #     return Boundary(grp)
        # else:
        #     return BoundaryGroup(grp)

    def _get_by_type(
            self,
            kind: str
    ) -> Union[Boundary, 'BoundaryGroup', None]:
        """
        Get boundary/boundaries of a given type
        Args:
            kind: type of boundary to get. Must be 'table', 'pocket',
            or 'bumper'

        Returns:
            Boundary instance if there is only one boundary matching,
            otherwise another BoundaryGroup containing the resulting
            subset, or None if an invalid type is passed
        """
        if kind not in ["table", "pocket", "bumper"]:
            return None
        grp = self._df[self._df["type"] == kind]
        return grp
        # if len(grp) == 1:
        #     return Boundary(grp)
        # else:
        #     return BoundaryGroup(grp)

    def _get_corner(self, loc: str) -> Union[Point, None]:
        """
        Gets corner at the given location
        Args:
            loc: location of corner ('tl', 'tr', 'bl', or 'br' where
                'tl' = top-left, 'br' = bottom-right, etc.)

        Returns:
            pd.Series with x and y locations of the corner if the
            instance is a box with 4 lines and a valid loc string is
            passed, otherwise None
        """
        corners = self.corners
        if corners is not None and loc in ['tl', 'tr', 'bl', 'br']:
            return Point(corners.loc[loc])
