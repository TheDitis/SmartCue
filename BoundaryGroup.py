import pandas as pd
from Boundary import Boundary
from Box import Box


class BoundaryGroup(pd.DataFrame):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df.apply(Boundary))

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
    def corners(self):
        if len(self) == 4:
            return Box(self)
        else:
            return None

    def _get_by_side(self, side):
        grp = self[self["side"] == side]
        if len(grp) == 1:
            return Boundary(grp)
        else:
            return BoundaryGroup(grp)

    def _get_by_type(self, kind):
        grp = self[self["type"] == kind]
        if len(grp) == 1:
            return Boundary(grp)
        else:
            return BoundaryGroup(grp)

    def _get_corner(self, loc: str):
        corners = self.corners
        if corners is not None:
            return corners.iloc[loc]