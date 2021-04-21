import pandas as pd
import numpy as np


class Box(pd.DataFrame):
    """
    Takes a pd.Dataframe representing a set of boundaries (ie. all
    bumper boundaries) and converts it into a representation of the
    corner points, rather than connecting lines
    """
    def __init__(self, df: pd.DataFrame):
        """
        Converts rows of passed df representing lines to rows that
        represent corner points.
        Args:
            df: dataframe containing 4 lines with columns x1, y1, x2,
                & y2. These lines should represent sides of a box
        """
        if "x1" in df:
            x_min = df["x1"].min()
            x_max = df["x2"].max()
            y_min = df["y1"].min()
            y_max = df["y2"].max()

            super().__init__(
                np.array([
                    [x_min, y_min, "tl", 't', 'l'],
                    [x_max, y_min, "tr", 't', 'r'],
                    [x_min, y_max, "bl", 'b', 'l'],
                    [x_max, y_max, "br", 'b', 'r']
                ]),
                index=["tl", "tr", "bl", "br"],
                columns=['x', 'y', 'loc', 'v_loc', 'h_loc'],
            )
            self['x'] = self['x'].astype(int)
            self['y'] = self['y'].astype(int)
        else:
            super().__init__(df)

    @property
    def tl(self):
        return self._get_corner('tl')

    @property
    def tr(self):
        return self._get_corner('tr')

    @property
    def bl(self):
        return self._get_corner('bl')

    @property
    def br(self):
        return self._get_corner('br')

    def _get_corner(self, loc: str) -> pd.Series:
        """
        gets the x and y locations of the corner at the given location
        Args:
            loc: corner location shorthand, one of ['tl', 'tr', 'bl',
             'br'] ('tr' = top-right, 'bl' = bottom-left, etc.)

        Returns:
            x and y locations of the corner requested as pd.Series
        """
        row = self.loc[loc]
        return row['x':'y']
