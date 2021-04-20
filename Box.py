import pandas as pd
import numpy as np


class Box(pd.DataFrame):
    """
    Takes a pd.Dataframe representing a set of boundaries (ie. all
    bumper boundaries) and converts it into a prepresentation of the
    corner points, rather than connecting lines
    """
    def __init__(self, df: pd.DataFrame):
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

    def _get_corner(self, loc: str):
        row = self.loc[loc]
        return row['x':'y']
