import math

import cv2 as cv
import pandas as pd
import numpy as np
from typing import Tuple, Union
from numbers import Number
from PoolPredictor.utils import Point, distance
from PoolPredictor.Point import Point


# @pd.api.extensions.register_dataframe_accessor("box")
# class Box:
#     def __init__(self, df: pd.DataFrame):
#         if "x1" in df:
#             x_min = df["x1"].min()
#             x_max = df["x2"].max()
#             y_min = df["y1"].min()
#             y_max = df["y2"].max()
#
#             df = pd.DataFrame(
#                 np.array([
#                     [x_min, y_min, "tl", 't', 'l'],
#                     [x_max, y_min, "tr", 't', 'r'],
#                     [x_min, y_max, "bl", 'b', 'l'],
#                     [x_max, y_max, "br", 'b', 'r']
#                 ]),
#                 index=["tl", "tr", "bl", "br"],
#                 columns=['x', 'y', 'loc', 'v_loc', 'h_loc'],
#             )
#             df['x'] = df['x'].astype(int)
#             df['y'] = df['y'].astype(int)
#
#         self._df = df
#
#     def __repr__(self):
#         return self._df
#
#     @staticmethod
#     def _validate(obj):
#         if not all(map(lambda c: c in obj.cols, ['x1', 'y1', 'x2', 'y2'])):
#             raise AttributeError("Must have x1, y1, x2, y2")
#
#     @classmethod
#     def from_circle_outer(cls, c: Point, r: Number) -> 'Box':
#         d = r * 2
#         return cls.from_center_and_dims(c, d, d)
#
#     @classmethod
#     def from_circle(cls, c: Point, r: Number) -> 'Box':
#         hypotenuse = r * 2
#         side = math.sqrt((hypotenuse ** 2) / 2)
#         return cls.from_center_and_dims(c, side, side)
#
#     @classmethod
#     def from_center_and_dims(cls, c: Point, w: Number, h: Number) -> 'Box':
#         w = int(w / 2)
#         h = int(h / 2)
#         return cls(pd.DataFrame(
#             [
#                 [c.x - w, c.y - h, "tl", 't', 'l'],
#                 [c.x + w, c.y - h, "tr", 't', 'r'],
#                 [c.x - w, c.y + h, "bl", 'b', 'l'],
#                 [c.x + w, c.y + h, "br", 'b', 'r'],
#             ],
#             index=["tl", "tr", "bl", "br"],
#             columns=['x', 'y', 'loc', 'v_loc', 'h_loc'],
#         ))
#
#     @property
#     def list_corners(self):
#         return [self.tl, self.tr, self.bl, self.br]
#
#     @property
#     def width_bounding(self) -> int:
#         """
#         Returns:
#             Width that would fit the whole box
#         """
#         x_min = min(self.tl.x, self.bl.x)
#         x_max = max(self.tr.x, self.br.x)
#         return x_max - x_min
#
#     @property
#     def height_bounding(self) -> int:
#         """
#         Returns:
#             Height that would fit the whole box
#         """
#         y_min = min(self.tl.y, self.tr.y)
#         y_max = max(self.bl.y, self.br.y)
#         return y_max - y_min
#
#     @property
#     def size_bounding(self) -> Tuple[int, int]:
#         """
#         Returns:
#             (width, height) of the smallest box that the instance box
#             will fit in. Same as the box size if the box is not
#             rotated
#         """
#         return self.width_bounding, self.height_bounding
#
#     @property
#     def width(self):
#         return distance(self.tl, self.tr)
#
#     @property
#     def height(self):
#         return distance(self.tl, self.br)
#
#     @property
#     def size(self) -> Tuple[int, int]:
#         return self.width, self.height
#
#     @property
#     def bounding_rect(self) -> 'Box':
#         """
#         Returns:
#             A Box instance that can fit the calling instance box
#             without rotation
#         """
#         x_min = min(self.tl.x, self.bl.x)
#         y_min = min(self.tl.y, self.tr.y)
#         width = self.width_bounding
#         height = self.height_bounding
#         copy = self._df
#         copy.loc['tl', 'x':'y'] = [x_min, y_min]
#         copy.loc['tr', 'x':'y'] = [x_min + width, y_min]
#         copy.loc['bl', 'x':'y'] = [x_min, y_min + height]
#         copy.loc['br', 'x':'y'] = [x_min + width, y_min + height]
#         return Box(copy)
#
#     @property
#     def _constructor(self):
#         return Box
#
#     @property
#     def tl(self) -> Point:
#         return self._get_corner('tl')
#
#     @property
#     def tr(self) -> Point:
#         return self._get_corner('tr')
#
#     @property
#     def bl(self) -> Point:
#         return self._get_corner('bl')
#
#     @property
#     def br(self) -> Point:
#         return self._get_corner('br')
#
#     @property
#     def lines(self) -> Tuple[tuple, tuple, tuple, tuple]:
#         return (
#             (self.tl, self.tr),
#             (self.tr, self.br),
#             (self.bl, self.br),
#             (self.tl, self.bl)
#         )
#
#     def _get_corner(self, loc: str) -> Point:
#         """
#         gets the x and y locations of the corner at the given location
#         Args:
#             loc: corner location shorthand, one of ['tl', 'tr', 'bl',
#              'br'] ('tr' = top-right, 'bl' = bottom-left, etc.)
#
#         Returns:
#             x and y locations of the corner requested as pd.Series
#         """
#         row = self._df.loc[loc]
#         return Point(row['x'], row['y'])
#
#     def crop(self, frame: np.ndarray) -> np.ndarray:
#         bounding = self.bounding_rect
#         tl, br = bounding.tl, bounding.br
#         return frame[tl.y:br.y, tl.x:br.x]
#
#     def draw(
#             self,
#             frame: np.ndarray,
#             color: Tuple[int, int, int] = (255, 255, 0),
#             thickness: int = 1
#     ):
#         cv.rectangle(frame, tuple(self.tl), tuple(self.br), color, thickness)


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
        else:
            super().__init__(df)
        self['x'] = self['x'].astype(int)
        self['y'] = self['y'].astype(int)

    @classmethod
    def from_circle_outer(cls, c: Point, r: Number) -> 'Box':
        d = r * 2
        return cls.from_center_and_dims(c, d, d)

    @classmethod
    def from_circle(cls, c: Point, r: Number) -> 'Box':
        hypotenuse = r * 2
        side = math.sqrt((hypotenuse**2) / 2)
        return cls.from_center_and_dims(c, side, side)

    @classmethod
    def from_center_and_dims(cls, c: Point, w: Number, h: Number) -> 'Box':
        w = int(w / 2)
        h = int(h / 2)
        return cls(pd.DataFrame(
            [
                [c.x - w, c.y - h, "tl", 't', 'l'],
                [c.x + w, c.y - h, "tr", 't', 'r'],
                [c.x - w, c.y + h, "bl", 'b', 'l'],
                [c.x + w, c.y + h, "br", 'b', 'r'],
            ],
            index=["tl", "tr", "bl", "br"],
            columns=['x', 'y', 'loc', 'v_loc', 'h_loc'],
        ))

    @property
    def list_corners(self):
        return [self.tl, self.tr, self.bl, self.br]

    @property
    def width_bounding(self) -> int:
        """
        Returns:
            Width that would fit the whole box
        """
        x_min = min(self.tl.x, self.bl.x)
        x_max = max(self.tr.x, self.br.x)
        return x_max - x_min

    @property
    def height_bounding(self) -> int:
        """
        Returns:
            Height that would fit the whole box
        """
        y_min = min(self.tl.y, self.tr.y)
        y_max = max(self.bl.y, self.br.y)
        return y_max - y_min

    @property
    def size_bounding(self) -> Tuple[int, int]:
        """
        Returns:
            (width, height) of the smallest box that the instance box
            will fit in. Same as the box size if the box is not
            rotated
        """
        return self.width_bounding, self.height_bounding

    @property
    def width(self):
        return distance(self.tl, self.tr)

    @property
    def height(self):
        return distance(self.tl, self.br)

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    @property
    def bounding_rect(self) -> 'Box':
        """
        Returns:
            A Box instance that can fit the calling instance box
            without rotation
        """
        x_min = min(self.tl.x, self.bl.x)
        y_min = min(self.tl.y, self.tr.y)
        width = self.width_bounding
        height = self.height_bounding
        copy = self.copy()
        copy.loc['tl', 'x':'y'] = [x_min, y_min]
        copy.loc['tr', 'x':'y'] = [x_min + width, y_min]
        copy.loc['bl', 'x':'y'] = [x_min, y_min + height]
        copy.loc['br', 'x':'y'] = [x_min + width, y_min + height]
        return Box(copy)

    # @property
    # def _constructor(self):
    #     return Box

    @property
    def tl(self) -> Point:
        return self._get_corner('tl')

    @property
    def tr(self) -> Point:
        return self._get_corner('tr')

    @property
    def bl(self) -> Point:
        return self._get_corner('bl')

    @property
    def br(self) -> Point:
        return self._get_corner('br')

    @property
    def lines(self) -> Tuple[tuple, tuple, tuple, tuple]:
        return (
            (self.tl, self.tr),
            (self.tr, self.br),
            (self.bl, self.br),
            (self.tl, self.bl)
        )

    def _get_corner(self, loc: str) -> Point:
        """
        gets the x and y locations of the corner at the given location
        Args:
            loc: corner location shorthand, one of ['tl', 'tr', 'bl',
             'br'] ('tr' = top-right, 'bl' = bottom-left, etc.)

        Returns:
            x and y locations of the corner requested as pd.Series
        """
        row = self.loc[loc]
        return Point(row['x'], row['y'])

    def crop(self, frame: np.ndarray) -> np.ndarray:
        bounding = self.bounding_rect
        tl, br = bounding.tl, bounding.br
        return frame[tl.y:br.y, tl.x:br.x]
        # return frame

    def draw(
            self,
            frame: np.ndarray,
            color: Tuple[int, int, int] = (255, 255, 0),
            thickness: int = 1
    ):
        cv.rectangle(frame, tuple(self.tl), tuple(self.br), color, thickness)
