import numpy as np
import pandas as pd
import cv2 as cv
from typing import Union, Tuple, List
from PoolPredictor.Boundaries.TableBoundaries import TableBoundaries
from PoolPredictor.utils import canny_image


class BallGroup(pd.DataFrame):
    def __init__(self):
        super().__init__(columns=[
            'x', 'y', 'size', 'b', 'g', 'r', 'motion', 'on_field',
            'in_pocket'
        ])


class BallSet:
    def __init__(
        self,
        boundaries: TableBoundaries,
        settings: dict,
        count: int = 16,
        colors: Union[Tuple[int], List[int], None] = None
    ):
        setting_num = settings["setting_number"]
        self._settings = settings["ball_detect_settings"][setting_num]
        self._boundaries = boundaries
        self._playfield = boundaries.bumper
        self._defaults = [0, 0, 10, 0, 0, 0, None, False, False]
        self._max_count = 16
        self._target_colors = colors
        self._balls = BallGroup

    def find(self, frame: np.ndarray):
        self._find_circles(frame)

    def _find_circles(self, frame: np.ndarray, save: bool = False):
        # crop_box = self._boundaries.pocket.corners.bounding_rect
        # tl, _, _, br = crop_box.list_corners
        # crop = frame[tl.y:br.y, tl.x:br.x]
        crop = self._boundaries.pocket.crop_to(frame)
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 3)
        circles = cv.HoughCircles(
            blur, cv.HOUGH_GRADIENT, 2, 25, param1=60, param2=25,
            minRadius=16, maxRadius=20
        )

        circles = np.uint16(np.around(circles))
        # circles[0, :, 0:2] += np.array([tl.x, tl.y], dtype=np.uint16)
        circles[0, :, 0:2] += np.array(
            self._boundaries.pocket.tl.as_list,
            dtype=np.uint16
        )
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

