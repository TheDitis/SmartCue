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
        self._playfield = boundaries.bumper
        self._defaults = [0, 0, 10, 0, 0, 0, None, False, False]
        self._max_count = 16
        self._target_colors = colors
        self._balls = BallGroup

    def find(self, frame: np.ndarray, save: bool = False):
        # canny = canny_image(frame, self._settings["canny"])
        self._find_circles(frame)
        # if save:
        #     cv.imwrite("debug_images/8_ball_canny.png", canny)

    def _find_circles(self, frame: np.ndarray, save: bool = False):
        # cimg = frame.copy()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 5)
        circles = cv.HoughCircles(
            blur, cv.HOUGH_GRADIENT, 1, 12, param1=40, param2=20,
            minRadius=12, maxRadius=20
        )

        # circles = np.uint16(np.around(circles))
        # for i in circles[0, :]:
        #     # draw the outer circle
        #     cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
        #     # draw the center of the circle
        #     cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        # print("HERE2")
        # cv.imshow('detected circles', frame)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
