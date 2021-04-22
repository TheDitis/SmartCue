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
        setting_num = settings["ball_detection"]["setting_number"]
        self._settings = settings["ball_detection"]["ball_detect_settings"][setting_num]
        self._use_cuda = settings["program"]["CUDA"]
        self._boundaries = boundaries
        self._playfield = boundaries.bumper
        self._defaults = [0, 0, 10, 0, 0, 0, None, False, False]
        self._max_count = 16
        self._target_colors = colors
        self._balls = BallGroup
        self._detector = cv.cuda.createHoughCirclesDetector(
            dp=2, minDist=25, cannyThreshold=60, votesThreshold=25,
            minRadius=16, maxRadius=20, maxCircles=self._max_count
        )
        self._blur_filter = cv.cuda.createMedianFilter(cv.CV_8UC1, 3)
        self._gpu_frame = cv.cuda_GpuMat()

    def find(self, frame: np.ndarray):
        if self._use_cuda:
            self._find_circles_cuda(frame)
        else:
            self._find_circles(frame)

    def _find_circles(self, frame: np.ndarray):
        # crop the frame to the inside bumper lines and prepare it
        crop = self._boundaries.pocket.crop_to(frame)
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 3)

        # find circles
        circles = cv.HoughCircles(
            blur, cv.HOUGH_GRADIENT, 2, 25, param1=60, param2=25,
            minRadius=16, maxRadius=20
        )

        self._draw_circles(frame, circles)

    def _find_circles_cuda(self, frame: np.ndarray):
        # crop the frame to the inside bumper lines and prepare it
        crop = self._boundaries.pocket.crop_to(frame)
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 3)
        self._gpu_frame.upload(blur)

        # for some reason the cuda filters seem to be much slower
        # gray = cv.cuda.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
        # blur = self._blur_filter.apply(gray)

        # find circles
        circles = self._detector.detect(self._gpu_frame).download()
        self._draw_circles(frame, circles)

    def _draw_circles(self, frame: np.ndarray, circles: np.ndarray):
        # convert datatype and translate center to non-cropped pos
        circles = np.uint16(np.around(circles))
        circles[0, :, 0:2] += np.array(
            self._boundaries.pocket.tl.as_list,
            dtype=np.uint16
        )

        # draw circles
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)