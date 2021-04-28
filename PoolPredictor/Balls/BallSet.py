import numpy as np
import pandas as pd
import cv2 as cv
from typing import Union, Tuple, List
from multipledispatch import dispatch

from PoolPredictor.Boundaries.TableBoundaries import TableBoundaries
from PoolPredictor.Point import Point
from PoolPredictor.Boundaries.Box import Box

number = (int, float, np.number)
iterable = (list, tuple, pd.Series, np.ndarray)


class Circle(pd.Series):
    @dispatch(number, number, number)
    def __init__(self, x, y, r):
        # since you can't assign a series column names unless from a df
        df = pd.DataFrame(
            [[x, y, r]],
            columns=['x', 'y', 'r']
        )
        super().__init__(
            df.iloc[0]
        )

    @dispatch((pd.DataFrame, np.ndarray, pd.Series, list, tuple))
    def __init__(self, vals):
        if isinstance(vals, pd.DataFrame):
            vals = vals.iloc[0]
            self.__init__(vals['x'], vals['y'], vals['r'])
        elif isinstance(vals, pd.Series) and 'x' in vals:
            self.__init__(vals['x'], vals['y'], vals['r'])
        elif isinstance(vals, iterable) and len(vals) >= 3:
            x, y, r, *_ = vals
            self.__init__(x, y, r)

    @property
    def center(self):
        return Point(self['x'], self['y'])

    @property
    def radius(self):
        return self['r']

    @property
    def box_inner(self):
        return Box.from_circle(self.center, self.radius)

    @property
    def box_outer(self):
        return Box.from_circle_outer(self.center, self.radius)

    def draw(self, frame: np.ndarray):
        self.box_inner.draw(frame)
        # self.box_outer.draw(frame)

    def find_color(self, frame: np.ndarray):
        crop = self.box_inner.crop(frame)
        flat = crop.reshape(crop.shape[0] * crop.shape[1], 3)
        mean = np.mean(flat, axis=0)
        # print(mean)


class BallGroup(pd.DataFrame):
    def __init__(self, ):
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
        self._max_count = count
        self._target_colors = colors
        self._balls = BallGroup

        if self._use_cuda:
            try:
                self._detector = cv.cuda.createHoughCirclesDetector(
                    dp=1, minDist=13, cannyThreshold=60, votesThreshold=13,
                    minRadius=14, maxRadius=17, maxCircles=self._max_count
                )
                self._blur_filter = cv.cuda.createMedianFilter(cv.CV_8UC1, 3)
                self._gpu_frame = cv.cuda_GpuMat()
            except AttributeError:
                print("The CUDA setting is set to true in settings.json but "
                      "the installed OpenCV module was not built with CUDA "
                      "support. Switching to CPU.")
                self._use_cuda = False
        if not self._use_cuda:
            self._blur_filter = None
            self._gpu_frame = None

    def find(self, frame: np.ndarray):
        """
        Finds the balls in the given frame
        Args:
            frame: The frame to find balls in

        Returns:
            None. Modifies self
        """
        if self._use_cuda:
            circles = self._find_circles_cuda(frame)
        else:
            circles = self._find_circles(frame)
        # self._draw_circles(frame, circles)
        # circles = pd.DataFrame(
        #     circles,
        #     columns=['x', 'y', 'r']
        # )
        # e1 = cv.getTickCount()

        for circle in circles:
            circ = Circle(circle)
            circ.find_color(frame)
            circ.draw(frame)
            # circ.find_color(frame)

        # e2 = cv.getTickCount()
        # t = (e2 - e1) / cv.getTickFrequency()
        # print("T tick: ", t)

    def _find_circles(self, frame: np.ndarray) -> np.ndarray:
        """
        Find potential-ball circles in the passed frame using the CPU (no CUDA)
        Args:
            frame: The current frame

        Returns:
            None
        """
        # crop the frame to the inside bumper lines and prepare it
        crop = self._boundaries.pocket.Box.crop(frame)
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 3)
        # blur = gray

        # find circles
        circles = cv.HoughCircles(
            blur, cv.HOUGH_GRADIENT, dp=2, minDist=13, param1=60, param2=30,
            minRadius=14, maxRadius=17
        )

        # convert datatype and translate center to non-cropped pos
        circles = np.uint16(np.around(circles[0]))
        translation = np.array(self._boundaries.pocket.Box.tl, dtype=np.uint16)
        circles[:, 0:2] += translation

        return circles

    def _find_circles_cuda(self, frame: np.ndarray) -> np.ndarray:
        """
        Find potential-ball circles in the given frame with CUDA acceleration
        Args:
            frame: The current frame

        Returns:
            numpy array of circles
        """
        # crop the frame to the inside bumper lines and prepare it
        crop = self._boundaries.pocket.crop(frame)
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        blur = cv.medianBlur(gray, 3)
        # blur = gray
        self._gpu_frame.upload(blur)

        # for some reason the cuda filters seem to be much slower
        # gray = cv.cuda.cvtColor(self._frame, cv.COLOR_BGR2GRAY)
        # blur = self._blur_filter.apply(gray)

        # find circles
        circles = self._detector.detect(self._gpu_frame).download()

        # convert datatype and translate center to non-cropped pos
        circles = np.uint16(np.around(circles[0]))
        translation = np.array(self._boundaries.pocket.tl, dtype=np.uint16)
        circles[:, 0:2] += translation

        return circles

    @staticmethod
    def _draw_circles(frame: np.ndarray, circles: np.ndarray):
        """
        Draws the passed array of circles on the passed frame inplace
        Args:
            frame: The frame you want circles drawn on
            circles: The array of circles

        Returns:
            None. Modifies passed frame
        """
        # draw circles
        for i in circles:
            # draw the outer circle
            cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
            # draw the center of the circle
            cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
