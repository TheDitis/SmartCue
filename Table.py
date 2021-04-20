import cv2 as cv
import numpy as np

from TableBoundaries import TableBoundaries
from PocketSet import PocketSet


class Table:
    def __init__(self, capture: cv.CAP_V4L2, settings: dict):
        self._cap = capture
        self._settings = settings
        _, self._ref_frame = capture.read()
        self.boundaries = TableBoundaries(capture, settings)
        self.boundaries.find()
        self.pockets = PocketSet()
        self.pockets.find(self.boundaries)
        self.pockets.draw(self._ref_frame)

    def draw_boundary_lines(
            self,
            frame: np.ndarray,
            color: tuple = (0, 0, 255),
            thickness: int = 2
    ) -> np.ndarray:
        """Draws found lines on given frame
        Args:
            frame: the frame you want to draw the lines on
            color: BGR formatted tuple
            thickness: line thickness

        Returns:
            the given frame with the table boundaries found drawn on
        """
        return self.boundaries.draw_boundary_lines(
            frame,
            color,
            thickness
        )

