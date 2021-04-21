import cv2 as cv
import numpy as np

from TableBoundaries import TableBoundaries
from PocketSet import PocketSet


class Table:
    """
    Represents the billiards table. Most of the functionality of the
    program is nested in this class. Contains the TableBoundaries,
    PocketSet, and BallSet classes, where most of the core computer
    vision algorithms live.
    """
    def __init__(self, capture: cv.CAP_V4L2, settings: dict):
        """
        Initializes the table by locating table boundaries and pocket
        locations.
        Args:
            capture: OpenCV capture object to read frames from
            settings: dictionary of settings loaded from settings.json
        """
        self._cap = capture
        self._settings = settings
        # get initial frame for reference
        _, self._ref_frame = capture.read()
        # Initialize and locate table boundaries
        self.boundaries = TableBoundaries(capture, settings)
        self.boundaries.find()
        # Initialize and locate pockets
        self.pockets = PocketSet()
        self.pockets.find(self.boundaries)
        self.pockets.draw(self._ref_frame, save=True)

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

