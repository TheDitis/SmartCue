import math

import cv2 as cv
import numpy as np
from typing import Union, Dict
from utils import (
    draw_lines,
    canny_image,
    rho_theta_to_xy_lines,
    get_slope,
    draw_lines_by_group
)
from copy import deepcopy
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances, mean_squared_error
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from TableBoundaries import TableBoundaries


class Table:
    def __init__(self, capture: cv.CAP_V4L2, settings: dict):
        self._cap = capture
        self._settings = settings
        self._ref_frame = None
        self.boundaries = TableBoundaries(capture, settings)
        self.boundaries.find()
        self._pockets = None

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

