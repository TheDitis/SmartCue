import numpy as np
from typing import Union, Tuple
import cv2 as cv
import pandas as pd
from numba import jit, njit
import math
import matplotlib.colors as mcolors
from Point import Point

default_colors = [
    "red",
    "orange",
    "yellow",
    "limegreen",
    "blue",
    "violet",
    "blueviolet",
    "salmon",
    "darkorange",
    "lightseagreen",
    "aqua",
    "lightblue",
    "firebrick",
    "olivedrab"
]


def draw_line(
        frame: np.ndarray,
        line: Union[list, np.array, pd.Series],
        color: Union[str, tuple] = (0, 0, 255),
        thickness: int = 2
):
    """
    Draws the given line on the given frame
    Args:
        frame: the frame you want to draw on
        line: the line to draw (fist 4 elements must by x1, y1, x2,
            & y2)
        color: the color you want the line to be
        thickness: the thickness in px you want the line to be

    Returns:
        None, modifies the passed frame
    """
    x1, y1, x2, y2 = line
    pt1, pt2 = (x1, y1), (x2, y2)
    # if a color string was passed, convert it to BGR tuple
    if isinstance(color, str):
        color = mcolors.colorConverter.to_rgb(color)
        color = tuple((int(v * 255) for v in color))
        color = tuple(reversed(color))  # rgb to bgr
    cv.line(frame, pt1, pt2, color, thickness)


def find_intersection(
        line1: Union[np.array, list, tuple],
        line2: Union[np.array, list, tuple]
) -> Union[Tuple[int, int], None]:
    """
    Finds the intersection point of 2 lines
    Args:
        line1: array-like containing x1, y1, x2, & y2
        line2: array-like containing x1, y1, x2, & y2

    Returns:
        Tuple with the x and y value of their intersection if the
        lines are not parallel, otherwise None
    """
    x1, y1, x2, y2 = line1[0], line1[1], line1[2], line1[3]
    sx1, sy1, sx2, sy2 = line2[0], line2[1], line2[2], line2[3]
    line1 = ((x1, y1), (x2, y2))
    line2 = ((sx1, sy1), (sx2, sy2))
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)

    if div != 0:
        d = (det(*line1), det(*line2))
        x = int(det(d, x_diff) / div)
        y = int(det(d, y_diff) / div)
        pt = (x, y)
        return pt


def draw_lines(
        frame: np.ndarray,
        lines: Union[list, np.array, pd.DataFrame],
        color: tuple = (0, 0, 255),
        thickness: int = 2
) -> np.ndarray:
    """Draws given lines on given frame
    Args:
        frame: the frame you want to draw the lines on
        lines: the lines you want drawn
        color: BGR formatted tuple
        thickness: line thickness

    Returns:
        the given frame with the table boundaries found drawn on
    """

    # make frame copy so we don't mutate the original frame
    copy = frame.copy()
    # if frame is grayscale, convert so line colors work
    if len(copy.shape) == 2:
        copy = cv.cvtColor(copy, cv.COLOR_GRAY2BGR)

    if isinstance(lines, pd.DataFrame):
        type_colors = {"table": 0, "pocket": 2, "bumper": 3}
        for i, line in lines.iterrows():
            if "type" in line:
                color = default_colors[type_colors[line["type"]]]
            else:
                color = default_colors[i % len(default_colors)]
            draw_line(
                copy,
                line[["x1", "y1", "x2", "y2"]],
                color=color
            )
    else:
        for line_s in lines:
            # if line_s is a single line:
            if isinstance(line_s[0], (int, float, np.int32, np.int64)):
                draw_line(copy, line_s)
            # if line_s is a group of lines
            else:
                for line in line_s:
                    draw_line(copy, line)
    return copy


def draw_lines_by_group(
        frame: np.ndarray,
        lines: pd.DataFrame,
        inplace: bool = False
) -> np.ndarray:
    """
    Draws all lines in the given lines dataframe, where each line
    group has a unique color
    Args:
        frame: The frame you want lines drawn on
        lines: A dataframe of lines, where each row contains x1, y1,
            x2, y2 (all 4 float or int), & group (int)
        inplace: If true, the original frame will be modified rather
            than copied

    Returns:
        Returns modified version of the passed frame with lines
    """
    if not inplace:
        frame = frame.copy()
    # draw each line coloring by the cluster it was assigned
    for i, line in lines.iterrows():
        if line["group"] < 0:
            color = "white"
        else:
            color = default_colors[
                int(line["group"]) % len(default_colors)
            ]

        points = line["x1":"y2"]
        draw_line(frame, points.astype(np.int32), color)
    return frame


def canny_image(frame: np.ndarray, canny_setting: dict) -> np.ndarray:
    """
    Run canny edge detection on the passed image.
    Args:
        frame: The frame to detect edges of
        canny_setting: parameters for the edge detection algorithm

    Returns:
        Binary image as result of canny edge detection
    """
    blur = cv.bilateralFilter(frame, 15, 35, 175)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    cv.imwrite("./debug_images/1.5_blur.png", blur)
    try:
        # Run canny edge-detection based on setting
        if canny_setting["mode"] == "auto":
            kwargs = {
                k: v for k, v in canny_setting.items()
                if k != "mode" and v is not None
            }
            canny = auto_canny(gray, **kwargs)

        else:
            low = canny_setting["low"]
            thresh_ratio = canny_setting["thresh_ratio"]
            canny = cv.Canny(
                gray,
                low,
                low * thresh_ratio
            )

    except KeyError:
        raise KeyError(
            "Malformed setting. Every setting must "
            "include a 'canny' key, with an object that "
            "contains a 'mode' key with a value of "
            "either 'auto' or 'manual'"
        )
    return canny


def auto_canny(
        frame: np.ndarray,
        sigma: float = 0.33,
        upper_mod: float = 1.0,
        lower_mod: float = 1.0,
        aperture_size: int = None
) -> np.ndarray:
    """
    Calculates image mean and creates canny image based on it

    Args:
        frame: the frame to run canny edge detection on
        sigma: shift parameter for upper and lower thresholds
        upper_mod: the upper threshold
        lower_mod: the lower threshold
        aperture_size: aperture size param passed directly to cv.Canny

    Returns:
        Binary image as result of canny edge detection
    """
    # find the median of the single channel pixel intensities
    v = np.median(frame)
    # apply automatic Canny edge detection using the median
    lower = int(max(0, (1.0 - sigma) * v) * lower_mod)
    upper = int(min(255, (1.0 + sigma) * v) * upper_mod)
    edged = cv.Canny(frame, lower, upper, apertureSize=aperture_size)
    return edged


# @jit(nopython=True)
def rho_theta_to_xy_lines(lines: np.array) -> np.ndarray:
    return np.apply_along_axis(rho_theta_to_xy, 1, lines)


@jit(nopython=True)
def rho_theta_to_xy(row):
    rho, theta = row
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * -b)
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * -b)
    y2 = int(y0 - 1000 * a)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def get_slope(line: Union[list, pd.Series, np.array, tuple]) -> float:
    line = np.array(line)
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return float("-inf")
    return (y2 - y1) / (x2 - x1)


def distance(
        pt1: Point,
        pt2: Point
) -> Union[int, float]:
    x1, y1 = pt1
    x2, y2 = pt2
    x_diff = abs(x2 - x1)
    y_diff = abs(y2 - y1)
    return math.sqrt((x_diff ** 2) + (y_diff ** 2))
