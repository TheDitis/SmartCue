import numpy as np
from typing import Union
import cv2 as cv
import pandas as pd
from numba import jit, njit
import math
import matplotlib.colors as mcolors

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
    "lightblue"
]


def draw_line(
        frame: np.ndarray,
        line: Union[list, np.array],
        color: Union[str, tuple] = (0, 0, 255),
        thickness: int = 2
):
    x1, y1, x2, y2 = line
    pt1, pt2 = (x1, y1), (x2, y2)
    # if a color string was passed, convert it to BGR tuple
    if isinstance(color, str):
        color = mcolors.colorConverter.to_rgb(color)
        color = tuple((int(v * 255) for v in color))
        color = tuple(reversed(color))  # rgb to bgr
    cv.line(frame, pt1, pt2, color, thickness)


def draw_lines(
        frame: np.ndarray,
        lines: Union[list, np.array],
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
) -> np.ndarray:
    copy = frame.copy()
    # draw each line coloring by the cluster it was assigned
    for i, line in lines.iterrows():
        color = default_colors[
            int(line["group"]) % len(default_colors)
        ]
        points = line["x1":"y2"]
        draw_line(copy, points, color)
    return copy


def canny_image(frame: np.ndarray, canny_setting: dict) -> np.ndarray:
    # blur = frame
    # blur = cv.blur(frame, (5, 5))
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
    """Calculates image mean and creates canny image based on it

    Args:
        frame:
        sigma:
        upper_mod:
        lower_mod:
        aperture_size:

    Returns:

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


def get_slope(line):
    line = np.array(line)
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return float("-inf")
    return (y2 - y1) / (x2 - x1)


# def get_slope_