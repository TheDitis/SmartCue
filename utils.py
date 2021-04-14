import numpy as np
from typing import Union
import cv2 as cv


def draw_lines(
        frame: np.ndarray,
        lines: Union[list, np.array],
        color: tuple = (0, 0, 255),
        thickness: int = 2
) -> np.ndarray:
    """Draws given lines on given frame
    Args:
        frame: the frame you want to draw the lines on
        color: BGR formatted tuple
        thickness: line thickness

    Returns:
        the given frame with the table boundaries found drawn on
    """
    def draw_line(frame_, line_):
        x1, y1, x2, y2 = line_
        pt1, pt2 = (x1, y1), (x2, y2)
        cv.line(frame_, pt1, pt2, color, thickness)

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
                draw_line(copy, line_s)
    return copy


def canny_image(frame: np.ndarray, canny_setting: dict) -> np.ndarray:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
        lower_mod: float = 1.0
) -> np.ndarray:
    """Calculates image mean and creates canny image based on it

    Args:
        frame:
        sigma:
        upper_mod:
        lower_mod:

    Returns:

    """
    # find the median of the single channel pixel intensities
    v = np.median(frame)
    # apply automatic Canny edge detection using the median
    lower = int(max(0, (1.0 - sigma) * v) * lower_mod)
    upper = int(min(255, (1.0 + sigma) * v) * upper_mod)
    edged = cv.Canny(frame, lower, upper)
    return edged