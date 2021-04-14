import cv2 as cv
import numpy as np
from typing import Union
from utils import draw_lines, canny_image
from copy import deepcopy


class Table:
    def __init__(self, capture: cv.CAP_V4L2, settings: dict):
        self._cap = capture
        self._settings = settings
        self._found_lines = []
        self._boundaries = None
        self._pockets = None
        self._find_table_boundaries()

    def _find_table_boundaries(self):
        # Get settings relevant to table line detection
        try:
            setting_num = self._settings["table_detect_setting"]
            setting = self._settings[
                "table_detect_settings"
            ][setting_num]
            hough_settings = self._get_hough_lines_settings()
            min_line_length = hough_settings["min_line_length"]
            max_line_gap = hough_settings["max_line_gap"]
            rho = hough_settings["rho"]
        except IndexError:
            raise IndexError(
                "table_detect_setting index out of bounds. check "
                "settings file"
            )
        except KeyError:
            raise KeyError(
                "Malformed setting. Every setting must include: "
                "\nmin_line_length: int"
                "\nmax_line_gap: int"
                "\nrho: int or float"
            )

        for i in range(10):
            ret, frame = self._cap.read()
            if ret:
                canny_setting = setting["canny"]

                canny = canny_image(frame, canny_setting)

                cv.imshow("canny", canny)

                # save image to debug_images folder
                cv.imwrite(
                    "./debug_images/1_table_canny.png",
                    canny
                )

                lines = cv.HoughLinesP(
                    canny,
                    rho,
                    np.pi / 180,
                    300,
                    minLineLength=min_line_length,
                    maxLineGap=max_line_gap
                )
                if lines is not None:
                    for i in range(len(lines)):
                        self._found_lines.append(lines[i][0])

                print("lines found: ", len(self._found_lines))

    def _canny_image(self, frame: np.ndarray, canny_setting: dict):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        try:
            # Run canny edge-detection based on setting
            if canny_setting["mode"] == "auto":
                kwargs = {
                    k: v for k, v in canny_setting.items()
                    if k != "mode" and v is not None
                }
                canny = self._auto_canny(gray, **kwargs)
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


    @staticmethod
    def _auto_canny(
            frame: np.ndarray,
            sigma: float = 0.33,
            upper_mod: float = 1.0,
            lower_mod: float = 1.0
    ):
        """
        Returns a canny image
        :param frame:
        :param sigma:
        :param upper_mod:
        :param lower_mod:
        :return:
        """
        # find the median of the single channel pixel intensities
        v = np.median(frame)
        # apply automatic Canny edge detection using the median
        lower = int(max(0, (1.0 - sigma) * v) * lower_mod)
        upper = int(min(255, (1.0 + sigma) * v) * upper_mod)
        edged = cv.Canny(frame, lower, upper)
        return edged

    def _get_hough_lines_settings(self) -> dict:
        setting_num = self._settings["table_detect_setting"]
        setting = self._settings["table_detect_settings"][setting_num]
        defaults = self._settings["table_detect_defaults"]
        output = deepcopy(defaults)
        if "canny" in output:
            del output["canny"]
        for (key, val) in setting.items():
            if key != "canny":
                output[key] = val
        return output

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
        return draw_lines(
            frame,
            self._found_lines,
            color,
            thickness
        )

