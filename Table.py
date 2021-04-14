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
        self._ref_frame = None
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

        for j in range(5):
            ret, frame = self._cap.read()
            if ret:
                # save the first frame as the reference frame
                if j == 0:
                    self._ref_frame = frame
                canny_setting = setting["canny"]
                canny = canny_image(frame, canny_setting)

                # save image to debug_images folder
                cv.imwrite("./debug_images/1_table_canny.png", canny)

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
        with_lines = draw_lines(self._ref_frame, self._found_lines)
        cv.imwrite("./debug_images/2_table_lines.png", with_lines)

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

