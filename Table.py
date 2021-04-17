import math

import cv2 as cv
import numpy as np
from typing import Union
from utils import (
    draw_lines,
    canny_image,
    rho_theta_to_xy_lines,
    get_slope
)
from copy import deepcopy
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple

class Table:
    def __init__(self, capture: cv.CAP_V4L2, settings: dict):
        self._cap = capture
        self._settings = settings
        self._found_lines = np.array([])
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
            thresh = hough_settings["thresh"]
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

        # find lines in a couple different frames
        for j in range(2):
            ret, frame = self._cap.read()
            if ret:
                # save the first frame as the reference frame
                if self._ref_frame is None:
                    self._ref_frame = frame
                canny_setting = setting["canny"]

                canny = canny_image(frame, canny_setting)

                # save image to debug_images folder
                cv.imwrite("./debug_images/1_table_canny.png", canny)

                lines = cv.HoughLinesP(
                    canny,
                    rho,
                    np.pi / 180,
                    thresh,
                    minLineLength=min_line_length,
                    maxLineGap=max_line_gap
                )
                lines = lines.reshape((lines.shape[0], 4))

                # TODO: try non-probabilistic mode again later
                # lines = cv.HoughLines(
                #     canny,
                #     rho,  # replace with rho later
                #     np.pi / 180,
                #     200
                # )
                # # reformat the array shape if needed
                # if len(lines.shape) > 2:
                #     lines = lines.reshape((lines.shape[0], 2))
                #
                # vertical = lines[lines[:, 1] % 1 == 0]
                # horizontal = lines[lines[:, 1] - 1 > 0.5]
                #
                # lines = rho_theta_to_xy_lines(horizontal)

                # add the lines found this frame to the self._found_lines
                if len(self._found_lines.shape) == 1:
                    self._found_lines = lines
                else:
                    self._found_lines = np.concatenate((self._found_lines, lines), axis=0)

        self._group_found_lines()
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

    def _group_found_lines(self):
        # TODO: rotate some of the videos and adapt this to work with
        df = pd.DataFrame(self._found_lines, columns=["x1", "y1", "x2", "y2"])
        df["slope"] = df.apply(get_slope, axis=1)
        # get horizontal and vertical lines
        df_h = df.loc[df["slope"].abs() < 0.02]
        df_v = df.loc[(df["slope"].isna()) | (df["slope"].abs() > 300)]

        # get rid of lines that sometimes show up around the borders
        df_h, df_v = self._filter_edge_lines(df_h, df_v)

        # get horizontal and vertical into np arrays without slope
        # horizontal = df_h.values[:, :4]
        # vertical = df_v.values[:, :4]

        horizontal = df_h.drop("slope", axis=1)
        vertical = df_v.drop("slope", axis=1)

        # cluster lines into bumper edge, bumper back, & table edge
        h_clusters = KMeans(n_clusters=6).fit(
            horizontal[["x1", "x2"]]
        )
        v_clusters = KMeans(n_clusters=6).fit(
            vertical[["y1", "y2"]]
        )

        df_h["group"] = h_clusters.labels_
        df_v["group"] = v_clusters.labels_

        print(df_v.head())

        self._found_lines = np.array(
            np.concatenate([np.array(horizontal), np.array(vertical)])
        ).astype(int)

    def _filter_edge_lines(
            self,
            hor: pd.DataFrame,
            vert: pd.DataFrame,
            thresh: int = 20
    ) -> Tuple[np.array, np.array]:
        """Removes lines that sometimes show on the edges of the frame

        Args:
            hor: dataframe of horizontal lines
            vert: dataframe of vertical lines
            thresh: removes lines 'thresh' pixels from the frame edges

        Returns:
            filtered versions of the passed dataframes
        """
        h = self._ref_frame.shape[0]
        w = self._ref_frame.shape[1]
        hor_filtered = hor.loc[
            (hor["y1"] > thresh) | (abs(h - hor["y1"]) < thresh)
        ]
        vert_filtered = vert.loc[
            (vert["x1"] > thresh) | (abs(w - vert["x1"]) < thresh)
        ]
        return hor_filtered, vert_filtered

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

