import math

import cv2 as cv
import numpy as np
from typing import Union
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
        # and in the middle (ie. the kitchen line)
        df_h, df_v = self._filter_found_lines(df_h, df_v)

        horizontal = df_h.drop("slope", axis=1)
        vertical = df_v.drop("slope", axis=1)

        # cluster lines into bumper edge, bumper back, & table edge
        h_clusters = DBSCAN(eps=7, min_samples=3).fit(
            horizontal[["y1", "y2"]]
        )
        v_clusters = DBSCAN(eps=7, min_samples=3).fit(
            vertical[["x1", "x2"]]
        )

        df_h["group"] = h_clusters.labels_  # .astype(str)
        df_v["group"] = v_clusters.labels_  # .astype(str)

        # filter out lines that were not clustered
        df_h = df_h[df_h["group"] >= 0]
        df_v = df_v[df_v["group"] >= 0]

        self._split_clustered_lines_by_quadrant(df_h, df_v)

        self._save_found_line_cluster_debug_images(df_h, df_v)

        self._found_lines = np.array(
            np.concatenate([np.array(horizontal), np.array(vertical)])
        ).astype(int)

    def _filter_found_lines(
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
        h, w, *_ = self._ref_frame.shape
        # filter out the lines within 'thresh' px of the borders
        hor_filtered = hor.loc[
            (hor["y1"] > thresh) | (abs(h - hor["y1"]) < thresh)
        ]
        vert_filtered = vert.loc[
            (vert["x1"] > thresh) | (abs(w - vert["x1"]) < thresh)
        ]

        # filter out any lines in the middle of the table
        hor_filtered = hor_filtered.loc[
            (hor_filtered["y1"] < h / 4) |
            (hor_filtered["y1"] > h - (h / 4))
        ]
        vert_filtered = vert_filtered.loc[
            (vert_filtered["x1"] < w / 4) |
            (vert_filtered["x1"] > w - (w / 4))
        ]

        return hor_filtered, vert_filtered

    def _split_clustered_lines_by_quadrant(
            self,
            hor: pd.DataFrame,
            vert: pd.DataFrame
    ):
        h, w, *_ = self._ref_frame.shape

        hor["side"] = hor.apply(
            lambda l: "top" if l["y1"] < h / 2 else "bottom",
            axis=1
        )
        vert["side"] = vert.apply(
            lambda l: "left" if l["x1"] < h / 2 else "right",
            axis=1
        )

        top = hor[hor["side"] == "top"]
        bottom = hor[hor["side"] == "bottom"]
        left = vert[vert["side"] == "left"]
        right = vert[vert["side"] == "right"]

        h_centers = hor.groupby("group")["y1"].mean()
        v_centers = vert.groupby("group")["x1"].mean()

        t_centers = top.groupby("group")["y1"].mean().to_frame()
        b_centers = bottom.groupby("group")["y1"].mean().to_frame()
        l_centers = left.groupby("group")["x1"].mean().to_frame()
        r_centers = right.groupby("group")["x1"].mean().to_frame()

        # print(t_centers)

        t_dists = self._get_relative_distances(t_centers)
        b_dists = self._get_relative_distances(b_centers)
        l_dists = self._get_relative_distances(l_centers)
        r_dists = self._get_relative_distances(r_centers)

        merged = t_dists\
            .merge(b_dists, how="cross", suffixes=("_t", "_b"))\
            .merge(l_dists, how="cross")\
            .merge(r_dists, how="cross", suffixes=("_l", "_r"))

        value_cols = merged.columns[
            merged.columns.str.startswith("value")
        ]
        values = merged[value_cols].apply(list, axis=1)
        merged["values"] = values
        merged.drop(value_cols, axis=1)
        merged["mean"] = merged["values"].map(np.mean)

        def mse(x, y):
            mean = np.mean((x, y))
            return abs(x / mean - y / mean) ** 2

        def calc_err(row: pd.Series) -> int:
            return sum(map(
                lambda v: mse(v, row["mean"]),
                row["values"])
            )

        merged["err"] = merged.apply(calc_err, axis=1)
        merged.sort_values("err", inplace=True)

        top_3 = merged[:3]
        print(top_3)

        group_cols = top_3.columns[
            top_3.columns.str.startswith("group")
        ]
        groups = top_3[group_cols]

        sns.heatmap(t_dists, cmap=sns.cm.rocket_r)
        plt.show()

    def _get_relative_distances(self, df: pd.DataFrame):
        dists = pd.DataFrame(
            pairwise_distances(df),
            index=df.index,
            columns=df.index
        )
        melt = dists.melt(ignore_index=False)
        melt["value"] = melt["value"].round(2)
        melt = melt[
            (~melt["value"].duplicated()) & (melt["value"] != 0)
        ]
        melt.rename(columns={"group": "group2"}, inplace=True)
        return melt.reset_index()

    def _save_found_line_cluster_debug_images(
            self,
            hor: pd.DataFrame,
            vert: pd.DataFrame,
    ):
        """
        Saves 4 images, 2 for each line orientation. One of the lines
        drawn on the reference frame colored by cluster and one of the
        hist plot representing those clusters
        Args:
            hor: dataframe of clustered horizontal lines
            vert: dataframe of clustered vertical lines

        Returns:
            None
        """
        # save clustered-line images to debug folder
        cv.imwrite(
            "./debug_images/3_horizontal_groups.png",
            draw_lines_by_group(self._ref_frame, hor)
        )
        cv.imwrite(
            "./debug_images/3_vertical_groups.png",
            draw_lines_by_group(self._ref_frame, vert)
        )

        # change type of 'group' to string to improve plot colors
        df_h = hor.copy()
        df_h["group"] = df_h["group"].astype(str)
        df_v = vert.copy()
        df_v["group"] = df_v["group"].astype(str)

        # plotting of the groups
        h_plt = sns.histplot(bins=100, data=df_h, x="y1", hue="group")
        h_plt.get_figure().savefig(
            "./debug_images/4_horizontal_clusters_plot.png"
        )
        v_plt = sns.histplot(bins=100, data=df_v, x="x1", hue="group")
        v_plt.get_figure().savefig(
            "./debug_images/4_vertical_clusters_plot.png"
        )

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

