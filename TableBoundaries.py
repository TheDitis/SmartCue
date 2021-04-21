import cv2 as cv
import numpy as np
from copy import deepcopy
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from typing import Union, Tuple, Dict, List
from utils import (
    canny_image,
    draw_lines,
    draw_lines_by_group,
    get_slope,
    find_intersection,
)
from BoundaryGroup import BoundaryGroup


class TableBoundaries:
    """
    Represents the boundaries of the table. Finds the table boundaries
    with the find method, and provides access to them through getter
    properties like boundaries.top.bumper.
    """
    def __init__(self, cap: cv.CAP_V4L2, settings: dict):
        self._found_lines = np.array([])
        self._ref_frame = None
        self._settings = settings
        self._cap = cap
        self._boundaries = None

    @property
    def ready(self):
        return len(self) == 12

    @property
    def top(self):
        return self._boundaries.top

    @property
    def bottom(self):
        return self._boundaries.bottom

    @property
    def left(self):
        return self._boundaries.left

    @property
    def right(self):
        return self._boundaries.right

    @property
    def t(self):
        return self.top

    @property
    def b(self):
        return self.bottom

    @property
    def l(self):
        return self.left

    @property
    def r(self):
        return self.right

    @property
    def bumper(self):
        return self._boundaries.bumper

    @property
    def pocket(self):
        return self._boundaries.pocket

    @property
    def table(self):
        return self._boundaries.table

    def __repr__(self):
        return self._boundaries

    def __getitem__(self, item):
        return self._boundaries[item]

    def __len__(self):
        return len(self._boundaries)

    def find(self):
        """
        Iterates of a few frames of the capture and identifies the
        boundary lines of the table from those frames

        Returns:
            None
        """
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

                # add the lines found this frame to the self._found_lines
                if len(self._found_lines.shape) == 1:
                    self._found_lines = lines
                else:
                    self._found_lines = np.concatenate(
                        (self._found_lines, lines), axis=0)

        self._boundaries = self._group_found_lines()

        print("lines found: ", len(self._found_lines))
        with_lines = draw_lines(self._ref_frame, self._found_lines)
        cv.imwrite("./debug_images/2_table_lines.png", with_lines)

    def _get_hough_lines_settings(self) -> dict:
        """
        Loads the settings relevant to hough-line-detection
        Returns:
            dictionary of parameters
        """
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

    def _group_found_lines(self) -> BoundaryGroup:
        """
        groups the preliminary found-lines into clusters based on
        their relative distances, identifies the clusters that have
        similar relative distances on each side, and averages those
        clusters into individual boundary lines.
        Returns:
            BoundaryGroup representing the identified set of table
            boundaries
        """
        # TODO: rotate some of the videos and adapt this to work with
        df = pd.DataFrame(
            self._found_lines,
            columns=["x1", "y1", "x2", "y2"]
        )
        df["slope"] = df.apply(get_slope, axis=1)
        # get horizontal and vertical lines
        df_h = df.loc[df["slope"].abs() < 0.02]
        df_v = df.loc[
            (df["slope"].isna()) | (df["slope"].abs() > 300)
        ]

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

        df_h["group"] = h_clusters.labels_
        df_v["group"] = v_clusters.labels_

        self._save_found_line_cluster_debug_images(df_h, df_v)

        # filter out lines that were not clustered
        df_h = df_h[df_h["group"] >= 0]
        df_v = df_v[df_v["group"] >= 0]

        # get cluster numbers we want to keep for each side
        cluster_sides = self._find_cluster_numbers_by_side(df_h, df_v)

        # reassign found_lines with only horizontal and vertical lines
        self._found_lines = np.array(
            np.concatenate([np.array(horizontal), np.array(vertical)])
        ).astype(int)

        return BoundaryGroup(
            self._side_line_clusters_to_boundaries(
                df_h,
                df_v,
                cluster_sides
            )
        )

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

    def _find_cluster_numbers_by_side(
            self,
            hor: pd.DataFrame,
            vert: pd.DataFrame
    ) -> Dict[str, List[int]]:
        """
        Calculates the distances between each cluster on each side and
        finds the clusters that the distances between which are most
        consistent between all sides, indicating that they are the
        desired boundaries of the table
        Args:
            hor: dataframe of pre-clustered horizontal lines
            vert: dataframe of pre-clustered vertical lines

        Returns:
            dictionary with side indicators ('t' = top, 'l' = left) as
            keys and arrays of 3 cluster numbers as values
        """
        h, w, *_ = self._ref_frame.shape

        # assign side based on position relative to center of frame
        hor["side"] = hor.apply(
            lambda l: 't' if l["y1"] < h / 2 else 'b',
            axis=1
        )
        vert["side"] = vert.apply(
            lambda l: 'l' if l["x1"] < h / 2 else 'r',
            axis=1
        )

        # get separate df for each side
        top = hor[hor["side"] == 't']
        bottom = hor[hor["side"] == 'b']
        left = vert[vert["side"] == 'l']
        right = vert[vert["side"] == 'r']

        # compute the mean on the relevant dimension for each cluster
        t_centers = top.groupby("group")["y1"].mean().to_frame()
        b_centers = bottom.groupby("group")["y1"].mean().to_frame()
        l_centers = left.groupby("group")["x1"].mean().to_frame()
        r_centers = right.groupby("group")["x1"].mean().to_frame()

        # get distances between each pair of clusters for each side
        t_dists = self._get_relative_distances(t_centers)
        b_dists = self._get_relative_distances(b_centers)
        l_dists = self._get_relative_distances(l_centers)
        r_dists = self._get_relative_distances(r_centers)

        # merge all of the relative distance dfs into one big df
        merged = t_dists \
            .merge(b_dists, how="cross", suffixes=("_t", "_b")) \
            .merge(l_dists, how="cross") \
            .merge(r_dists, how="cross", suffixes=("_l", "_r"))

        # get the names of all of the value columns (cluster means)
        value_cols = merged.columns[
            merged.columns.str.startswith("value")
        ]

        # add column with cluster distances of each side as list
        values = merged[value_cols].apply(list, axis=1)
        merged["values"] = values
        # get rid of the individual value columns
        merged.drop(value_cols, axis=1)
        # add column for the mean of the distances
        merged["mean"] = merged["values"].map(np.mean)

        def mse(x, y):
            # mean squared error function that corrects for position
            mean = np.mean((x, y))
            return abs((x / mean) - (y / mean)) ** 2

        def calc_err(row: pd.Series) -> int:
            # calculates the similarity of each distance value to the mean
            return sum(map(
                lambda v: mse(v, row["mean"]),
                row["values"])
            )

        # add the error as a column and sort by it
        merged["err"] = merged.apply(calc_err, axis=1)
        merged.sort_values("err", inplace=True)

        # 3 most consistent cluster-distance
        top_3 = merged[:3]
        group_cols = top_3.columns[
            top_3.columns.str.startswith("group")
        ]
        groups = top_3[group_cols]

        # get the cluster numbers we want to keep for each side
        top_clusters = pd.unique(
            groups[["group_t", "group2_t"]].values.ravel("K")
        )
        # get the cluster numbers we want to keep for each side
        bottom_clusters = pd.unique(
            groups[["group_b", "group2_b"]].values.ravel("K")
        )
        # get the cluster numbers we want to keep for each side
        left_clusters = pd.unique(
            groups[["group_l", "group2_l"]].values.ravel("K")
        )
        # get the cluster numbers we want to keep for each side
        right_clusters = pd.unique(
            groups[["group_r", "group2_r"]].values.ravel("K")
        )
        clusters_by_side = {
            "t": list(top_clusters),
            "b": list(bottom_clusters),
            "l": list(left_clusters),
            "r": list(right_clusters)
        }
        return clusters_by_side

    def _side_line_clusters_to_boundaries(
            self,
            hor: pd.DataFrame,
            vert: pd.DataFrame,
            cluster_sides: Dict[str, List[int]]
    ) -> pd.DataFrame:
        """
        finds the final table boundaries from the clustered horizontal
        and vertical groups of lines using dictionary of clusters per
        side
        Args:
            hor: dataframe of clustered horizontal lines
            vert: dataframe of clustered vertical lines
            cluster_sides: dict with direction letter ('t', 'l', etc.)
            as keys and a list of 3 cluster numbers as values, which
            are the desired cluster numbers for that side

        Returns:
            pandas DataFrame of all 12 table boundaries where lines
            have been averaged from cluster lines and their ends
            have been extended to their intersection points with their
            neighboring boundaries of their type
        """
        # filter out any lines that aren't in the desired clusters
        hor = hor.loc[
            hor["group"].isin(
                cluster_sides['t'] + cluster_sides['b']
            )
        ]
        vert = vert.loc[
            vert["group"].isin(
                cluster_sides['l'] + cluster_sides['r']
            )
        ]

        combined = []
        # for each side of the table and cluster numbers for that side
        for k, group_nums in cluster_sides.items():
            labels = ["table", "pocket", "bumper"]
            labels = labels if k in ['t', 'l'] else labels[::-1]

            # for top or bottom (horizontal lines)
            if k in ['t', 'b']:
                df = hor
                orientation = 'h'
                # coords 'parallel' to line orientation
                par_coords = ['y1', 'y2']
                # coords 'perpendicular' to line orientation
                perp_coords = ['x1', 'x2']
            # for left or right (vertical lines)
            else:
                df = vert
                orientation = 'v'
                par_coords = ['x1', 'x2']
                perp_coords = ['y2', 'y1']  # reversed for vertical

            # filter out irrelevant clusters and group by cluster
            groups = df[
                df["group"].isin(group_nums)
            ].groupby("group")
            # get means for the parallel plane (y values for horizontal)
            means = groups[par_coords].mean().round().astype(int)
            # get min and max for the perpendicular plane (x for horizontal)
            pt1 = groups[perp_coords[0]].min()
            pt2 = groups[perp_coords[1]].max()
            # combine these values into one df (3 lines, one for each cluster)
            lines = pd.concat([means, pt1, pt2], axis=1)
            # add types of each boundary ('bumper', 'table', 'pocket')
            lines.sort_values(par_coords[0], inplace=True)
            lines["type"] = lines.reset_index().index.map(
                lambda x: labels[x]
            )

            # add side and orientation
            lines["side"] = k
            lines["orientation"] = orientation
            combined.append(lines)

        # combine list of dataframes into one
        combined = pd.concat(combined)

        # move group into its own column
        combined.reset_index(inplace=True)
        combined.loc[combined["orientation"] == "v", 'group'] += 6

        # reorder columns
        combined = combined[
            [
                'x1', 'y1', 'x2', 'y2', 'orientation', 'side',
                'type', "group"
            ]
        ]

        # sets the corners of each point to their intersection with neighbors
        def join_points(row: pd.Series) -> pd.Series:
            # coordinate that we want to adjust based on
            coord = 'x' if row.orientation == 'h' else 'y'
            # get the adjacent boundaries of the same type
            adjacent = combined[
                (combined.type == row.type) &
                (combined.orientation != row.orientation)
            ].sort_values('y1' if coord == 'x' else 'x1')
            # adjust points to intersection point with each neighbor
            row[f'{coord}1'] = find_intersection(
                row["x1":"y2"],
                adjacent.iloc[0]["x1":"y2"]
            )[0 if coord == 'x' else 1]
            row[f'{coord}2'] = find_intersection(
                row["x1":"y2"],
                adjacent.iloc[1]["x1":"y2"],
            )[0 if coord == 'x' else 1]
            return row

        # get df where point ends are
        adjusted = combined.apply(join_points, axis=1)

        # save debug images
        self._save_averaged_cluster_debug_images(combined, adjusted)

        return adjusted

    @staticmethod
    def _get_relative_distances(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a dataframe of centers of each cluster into a
        dataframe of the distances between each pair of clusters
        Args:
            df: A dataframe with cluster (or 'group') number as
            indices and cluster centers as the only column

        Returns:
            A dataframe with columns group, group2, and value, where
            group and group 2 are cluster numbers and value is the
            distance between the centers of those clusters.
        """
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

    def _save_averaged_cluster_debug_images(
            self,
            averaged: pd.DataFrame,
            joined_corners: pd.DataFrame
    ):
        """
        Saves debug images for the averaged line clusters and the
        intersection points for the different boundary types

        Args:
            averaged: dataframe of lines that have been averaged from
                each cluster
            joined_corners: dataframe same as averaged, but where the
                endpoints of each line has been moved to where its
                intersection point with its neighbors would be

        Returns:
            None
        """
        # Saving the image for cluster average lines
        averaged = draw_lines_by_group(
            self._ref_frame,
            averaged
        )
        cv.imwrite("./debug_images/5_averaged_clusters.png", averaged)

        # Saving the image for joined-corner boundaries
        # make a copy where the group number is type for line image
        joined_groupby_type = joined_corners.copy()
        types = ["table", "pocket", "bumper"]
        joined_groupby_type["group"] = joined_corners.apply(
            lambda x: types.index(x["type"]), axis=1
        )
        borders = draw_lines_by_group(
            self._ref_frame,
            joined_groupby_type
        )
        cv.imwrite("./debug_images/6_boundaries_joined.png", borders)

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
            self._boundaries,
            color,
            thickness
        )
