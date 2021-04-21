import numpy as np
import pandas as pd
import cv2 as cv
from TableBoundaries import TableBoundaries
from Pocket import Pocket
from Box import Box
from typing import Union


class PocketSet(pd.DataFrame):
    """
    Wraps pandas DataFrame. Calculates locations and sizes of table
    pockets, providing getters the wrap rows in Pocket class
    """
    def __init__(self):
        super().__init__(
            columns=['x', 'y', 'r', 'loc', 'h_loc', 'v_loc', 'num']
        )

    @property
    def ready(self):
        return len(self) == 6

    @property
    def pocket_nums(self):
        """
        Returns:
            Mapping of locations aliases ('tl', 'bm', etc.) to
            pocket numbers
        """
        return {'tl': 1, 'tm': 2, 'tr': 3, 'bl': 4, 'bm': 5, 'br': 6}

    @property
    def pocket_num_loc(self):
        """
        Returns:
            Mapping of pocket numbers (1-6) to location aliases ('tl',
            'bm', etc.)
        """
        return {v: k for k, v in self.pocket_nums.items()}

    def find(self, borders: TableBoundaries):
        """
        Calculates the location of the 6 pockets based on the border
        locations. Fills self (pd.DataFrame) with rows of pockets
        Args:
            borders: The TableBoundaries object that has already run
            its 'find' algorithm and has all 3 boundary boxes found

        Returns:
            None
        """
        bumper_box = borders.bumper.corners
        pocket_box = borders.pocket.corners
        table_box = borders.table.corners

        # calculate size of the pockets
        border_width = bumper_box.tr['y'] - table_box.tr['y']
        r = border_width * 0.4

        # calculate box halfway between the bumper and pocket boxes
        combined = pd.concat([bumper_box, pocket_box]) \
            .groupby("loc", as_index=False) \
            .agg({
                'x': np.mean, 'y': np.mean, 'loc': min,
                'h_loc': min, 'v_loc': min
            })
        combined.index = combined["loc"]
        combined = Box(combined)

        # find side pocket locations
        x_mid_t = (combined.tl['x'] + combined.tr['x']) / 2
        x_mid_b = (combined.bl['x'] + combined.br['x']) / 2
        y_mid_t = (pocket_box.tl['y'] + pocket_box.tr['y']) / 2
        y_mid_b = (pocket_box.bl['y'] + pocket_box.br['y']) / 2

        # tack on the pocket number row and radius
        combined["num"] = combined["loc"].apply(
            lambda x: self.pocket_nums[x]
        )
        combined["r"] = combined.apply(lambda _: r, axis=1)

        # add all the corner pockets
        for i, row in combined.iterrows():
            self.loc[row["loc"]] = Pocket(row)

        # add the top side pocket
        self.loc['tm'] = Pocket(pd.Series({
            'x': x_mid_t, 'y': y_mid_t, 'r': r, "loc": "tm",
            "h_loc": 'm', "v_loc": 't', "num": self.pocket_nums['tm']
        }))
        # # add the bottom side pocket
        self.loc['bm'] = Pocket(pd.Series({
            'x': x_mid_b, 'y': y_mid_b, 'r': r, "loc": "bm",
            "h_loc": 'm', "v_loc": 'b', "num": self.pocket_nums['tm']
        }))

    def __getitem__(self, item) -> Union[Pocket, pd.Series]:
        """
        Wraps __getitem__ of super by returning pocket instance of a
        row if the item is a row identifier

        Args:
            item: anything you can index a pd.DataFrame with

        Returns:
            A Pocket if a row identifier is passed, otherwise whatever
            the pd.DataFrame match would be
        """
        if item in self.pocket_nums.keys():
            return Pocket(self.loc[item])
        else:
            return super().__getitem__(item)

    def num(self, num: int) -> Union[Pocket, None]:
        """
        get pocket by pocket number. pocket numbers go from 1 through
        6 and are in the order of left-to-right, top then bottom
        (bottom left is 4, top right is 3, bottom middle is 5, etc.)
        Args:
            num: the number of the pocket you want

        Returns:
            a Pocket if the number is between 1 & 6, otherwise None
        """
        if num not in self.pocket_num_loc.keys():
            return None
        row = self.loc[self.pocket_num_loc[num]]
        return Pocket(row)

    def iter(self) -> Pocket:
        """
        Wraps iterrows from superclass, returning row as Pocket class
        Yields:
            Pocket instance
        """
        for _, row in super().iterrows():
            yield Pocket(row)

    def draw(
            self,
            frame: np.ndarray,
            inplace: bool = False,
            save: bool = False
    ) -> np.ndarray:
        """
        draws the pockets on the given frame
        Args:
            frame: the frame you want pocket circles drawn on
            inplace: modifies the passed frame if true
            save: saves image to debug_images folder if true

        Returns:
            A copy of the passed frame with pocket circles drawn on it
        """
        if not inplace:
            frame = frame.copy()
        for pocket in self.iter():
            cv.circle(frame, tuple(pocket.center), 5, (0, 0, 255), -1)
            cv.circle(
                frame,
                tuple(pocket.center),
                int(pocket.r),
                (0, 255, 0),
                2,
            )
        if save:
            cv.imwrite("./debug_images/7_pockets.png", frame)
        return frame
