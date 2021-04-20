import numpy as np
import pandas as pd
import cv2 as cv
from TableBoundaries import TableBoundaries, Box

from typing import Union


class Pocket(pd.Series):
    def __init__(
        self,
        row: Union[list, pd.Series]
    ):
        super().__init__(row)

    @property
    def radius(self):
        return self['r'].astype(int)

    @property
    def r(self):
        return self.radius

    @property
    def center(self):
        return self[['x', 'y']].astype(int)

    @property
    def num(self):
        return self['num']


class PocketSet(pd.DataFrame):
    def __init__(self):
        super().__init__(
            columns=['x', 'y', 'r', 'loc', 'h_loc', 'v_loc', 'num']
        )

    @property
    def pocket_nums(self):
        return {'tl': 1, 'tm': 2, 'tr': 3, 'bl': 4, 'bm': 5, 'br': 6}

    def find(self, borders: TableBoundaries):
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
            print(row)
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

    def __getitem__(self, item):
        if item in self.pocket_nums.keys():
            return Pocket(super().__getitem__(item))
        else:
            return super().__getitem__(item)

    def iterrows(self) -> Pocket:
        for _, row in super().iterrows():
            yield Pocket(row)

    def draw(
            self,
            frame: np.ndarray
    ) -> np.ndarray:
        copy = frame.copy()
        print("length of self: ", len(self))
        for pocket in self.iterrows():
            print("POCKET")
            print(pocket)
            print("pocket.r: ", pocket.r)
            print("pocket.center: ", pocket.center)
            cv.circle(copy, tuple(pocket.center), int(pocket.r), (0, 255, 0), 2)
        cv.imwrite("./debug_images/7_pockets.png", copy)
        return copy
