import cv2 as cv
import numpy as np
from scipy.stats import mode
from PoolPredictor.Boundaries.TableBoundaries import TableBoundaries
from PoolPredictor.Pockets.PocketSet import PocketSet
from PoolPredictor.Balls.BallSet import BallSet


class Table:
    """
    Represents the billiards table. Most of the functionality of the
    program is nested in this class. Contains the TableBoundaries,
    PocketSet, and BallSet classes, where most of the core computer
    vision algorithms live.
    """
    def __init__(self, capture: cv.CAP_V4L2, settings: dict):
        """
        Initializes the table by locating table boundaries and pocket
        locations.
        Args:
            capture: OpenCV capture object to read frames from
            settings: dictionary of settings loaded from settings.json
        """
        self._cap = capture
        self._settings = settings
        # get initial frame for reference
        _, self._ref_frame = capture.read()
        # Initialize and locate table boundaries
        self.boundaries = TableBoundaries(
            capture,
            settings["table_detection"]
        )
        self.boundaries.find()

        # detect cloth color
        self.color = self._detect_color()
        print("color: ", self.color)

        # Initialize and locate pockets
        self.pockets = PocketSet()
        self.pockets.find(self.boundaries)
        self.pockets.draw(self._ref_frame, save=True)

        self.balls = BallSet(
            self.boundaries,
            settings
        )
        
    @property
    def ready(self):
        return self.boundaries.ready and self.pockets.ready
    
    def _detect_color(self):
        if self.boundaries.ready:
            crop = self.boundaries.bumper.crop(self._ref_frame)
            flat = crop.reshape(crop.shape[0] * crop.shape[1], 3)
            mean = np.mean(flat, axis=0)
            print("mean: ", mean)
            modes, counts = mode(flat)
            return modes[0]


    def draw_boundary_lines(
            self,
            frame: np.ndarray,
            color: tuple = (0, 0, 255),
            thickness: int = 2,
            inplace: bool = False
    ) -> np.ndarray:
        """Draws found lines on given frame
        Args:
            frame: the frame you want to draw the lines on
            color: BGR formatted tuple
            thickness: line thickness
            inplace: original frame will be modified if true

        Returns:
            the given frame with the table boundaries found drawn on
        """
        return self.boundaries.draw_boundary_lines(
            frame,
            color,
            thickness,
            inplace
        )
    
    @property
    def SetupError(self):
        if not self.boundaries.ready:
            message = f"Boundaries were not detected successfully. " \
                      f"Check the debug_images folder and tweak " \
                      f"the settings.json file, or try a different " \
                      f"setting number."
        elif not self.pockets.ready:
            message = f"Pockets not located successfully. Check " \
                      f"the debug_images folder to make sure that " \
                      f"all boundaries were found successfully. " \
                      f"tweak the settings.json file or try a " \
                      f"different setting number"
        else:
            message = f"An unknown error occurred during setup. " \
                      f"Check debug_images folder for clues."
        return TableSetupError(message)


class TableSetupError(Exception):
    pass
