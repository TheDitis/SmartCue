from Table import Table
import json
import cv2 as cv


class PoolPredictor:
    def __init__(self, video_file="clips/2019_PoolChamp_Clip4.mp4"):
        self._cap = cv.VideoCapture(video_file)
        self._settings = self._load_settings()
        self._table = Table(self._cap, self._settings)
        self._frame = None

    def run(self):
        while True:
            ret, frame = self._cap.read()
            if ret:
                self._frame = frame
                frame = self._table.draw_boundary_lines(frame)
                cv.imshow('frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self._cap.release()
        cv.destroyAllWindows()

    @staticmethod
    def _load_settings():
        """Loads the settings.json file.

        Returns:
            A dict containing settings
        """
        # defaults in case the settings.json file is missing
        data = {
            "table_detect_setting": 0,
            "table_detect_settings": [
                {
                    "canny": "auto",
                    "min_line_length": 100000,
                    "max_line_gap": 10,
                    "rho": 1
                }
            ],
            "table_detect_defaults": {
                "canny": {
                    "thresh_ratio": 3,
                    "low": 30
                },
                "min_line_length": 100000,
                "max_line_gap": 10,
                "rho": 1
            },
        }
        # attempt to load the settings file
        try:
            with open("settings.json", "r") as file:
                data = json.load(file)
        # if the settings file is not found, create it
        except FileNotFoundError:
            print("SETTINGS FILE NOT FOUND. CREATING...")
            with open("settings.json", "w") as file:
                json.dump(data, file, indent=4, sort_keys=True)
        return data





