from PoolPredictor.Table import Table
import pyglview
import cProfile
import OpenGL
import json
import cv2 as cv


class PoolPredictor:
    def __init__(self, video_file="clips/2019_PoolChamp_Clip4.mp4"):
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        self._cap = cv.VideoCapture(video_file)
        self._settings = self._load_settings()
        self.table = Table(self._cap, self._settings)
        self._frame = None

    def run(self):
        # try:
        # self._run_opengl()
        self._run_no_opengl()
        # except (OpenGL.error.NullFunctionError, ModuleNotFoundError):
        #     self._run_no_opengl()

    def _run_opengl(self):
        def stop_loop():
            self._profiler.disable()
            self._profiler.print_stats(sort="time")
            self._cap.release()
            cv.destroyAllWindows()

        def play_frame():
            ret, frame = self._cap.read()
            if ret:
                self.table.draw_boundary_lines(frame, inplace=True)
                self.table.balls.find(frame)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                viewer.set_image(frame)
            else:
                viewer.destructor_function()
                exit(9)

        viewer = pyglview.Viewer(
            window_width=2000, window_height=1000, fullscreen=False,
            opengl_direct=True
        )
        viewer.set_destructor(stop_loop)
        viewer.set_loop(play_frame)
        viewer.start()

    def _run_no_opengl(self):
        if self.table.ready:
            while True:
                ret, frame = self._cap.read()
                if ret:
                    self._frame = frame
                    self.table.draw_boundary_lines(frame, inplace=True)
                    self.table.balls.find(frame)
                    cv.imshow('frame', frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            self._cap.release()
            cv.destroyAllWindows()
            self._profiler.disable()
            self._profiler.print_stats(sort="time")
        else:
            raise self.table.SetupError

    @staticmethod
    def _load_settings():
        """Loads the settings.json file.

        Returns:
            A dict containing settings
        """
        # defaults in case the settings.json file is missing
        data = {
            "table_detection": {
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