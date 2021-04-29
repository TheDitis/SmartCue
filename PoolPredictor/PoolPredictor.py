import time

from PoolPredictor.Table import Table
import pyglview
import cProfile
import json
import cv2 as cv
from imutils.video import FPS


class PoolPredictor:
    def __init__(
        self,
        video_file: str = "clips/2019_PoolChamp_Clip3.mp4"
    ):
        self._cap = cv.VideoCapture(video_file)
        self._settings = self._load_settings()
        self._profiler = cProfile.Profile()
        self._delay = self._settings["program"]["frame_delay"]
        self._fps = FPS()
        self.table = Table(self._cap, self._settings)
        self._frame = None

    def run(self):
        self._fps.start()
        # try:
        if self._settings["program"]["OpenGL"]:
            self._run_opengl()
        else:
            self._run_without_opengl()
        # except (OpenGL.error.NullFunctionError, ModuleNotFoundError):
        #     self._run_no_opengl()

    def _run_opengl(self):
        def stop_loop():
            self.stop()

        def play_frame():
            ret, frame = self._cap.read()
            if ret:
                # self.table.draw_boundary_lines(frame, inplace=True)
                self.table.balls.find(frame)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                viewer.set_image(frame)
                self._fps.update()
                if self._delay:
                    time.sleep(self._delay / 1000)
            else:
                viewer.destructor_function()
                exit(9)

        viewer = pyglview.Viewer(
            window_width=2000, window_height=1000, fullscreen=False,
            opengl_direct=True
        )
        viewer.set_destructor(stop_loop)
        viewer.set_loop(play_frame)
        self._on_start()
        viewer.start()

    def _run_without_opengl(self):
        if self.table.ready:
            self._on_start()
            first = True
            while True:
                ret, frame = self._cap.read()
                if ret:
                    self._frame = frame
                    # if first:
                    #     print("running test loop")
                        # prof = cProfile.Profile()
                        # prof.enable()
                        # start = time.time()
                        # for _ in range(1000):
                            # self.table.boundaries.pocket  # .crop(frame)
                            # self.table.boundaries.pocket
                        # first = False
                        # prof.disable()
                        # prof.print_stats(sort="time")
                        # loop_time = time.time() - start
                        # print(f"LOOP TOOK {int(loop_time * 1000)}ms")
                    self.table.draw_boundary_lines(frame, inplace=True)
                    self.table.balls.find(frame)
                    cv.imshow('frame', frame)
                    self._fps.update()
                    if cv.waitKey(max(self._delay, 1)) & 0xFF == ord('q'):
                        break
                else:
                    break
            self.stop()
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
            "program": {
                "profile": True
            },
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

    def _on_start(self):
        if self._settings["program"]["profile"]:
            self._profiler.enable()

    def stop(self):
        self._cap.release()
        cv.destroyAllWindows()
        self._fps.stop()
        if self._settings["program"]["profile"]:
            self._profiler.disable()
            self._profiler.print_stats(sort="time")
        print(f"time elapsed: {self._fps.elapsed():.2f}")
        print(f"FPS: {self._fps.fps():.2f}")
