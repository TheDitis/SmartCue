

class Settings:
    def __init__(self):
        pass

    def load(self):
        pass

    def save(
            self,
            category: str,
            parameters: dict
    ):
        pass

    def save_defaults(self):
        defaults = {
            "table_detection": {
                "table_detect_setting": 0,
                "table_detect_settings": [
                    {
                        "canny": {
                            "mode": "auto",
                            "sigma": 0.5,
                            "upper_mod": 6,
                            "lower_mod": 0,
                            "aperture_size": 5
                        },
                        "min_line_length": 70,
                        "max_line_gap": 10,
                        "rho": 1,
                        "thresh": 100
                    },
                    {
                        "canny": {
                            "mode": "auto",
                            "sigma": 1.5,
                            "upper_mod": 0.3,
                            "lower_mod": 0
                        },
                        "min_line_length": 80,
                        "max_line_gap": 10,
                        "rho": 1,
                        "thresh": 100
                    },
                    {
                        "canny": {
                            "mode": "auto",
                            "sigma": 1.1,
                            "upper_mod": 0.33,
                            "lower_mod": 0
                        },
                        "min_line_length": 80,
                        "max_line_gap": 10,
                        "rho": 1,
                        "thresh": 100
                    },
                    {
                        "canny": {
                            "mode": "auto",
                            "sigma": 0.5,
                            "upper_mod": 0.4,
                            "lower_mod": 0.3
                        },
                        "min_line_length": 100,
                        "max_line_gap": 20,
                        "rho": 0.6
                    },
                    {
                        "canny": {
                            "mode": "manual",
                            "thresh_ratio": 3,
                            "low": 35
                        }
                    },
                    {
                        "canny": {
                            "mode": "manual",
                            "thresh_ratio": 3,
                            "low": 30
                        },
                        "min_line_length": 5,
                        "max_line_gap": 1,
                        "rho": 1.5
                    }
                ],
                "table_detect_defaults": {
                    "canny": {
                        "thresh_ratio": 3,
                        "low": 30
                    },
                    "min_line_length": 100,
                    "max_line_gap": 10,
                    "rho": 1,
                    "thresh": 200
                }
            }

        }