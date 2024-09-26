import json
import os
from pathlib import Path

from src.config.config import Config


def transforms_for_stokes(config: Config):
    transforms_dir = config.options["paths"]["transforms_dir"]
    transforms_stokes_dir = config.options["paths"]["transforms_stokes_dir"]
    stokes_dir = config.options["paths"]["stokes_dir"]
    os.makedirs(transforms_stokes_dir, exist_ok=True)

    def remake_frame_path(frame):
        path = Path(os.path.normpath(frame["file_path"]))
        frame["file_path"] = stokes_dir + "/" + (".".join(path.name.split(".")[:-1])) + "_stokes.npy"
        return frame

    for transform_filename in os.listdir(transforms_dir):
        with open(transforms_dir + "/" + transform_filename, "r") as source, open(transforms_stokes_dir + "/" + transform_filename, "w") as destination:
            transforms = json.load(source)
            transforms["frames"] = [remake_frame_path(frame) for frame in transforms["frames"]]
            json.dump(transforms, destination, indent=4)