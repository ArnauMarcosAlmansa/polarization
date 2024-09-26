import json
import os
from pathlib import Path

import numpy as np

from src.config.config import Config
from src.data_generation.StokesImageGenerator import StokesImageGenerator
from src.load_polarimetric import PolarimetricImage

def generate_stokes_images(config: Config):
    transforms_dir = config.options["paths"]["transforms_dir"]
    stokes_dir = config.options["paths"]["stokes_dir"]

    downscale = config.options["tasks"]["generate_rays"]["downscale"]

    transforms_files = os.listdir(transforms_dir)
    os.makedirs(stokes_dir, exist_ok=True)

    def make_stokes_filename(image_filename):
        only_filename = Path(image_filename).name
        clean_filename = only_filename.split(".")[0]
        return clean_filename + "_stokes.npy"

    for filename in transforms_files:
        path = os.path.join(transforms_dir, filename)
        with open(path, "r") as f:
            transforms = json.load(f)
            stokes_generator = StokesImageGenerator()
            for frame in transforms["frames"]:
                pose = np.array(frame["transform_matrix"])
                image = PolarimetricImage.load(os.path.normpath(os.path.join(transforms_dir, frame["file_path"])), downscale=downscale)
                stokes_image = stokes_generator.generate_stokes_image(pose.astype(np.float32), image)
                np.save(f"{stokes_dir}/{make_stokes_filename(frame["file_path"])}", stokes_image)
