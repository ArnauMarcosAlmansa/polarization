import json
import os
from pathlib import Path

import numpy as np

from src.config.config import Config
from src.data_generation.StokesImageGenerator import StokesImageGenerator
from src.load_polarimetric import PolarimetricImage
from statistics import mean, stdev, quantiles

class StokesErrorsCounter:
    def __init__(self):
        self.error_count = 0
        self.ok_count = 0
        self.errors = []

    def detect_errors(self, stokes_image: np.ndarray) -> None:
        error_mask = stokes_image[:, :, 0] > 1
        self.error_count += error_mask.sum()
        self.ok_count += (~error_mask).sum()
        self.errors += stokes_image[error_mask, 0].flatten().tolist()

    def print_stats(self) -> None:
        total = self.error_count + self.ok_count
        error_mean = mean(self.errors)
        error_stdev = stdev(self.errors)
        error_quantiles = quantiles(self.errors)
        error_max = max(self.errors)
        error_min = min(self.errors)
        print("StokesErrorsCounter STATS:")
        print(f"TOTAL PIXELS = {total}")
        print(f"ERROR PIXELS = {self.error_count} ({self.error_count / total * 100:.05f} %)")
        print(f"OK PIXELS = {self.ok_count} ({self.ok_count / total * 100:.05f} %)")
        print(f"ERROR DISTRIBUTION:\n\tmean = {error_mean:.05f}\n\tstdev = {error_stdev:.05f}\n\terror_quantiles = {error_quantiles}\n\tmax = {error_max:.05f}\n\tmin = {error_min:.05f}")


class IntensitySaturationCounter:
    def __init__(self):
        self.saturated_count = 0
        self.max_intensity = 0

    def detect(self, image: np.ndarray) -> None:
        self.saturated_count += (image > 0.98).sum()
        self.max_intensity = max(self.max_intensity, image.max())

    def print_stats(self) -> None:
        print("IntensitySaturationCounter STATS")
        print(f"SATURATED PIXELS = {self.saturated_count}")
        print(f"MAX INTENSITY = {self.max_intensity}")


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

    error_counter = StokesErrorsCounter()
    saturation_counter = IntensitySaturationCounter()

    for filename in transforms_files:
        path = os.path.join(transforms_dir, filename)
        with open(path, "r") as f:
            transforms = json.load(f)
            stokes_generator = StokesImageGenerator()
            for frame in transforms["frames"]:
                pose = np.array(frame["transform_matrix"])
                image = PolarimetricImage.load(os.path.normpath(os.path.join(transforms_dir, frame["file_path"])), downscale=downscale)
                saturation_counter.detect(image.I0)
                saturation_counter.detect(image.I45)
                saturation_counter.detect(image.I90)
                saturation_counter.detect(image.I135)
                stokes_image = stokes_generator.generate_stokes_image(pose.astype(np.float32), image)
                np.save(f"{stokes_dir}/{make_stokes_filename(frame["file_path"])}", stokes_image)
                error_counter.detect_errors(stokes_image)

    error_counter.print_stats()
    saturation_counter.print_stats()
