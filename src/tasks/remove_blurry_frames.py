import os
from collections import OrderedDict

import cv2

from src.config.config import Config


def laplacian_var(im):
    return cv2.Laplacian(im, cv2.CV_64F).var()


def remove_blurry_frames(config: Config):
    frames_dir = config.options["paths"]["frames_dir"]
    threshold = config.options["tasks"]["remove_blurry_frames"]["threshold"]

    laplacian_vars = OrderedDict()
    for filename in os.listdir(frames_dir):
        frame = cv2.imread(os.path.join(frames_dir, filename))
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_vars[filename] = laplacian_var(grayscale)

    for filename, var in laplacian_vars.items():
        if var <= threshold:
            os.unlink(os.path.join(frames_dir, filename))