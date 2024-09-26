import argparse
import copy
import json
from collections import OrderedDict
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config.config import Config
from src.data_generation.RayGenerator import RayGenerator
from src.load_polarimetric import PolarimetricImage
from src.pipeline import Pipeline, sequential, command, function
import os

from src.tasks import extract_frames, remove_blurry_frames, run_colmap, train_test_split_transforms, generate_stokes_images, \
    run_colmap2nerf, train_nerfs, estimate_near_far
from src.tasks.transforms_for_stokes import transforms_for_stokes


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", default="../../config.yaml")
    return argparser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    config = Config.parse(args.config)

    this_fle_path = os.path.dirname(os.path.realpath(__file__))

    extract_frames_tasks = sequential(extract_frames(config))

    remove_blurry_frames_task = function(lambda: remove_blurry_frames(config))

    run_colmap_task = sequential(run_colmap(config))

    estimate_near_far_task = estimate_near_far(config)

    colmap2nerf = sequential(run_colmap2nerf(config))

    split_transforms_task = function(lambda: train_test_split_transforms(config))

    generate_stokes_images_task = function(lambda: generate_stokes_images(config))

    transforms_for_stokes_task = function(lambda: transforms_for_stokes(config))

    pipeline = Pipeline([
        # extract_frames_tasks,
        # remove_blurry_frames_task,
        # run_colmap_task,
        # colmap2nerf,
        # split_transforms_task,
        # generate_rays_task,
        generate_stokes_images_task,
        # transforms_for_stokes_task
    ])

    pipeline.run()
