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

from src.tasks import extract_frames, remove_blurry_frames, run_colmap, train_test_split_transforms, generate_rays, \
    run_colmap2nerf


def train_nerfs(rays_dir, nerfs_dir):
    rays_files = os.listdir(rays_dir)
    os.makedirs(nerfs_dir, exist_ok=True)

    def extract_model_name(rays_filename):
        name_without_extension = ".".join(rays_filename.split(".")[:-1])
        name = "_".join(name_without_extension.split("_")[:-2])
        return name

    model_names = list(set([extract_model_name(filename) for filename in rays_files]))
    tasks = []
    for model_name in model_names:
        tasks.append(lambda: train_nerf(rays_dir, nerfs_dir, model_name))

    return tasks


def train_nerf(rays_dir, nerfs_dir, model_name):
    pass



if __name__ == '__main__':

    config = Config.parse("../../config.yaml")

    this_fle_path = os.path.dirname(os.path.realpath(__file__))

    extract_frames_tasks = sequential(extract_frames(config))

    remove_blurry_frames_task = function(lambda: remove_blurry_frames(config))

    run_colmap_task = sequential(run_colmap(config))

    colmap2nerf = sequential(run_colmap2nerf(config))

    split_transforms_task = function(lambda: train_test_split_transforms(config))

    generate_rays_task = function(lambda: generate_rays(config))

    # train_nerfs_task = sequential(train_nerfs())

    pipeline = Pipeline([
        # extract_frames_tasks,
        # remove_blurry_frames_task,
        # run_colmap_task,
        # colmap2nerf,
        # split_transforms_task,
        # generate_rays_task,
        # train_nerfs_task,
    ])

    pipeline.run()
