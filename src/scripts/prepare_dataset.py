import copy
import json
from collections import OrderedDict
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.data_generation.RayGenerator import RayGenerator
from src.load_polarimetric import PolarimetricImage
from src.pipeline import Pipeline, sequential, command, function
import os


def extract_frames(videos_dir, frames_dir):
    tasks = [function(lambda: os.makedirs(frames_dir, exist_ok=True))]
    for filename in os.listdir(videos_dir):
        filename_without_extension = ".".join(filename.split('.')[0:-1])
        tasks.append(command(f"ffmpeg -i {videos_dir}/{filename} {frames_dir}/{filename_without_extension}_%05d.png"))
    return tasks


def laplacian_var(im):
    return cv2.Laplacian(im, cv2.CV_64F).var()


def remove_blurry_frames(frames_dir, threshold):
    laplacian_vars = OrderedDict()
    for filename in os.listdir(frames_dir):
        frame = cv2.imread(os.path.join(frames_dir, filename))
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_vars[filename] = laplacian_var(grayscale)

    for filename, var in laplacian_vars.items():
        if var <= threshold:
            os.unlink(os.path.join(frames_dir, filename))


def train_test_split(samples, test=0.2):
    l = len(samples)
    random.shuffle(samples)
    boundary = int(l * test)
    return samples[boundary:], samples[:boundary]


def split_transforms(transforms_path, transforms_dir):
    os.makedirs(transforms_dir, exist_ok=True)
    with open(transforms_path) as f:
        transforms = json.load(f)

    frame_paths = [filename for filename in map(lambda frame: frame["file_path"], transforms["frames"])]

    def clean_filename(fn):
        fn = os.path.basename(fn)
        fn_no_ext = ".".join(fn.split(".")[:-1])
        fn_no_idx = "_".join(fn_no_ext.split("_")[:-1])
        return fn_no_idx

    relative = os.path.relpath(os.path.dirname(transforms_path), transforms_dir)

    def update_file_path(frame):
        frame["file_path"] = os.path.join(relative, frame["file_path"])
        return frame

    transforms["frames"] = list(map(update_file_path, transforms["frames"]))

    video_names = list(set(map(clean_filename, frame_paths)))
    for video_name in video_names:
        transforms_for_video = copy.deepcopy(transforms)
        frames = [frame for frame in transforms_for_video["frames"] if
                 "_".join(frame["file_path"].split("/")[-1].split("_")[:-1]) == video_name]

        poses_train, poses_test = train_test_split(frames, test=0.2)

        transforms_for_video["frames"] = poses_train
        with open(os.path.join(transforms_dir, f"{video_name}_train_transforms.json"), "w") as f:
            json.dump(transforms_for_video, f, indent=4)

        transforms_for_video["frames"] = poses_test
        with open(os.path.join(transforms_dir, f"{video_name}_test_transforms.json"), "w") as f:
            json.dump(transforms_for_video, f, indent=4)


def generate_rays(transforms_dir, rays_dir):
    transforms_files = os.listdir(transforms_dir)
    os.makedirs(rays_dir, exist_ok=True)

    for filename in transforms_files:
        path = os.path.join(transforms_dir, filename)
        out_path = os.path.join(rays_dir, filename.split(".")[0] + "_rays.polrays")
        with open(path, "r") as f, open(out_path, "wb") as out:
            transforms = json.load(f)
            ray_generator = RayGenerator(transforms)
            for frame in transforms["frames"]:
                pose = np.array(frame["transform_matrix"])
                image = PolarimetricImage.load(os.path.normpath(os.path.join(transforms_dir, frame["file_path"])))
                ir0, ir45, ir90, ir135 = ray_generator.get_rays_for_pose_and_image(pose, image)
                bytes_ir0 = bytearray(ir0.to_raw_data().astype(np.float32))
                bytes_ir45 = bytearray(ir45.to_raw_data().astype(np.float32))
                bytes_ir90 = bytearray(ir90.to_raw_data().astype(np.float32))
                bytes_ir135 = bytearray(ir135.to_raw_data().astype(np.float32))
                out.write(bytes_ir0)
                out.write(bytes_ir45)
                out.write(bytes_ir90)
                out.write(bytes_ir135)


def train_nerfs(rays_dir):
    tasks = []

    return tasks


if __name__ == '__main__':
    videos_dir = "/data1tb/polarization/data/grapadora/videos"
    frames_dir = "/data1tb/polarization/data/grapadora/frames"
    colmap_dir = "/data1tb/polarization/data/grapadora/colmap"
    transforms_dir = "/data1tb/polarization/data/grapadora/transforms"
    rays_dir = "/data1tb/polarization/data/grapadora/rays"
    this_fle_path = os.path.dirname(os.path.realpath(__file__))

    extract_frames_tasks = sequential(extract_frames(
        videos_dir,
        frames_dir
    ))

    remove_blurry_frames_task = function(lambda: remove_blurry_frames(frames_dir, 5))

    run_colmap_task = sequential([
        function(lambda: os.makedirs(colmap_dir, exist_ok=True)),
        command(
            f"QT_QPA_PLATFORM=offscreen colmap feature_extractor --database_path {colmap_dir}/database.db --image_path {frames_dir} --ImageReader.single_camera_per_folder 1"),
        command(f"QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher --database_path {colmap_dir}/database.db"),
        command(f"mkdir {colmap_dir}/sparse"),
        command(
            f"QT_QPA_PLATFORM=offscreen colmap mapper --database_path {colmap_dir}/database.db --image_path {frames_dir} --output_path {colmap_dir}/sparse"),
        command(
            f"colmap model_converter --input_path {colmap_dir}/sparse/0 --output_path {colmap_dir}/sparse/0 --output_type TXT"),
    ])

    colmap2nerf = sequential([
        command(
            f"python3 {this_fle_path}/colmap2nerf.py --text {colmap_dir}/sparse/0 --images {frames_dir} --out {colmap_dir}/transforms.json"),
    ])

    split_transforms_task = function(lambda: split_transforms(f"{colmap_dir}/transforms.json", transforms_dir))

    generate_rays_task = function(lambda: generate_rays(transforms_dir, rays_dir))

    train_nerfs_task = sequential(train_nerfs())

    pipeline = Pipeline([
        # extract_frames_tasks,
        # remove_blurry_frames_task,
        # run_colmap_task,
        # colmap2nerf,
        # split_transforms_task,
        # generate_rays_task,
        train_nerfs_task,
    ])

    pipeline.run()
