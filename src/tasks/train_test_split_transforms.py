import copy
import json
import os
import random

from src.config.config import Config


def train_test_split(samples, test=0.2):
    l = len(samples)
    random.shuffle(samples)
    boundary = int(l * test)
    return samples[boundary:], samples[:boundary]


def train_test_split_transforms(config: Config):
    transforms_dir = config.options["paths"]["transforms_dir"]
    transforms_path = config.options["paths"]["colmap_dir"] + "/transforms.json"
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

        poses_train, poses_test = train_test_split(frames, test=0.1)

        transforms_for_video["frames"] = poses_train
        with open(os.path.join(transforms_dir, f"{video_name}_train_transforms.json"), "w") as f:
            json.dump(transforms_for_video, f, indent=4)

        transforms_for_video["frames"] = poses_test
        with open(os.path.join(transforms_dir, f"{video_name}_test_transforms.json"), "w") as f:
            json.dump(transforms_for_video, f, indent=4)