from collections import OrderedDict

import cv2
import cv2.videostab
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    videos_dir = "/home/amarcos/workspace/polarization/data/videos_casa"
    frames_dir = "/home/amarcos/workspace/polarization/data/frames_casa"
    colmap_dir = "/home/amarcos/workspace/polarization/data/colmap_casa"
    this_fle_path = os.path.dirname(os.path.realpath(__file__))

    extract_frames_tasks = sequential(extract_frames(
        videos_dir,
        frames_dir
    ))

    remove_blurry_frames_task = function(lambda: remove_blurry_frames(frames_dir, 5))

    run_colmap_task = sequential([
        function(lambda: os.makedirs(colmap_dir, exist_ok=True)),
        command(f"QT_QPA_PLATFORM=offscreen colmap feature_extractor --database_path {colmap_dir}/database.db --image_path {frames_dir} --ImageReader.single_camera_per_folder 1"),
        command(f"QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher --database_path {colmap_dir}/database.db"),
        command(f"mkdir {colmap_dir}/sparse"),
        command(f"QT_QPA_PLATFORM=offscreen colmap mapper --database_path {colmap_dir}/database.db --image_path {frames_dir} --output_path {colmap_dir}/sparse"),
    ])

    colmap2nerf = sequential([
        command(f"python3 {this_fle_path}/colmap2nerf.py {colmap_dir}/sparse")
    ])

    pipeline = Pipeline([
        extract_frames_tasks,
        remove_blurry_frames_task,
        run_colmap_task,
        colmap2nerf
    ])

    pipeline.run()

