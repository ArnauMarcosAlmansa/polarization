import os

from src.config.config import Config
from src.pipeline import function, command


def extract_frames(config: Config):
    videos_dir = config.options["paths"]["videos_dir"]
    frames_dir = config.options["paths"]["frames_dir"]

    tasks = [function(lambda: os.makedirs(frames_dir, exist_ok=True))]

    for filename in os.listdir(videos_dir):
        filename_without_extension = ".".join(filename.split('.')[0:-1])
        tasks.append(command(f"ffmpeg -i {videos_dir}/{filename} {frames_dir}/{filename_without_extension}_%05d.png"))

    return tasks
