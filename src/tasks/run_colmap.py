import os

from src.config.config import Config
from src.pipeline import function, command


def run_colmap(config: Config):
    colmap_dir = config.options["paths"]["colmap_dir"]
    frames_dir = config.options["paths"]["frames_dir"]

    return [
        function(lambda: os.makedirs(colmap_dir, exist_ok=True)),
        command(
            f"QT_QPA_PLATFORM=offscreen colmap feature_extractor --database_path {colmap_dir}/database.db --image_path {frames_dir} --ImageReader.single_camera_per_folder 1 --ImageReader.camera_model PINHOLE"),
        command(f"QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher --database_path {colmap_dir}/database.db"),
        command(f"mkdir {colmap_dir}/sparse"),
        command(
            f"QT_QPA_PLATFORM=offscreen colmap mapper --database_path {colmap_dir}/database.db --image_path {frames_dir} --output_path {colmap_dir}/sparse"),
        command(f"mkdir {colmap_dir}/dense"),
        command(f"colmap image_undistorter --image_path {frames_dir} --input_path {colmap_dir}/sparse/0 --output_path {colmap_dir}/dense --output_type COLMAP --max_image_size 9999"),
        command(
            f"colmap model_converter --input_path {colmap_dir}/sparse/0 --output_path {colmap_dir}/sparse/0 --output_type TXT"),
    ]