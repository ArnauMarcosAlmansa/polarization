import os

from src.config.config import Config
from src.pipeline import function, command


def run_colmap(config: Config):
    colmap_dir = config.options["paths"]["colmap_dir"]
    frames_dir = config.options["paths"]["frames_dir"]

    return [
        function(lambda: os.makedirs(colmap_dir, exist_ok=True)),
        command(
            f"QT_QPA_PLATFORM=offscreen colmap feature_extractor --database_path {colmap_dir}/database.db --image_path {frames_dir} --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE     --ImageReader.camera_params \"1638.9357202716612,1630.9098989804822,960,540\" "),
        # command(f"QT_QPA_PLATFORM=offscreen colmap exhaustive_matcher --database_path {colmap_dir}/database.db"),
        command(f"QT_QPA_PLATFORM=offscreen colmap sequential_matcher --database_path {colmap_dir}/database.db"),
        function(lambda: os.makedirs(f"{colmap_dir}/sparse", exist_ok=True)),
        command(
            f"QT_QPA_PLATFORM=offscreen colmap mapper --database_path {colmap_dir}/database.db --image_path {frames_dir} --output_path {colmap_dir}/sparse     --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0"),
        function(lambda: os.makedirs(f"{colmap_dir}/dense", exist_ok=True)),
        command(f"colmap image_undistorter --image_path {frames_dir} --input_path {colmap_dir}/sparse/0 --output_path {colmap_dir}/dense --output_type COLMAP --max_image_size 9999"),
        command(
            f"colmap model_converter --input_path {colmap_dir}/sparse/0 --output_path {colmap_dir}/sparse/0 --output_type TXT"),
    ]