from src.config.config import Config
from src.pipeline import command


def run_colmap2nerf(config: Config):
    colmap2nerf_path = config.options["paths"]["colmap2nerf"]
    colmap_dir = config.options["paths"]["colmap_dir"]
    frames_dir = config.options["paths"]["frames_dir"]
    return [
        command(
            f"python3 {colmap2nerf_path} --keep_colmap_coords --text {colmap_dir}/sparse/0 --images {frames_dir} --out {colmap_dir}/transforms.json"),
    ]