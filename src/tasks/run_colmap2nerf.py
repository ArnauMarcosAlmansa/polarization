from src.config.config import Config
from src.pipeline import command


def run_colmap2nerf(config: Config):
    colmap2nerf_path = config.options["path"]["colmap2nerf"]
    colmap_dir = config.options["path"]["colmap_dir"]
    frames_dir = config.options["path"]["frames_dir"]
    return [
        command(
            f"python3 {colmap2nerf_path} --text {colmap_dir}/sparse/0 --images {frames_dir} --out {colmap_dir}/transforms.json"),
    ]