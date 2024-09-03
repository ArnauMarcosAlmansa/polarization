import json
import os

import numpy as np

from src.config.config import Config
from src.data_generation.RayGenerator import RayGenerator
from src.load_polarimetric import PolarimetricImage


def generate_rays(config: Config):
    transforms_dir = config.options["paths"]["transforms_dir"]
    rays_dir = config.options["paths"]["rays_dir"]

    half_res = config.options["tasks"]["generate_rays"]["half_res"]

    transforms_files = os.listdir(transforms_dir)
    os.makedirs(rays_dir, exist_ok=True)

    def make_rays_filename(transform_filename):
        full_filename = transform_filename.split(".")[0]
        return "_".join(full_filename.split("_")[:-1]) + "_rays.polrays"

    for filename in transforms_files:
        path = os.path.join(transforms_dir, filename)
        out_path = os.path.join(rays_dir, make_rays_filename(filename))
        with open(path, "r") as f, open(out_path, "wb") as out:
            transforms = json.load(f)
            ray_generator = RayGenerator(transforms, half_res=half_res)
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