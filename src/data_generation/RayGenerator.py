import os

import cv2
import numpy as np
from xdg.IconTheme import basedir

from src.load_polarimetric import PolarimetricImage, ImageWithRays, rotation_to_angle, PolarRotation
from src.run_nerf_helpers import get_rays_np_with_camera_orientation, rotate_up_right_rays


class RayGenerator:
    def __init__(self, transforms, half_res:bool=False):
        self.half_res = half_res
        scale = 4 if half_res else 2
        self.fl_x = transforms['fl_x'] / scale
        self.fl_y = transforms['fl_y'] / scale
        self.c_x = transforms['cx'] / scale
        self.c_y = transforms['cy'] / scale
        self.h = transforms['h'] / scale
        self.w = transforms['w'] / scale

    def get_rays_for_pose_and_image(self, camera_pose, image: PolarimetricImage) -> tuple[ImageWithRays, ImageWithRays, ImageWithRays, ImageWithRays]:
        K = np.array([
            [self.fl_x, 0.0, self.c_x],
            [0.0, self.fl_y, self.c_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        r_o, r_f, r_u, r_r = get_rays_np_with_camera_orientation(self.h, self.w, K, camera_pose)

        return (
            ImageWithRays(image.I0, (r_o, r_f, r_u, r_r)),
            ImageWithRays(image.I45, (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R45)))),
            ImageWithRays(image.I90, (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R90)))),
            ImageWithRays(image.I135, (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R135)))),
        )