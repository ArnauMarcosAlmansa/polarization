import math
import os
from itertools import pairwise

import cv2
import numpy as np
from xdg.IconTheme import basedir

from src.load_polarimetric import PolarimetricImage, ImageWithRays, rotation_to_angle, PolarRotation
from src.run_nerf_helpers import get_rays_np_with_camera_orientation, rotate_up_right_rays

def angle_between_vectors(vec1, vec2):
    return math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

type Downscale = 1 | 2 | 4 | 8

class RayGenerator:
    def __init__(self, transforms, downscale: Downscale = 1):
        self.downscale = downscale
        scale = 2 * downscale
        self.fl_x = transforms['fl_x'] / scale
        self.fl_y = transforms['fl_y'] / scale
        self.c_x = transforms['cx'] / scale
        self.c_y = transforms['cy'] / scale
        self.h = transforms['h'] / scale
        self.w = transforms['w'] / scale

    def get_nonrotated_rays_for_pose(self, camera_pose):
        K = np.array([
            [self.fl_x, 0.0, self.c_x],
            [0.0, self.fl_y, self.c_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        r_o, r_f, r_u, r_r = get_rays_np_with_camera_orientation(self.h, self.w, K, camera_pose)
        return r_o, r_f, r_u, r_r

    def get_rays_for_pose(self, camera_pose):
        r_o, r_f, r_u, r_r = self.get_nonrotated_rays_for_pose(camera_pose)

        return (
            (r_o, r_f, r_u, r_r),
            (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R45))),
            (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R90))),
            (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R135))),
        )

    def get_rays_for_pose_and_image(self, camera_pose, image: PolarimetricImage) -> tuple[ImageWithRays, ImageWithRays, ImageWithRays, ImageWithRays]:
        r_o, r_f, r_u, r_r = self.get_nonrotated_rays_for_pose(camera_pose)

        return (
            ImageWithRays(image.I0, (r_o, r_f, r_u, r_r)),
            ImageWithRays(image.I45, (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R45)))),
            ImageWithRays(image.I90, (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R90)))),
            ImageWithRays(image.I135, (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R135)))),
        )

    def _verify_ray_rotations(self, camera_pose):
        rays_per_polarization = self.get_rays_for_pose(camera_pose)
        ups: list[np.ndarray] = [rays_per_polarization[0][2], rays_per_polarization[1][2], rays_per_polarization[2][2], rays_per_polarization[3][2]]
        rights: list[np.ndarray] = [rays_per_polarization[0][3], rays_per_polarization[1][3], rays_per_polarization[2][3], rays_per_polarization[3][3]]

        for (ups1, ups2), (rights1, rights2) in zip(pairwise(ups), pairwise(rights)):
            for y in range(int(self.h)):
                for x in range(int(self.w)):
                    up1 = ups1[y, x]
                    right1 = rights1[y, x]
                    up2 = ups2[y, x]
                    right2 = rights2[y, x]
                    assert math.isclose(angle_between_vectors(up1, up2) / np.pi * 180, 45.0)
                    assert math.isclose(angle_between_vectors(right1, right2) / np.pi * 180, 45.0)
                    assert math.isclose(angle_between_vectors(up1, right1) / np.pi * 180, 90.0)
                    assert math.isclose(angle_between_vectors(up2, right2) / np.pi * 180, 90.0)


        print("ROTATIONS ARE OK")