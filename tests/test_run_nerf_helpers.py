from unittest import TestCase

import numpy as np
import torch

from src.run_nerf_helpers import get_rays_with_camera_orientation, get_rays_np_with_camera_orientation, \
    rotate_up_right_rays


class Test(TestCase):
    def test_get_rays_with_camera_orientation(self):
        H = 5
        W = 5
        K = torch.tensor([
            [1, 0, W / 2],
            [0, 1, H / 2],
            [0, 0, 1],
        ])
        pose = torch.eye(4)
        rays = get_rays_with_camera_orientation(H, W, K, pose)

        K = np.array([
            [1, 0, W / 2],
            [0, 1, H / 2],
            [0, 0, 1],
        ])
        pose = np.eye(4)
        rays_np = get_rays_np_with_camera_orientation(H, W, K, pose)
        print(rays)
        print(rays_np)

    def test_rotate_up_right_rays(self):
        rays_forward = np.array([[[0, 0, 1]]])
        rays_up = np.array([[[0, -1, 0]]])
        rays_right = np.array([[[1, 0, 0]]])

        r_u, r_r = rotate_up_right_rays(rays_forward, rays_up, rays_right, np.pi / 2)
        print(r_u)
        print(r_r)
