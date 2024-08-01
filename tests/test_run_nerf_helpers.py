from unittest import TestCase

import numpy as np
import torch

from src.run_nerf_helpers import get_rays_with_camera_orientation, get_rays_np_with_camera_orientation


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
