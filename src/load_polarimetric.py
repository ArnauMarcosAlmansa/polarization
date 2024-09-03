import enum
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from src.run_nerf_helpers import get_rays_with_camera_orientation, get_rays_np_with_camera_orientation, \
    rotate_up_right_rays


@dataclass
class PolarimetricImage:
    I0: np.ndarray
    I45: np.ndarray
    I90: np.ndarray
    I135: np.ndarray

    @staticmethod
    def from_raw_image(image, half_res:bool=False):
        i0 = image[0::2, 0::2]
        i45 = image[0::2, 1::2]
        i90 = image[1::2, 0::2]
        i135 = image[1::2, 1::2]
        if half_res:
            i0 = i0[::2, ::2]
            i45 = i45[::2, ::2]
            i90 = i90[::2, ::2]
            i135 = i135[::2, ::2]
        return PolarimetricImage(i0, i45, i90, i135)

    @staticmethod
    def load(filename, half_res:bool=False):
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im = im.astype(np.float32) / 255
        return PolarimetricImage.from_raw_image(im, half_res=half_res)


class PolarRotation(enum.Enum):
    R0 = 0
    R45 = 45
    R90 = 90
    R135 = 135


def rotation_to_angle(rot: PolarRotation):
    match rot:
        case PolarRotation.R0:
            return 0
        case PolarRotation.R45:
            return np.pi / 4
        case PolarRotation.R90:
            return np.pi / 2
        case PolarRotation.R135:
            return np.pi / 4 * 3


@dataclass
class ImageWithRays:
    image: np.ndarray
    rays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    def to_raw_data(self):
        rays_origins = self.rays[0].reshape((self.image.shape[0] * self.image.shape[1], 3))
        rays_forwards = self.rays[1].reshape((self.image.shape[0] * self.image.shape[1], 3))
        rays_ups = self.rays[2].reshape((self.image.shape[0] * self.image.shape[1], 3))
        rays_rights = self.rays[3].reshape((self.image.shape[0] * self.image.shape[1], 3))
        colors = self.image.reshape((self.image.shape[0] * self.image.shape[1], 1))

        return np.hstack([rays_origins, rays_forwards, rays_ups, rays_rights, colors])


class PolarimetricDataset:
    def __init__(self, transforms_filename, halfres=False):
        self.camera_poses = []
        self.images: list[PolarimetricImage] = []
        self.halfres = halfres

        basedir = os.path.dirname(transforms_filename)

        with open(transforms_filename) as f:
            transforms = json.load(f)
            # porque separamos las imagenes de intensidad segun el patrón de los filtros del sensor
            self.fl_x = transforms['fl_x'] / 2
            self.fl_y = transforms['fl_y'] / 2
            self.c_x = transforms['c_x'] / 2
            self.c_y = transforms['c_y'] / 2
            self.h = transforms['h'] / 2
            self.w = transforms['w'] / 2

            for frame in transforms['frames']:
                pose = np.array(frame['transform_matrix'])
                self.camera_poses.append(pose)
                image = cv2.imread(os.path.join(basedir, frame['file_path']))
                image = image.astype(np.float32) / 255.0
                self.images.append(PolarimetricImage.from_raw_image(image))

    def __len__(self):
        return len(self.camera_poses)

    def __getitem__(self, idx):
        return self.camera_poses[idx], self.images[idx]

    def get_rays_for_pose_and_image(self, camera_pose, image: PolarimetricImage) -> tuple[
        ImageWithRays, ImageWithRays, ImageWithRays, ImageWithRays]:
        K = np.array([
            [self.fl_x, 0.0, self.c_x],
            [0.0, self.fl_y, self.c_y],
            [0.0, 0.0, 1.0]
        ])
        r_o, r_f, r_u, r_r = get_rays_np_with_camera_orientation(self.h, self.w, K, camera_pose)

        return (
            ImageWithRays(image.I0, (r_o, r_f, r_u, r_r)),
            ImageWithRays(image.I45,
                          (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R45)))),
            ImageWithRays(image.I90,
                          (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R90)))),
            ImageWithRays(image.I135,
                          (r_o, r_f, *rotate_up_right_rays(r_f, r_u, r_r, rotation_to_angle(PolarRotation.R135)))),
        )
