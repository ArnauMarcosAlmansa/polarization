from __future__ import annotations
import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch

from src.config.config import Config
from src.data_generation.RayGenerator import RayGenerator
from src.device import device
from src.scripts.solve_mueller_matrices import image_4channel_to_stokes, solve_muellers_for_image, \
    make_mueller_image_validity_map, make_stokes_mask
from src.training import CRANeRFModel


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", default="../../config.yaml")
    return argparser.parse_args()


def load_last_available_checkpoint_for_model(model: CRANeRFModel, nerfs_dir: str, model_name: str):
    checkpoints = os.listdir(f"{nerfs_dir}/{model_name}")
    if not checkpoints:
        # raise Exception(f"No checkpoints available for '{model_name}'.")
        print(f"No checkpoints available for '{model_name}'")
        return (model, 1)

    checkpoints = sorted(checkpoints)
    checkpoint = checkpoints[-1]
    model.load(f"{nerfs_dir}/{model_name}/{checkpoint}")
    start = int(checkpoint[:10])

    return (model, start)


def camera_transform_to_pose(transform: list[list[float]]) -> np.ndarray:
    return np.array(transform)

def compute_screen_normals(depth_map: np.ndarray, fx: float, fy: float):
    dz_dv, dz_du = np.gradient(depth_map)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth_map  # x is xyz of camera coordinate
    dv_dy = fy / depth_map

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth_map)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit

@dataclass
class Int2:
    x: int
    y: int


def scaled_grid_coordinates(pos: Int2, grid: np.ndarray, maximums: Int2):
    pos = copy.copy(pos)
    while maximums.x > grid.shape[1]:
        maximums.x //= 2
        maximums.y //= 2

        pos.x //= 2
        pos.y //= 2

    return pos


@dataclass
class Stokes:
    s0: float
    s1: float
    s2: float

    @staticmethod
    def from_intensities(i000: float, i045: float, i090: float, i135: float) -> Stokes:
        s0 = (i000 + i045 + i090 + i135) / 2
        s1 = i000 - i090
        s2 = i045 - i135
        return Stokes(s0, s1, s2)

    def numpy(self) -> np.ndarray:
        return np.array([self.s0, self.s1, self.s2])


def solve_mueller(Sout_000, Sout_045, Sout_090, Sout_135, Sneutral, Sin_000, Sin_045, Sin_090, Sin_135):
    Sout_000 = Sout_000 - Sneutral
    Sout_045 = Sout_045 - Sneutral
    Sout_090 = Sout_090 - Sneutral
    Sout_135 = Sout_135 - Sneutral

    A = np.concatenate([Sout_000, Sout_045, Sout_090, Sout_135])
    B = np.vstack([
        np.kron(np.eye(3)[0], Sin_000),  # Row for A1
        np.kron(np.eye(3)[1], Sin_000),
        np.kron(np.eye(3)[2], Sin_000),
        np.kron(np.eye(3)[0], Sin_045),  # Row for A2
        np.kron(np.eye(3)[1], Sin_045),
        np.kron(np.eye(3)[2], Sin_045),
        np.kron(np.eye(3)[0], Sin_090),  # Row for A3
        np.kron(np.eye(3)[1], Sin_090),
        np.kron(np.eye(3)[2], Sin_090),
        np.kron(np.eye(3)[0], Sin_135),  # Row for A4
        np.kron(np.eye(3)[1], Sin_135),
        np.kron(np.eye(3)[2], Sin_135)
    ])

    M_vector, residuals, rank, s = np.linalg.lstsq(B, A, rcond=None)

    M = M_vector.reshape(3, 3)

    return M


INPUT_STOKES = [
    Stokes(1, 1, 0),
    Stokes(1, 0, 1),
    Stokes(1, -1, 0),
    Stokes(1, 0, -1),
]


class Demo:
    def __init__(self, config: Config, models: Models, transforms: ModelTransforms):
        self.config = config
        self.models = models
        self.transforms = transforms
        self.ray_generator = RayGenerator(transforms.neutral.test, downscale=2)

        pygame.init()
        self.SCREEN = pygame.display.set_mode((1224, 1024))
        self.CLOCK = pygame.time.Clock()
        self.SCREEN.fill((0, 0, 0))

        self.last_image = np.zeros((1024, 1224), dtype=np.uint8)
        self.last_surface = pygame.surfarray.make_surface(self.last_image)
        self.last_rays = np.zeros((1, 1))
        self.camera_pose = camera_transform_to_pose(self.transforms.lighted_135.test["frames"][0]["transform_matrix"])
        # self.render(self.models.lighted, self.camera_pose)
        # self.render_max_difference(self.camera_pose)
        # self.compute_mueller_matrix_for_point(500, 500)
        # self.ray_generator._verify_ray_rotations(self.camera_pose)
        # pol_im = self.render_polarimetric(self.models.lighted_135, self.camera_pose)
        #
        # plt.imsave("test_000.png", pol_im[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
        # plt.imsave("test_045.png", pol_im[:, :, 1], cmap="gray", vmin=0.0, vmax=1.0)
        # plt.imsave("test_090.png", pol_im[:, :, 2], cmap="gray", vmin=0.0, vmax=1.0)
        # plt.imsave("test_135.png", pol_im[:, :, 3], cmap="gray", vmin=0.0, vmax=1.0)
        self.render_depth(self.models.lighted_135, self.camera_pose)
        # self.render(self.models.lighted_135, self.camera_pose)
        # self.render_normals(self.models.lighted_135, self.camera_pose)
        # self.compute_mueller_matrix_for_entire_image()
        print("DONE")

    def step(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                self.compute_mueller_matrix_for_point(pos[0], pos[1])
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.SCREEN.fill((0, 0, 0))
        self.SCREEN.blit(self.last_surface, (0, 0))
        pygame.display.flip()

    @torch.no_grad()
    def render_max_difference(self, model: CRANeRFModel, camera_pose):
        rays_per_polariation = self.ray_generator.get_rays_for_pose(camera_pose)

        images = []
        for rays in rays_per_polariation:
            image_rays = torch.from_numpy(
                np.dstack(rays).reshape((int(self.ray_generator.h * self.ray_generator.w), 12)).astype(np.float32)).to(
                device)
            image = self.render_rays(model, image_rays).reshape((int(self.ray_generator.h), int(self.ray_generator.w)))
            images.append(image)

        stack = np.dstack(images)
        maximums = stack.max(2)
        minimums = stack.min(2)

        self.last_image = ((maximums - minimums) * 255).astype(np.uint8).transpose()
        self.last_surface = pygame.surfarray.make_surface(
            np.dstack([self.last_image, self.last_image, self.last_image]))
        self.last_surface = pygame.transform.scale(self.last_surface, (1224, 1024))

    def render_rays(self, model: CRANeRFModel, image_rays):
        resulting_image = np.zeros((int(self.ray_generator.h * self.ray_generator.w)))

        batch_start = 0
        batch_end = 0
        for ray_batch in image_rays.split(2048, 0):
            batch_end += ray_batch.shape[0]
            ret = model.render_rays(ray_batch)
            resulting_image[batch_start:batch_end] = ret["rgb_map"].cpu().numpy().squeeze(1)
            batch_start += ray_batch.shape[0]

        return resulting_image

    def render_depth_rays(self, model: CRANeRFModel, image_rays):
        resulting_image = np.zeros((int(self.ray_generator.h * self.ray_generator.w)))

        batch_start = 0
        batch_end = 0
        for ray_batch in image_rays.split(2048, 0):
            batch_end += ray_batch.shape[0]
            ret = model.render_rays(ray_batch)
            resulting_image[batch_start:batch_end] = ret["depth_map"].cpu().numpy()
            batch_start += ray_batch.shape[0]

        return resulting_image

    @torch.no_grad()
    def render_polarimetric(self, model: CRANeRFModel, camera_pose):
        rays_per_polariation = self.ray_generator.get_rays_for_pose(camera_pose)
        images = []
        for rays in rays_per_polariation:
            image_rays = torch.from_numpy(
                np.dstack(rays).reshape((int(self.ray_generator.h * self.ray_generator.w), 12)).astype(np.float32)).to(
                device)
            image = self.render_rays(model, image_rays).reshape((int(self.ray_generator.h), int(self.ray_generator.w)))
            images.append(image)

        polarimetric_image = np.dstack(images)
        return polarimetric_image

    @torch.no_grad()
    def render(self, model: CRANeRFModel, camera_pose):
        r_o, r_d, r_u, r_r = self.ray_generator.get_nonrotated_rays_for_pose(camera_pose)
        self.last_rays = np.dstack([r_o, r_d, r_u, r_r]).astype(np.float32)
        image_rays = torch.from_numpy(
            self.last_rays.reshape((int(self.ray_generator.h * self.ray_generator.w), 12))).to(device)

        resulting_image = self.render_rays(model, image_rays)

        self.last_image = resulting_image.reshape((int(self.ray_generator.h), int(self.ray_generator.w)))
        self.last_image = (self.last_image * 255).astype(np.uint8).transpose()
        self.last_surface = pygame.surfarray.make_surface(
            np.dstack([self.last_image, self.last_image, self.last_image]))
        self.last_surface = pygame.transform.scale(self.last_surface, (1224, 1024))

    @torch.no_grad()
    def render_depth(self, model: CRANeRFModel, camera_pose):
        r_o, r_d, r_u, r_r = self.ray_generator.get_nonrotated_rays_for_pose(camera_pose)
        self.last_rays = np.dstack([r_o, r_d, r_u, r_r]).astype(np.float32)
        image_rays = torch.from_numpy(
            self.last_rays.reshape((int(self.ray_generator.h * self.ray_generator.w), 12))).to(device)

        resulting_image = self.render_depth_rays(model, image_rays)

        self.last_image = resulting_image.reshape((int(self.ray_generator.h), int(self.ray_generator.w)))
        self.last_image = self.last_image.clip(3, 4.5)
        self.last_image = 1 - ((self.last_image - self.last_image.min()) / (self.last_image.max() - self.last_image.min()))
        self.last_image = (self.last_image * 255).astype(np.uint8).transpose()
        self.last_surface = pygame.surfarray.make_surface(
            np.dstack([self.last_image, self.last_image, self.last_image]))
        self.last_surface = pygame.transform.scale(self.last_surface, (1224, 1024))

    @torch.no_grad()
    def render_normals(self, model: CRANeRFModel, camera_pose):
        r_o, r_d, r_u, r_r = self.ray_generator.get_nonrotated_rays_for_pose(camera_pose)
        self.last_rays = np.dstack([r_o, r_d, r_u, r_r]).astype(np.float32)
        image_rays = torch.from_numpy(
            self.last_rays.reshape((int(self.ray_generator.h * self.ray_generator.w), 12))).to(device)

        resulting_image = self.render_depth_rays(model, image_rays)

        depth_image = resulting_image.reshape((int(self.ray_generator.h), int(self.ray_generator.w)))
        screen_normals_image = compute_screen_normals(depth_image, self.ray_generator.fl_x, self.ray_generator.fl_y)

        self.last_image = (screen_normals_image + 1.0) / 2.0
        self.last_image = (self.last_image * 255).astype(np.uint8).transpose((1, 0, 2))
        self.last_surface = pygame.surfarray.make_surface(self.last_image)
        self.last_surface = pygame.transform.scale(self.last_surface, (1224, 1024))


    @torch.no_grad()
    def compute_mueller_matrix_for_point(self, mouse_x, mouse_y):
        pos = scaled_grid_coordinates(Int2(mouse_x, mouse_y), self.last_rays, Int2(1224, 1024))
        rays = self.ray_generator.get_rays_for_pose(self.camera_pose)
        ray_000 = rays[0][0][pos.y, pos.x], rays[0][1][pos.y, pos.x], rays[0][2][pos.y, pos.x], rays[0][3][pos.y, pos.x]
        ray_045 = rays[1][0][pos.y, pos.x], rays[1][1][pos.y, pos.x], rays[1][2][pos.y, pos.x], rays[1][3][pos.y, pos.x]
        ray_090 = rays[2][0][pos.y, pos.x], rays[2][1][pos.y, pos.x], rays[2][2][pos.y, pos.x], rays[2][3][pos.y, pos.x]
        ray_135 = rays[3][0][pos.y, pos.x], rays[3][1][pos.y, pos.x], rays[3][2][pos.y, pos.x], rays[3][3][pos.y, pos.x]

        neutral_000 = self.render_ray_for_model(self.models.neutral, ray_000)
        neutral_045 = self.render_ray_for_model(self.models.neutral, ray_045)
        neutral_090 = self.render_ray_for_model(self.models.neutral, ray_090)
        neutral_135 = self.render_ray_for_model(self.models.neutral, ray_135)

        lighted_000 = self.render_ray_for_model(self.models.lighted, ray_000)
        lighted_045 = self.render_ray_for_model(self.models.lighted, ray_045)
        lighted_090 = self.render_ray_for_model(self.models.lighted, ray_090)
        lighted_135 = self.render_ray_for_model(self.models.lighted, ray_135)

        lighted_000_000 = self.render_ray_for_model(self.models.lighted_000, ray_000)
        lighted_000_045 = self.render_ray_for_model(self.models.lighted_000, ray_045)
        lighted_000_090 = self.render_ray_for_model(self.models.lighted_000, ray_090)
        lighted_000_135 = self.render_ray_for_model(self.models.lighted_000, ray_135)

        lighted_045_000 = self.render_ray_for_model(self.models.lighted_045, ray_000)
        lighted_045_045 = self.render_ray_for_model(self.models.lighted_045, ray_045)
        lighted_045_090 = self.render_ray_for_model(self.models.lighted_045, ray_090)
        lighted_045_135 = self.render_ray_for_model(self.models.lighted_045, ray_135)

        lighted_090_000 = self.render_ray_for_model(self.models.lighted_090, ray_000)
        lighted_090_045 = self.render_ray_for_model(self.models.lighted_090, ray_045)
        lighted_090_090 = self.render_ray_for_model(self.models.lighted_090, ray_090)
        lighted_090_135 = self.render_ray_for_model(self.models.lighted_090, ray_135)

        lighted_135_000 = self.render_ray_for_model(self.models.lighted_135, ray_000)
        lighted_135_045 = self.render_ray_for_model(self.models.lighted_135, ray_045)
        lighted_135_090 = self.render_ray_for_model(self.models.lighted_135, ray_090)
        lighted_135_135 = self.render_ray_for_model(self.models.lighted_135, ray_135)

        stokes_neutral = Stokes.from_intensities(neutral_000.item(), neutral_045.item(), neutral_090.item(),
                                                 neutral_135.item())
        stokes_000 = Stokes.from_intensities(lighted_000_000.item(), lighted_000_045.item(), lighted_000_090.item(),
                                             lighted_000_135.item())
        stokes_045 = Stokes.from_intensities(lighted_045_000.item(), lighted_045_045.item(), lighted_045_090.item(),
                                             lighted_045_135.item())
        stokes_090 = Stokes.from_intensities(lighted_090_000.item(), lighted_090_045.item(), lighted_090_090.item(),
                                             lighted_090_135.item())
        stokes_135 = Stokes.from_intensities(lighted_135_000.item(), lighted_135_045.item(), lighted_135_090.item(),
                                             lighted_135_135.item())

        M = solve_mueller(
            stokes_000.numpy(),
            stokes_045.numpy(),
            stokes_090.numpy(),
            stokes_135.numpy(),
            stokes_neutral.numpy(),
            INPUT_STOKES[0].numpy(),
            INPUT_STOKES[1].numpy(),
            INPUT_STOKES[2].numpy(),
            INPUT_STOKES[3].numpy()
        )

        print("Mueller: ")
        print(M)
        print(f"VALID DETERMINANT? {np.linalg.det(M) >= 0}")
        print(f"VALID M00 CONSTRAINT? {np.abs(M).max() == M[0, 0]}")
        print(f"VALID RANGE? {M.max() <= 1 and M.min() >= -1}")

    def render_ray_for_model(self, model: CRANeRFModel, ray):
        ray_batch = torch.from_numpy(np.concatenate(ray).reshape((1, 12)).astype(np.float32)).to(device)
        ret = model.render_rays(ray_batch)
        return ret["rgb_map"][0, 0]

    @torch.no_grad()
    def compute_mueller_matrix_for_entire_image(self):
        neutral_pol_im = self.render_polarimetric(self.models.neutral, self.camera_pose)
        lighted_000_pol_im = self.render_polarimetric(self.models.lighted_000, self.camera_pose) - neutral_pol_im
        lighted_045_pol_im = self.render_polarimetric(self.models.lighted_045, self.camera_pose) - neutral_pol_im
        lighted_090_pol_im = self.render_polarimetric(self.models.lighted_090, self.camera_pose) - neutral_pol_im
        lighted_135_pol_im = self.render_polarimetric(self.models.lighted_135, self.camera_pose) - neutral_pol_im

        # neutral_stokes = image_4channel_to_stokes(neutral_pol_im)
        lighted_000_stokes = image_4channel_to_stokes(lighted_000_pol_im)
        lighted_045_stokes = image_4channel_to_stokes(lighted_045_pol_im)
        lighted_090_stokes = image_4channel_to_stokes(lighted_090_pol_im)
        lighted_135_stokes = image_4channel_to_stokes(lighted_135_pol_im)

        total_mask = make_stokes_mask(lighted_000_stokes) & make_stokes_mask(lighted_045_stokes) & make_stokes_mask(lighted_090_stokes) & make_stokes_mask(lighted_135_stokes)

        mueller_image = solve_muellers_for_image(lighted_000_stokes, lighted_045_stokes, lighted_090_stokes, lighted_135_stokes, total_mask)

        validity_map = make_mueller_image_validity_map(mueller_image)

        plt.imshow(validity_map),
        plt.show()


@dataclass
class ModelNames:
    neutral: str
    lighted: str
    lighted_000: str
    lighted_045: str
    lighted_090: str
    lighted_135: str


class Models:
    def __init__(self, neutral: CRANeRFModel, lighted: CRANeRFModel, lighted_000: CRANeRFModel,
                 lighted_045: CRANeRFModel, lighted_090: CRANeRFModel, lighted_135: CRANeRFModel):
        self.neutral = neutral
        self.lighted = lighted
        self.lighted_000 = lighted_000
        self.lighted_045 = lighted_045
        self.lighted_090 = lighted_090
        self.lighted_135 = lighted_135

    @staticmethod
    def load_from_folder(config: Config, names: ModelNames) -> Models:
        nerfs_dir = config.options["paths"]["nerfs_dir"]
        neutral = load_last_available_checkpoint_for_model(CRANeRFModel(config), nerfs_dir, names.neutral)
        lighted = load_last_available_checkpoint_for_model(CRANeRFModel(config), nerfs_dir, names.lighted)
        lighted_000 = load_last_available_checkpoint_for_model(CRANeRFModel(config), nerfs_dir, names.lighted_000)
        lighted_045 = load_last_available_checkpoint_for_model(CRANeRFModel(config), nerfs_dir, names.lighted_045)
        lighted_090 = load_last_available_checkpoint_for_model(CRANeRFModel(config), nerfs_dir, names.lighted_090)
        lighted_135 = load_last_available_checkpoint_for_model(CRANeRFModel(config), nerfs_dir, names.lighted_135)

        return Models(neutral[0], lighted[0], lighted_000[0], lighted_045[0], lighted_090[0], lighted_135[0])


@dataclass
class TransformPair:
    train: dict[str, Any]
    test: dict[str, Any]


def load_transform(filename: str) -> dict[str, Any]:
    with open(filename, "rt") as f:
        return json.load(f)


@dataclass
class ModelTransforms:
    neutral: TransformPair
    lighted: TransformPair
    lighted_000: TransformPair
    lighted_045: TransformPair
    lighted_090: TransformPair
    lighted_135: TransformPair

    @staticmethod
    def load_from_folder(config: Config, names: ModelNames) -> ModelTransforms:
        transforms_dir = config.options["paths"]["transforms_dir"]

        return ModelTransforms(
            neutral=TransformPair(
                train=load_transform(f"{transforms_dir}/{names.neutral}_train_transforms.json"),
                test=load_transform(f"{transforms_dir}/{names.neutral}_test_transforms.json"),
            ),
            lighted=TransformPair(
                train=load_transform(f"{transforms_dir}/{names.lighted}_train_transforms.json"),
                test=load_transform(f"{transforms_dir}/{names.lighted}_test_transforms.json"),
            ),
            lighted_000=TransformPair(
                train=load_transform(f"{transforms_dir}/{names.lighted_000}_train_transforms.json"),
                test=load_transform(f"{transforms_dir}/{names.lighted_000}_test_transforms.json"),
            ),
            lighted_045=TransformPair(
                train=load_transform(f"{transforms_dir}/{names.lighted_045}_train_transforms.json"),
                test=load_transform(f"{transforms_dir}/{names.lighted_045}_test_transforms.json"),
            ),
            lighted_090=TransformPair(
                train=load_transform(f"{transforms_dir}/{names.lighted_090}_train_transforms.json"),
                test=load_transform(f"{transforms_dir}/{names.lighted_090}_test_transforms.json"),
            ),
            lighted_135=TransformPair(
                train=load_transform(f"{transforms_dir}/{names.lighted_135}_train_transforms.json"),
                test=load_transform(f"{transforms_dir}/{names.lighted_135}_test_transforms.json"),
            ),
        )


if __name__ == '__main__':
    args = parse_args()

    config = Config.parse(args.config)
    names = ModelNames(
        neutral="neutral",
        lighted="pos1",
        lighted_000="pos1_000",
        lighted_045="pos1_045",
        lighted_090="pos1_090",
        lighted_135="pos1_135",
    )

    # names = ModelNames(
    #     neutral="neutral",
    #     lighted="neutral",
    #     lighted_000="neutral",
    #     lighted_045="neutral",
    #     lighted_090="neutral",
    #     lighted_135="neutral",
    # )
    models = Models.load_from_folder(config, names)
    transforms = ModelTransforms.load_from_folder(config, names)

    demo = Demo(config, models, transforms)

    while True:
        demo.step()
