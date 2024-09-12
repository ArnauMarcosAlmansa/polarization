from __future__ import annotations
import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch

from src.config.config import Config
from src.data_generation.RayGenerator import RayGenerator
from src.device import device
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


@dataclass
class Int2:
    x: int
    y: int

def scaled_grid_coordinates(pos: Int2, grid: np.ndarray, maximums: Int2):
    pos = copy.copy(pos)
    while maximums.x > grid.shape[1]:
        maximums.x //=2
        maximums.y //= 2

        pos.x //=2
        pos.y //= 2

    return pos


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
        self.camera_pose = camera_transform_to_pose(self.transforms.lighted.test["frames"][0]["transform_matrix"])
        self.render(self.camera_pose)
        # self.render_max_difference(self.camera_pose)

    def step(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                self.compute_mueller_matrix_for_pos(pos[0], pos[1])
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.SCREEN.fill((0, 0, 0))
        self.SCREEN.blit(self.last_surface, (0, 0))
        pygame.display.flip()

    @torch.no_grad()
    def render_max_difference(self, camera_pose):
        rays_per_polariation = self.ray_generator.get_rays_for_pose(camera_pose)

        images = []
        for rays in rays_per_polariation:
            image_rays = torch.from_numpy(np.dstack(rays).reshape((int(self.ray_generator.h * self.ray_generator.w), 12)).astype(np.float32)).to(device)
            image = self.render_rays(image_rays).reshape((int(self.ray_generator.h), int(self.ray_generator.w)))
            images.append(image)

        stack = np.dstack(images)
        maximums = stack.max(2)
        minimums = stack.min(2)

        self.last_image = ((maximums - minimums) * 255).astype(np.uint8).transpose()
        self.last_surface = pygame.surfarray.make_surface(np.dstack([self.last_image, self.last_image, self.last_image]))
        self.last_surface = pygame.transform.scale(self.last_surface, (1224, 1024))

    def render_rays(self, image_rays):
        resulting_image = np.zeros((int(self.ray_generator.h * self.ray_generator.w)))

        batch_start = 0
        batch_end = 0
        for ray_batch in image_rays.split(2048, 0):
            batch_end += ray_batch.shape[0]
            ret = self.models.lighted.render_rays(ray_batch)
            resulting_image[batch_start:batch_end] = ret["rgb_map"].cpu().numpy().squeeze(1)
            batch_start += ray_batch.shape[0]

        return resulting_image

    @torch.no_grad()
    def render(self, camera_pose):
        r_o, r_d, r_u, r_r = self.ray_generator.get_nonrotated_rays_for_pose(camera_pose)
        self.last_rays = np.dstack([r_o, r_d, r_u, r_r]).astype(np.float32)
        image_rays = torch.from_numpy(self.last_rays.reshape((int(self.ray_generator.h * self.ray_generator.w), 12))).to(device)

        resulting_image = self.render_rays(image_rays)

        self.last_image = resulting_image.reshape((int(self.ray_generator.h), int(self.ray_generator.w)))
        self.last_image = (self.last_image * 255).astype(np.uint8).transpose()
        self.last_surface = pygame.surfarray.make_surface(np.dstack([self.last_image, self.last_image, self.last_image]))
        self.last_surface = pygame.transform.scale(self.last_surface, (1224, 1024))

    @torch.no_grad()
    def compute_mueller_matrix_for_pos(self, mouse_x, mouse_y):
        return
        pos = scaled_grid_coordinates(Int2(mouse_x, mouse_y), self.last_rays, Int2(1224, 1024))
        rays = self.ray_generator.get_rays_for_pose(self.camera_pose)
        ray_000 = rays[0][pos.y, pos.x]
        ray_045 = rays[1][pos.y, pos.x]
        ray_090 = rays[2][pos.y, pos.x]
        ray_135 = rays[3][pos.y, pos.x]

    def render_ray_for_model(self, model: CRANeRFModel, ray):
        pass


@dataclass
class ModelNames:
    neutral: str
    lighted: str
    lighted_000: str
    lighted_045: str
    lighted_090: str
    lighted_135: str

class Models:
    def __init__(self, neutral: CRANeRFModel, lighted: CRANeRFModel, lighted_000: CRANeRFModel, lighted_045: CRANeRFModel, lighted_090: CRANeRFModel, lighted_135: CRANeRFModel):
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

    names = ModelNames(
        neutral="neutral",
        lighted="neutral",
        lighted_000="neutral",
        lighted_045="neutral",
        lighted_090="neutral",
        lighted_135="neutral",
    )
    models = Models.load_from_folder(config, names)
    transforms = ModelTransforms.load_from_folder(config, names)

    demo = Demo(config, models, transforms)

    while True:
        demo.step()