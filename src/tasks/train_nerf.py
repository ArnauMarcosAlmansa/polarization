import cProfile
import os
from math import log10

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from click.core import batch
from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.config.config import Config
from src.device import device
from src.metrics.metrics import psnr
from src.pipeline import function
from src.training import CRANeRFModel, get_rays_dataset, RaysDataset


def train_nerf(name: str, config: Config):
    rays_dir = config.options["paths"]["rays_dir"]
    nerfs_dir = config.options["paths"]["nerfs_dir"]
    os.makedirs(nerfs_dir, exist_ok=True)

    tasks = []
    # tasks.append(function(lambda: train_nerf(rays_dir + f"/{model_name}_train_rays.polrays", model_name, config)))
    tasks.append(function(lambda: _train_nerf(rays_dir + f"/{name}_train_rays.polrays",
                                                 rays_dir + f"/{name}_test_rays.polrays", name, config)))

    return tasks



def _train_nerf(rays_filename: str, test_rays_filename: str, model_name: str, config: Config):
    print(f"TRAINING MODEL {model_name}")
    summary = SummaryWriter("./runs/" + model_name)
    model = CRANeRFModel(config)
    train_dataset = get_rays_dataset(rays_filename)
    test_dataset = get_rays_dataset(test_rays_filename)
    nerfs_dir = config.options["paths"]["nerfs_dir"]
    # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.options["tasks"]["train_nerfs"]["train"]["batch_size"], shuffle=True)
    batch_size = config.options["tasks"]["train_nerfs"]["train"]["batch_size"]
    iters = config.options["tasks"]["train_nerfs"]["train"]["n_iterations"]
    # batch = torch.from_numpy(train_dataset.get_batch(2048)).to(device)
    # im = train_dataset.matrix[:1024 * 1224, 12].reshape((1024, 1224))

    log_every_n = config.options["tasks"]["train_nerfs"]["train"]["log_every_n_iterations"]
    save_every_n = config.options["tasks"]["train_nerfs"]["train"]["save_every_n_iterations"]
    test_every_n = config.options["tasks"]["train_nerfs"]["train"]["test_every_n_iterations"]
    do_test = config.options["tasks"]["train_nerfs"]["train"]["do_test"]
    render_every_n = config.options["tasks"]["train_nerfs"]["train"]["render_every_n_iterations"]
    do_render = config.options["tasks"]["train_nerfs"]["train"]["do_render"]

    os.makedirs(f"{nerfs_dir}/{model_name}", exist_ok=True)

    checkpoints = os.listdir(f"{nerfs_dir}/{model_name}")
    if checkpoints:
        checkpoints = sorted(checkpoints)
        checkpoint = checkpoints[-1]
        model.load(f"{nerfs_dir}/{model_name}/{checkpoint}")
        start = int(checkpoint[:10])
    else:
        start = 1

    mean_mse = 0
    for i in trange(start, iters + 1):
        model.optimizer.zero_grad()
        batch = torch.from_numpy(train_dataset.get_batch(batch_size)).to(device)
        target_color = batch[:, 12]
        ret = model.render_rays(batch[:, 0:12])

        mse = torch.nn.functional.mse_loss(ret["rgb_map"].flatten(), target_color)
        mean_mse += mse.item()

        if 'coarse_rgb' in ret:
            coarse_loss = torch.nn.functional.mse_loss(ret['coarse_rgb'].flatten(), target_color)
            mse = mse + coarse_loss

        mse.backward()
        model.optimizer.step()

        if i % log_every_n == 0:
            print(f"ITER {i}\tMSE = {mean_mse / log_every_n:.5f}\tPSNR = {psnr(mean_mse / log_every_n, 1.0):.5f}")
            summary.add_scalar("train_loss", mean_mse / log_every_n, i)
            mean_mse = 0

        del mse
        del ret

        if do_test and i % test_every_n == 0:
            test_nerf(model, test_dataset, summary, i)

        if do_render and i % render_every_n == 0:
            render_nerf(model, test_dataset, summary, i)

        if i % save_every_n == 0:
            model.save(f"{nerfs_dir}/{model_name}/{i:010d}.nerf")

    model.save(f"{nerfs_dir}/{model_name}/{i:010d}.nerf")


@torch.no_grad()
def test_nerf(model: CRANeRFModel, dataset, summary: SummaryWriter, iteration: int):
    mean_mse = 0
    i = 0
    for i, batch in enumerate(dataset.sequential_batches(2048)):
        batch = torch.from_numpy(batch).to(device)
        ret = model.render_rays(batch[:, 0:12])
        mse = torch.nn.functional.mse_loss(ret["rgb_map"].flatten(), batch[:, 12])
        mean_mse += mse.item()

    print(f"TEST\tMSE = {mean_mse / (i + 1):.5f}\tPSNR = {psnr(mean_mse / (i + 1), 1.0):.5f}")
    summary.add_scalar("test_loss", mean_mse / (i + 1), global_step=iteration)


@torch.no_grad()
def render_nerf(model: CRANeRFModel, dataset: RaysDataset, summary: SummaryWriter, iteration: int):
    rgb_row = []
    for batch in dataset.get_first_image_batches(1224, 1024, 512):
        rays = torch.from_numpy(batch).to(device)
        ret = model.render_rays(rays[:, 0:12])
        rgb_row.append(ret["rgb_map"].cpu())

    image = torch.cat(rgb_row).reshape((1024, 1224))

    summary.add_image("render", image, global_step=iteration, dataformats="HW")