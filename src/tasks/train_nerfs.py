import cProfile
import os
from math import log10

import matplotlib.pyplot as plt
import torch.utils.data
from click.core import batch
from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from src.config.config import Config
from src.device import device
from src.pipeline import function
from src.training import CRANeRFModel, OnDiskRaysDataset, InMemoryRaysDataset, get_rays_dataset, RaysDataset


def train_nerfs(config: Config):
    rays_dir = config.options["paths"]["rays_dir"]
    nerfs_dir = config.options["paths"]["nerfs_dir"]
    rays_files = sorted(os.listdir(rays_dir))
    os.makedirs(nerfs_dir, exist_ok=True)

    def extract_model_name(rays_filename):
        name_without_extension = ".".join(rays_filename.split(".")[:-1])
        name = "_".join(name_without_extension.split("_")[:-2])
        return name

    model_names = list(set([extract_model_name(filename) for filename in rays_files]))
    tasks = []
    for model_name in model_names:
        # tasks.append(function(lambda: train_nerf(rays_dir + f"/{model_name}_train_rays.polrays", model_name, config)))
        tasks.append(function(lambda: train_nerf(rays_dir + f"/{model_name}_train_rays.polrays", rays_dir + f"/{model_name}_test_rays.polrays", model_name, config)))

    return tasks


def psnr(mse, max):
    return 10 * log10(max ** 2 / mse)

def train_nerf(rays_filename: str, test_rays_filename: str, model_name: str, config: Config):
    summary = SummaryWriter("./runs/" + model_name)
    model = CRANeRFModel(config)
    train_dataset = get_rays_dataset(rays_filename)
    test_dataset = get_rays_dataset(test_rays_filename)
    prof = cProfile.Profile()
    prof.enable()
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

    mean_mse = 0
    for i in trange(1, iters + 1):
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
            print(f"ITER {i}\tMSE = {mean_mse/log_every_n:.5f}\tPSNR = {psnr(mean_mse/log_every_n, 1.0):.5f}")
            summary.add_scalar("train_loss", mean_mse/log_every_n, i)
            mean_mse = 0

        if do_test and i % test_every_n == 0:
            test_nerf(model, test_dataset, summary, i)

        if do_render and i % render_every_n == 0:
            render_nerf(model, test_dataset, summary, i)

        if i % save_every_n == 0:
            model.save(f"{nerfs_dir}/{model_name}/{i:010d}.nerf")

    model.save(f"{nerfs_dir}/{model_name}/{i:010d}.nerf")

    prof.disable()
    prof.dump_stats("profile.txt")


@torch.no_grad()
def test_nerf(model: CRANeRFModel, dataset, summary: SummaryWriter, iteration: int):
    mean_mse = 0
    i = 0
    for i, batch in enumerate(dataset.sequential_batches(2048)):
        batch = torch.from_numpy(batch)
        ret = model.render_rays(batch[:, 0:12])
        mse = torch.nn.functional.mse_loss(ret["rgb_map"].flatten(), batch[:, 12])
        mean_mse += mse.item()

    print(f"TEST\tMSE = {mean_mse / (i+1):.5f}\tPSNR = {psnr(mean_mse / (i+1), 1.0):.5f}")
    summary.add_scalar("test_loss", mean_mse / (i+1), global_step=iteration)



@torch.no_grad()
def render_nerf(model: CRANeRFModel, dataset: RaysDataset, summary: SummaryWriter, iteration: int):
    rays = torch.from_numpy(dataset.get_first_image_batch(1224, 1024))
    ret = model.render_rays(rays[:, 0:12])
    summary.add_image("render", ret["rgb_map"], global_step=iteration)


