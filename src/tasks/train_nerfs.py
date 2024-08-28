import os

import torch.utils.data
from click.core import batch

from src.config.config import Config
from src.device import device
from src.pipeline import function
from src.training import CRANeRFModel, RaysDataset


def train_nerfs(config: Config):
    rays_dir = config.options["paths"]["rays_dir"]
    nerfs_dir = config.options["paths"]["nerfs_dir"]
    rays_files = os.listdir(rays_dir)
    os.makedirs(nerfs_dir, exist_ok=True)

    def extract_model_name(rays_filename):
        name_without_extension = ".".join(rays_filename.split(".")[:-1])
        name = "_".join(name_without_extension.split("_")[:-2])
        return name

    model_names = list(set([extract_model_name(filename) for filename in rays_files]))
    tasks = []
    for model_name in model_names:
        tasks.append(function(lambda: train_nerf(rays_dir + f"/{model_name}_train_rays.polrays", model_name, config)))

    return tasks


def train_nerf(rays_filename: str, model_name: str, config: Config):
    model = CRANeRFModel(config)
    dataset = RaysDataset(rays_filename)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.options["tasks"]["train_nerfs"]["train"]["batch_size"], shuffle=True)

    iters = config.options["tasks"]["train_nerfs"]["train"]["n_iterations"]

    for _ in range(iters):
        batch = torch.from_numpy(dataset.get_batch(512)).to(device)
        ret = model.render_rays(batch[0:12])
        mse = torch.nn.functional.mse_loss(ret["rgb_map"], batch[12])
        mse.backward()
        model.optimizer.step()
        print(f"MSE IS {mse.item():.5f}")