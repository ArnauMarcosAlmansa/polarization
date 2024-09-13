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
from src.tasks.train_nerf import _train_nerf
from src.training import CRANeRFModel, get_rays_dataset, RaysDataset


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
        tasks.append(function(lambda: _train_nerf(rays_dir + f"/{model_name}_train_rays.polrays",
                                                  rays_dir + f"/{model_name}_test_rays.polrays", model_name, config)))

    return tasks
