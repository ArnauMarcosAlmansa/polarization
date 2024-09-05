import os
import random

import numpy as np
import torch
import psutil

from src.config.config import Config
from src.device import device
from src.models.CRANeRF import CRANeRF
from src.models.Embedder import get_embedder
from src.run_nerf_helpers import sample_pdf
from src.scripts.run_nerf import run_network, raw2outputs


class CRANeRFModel:
    def __init__(self, config: Config):
        opts = config.options["tasks"]["train_nerfs"]
        lrate = opts["train"]["learning_rate"]

        position_embed_fn, position_input_ch = get_embedder(3, opts["render"]["position_encoding_resolution"], 0)
        view_embed_fn, view_input_ch = get_embedder(9, opts["render"]["view_encoding_resolution"], 0)

        output_ch = 2
        skips = [4]
        self.coarse_network = CRANeRF(D=8, W=256,
                                      input_ch=position_input_ch, output_ch=output_ch, skips=skips,
                                      input_ch_views=view_input_ch, use_viewdirs=True).to(device)
        self.grad_vars = list(self.coarse_network.parameters())

        self.fine_model = CRANeRF(D=8, W=256,
                                  input_ch=position_input_ch, output_ch=output_ch, skips=skips,
                                  input_ch_views=view_input_ch, use_viewdirs=True).to(device)
        self.grad_vars += list(self.fine_model.parameters())

        self.optimizer = torch.optim.Adam(params=self.grad_vars, lr=lrate, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(params=self.grad_vars, lr=lrate, momentum=0)

        # FIXME: load checkpoints maybe???

        self.perturbation = opts["render"]["perturbation"]

        self.network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                                 embed_fn=position_embed_fn,
                                                                                 embeddirs_fn=view_embed_fn,
                                                                                 netchunk=opts["train"]["chunk_size"])

        self.n_coarse_samples = opts["render"]["n_coarse_samples"]
        self.n_fine_samples = opts["render"]["n_fine_samples"]
        self.near = opts["render"]["near"]
        self.far = opts["render"]["far"]
        self.debug = opts["train"]["debug"]

    def render_rays(self, ray_batch):
        n_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, 3:12] if ray_batch.shape[-1] > 8 else None
        # bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        # near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
        near, far = self.near, self.far  # [-1,1]
        t_vals = torch.linspace(0., 1., steps=self.n_coarse_samples, device=device)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([n_rays, self.n_coarse_samples])

        if self.perturbation > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=device)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        raw = self.network_query_fn(pts, viewdirs, self.coarse_network)
        coarse_rgb_map, coarse_disp_map, coarse_acc_map, coarse_weights, coarse_depth_map = raw2outputs(raw, z_vals,
                                                                                                        rays_d, 0, True,
                                                                                                        pytest=False)

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, coarse_weights[..., 1:-1], self.n_fine_samples,
                               det=(self.perturbation == 0.), pytest=False)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = self.fine_model
        #         raw = run_network(pts, fn=run_fn)
        raw = self.network_query_fn(pts, viewdirs, run_fn)

        fine_rgb_map, fine_disp_map, fine_acc_map, fine_weights, fine_depth_map = raw2outputs(raw, z_vals, rays_d, 0,
                                                                                              True,
                                                                                              pytest=False)
        ret = dict()
        ret["raw"] = raw
        ret['coarse_rgb'] = coarse_rgb_map
        ret['coarse_disp'] = coarse_disp_map
        ret['coarse_acc'] = coarse_acc_map
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['rgb_map'] = fine_rgb_map
        ret['disp_map'] = fine_disp_map
        ret['acc_map'] = fine_acc_map

        for k in ret:
            if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and self.debug:
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return ret

    def save(self, filename):
        obj = {
            "coarse_network": self.coarse_network.state_dict(),
            "fine_network": self.fine_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(obj, filename)

    def load(self, filename):
        obj = torch.load(filename)
        self.coarse_network.load_state_dict(obj["coarse_network"])
        self.fine_model.load_state_dict(obj["fine_network"])
        self.optimizer.load_state_dict(obj["optimizer"])


class OnDiskRaysDataset:
    def __init__(self, filename: str):
        self.n_rays = os.stat(filename).st_size // 13 // 4
        self.matrix = np.memmap(filename, shape=(self.n_rays, 13), dtype=np.float32)

    def __getitem__(self, item):
        return self.matrix[item]

    def __len__(self):
        return self.n_rays

    def get_batch(self, size: int):
        indexes = set()
        while len(indexes) < size:
            random_idx = random.randint(0, self.n_rays)
            indexes.add(random_idx)

        samples = np.zeros((size, 13), dtype=np.float32)
        for i, idx in enumerate(indexes):
            samples[i] = self[idx]

        return samples

    def sequential_batches(self, size: int):
        current_index = 0
        samples = np.zeros((size, 13), dtype=np.float32)
        while current_index < self.n_rays:
            samples[current_index % size] = self.matrix[current_index]
            if (current_index + 1) % size == 0:
                yield samples
                samples = np.zeros((size, 13), dtype=np.float32)

            current_index += 1


class InMemoryRaysDataset:
    def __init__(self, filename: str):
        self.n_rays = os.stat(filename).st_size // 13 // 4
        raw_data = np.fromfile(filename, dtype=np.float32, count=-1)
        self.matrix = raw_data.reshape((raw_data.shape[0] // 13, 13))

    def __getitem__(self, item):
        return self.matrix[item]

    def __len__(self):
        return self.n_rays

    def get_batch(self, size: int):
        indexes = set()
        while len(indexes) < size:
            random_idx = random.randint(0, self.n_rays)
            indexes.add(random_idx)

        samples = np.zeros((size, 13), dtype=np.float32)
        for i, idx in enumerate(indexes):
            samples[i] = self[idx]

        return samples

    def sequential_batches(self, size: int):
        current_index = 0
        samples = np.zeros((size, 13), dtype=np.float32)
        while current_index < self.n_rays:
            samples[current_index % size] = self.matrix[current_index]
            if (current_index + 1) % size == 0:
                yield samples
                samples = np.zeros((size, 13), dtype=np.float32)

            current_index += 1


def get_rays_dataset(filename: str):
    available_memory = psutil.virtual_memory().available
    required_memory = os.stat(filename).st_size * 2

    if available_memory < required_memory:
        return OnDiskRaysDataset(filename)
    else:
        return InMemoryRaysDataset(filename)
