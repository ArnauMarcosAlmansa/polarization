import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.device import device

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)




def get_rays_with_camera_orientation(H, W, K, c2w) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_forward = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    unit_rays_forward = rays_forward / torch.linalg.vector_norm(rays_forward, dim=-1, keepdim=True)

    camera_right = torch.tensor([[1, 0, 0]], dtype=torch.float32) @ c2w[:3, :3]
    rays_up = torch.cross(unit_rays_forward, camera_right.expand(unit_rays_forward.shape), dim=2)
    rays_right = torch.cross(unit_rays_forward, rays_up)


    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_origins = c2w[:3, -1].expand(rays_forward.shape)
    return rays_origins, rays_forward, rays_up, rays_right


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np_with_camera_orientation(H, W, K, c2w) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_forward = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.

    unit_rays_forward = rays_forward / np.linalg.norm(rays_forward, axis=-1, keepdims=True)

    camera_right = np.array([[1, 0, 0]], dtype=np.float32) @ c2w[:3, :3]
    rays_up = np.cross(unit_rays_forward, np.broadcast_to(camera_right, np.shape(unit_rays_forward)), axis=2)
    rays_right = np.cross(unit_rays_forward, rays_up)

    rays_origins = np.broadcast_to(c2w[:3, -1], np.shape(rays_forward))
    return rays_origins, rays_forward, rays_up, rays_right


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


# def rotate_up_right_rays(rays_forward, rays_up, rays_right, angle) -> tuple[np.ndarray, np.ndarray]:
#     q = np.stack([
#         np.full((rays_up.shape[0] * rays_up.shape[1]), np.cos(angle / 2)),
#         rays_forward[:, :, 0].flatten() * np.sin(angle / 2),
#         rays_forward[:, :, 1].flatten() * np.sin(angle / 2),
#         rays_forward[:, :, 2].flatten() * np.sin(angle / 2),
#     ])
#
#     conj_q = np.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]])
#
#     rays_up_homogeneous = np.append(rays_up, np.full((rays_up.shape[0], rays_up.shape[1], 1), 0), axis=2)
#     rays_right_homogeneous = np.append(rays_right, np.full((rays_right.shape[0], rays_right.shape[1], 1), 0), axis=2)
#
#     rays_up_homogeneous_rotated = q @ rays_up_homogeneous.reshape((rays_up.shape[0] * rays_up.shape[1], 4)) @ conj_q
#     rays_right_homogeneous_rotated = q @ rays_right_homogeneous.reshape((rays_right.shape[0] * rays_right.shape[1], 4)) @ conj_q
#
#     return rays_up_homogeneous_rotated.reshape((rays_up.shape[0], rays_up.shape[1], 4))[:, :, :3], rays_right_homogeneous_rotated.reshape((rays_right.shape[0], rays_right.shape[1], 4))[:, :, :3]


def quaternion_multiply(q, r):
    """ Multiply two quaternions q and r. """
    w0, x0, y0, z0 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w1, x1, y1, z1 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ]).T


def rotate_up_right_rays(rays_forward, rays_up, rays_right, angle) -> tuple[np.ndarray, np.ndarray]:
    q = np.dstack([
        np.full((rays_up.shape[0] * rays_up.shape[1]), np.cos(angle / 2)),
        rays_forward[:, :, 0].flatten() * np.sin(angle / 2),
        rays_forward[:, :, 1].flatten() * np.sin(angle / 2),
        rays_forward[:, :, 2].flatten() * np.sin(angle / 2),
    ])[0, :, :]

    conj_q = q * np.array([[1, -1, -1, -1]])

    rays_up_quat = np.concatenate([np.zeros((rays_up.shape[0] * rays_up.shape[1], 1)), rays_up.reshape((rays_up.shape[0] * rays_up.shape[1], 3))], axis=1)
    rays_right_quat = np.concatenate([np.zeros((rays_right.shape[0] * rays_right.shape[1], 1)), rays_right.reshape((rays_right.shape[0] * rays_right.shape[1], 3))], axis=1)

    rays_up_rotated_quat = quaternion_multiply(quaternion_multiply(q, rays_up_quat), conj_q)
    rays_right_rotated_quat = quaternion_multiply(quaternion_multiply(q, rays_right_quat), conj_q)

    return rays_up_rotated_quat[:, 1:].reshape((rays_up.shape[0], rays_up.shape[1], 3)), rays_right_rotated_quat[:, 1:].reshape((rays_right.shape[0], rays_right.shape[1], 3))
