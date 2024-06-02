import random
import sys
import time
from math import cos, sin, sqrt, atan2, isclose, asin, tan, atan
from enum import Enum
from typing import Callable, Tuple

import cv2
import numba
import numpy as np
from matplotlib import pyplot as plt
from numba import cuda

import math

from tqdm import tqdm

from src.ellipse import Ellipse

from numba import config

config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True


def clamp(min_value, max_value):
    def clamp_fn(value):
        return max(min(value, max_value), min_value)

    return clamp_fn


clamp_grad = cuda.jit(clamp(-0.25, 0.25), device=True)


@cuda.jit(device=True)
def isclose(a: float, b: float, abs_tol):
    return abs(a - b) <= max(1e-05 * max(abs(a), abs(b)), abs_tol)


@cuda.jit(device=True)
def diff_a(a: float, b: float, A: float, x: float, y: float) -> float:
    return -4 * (x * cos(A) + y * sin(A)) ** 2 * (
            (x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) / a ** 3


@cuda.jit(device=True)
def diff_b(a: float, b: float, A: float, x: float, y: float) -> float:
    return -4 * (y * cos(A) - x * sin(A)) ** 2 * (
            (x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) / b ** 3


@cuda.jit(device=True)
def diff_A(a: float, b: float, A: float, x: float, y: float) -> float:
    return 4 * ((x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) * (
            (x * cos(A) + y * sin(A)) * (y * cos(A) - x * sin(A)) / a ** 2 - (x * cos(A) + y * sin(A)) * (
            y * cos(A) - x * sin(A)) / b ** 2)


@cuda.jit
def kernel_make_elipse_image(I0: np.ndarray, I45: np.ndarray, I90: np.ndarray, I135: np.ndarray, results: np.ndarray,
                             max_iters=100000, tolerance=0.00001, black_threshold=1.0 / 255):
    x = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    y = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y

    if x >= I0.shape[0] or y >= I0.shape[1]:
        return

    i0 = I0[x, y]
    i45 = I45[x, y]
    i90 = I90[x, y]
    i135 = I135[x, y]

    s0 = (i0 + i45 + i90 + i135) / 2
    s1 = (i0 - i90)
    s2 = (i45 - i135)

    if s0 < black_threshold:
        results[x, y, 0] = 1.0 / 512
        results[x, y, 1] = 1.0 / 512
        results[x, y, 2] = 0
        results[x, y, 3] = False
        return

    dolp = sqrt(s1 ** 2 + s2 ** 2) / s0

    points = cuda.local.array((8, 2), dtype=numba.float32)

    points[0, 0] = numba.float32(0.0)
    points[0, 1] = i0
    points[1, 0] = numba.float32(0.0)
    points[1, 1] = -i0
    points[2, 0] = i45 * sin(np.pi / 4)
    points[2, 1] = i45 * cos(np.pi / 4)
    points[3, 0] = -i45 * sin(np.pi / 4)
    points[3, 1] = -i45 * cos(np.pi / 4)
    points[4, 0] = i90
    points[4, 1] = numba.float32(0.0)
    points[5, 0] = -i90
    points[5, 1] = numba.float32(0.0)
    points[6, 0] = i135 * sin(np.pi / 4)
    points[6, 1] = -i135 * cos(np.pi / 4)
    points[7, 0] = -i135 * sin(np.pi / 4)
    points[7, 1] = i135 * cos(np.pi / 4)

    a_candidate = 0
    b_candidate = 9999999

    for i in range(8):
        a_candidate = max(a_candidate, sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))
        b_candidate = min(b_candidate, sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

    a = b_candidate
    b = b_candidate
    A = 1 / 2 * atan2(s2, s1)

    lr = 0.005

    lmbda_a = 0.5
    lmbda_b = 0.5

    converged = True
    for i in range(max_iters):
        delta_a = 0
        delta_b = 0
        delta_A = 0

        for point in points:
            grad_a = diff_a(a, b, A, point[0], point[1])
            grad_b = diff_b(a, b, A, point[0], point[1])
            grad_A = diff_A(a, b, A, point[0], point[1])

            delta_a += clamp_grad(grad_a)
            delta_b += clamp_grad(grad_b)
            delta_A += clamp_grad(grad_A)

        # reg_a = lmbda_a * (a * (1 - dolp / 2)) ** 2
        # reg_b = lmbda_b * (b * (dolp / 2)) ** 2

        # reg_a = lmbda_a * a ** 2
        # reg_b = lmbda_b * b ** 2

        reg_a = lmbda_a * abs(a)
        reg_b = lmbda_b * abs(b)

        new_a = a - ((delta_a / len(points) + reg_a) * lr)
        new_b = b - ((delta_b / len(points) + reg_b) * lr)
        new_A = A - ((delta_A / len(points)) * lr)

        if isclose(a, new_a, tolerance) and isclose(b, new_b, tolerance) and isclose(A, new_A, tolerance):
            break

        a = new_a
        b = new_b
        A = new_A

    else:
        converged = False

    results[x, y, 0] = a
    results[x, y, 1] = b
    results[x, y, 2] = A
    results[x, y, 3] = converged


def make_elipse_image(I0: np.ndarray, I45: np.ndarray, I90: np.ndarray, I135: np.ndarray) -> list[list[Ellipse]]:
    results = np.ndarray(shape=(I0.shape[0], I0.shape[1], 4))
    threads_per_block = (16, 16)
    grid_x = math.ceil(I0.shape[0] / threads_per_block[0])
    grid_y = math.ceil(I0.shape[1] / threads_per_block[1])
    grid = (grid_x, grid_y)
    start = time.time()
    kernel_make_elipse_image[grid, threads_per_block](I0, I45, I90, I135, results, 100000, 0.00001, 1.0 / 255 * 10)
    end = time.time()
    print(f"Kernel make_elipse_image took {end - start:.5f} seconds")

    print(f"Did converge: {(results[:, :, 3] == True).sum()}")
    print(f"Did not converge: {(results[:, :, 3] == False).sum()}")

    image = []
    for i in tqdm(range(results.shape[0])):
        row = []
        for j in range(results.shape[1]):
            a = results[i, j, 0]
            b = results[i, j, 1]
            A = results[i, j, 2]
            converged = results[i, j, 3]

            if converged or True:
                ellipse = Ellipse(a=a, b=b, A=A)
            else:
                ellipse = Ellipse(a=0.00001, b=0.00001, A=0)

            # if i == 20 and j == 82:
            #     i0 = I0[i, j]
            #     i45 = I45[i, j]
            #     i90 = I90[i, j]
            #     i135 = I135[i, j]
            #
            #     points = np.ndarray((8, 2))
            #
            #     points[0, 0] = numba.float32(0.0)
            #     points[0, 1] = i0
            #     points[1, 0] = numba.float32(0.0)
            #     points[1, 1] = -i0
            #     points[2, 0] = i45 * sin(np.pi / 4)
            #     points[2, 1] = i45 * cos(np.pi / 4)
            #     points[3, 0] = -i45 * sin(np.pi / 4)
            #     points[3, 1] = -i45 * cos(np.pi / 4)
            #     points[4, 0] = i90
            #     points[4, 1] = numba.float32(0.0)
            #     points[5, 0] = -i90
            #     points[5, 1] = numba.float32(0.0)
            #     points[6, 0] = i135 * sin(np.pi / 4)
            #     points[6, 1] = -i135 * cos(np.pi / 4)
            #     points[7, 0] = -i135 * sin(np.pi / 4)
            #     points[7, 1] = i135 * cos(np.pi / 4)
            #
            #     plt.plot(*ellipse.samples(), linestyle='None', marker='v')
            #     plt.plot(points[:, 0], points[:, 1], linestyle='None', marker='x')
            #     plt.show()
            #     print()

            row.append(ellipse)

        image.append(row)

    return image


def render_ellipse(image: list[list[Ellipse]], angle: float) -> np.ndarray:
    rendered = []
    for i in range(len(image)):
        row = []
        for j in range(len(image[0])):
            row.append(image[i][j].intensity(angle))
        rendered.append(row)

    im = np.array(rendered)
    return im


if __name__ == '__main__':
    # e = ellipse_from_gradient_descent(
    #     [(-5, 1), (5, 1), (0, 0), (0, 2), (-3, 0.2), (3, 0.2), (-3, 1.8), (3, 1.8), (1, 0.02), (-1, 0.02), (1, 1.98),
    #      (-1, 1.98)])
    # print(e)

    # I0 = cv2.imread("../../data/video4-frame1/00001_000.png")[795:850, 750:850, 0].astype(np.float32) / 255
    # I45 = cv2.imread("../../data/video4-frame1/00001_045.png")[795:850, 750:850, 0].astype(np.float32) / 255
    # I90 = cv2.imread("../../data/video4-frame1/00001_090.png")[795:850, 750:850, 0].astype(np.float32) / 255
    # I135 = cv2.imread("../../data/video4-frame1/00001_135.png")[795:850, 750:850, 0].astype(np.float32) / 255

    # I0 = cv2.imread("../../data/video4-frame1/00001_000.png")[600:900, 600:900, 0].astype(np.float32) / 255
    # I45 = cv2.imread("../../data/video4-frame1/00001_045.png")[600:900, 600:900, 0].astype(np.float32) / 255
    # I90 = cv2.imread("../../data/video4-frame1/00001_090.png")[600:900, 600:900, 0].astype(np.float32) / 255
    # I135 = cv2.imread("../../data/video4-frame1/00001_135.png")[600:900, 600:900, 0].astype(np.float32) / 255

    I0 = cv2.imread("../../data/video4-frame1/00001_000.png")[:, :, 0].astype(np.float32) / 255
    I45 = cv2.imread("../../data/video4-frame1/00001_045.png")[:, :, 0].astype(np.float32) / 255
    I90 = cv2.imread("../../data/video4-frame1/00001_090.png")[:, :, 0].astype(np.float32) / 255
    I135 = cv2.imread("../../data/video4-frame1/00001_135.png")[:, :, 0].astype(np.float32) / 255

    plt.imshow(I0, cmap='gray')
    plt.show()

    s0 = (I0 + I45 + I90 + I135) / 2
    s1 = I0 - I90
    s2 = I45 - I135
    dolp = np.nan_to_num(np.sqrt(s1 ** 2 + s2 ** 2) / s0)

    ellipses = make_elipse_image(I0, I45, I90, I135)

    print(ellipses[0][0])

    I = np.dstack([I0, I45, I90, I135])

    im0 = np.nan_to_num(render_ellipse(ellipses, np.pi / 2))
    im45 = np.nan_to_num(render_ellipse(ellipses, np.pi / 4))
    im90 = np.nan_to_num(render_ellipse(ellipses, 0))
    im135 = np.nan_to_num(render_ellipse(ellipses, np.pi / 4 * 3))

    im = np.dstack([im0, im45, im90, im135])

    mae = np.abs(im - I).mean()

    print(f"MAE = {mae}")

    exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.mp4', fourcc, 30, (I0.shape[1], I0.shape[0]))

    images = []
    angle = 0
    while angle < np.pi * 2:
        im = render_ellipse(ellipses, angle)
        images.append(im)
        angle += 0.05

    imstack = np.dstack(images)
    imstack = np.nan_to_num(imstack)
    # imstack_min = imstack.min()
    # imstack_max = imstack.max()
    # imstack_normalized = (imstack - imstack_min) / (imstack_max - imstack_min)
    imstack_normalized = np.clip(imstack, 0, 1)

    frames = []
    for i in range(imstack_normalized.shape[2]):
        frames.append(np.repeat((imstack_normalized[:, :, i] * 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2))

    for _ in range(10):
        plt.imshow(frames[0])
        plt.show()
        for frame in frames:
            video.write(frame)

    video.release()
