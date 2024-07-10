import Imath
import OpenEXR
import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_stokes_image(index: int):
    file = f"../../data/video4-stokes/{index:05}.exr"
    exrfile = OpenEXR.InputFile(file)
    raw_S0_bytes = exrfile.channel('S0', Imath.PixelType(Imath.PixelType.FLOAT))
    raw_S1_bytes = exrfile.channel('S1', Imath.PixelType(Imath.PixelType.FLOAT))
    raw_S2_bytes = exrfile.channel('S2', Imath.PixelType(Imath.PixelType.FLOAT))
    raw_S3_bytes = exrfile.channel('S3', Imath.PixelType(Imath.PixelType.FLOAT))
    S0_vector = np.frombuffer(raw_S0_bytes, dtype=np.float32)
    S1_vector = np.frombuffer(raw_S1_bytes, dtype=np.float32)
    S2_vector = np.frombuffer(raw_S2_bytes, dtype=np.float32)
    S3_vector = np.frombuffer(raw_S3_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    S0 = np.reshape(S0_vector, (height, width))
    S1 = np.reshape(S1_vector, (height, width))
    S2 = np.reshape(S2_vector, (height, width))
    S3 = np.reshape(S3_vector, (height, width))
    return np.dstack([S0, S1, S2, S3])


def stokes_image_to_intensity(stokes_image, theta, phi):
    S0 = stokes_image[:, :, 0]
    S1 = stokes_image[:, :, 1]
    S2 = stokes_image[:, :, 2]
    S3 = stokes_image[:, :, 3]

    I = 0.5 * (S0 + S1 * np.cos(2 * theta) + S2 * np.cos(phi) * np.sin(2 * theta) + S3 * np.sin(phi) * np.sin(
        2 * theta))

    return I


if __name__ == '__main__':

    S_im = load_stokes_image(1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video2.mp4', fourcc, 30, (S_im.shape[1], S_im.shape[0]))

    images = []
    angle = 0
    while angle < np.pi * 2:
        im = stokes_image_to_intensity(S_im, angle, 0)
        images.append(im)
        angle += 0.05

    imstack = np.dstack(images)
    imstack = np.nan_to_num(imstack)
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