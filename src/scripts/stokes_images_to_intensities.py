import Imath
import OpenEXR
import cv2
import numpy as np


def load_intensity_images(index: int):
    index_str = f"{index:05}"
    I0 = cv2.imread(f'../../data/video4/{index_str}_045.png', cv2.IMREAD_GRAYSCALE)
    I45 = cv2.imread(f'../../data/video4/{index_str}_045.png', cv2.IMREAD_GRAYSCALE)
    I90 = cv2.imread(f'../../data/video4/{index_str}_090.png', cv2.IMREAD_GRAYSCALE)
    I135 = cv2.imread(f'../../data/video4/{index_str}_135.png', cv2.IMREAD_GRAYSCALE)

    I0 = I0.astype(np.float32) / 255
    I45 = I45.astype(np.float32) / 255
    I90 = I90.astype(np.float32) / 255
    I135 = I135.astype(np.float32) / 255

    return np.dstack([I0, I45, I90, I135])


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
    loss = 0
    for i in range(1, 201):
        I = load_intensity_images(i)
        S_im = load_stokes_image(i)

        I0 = stokes_image_to_intensity(S_im, 0, 0)
        I45 = stokes_image_to_intensity(S_im, np.pi / 4, 0)
        I90 = stokes_image_to_intensity(S_im, np.pi / 2, 0)
        I135 = stokes_image_to_intensity(S_im, 3 * np.pi / 4, 0)

        I_reconstructed = np.dstack([I0, I45, I90, I135])

        loss += np.square(I - I_reconstructed).mean()

    print(f"loss = {(loss / 200):5f}")
