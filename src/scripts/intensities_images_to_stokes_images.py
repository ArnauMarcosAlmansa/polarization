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

    return I0, I45, I90, I135


def convert_to_stokes_image(I0, I45, I90, I135):
    S0 = (I0 + I45 + I90 + I135) / 2
    S1 = I0 - I90
    S2 = I45 - I135
    S3 = np.zeros_like(I0)

    return np.dstack([S0, S1, S2, S3])

def save_stokes_image(S_im, index: int):
    index_str = f"{index:05}"
    size = S_im.shape
    exrHeader = OpenEXR.Header(size[1], size[0])

    exrHeader['channels'] = {
        "S0": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT), 1, 1),
        "S1": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT), 1, 1),
        "S2": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT), 1, 1),
        "S3": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT), 1, 1),
    }

    exrOut = OpenEXR.OutputFile(f"../../data/video4-stokes/{index_str}.exr", exrHeader)
    S0 = (S_im[:, :, 0]).astype(np.float32).tobytes()
    S1 = (S_im[:, :, 1]).astype(np.float32).tobytes()
    S2 = (S_im[:, :, 2]).astype(np.float32).tobytes()
    S3 = (S_im[:, :, 3]).astype(np.float32).tobytes()
    exrOut.writePixels({
        'S0': S0,
        'S1': S1,
        'S2': S2,
        'S3': S3,
    })
    exrOut.close()


if __name__ == '__main__':
    for index in range(1, 201):
        I0, I45, I90, I135 = load_intensity_images(index)
        sim = convert_to_stokes_image(I0, I45, I90, I135)
        save_stokes_image(sim, index)
