from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def load_4channel_image(filename):
    return np.load(filename, allow_pickle=True)


def image_4channel_to_stokes(image):
    if image.shape[2] == 3:
        image = np.dstack([image, np.zeros((image.shape[0], image.shape[1]))])
    I000 = image[:, :, 0]
    I045 = image[:, :, 1]
    I090 = image[:, :, 2]
    I135 = image[:, :, 3]

    S0 = (I000 + I045 + I090 + I135) / 2
    S1 = I000 - I090
    S2 = I045 - I135

    return np.dstack((S0, S1, S2))


def make_stokes_mask(S):
    S0 = S[:, :, 0]
    threshold = 0.001
    threshold = 0
    return (S0 >= threshold).astype(np.uint8) * 255


# https://www.fiberoptics4sale.com/blogs/wave-optics/104730310-mueller-matrices-for-polarizing-elements
def mueller_for_linear_polarizer(angle):
    return np.array([
        [1, np.cos(2 * angle), 0],
        [np.cos(2 * angle), 1, 0],
        [0, 0, np.sin(2 * angle)],
    ]) * 0.5


def solve_mueller_for_pixel(s000_sn, s045_sn, s090_sn, s135_sn):
    M000 = mueller_for_linear_polarizer(0)
    M045 = mueller_for_linear_polarizer(np.pi / 4)
    M090 = mueller_for_linear_polarizer(np.pi / 2)
    M135 = mueller_for_linear_polarizer(np.pi / 4 * 3)

    unpoliarized_stokes = np.array([1, 0, 0])

    Sin_000 = M000 @ unpoliarized_stokes
    Sin_045 = M045 @ unpoliarized_stokes
    Sin_090 = M090 @ unpoliarized_stokes
    Sin_135 = M135 @ unpoliarized_stokes

    Sin = np.transpose(np.array([Sin_000, Sin_045, Sin_090, Sin_135]))
    Sout = np.transpose(np.array([s000_sn, s045_sn, s090_sn, s135_sn]))

    Sin_inverse = np.linalg.pinv(Sin)

    M = Sout @ Sin_inverse

    return M


def solve_muellers_for_image(S000_Sn, S045_Sn, S090_Sn, S135_Sn, mask):
    mueller_image = [[None for _ in range(S000_Sn.shape[1])] for _ in range(S000_Sn.shape[0])]
    for i in tqdm(range(S000_Sn.shape[0])):
        for j in range(S000_Sn.shape[1]):
            if mask[i, j] > 0:
                mueller_image[i][j] = solve_mueller_for_pixel(S000_Sn[i, j], S045_Sn[i, j], S090_Sn[i, j],
                                                              S135_Sn[i, j])

    return mueller_image


def make_mueller_image_validity_map(mueller_im):
    validity = np.zeros((len(mueller_im), len(mueller_im[0]), 3))
    for i in range(len(mueller_im)):
        for j in range(len(mueller_im[0])):
            if mueller_im[i][j] is not None:
                M = mueller_im[i][j]
                constraint1 = M.max() <= 1 and M.min() >= -1
                constraint2 = M[0, 0] == M.max()
                constraint3 = (M[0, 0] + np.sqrt(M[0, 1] ** 2 + M[0, 2] ** 2)) <= 1
                constraint4 = np.linalg.det(M) >= 0
                if constraint1 and constraint2 and constraint3 and constraint4:
                    validity[i, j] = [0, 1, 0]
                else:
                    validity[i, j] = [1, 0, 0]

    return validity


def simplify_mueller_image(mueller_im, x, y):
    im = np.zeros((len(mueller_im), len(mueller_im[0])))
    for i in range(len(mueller_im)):
        for j in range(len(mueller_im[0])):
            if mueller_im[i][j] is None:
                im[i, j] = 0
            else:
                im[i, j] = mueller_im[i][j][x, y]

    return im


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--neutral", type=str, required=True)
    parser.add_argument("--i000", type=str, required=True)
    parser.add_argument("--i045", type=str, required=True)
    parser.add_argument("--i090", type=str, required=True)
    parser.add_argument("--i135", type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    neutral_path = args.neutral
    i000_path = args.i000
    i045_path = args.i045
    i090_path = args.i090
    i135_path = args.i135


    # N = load_4channel_image(neutral_path)
    # Sn = image_4channel_to_stokes(N)
    # im_000 = load_4channel_image(i000_path)
    # S000 = image_4channel_to_stokes(im_000)
    # im_045 = load_4channel_image(i045_path)
    # S045 = image_4channel_to_stokes(im_045)
    # im_090 = load_4channel_image(i090_path)
    # S090 = image_4channel_to_stokes(im_090)
    # im_135 = load_4channel_image(i135_path)
    # S135 = image_4channel_to_stokes(im_135)

    Sn = load_4channel_image(neutral_path)
    S000 = load_4channel_image(i000_path)
    S045 = load_4channel_image(i045_path)
    S090 = load_4channel_image(i090_path)
    S135 = load_4channel_image(i135_path)

    S000_Sn = S000 # - Sn
    S045_Sn = S045 # - Sn
    S090_Sn = S090 # - Sn
    S135_Sn = S135 # - Sn

    total_mask = make_stokes_mask(S000_Sn) & make_stokes_mask(S045_Sn) & make_stokes_mask(S090_Sn) & make_stokes_mask(
        S135_Sn)

    plt.imshow(total_mask, cmap='gray')
    plt.show()

    muellers_im = solve_muellers_for_image(S000_Sn, S045_Sn, S090_Sn, S135_Sn, total_mask)
    validity = make_mueller_image_validity_map(muellers_im)
    inteisity_changes = simplify_mueller_image(muellers_im, 0, 0)
    plt.imshow(validity, cmap='gray')
    plt.show()
    plt.imshow(inteisity_changes, cmap='gray')
    plt.show()
