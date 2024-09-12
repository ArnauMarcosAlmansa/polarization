from math import log10


def psnr(mse, max):
    return 10 * log10(max ** 2 / mse)