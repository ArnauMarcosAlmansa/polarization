from math import sin, cos, pi

import numpy as np
from matplotlib import pyplot as plt

from make_ellipsis_video import Ellipse, ellipse_from_stokes_vector

I = np.array([1, 0.1, 0.1, 0])


K = np.array([
    [0.5, 0.5, 0.5, 0.5],
    [1, 0, -1, 0],
    [0, 1, 0, -1],
    [0, 0, 0, 0],
])

S = K @ I

S = np.array([1, 0.01, 0.99, 0])

print(S)
theta = pi / 4
R = np.array([
    [1, 0, 0, 0],
    [0, cos(2 * theta), sin(2 * theta), 0],
    [0, -sin(2 * theta), cos(2 * theta), 0],
    [0, 0, 0, 1],
])

S2 = R @ S  # @ R.T

print(S2)

revK = np.array([
    [0.5, 0.5, 0, 0],
    [0.5, -0.5, 0, 0],
    [0.5, 0, 0.5, 0],
    [0.5, 0, -0.5, 0],
])

I2 = revK @ S2

print(I2)

ellipse = ellipse_from_stokes_vector(tuple(S))
plt.plot(*ellipse.samples(), linestyle='None', marker='o')
plt.show()

ellipse = ellipse_from_stokes_vector(tuple(S2))
plt.plot(*ellipse.samples(), linestyle='None', marker='x')
plt.show()
