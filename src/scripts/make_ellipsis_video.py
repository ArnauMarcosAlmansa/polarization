import random
from math import cos, sin, sqrt, atan2, isclose
from enum import Enum
from typing import Callable, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

StokesVector = tuple[float, float, float, float]


class PointRelationship(Enum):
    INSIDE = 0
    BELONGS = 1
    OUTSIDE = 2


def g(a: float, b: float, A: float):
    def f(x: float, y: float) -> float:
        return ((x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) ** 2

    return f


def diff_a(a: float, b: float, A: float) -> Callable[[float, float], float]:
    def grad_a(x: float, y: float) -> float:
        return -4 * (x * cos(A) + y * sin(A)) ** 2 * (
                (x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) / a ** 3

    return grad_a


def diff_b(a: float, b: float, A: float) -> Callable[[float, float], float]:
    def grad_b(x: float, y: float) -> float:
        return -4 * (y * cos(A) - x * sin(A)) ** 2 * (
                (x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) / b ** 3

    return grad_b


def diff_A(a: float, b: float, A: float) -> Callable[[float, float], float]:
    def grad_A(x: float, y: float) -> float:
        return 4 * ((x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) * (
                (x * cos(A) + y * sin(A)) * (y * cos(A) - x * sin(A)) / a ** 2 - (x * cos(A) + y * sin(A)) * (
                y * cos(A) - x * sin(A)) / b ** 2)

    return grad_A


class Ellipse:
    A: float
    a: float
    b: float

    def __init__(self, a: float, b: float, A: float):
        self.a = a
        self.b = b
        self.A = A

    def _compute_right_term(self, x: float, y: float) -> float:
        u1 = ((x * cos(self.A)) + (y * sin(self.A))) ** 2
        u2 = ((x * sin(self.A)) - (y * cos(self.A))) ** 2
        d1 = self.a ** 2
        d2 = self.b ** 2

        right = u1 / d1 + u2 / d2
        return right

    def point_related_to_ellipsis(self, x: float, y: float) -> PointRelationship:
        right = self._compute_right_term(x, y)

        match right:
            case _ if isclose(right, 1):
                return PointRelationship.BELONGS
            case _ if right < 1:
                return PointRelationship.INSIDE
            case _ if right > 1:
                return PointRelationship.OUTSIDE

    def _y(self, x: float) -> tuple[float, float]:
        a = self.a
        b = self.b
        A = self.A

        y1 = -(sqrt(a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2 - (
                cos(A) ** 4 + 2 * cos(A) ** 2 * sin(A) ** 2 + sin(A) ** 4) * x ** 2) * a * b - (
                       a ** 2 * cos(A) * sin(A) - b ** 2 * cos(A) * sin(A)) * x) / (
                     a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2)

        y2 = (sqrt(a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2 - (
                cos(A) ** 4 + 2 * cos(A) ** 2 * sin(A) ** 2 + sin(A) ** 4) * x ** 2) * a * b + (
                      a ** 2 * cos(A) * sin(A) - b ** 2 * cos(A) * sin(A)) * x) / (
                     a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2)
        return y1, y2

    def samples(self):
        x_samples: list[float] = list()
        y_samples: list[float] = list()
        major_axis = max(self.a, self.b) * cos(self.A)
        x = -major_axis
        while x <= major_axis:
            try:
                y1, y2 = self._y(x)
                if self.point_related_to_ellipsis(x, y1) == PointRelationship.BELONGS:
                    x_samples.append(x)
                    x_samples.append(x)
                    y_samples.append(y1)
                    y_samples.append(y2)
            except ValueError:
                pass
            x += 0.01

        return x_samples, y_samples

    def intensity(self, theta: float):
        a = self.a
        b = self.b
        A = self.A

        x = -(sqrt(-cos(A) ** 4 - 2 * cos(A) ** 2 * sin(A) ** 2 - sin(A) ** 4 + (
                    a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2) * cos(theta) ** 2 + 2 * (
                               a ** 2 * cos(A) * sin(A) - b ** 2 * cos(A) * sin(A)) * cos(theta) * sin(theta) + (
                               b ** 2 * cos(A) ** 2 + a ** 2 * sin(A) ** 2) * sin(theta) ** 2) * a * b * sin(theta) - (
                          a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2) * cos(theta) - (
                          a ** 2 * cos(A) * sin(A) - b ** 2 * cos(A) * sin(A)) * sin(theta)) / (
                        (a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2) * cos(theta) ** 2 + 2 * (
                            a ** 2 * cos(A) * sin(A) - b ** 2 * cos(A) * sin(A)) * cos(theta) * sin(theta) + (
                                    b ** 2 * cos(A) ** 2 + a ** 2 * sin(A) ** 2) * sin(theta) ** 2)
        y = (sqrt(-cos(A) ** 4 - 2 * cos(A) ** 2 * sin(A) ** 2 - sin(A) ** 4 + (
                    a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2) * cos(theta) ** 2 + 2 * (
                              a ** 2 * cos(A) * sin(A) - b ** 2 * cos(A) * sin(A)) * cos(theta) * sin(theta) + (
                              b ** 2 * cos(A) ** 2 + a ** 2 * sin(A) ** 2) * sin(theta) ** 2) * a * b * cos(theta) + (
                         a ** 2 * cos(A) * sin(A) - b ** 2 * cos(A) * sin(A)) * cos(theta) + (
                         b ** 2 * cos(A) ** 2 + a ** 2 * sin(A) ** 2) * sin(theta)) / (
                        (a ** 2 * cos(A) ** 2 + b ** 2 * sin(A) ** 2) * cos(theta) ** 2 + 2 * (
                            a ** 2 * cos(A) * sin(A) - b ** 2 * cos(A) * sin(A)) * cos(theta) * sin(theta) + (
                                    b ** 2 * cos(A) ** 2 + a ** 2 * sin(A) ** 2) * sin(theta) ** 2)
        return sqrt(x ** 2 + y ** 2)

    def __str__(self):
        return f"Ellipse(a={self.a}, b={self.b}, A={self.A})"

def centered_ellipse_from_gradient_descent(points: list[tuple[float, float]]) -> Ellipse:
    state = {"a": 1, "b": 1, "A": 0}
    # state = dict(a=1.4106914984369574 * 2, b=0.09974716155205582 * 2, A=0.7803478301047874 * 2, h=0, k=0)
    lr = 0.005
    for i in range(50000):
        if i % 10 == 0:
            cost = 0
            for p in points:
                cost += g(**state)(p[0], p[1])
            print(cost / len(points))

        point = random.choice(points)

        grad_a = diff_a(**state)
        grad_b = diff_b(**state)
        grad_A = diff_A(**state)

        state["a"] -= grad_a(point[0], point[1]) * lr
        state["b"] -= grad_b(point[0], point[1]) * lr
        state["A"] -= grad_A(point[0], point[1]) * lr

        if i % 10 == 0:
            print(grad_a(point[0], point[1]), grad_b(point[0], point[1]), grad_A(point[0], point[1]))
            print(state)

    return Ellipse(**state)


def ellipse_from_stokes_vector(stokes_vector: StokesVector) -> Ellipse:
    s0, s1, s2, s3 = stokes_vector
    p = sqrt((s1 ** 2 + s2 ** 2 + s3 ** 2) / s0)
    A = atan2(s2, s1) / 2
    a = sqrt(s0 * (1 + p))
    b = sqrt(s0 * (1 - p))

    return Ellipse(a, b, A)


def make_elipse_image(I0: np.ndarray, I45: np.ndarray, I90: np.ndarray, I135: np.ndarray) -> list[list[Ellipse]]:
    image = []
    for i in range(I0.shape[0]):
        row = []
        for j in range(I0.shape[1]):
            i0: float = I0[i, j]
            i45: float = I45[i, j]
            i90: float = I90[i, j]
            i135: float = I135[i, j]
            s0 = (i0 + i45 + i90 + i135) / 2
            s1 = i0 - i90
            s2 = i45 - i135

            row.append(ellipse_from_stokes_vector((s0, s1, s2, 0)))

        image.append(row)

    return image


def render_ellipse(image: list[list[Ellipse]], angle: float) -> np.ndarray:
    rendered = []
    for i in range(len(image)):
        row = []
        for j in range(len(image[0])):
            row.append(image[i][j].intensity(angle))
        rendered.append(row)

    im = np.matrix(rendered)
    return im


if __name__ == '__main__':

    # e = ellipse_from_gradient_descent(
    #     [(-5, 1), (5, 1), (0, 0), (0, 2), (-3, 0.2), (3, 0.2), (-3, 1.8), (3, 1.8), (1, 0.02), (-1, 0.02), (1, 1.98),
    #      (-1, 1.98)])
    # print(e)

    e = centered_ellipse_from_gradient_descent(
        [
            (0.993, 1.002),
            (-0.993, -1.002),
            (0, -0.141),
            (0, 0.141),
            (0.141, 0),
            (-0.141, 0),
        ]
    )
    print(e)
    x, y = e.samples()
    plt.plot(x, y, linestyle='None', marker='o')

    # I0 = cv2.imread("../../data/video4-frame1/00001_000.png")[:500, :500, 0].astype(np.float32) / 255
    # I45 = cv2.imread("../../data/video4-frame1/00001_045.png")[:500, :500, 0].astype(np.float32) / 255
    # I90 = cv2.imread("../../data/video4-frame1/00001_090.png")[:500, :500, 0].astype(np.float32) / 255
    # I135 = cv2.imread("../../data/video4-frame1/00001_135.png")[:500, :500, 0].astype(np.float32) / 255
    #
    # ellipses = make_elipse_image(I0, I45, I90, I135)
    # im = render_ellipse(ellipses, 0)
    # plt.imshow(im, cmap='gray')
    # plt.show()
    # im = render_ellipse(ellipses, np.pi / 2)
    # plt.imshow(im, cmap='gray')
    # plt.show()

    e = ellipse_from_stokes_vector((1, 0.01, 0.99, 0))
    print(e)

    plt.plot(*e.samples(), linestyle='None', marker='x')
    plt.show()
