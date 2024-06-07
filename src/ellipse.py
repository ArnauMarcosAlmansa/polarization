from math import cos, sin, sqrt, atan2, isclose, asin, tan, atan
from enum import Enum
from typing import Callable, Tuple, Self


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def clamp(min_value, max_value):
    def clamp_fn(value):
        return max(min(value, max_value), min_value)

    return clamp_fn


clamp_grad = clamp(-0.25, 0.25)

StokesVector = tuple[float, float, float, float]

class PointRelationship(Enum):
    INSIDE = 0
    BELONGS = 1
    OUTSIDE = 2


def g(a: float, b: float, A: float):
    def f(x: float, y: float) -> float:
        return ((x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) ** 2

    return f


def diff_a(a: float, b: float, A: float, x: float, y: float) -> float:
    return -4 * (x * cos(A) + y * sin(A)) ** 2 * (
            (x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) / a ** 3


def diff_b(a: float, b: float, A: float, x: float, y: float) -> float:
    return -4 * (y * cos(A) - x * sin(A)) ** 2 * (
            (x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) / b ** 3


def diff_A(a: float, b: float, A: float, x: float, y: float) -> float:
    return 4 * ((x * cos(A) + y * sin(A)) ** 2 / a ** 2 + (y * cos(A) - x * sin(A)) ** 2 / b ** 2 - 1) * (
            (x * cos(A) + y * sin(A)) * (y * cos(A) - x * sin(A)) / a ** 2 - (x * cos(A) + y * sin(A)) * (
            y * cos(A) - x * sin(A)) / b ** 2)


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

        theta -= A
        r = (a * b) / sqrt(a ** 2 * sin(theta) ** 2 + b ** 2 * cos(theta) ** 2)
        return r

    @staticmethod
    def from_points_using_gd(points: list[tuple[float, float]], tolerance=0.00001, max_iters=1000000):
        a = 1
        b = 1
        A = 0

        lr = 0.005
        for i in range(max_iters):
            delta_a = 0
            delta_b = 0
            delta_A = 0

            for point in points:
                grad_a = diff_a(a=a, b=b, A=A, x=point[0], y=point[1])
                grad_b = diff_b(a=a, b=b, A=A, x=point[0], y=point[1])
                grad_A = diff_A(a=a, b=b, A=A, x=point[0], y=point[1])

                delta_a += clamp_grad(grad_a)
                delta_b += clamp_grad(grad_b)
                delta_A += clamp_grad(grad_A)

            new_a = a - delta_a / len(points) * lr
            new_b = b - delta_b / len(points) * lr
            new_A = A - delta_A / len(points) * lr

            if isclose(a, new_a, abs_tol=tolerance) and isclose(b, new_b, abs_tol=tolerance) and isclose(A, new_A,
                                                                                                         abs_tol=tolerance):
                break

            a = new_a
            b = new_b
            A = new_A

        else:
            print("DID NOT CONVERGE")

        return Ellipse(a, b, A)

    def __str__(self):
        return f"Ellipse(a={self.a}, b={self.b}, A={self.A})"
