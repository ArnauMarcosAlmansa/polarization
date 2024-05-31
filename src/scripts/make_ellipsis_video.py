from math import cos, sin
from enum import Enum
from typing import Callable


class PointRelationship(Enum):
    INSIDE = 0
    BELONGS = 1
    OUTSIDE = 2


def diff_a(a: float, b: float, A: float, h: float, k: float) -> Callable[[float, float], float]:
    def grad_a(x: float, y: float) -> float:
        return -2 * ((h - x) * cos(A) + (k - y) * sin(A)) ** 2 / a ** 3

    return grad_a


def diff_b(a: float, b: float, A: float, h: float, k: float) -> Callable[[float, float], float]:
    def grad_b(x: float, y: float) -> float:
        return -2 * ((k - y) * cos(A) - (h - x) * sin(A)) ** 2 / b ** 3

    return grad_b


def diff_A(a: float, b: float, A: float, h: float, k: float) -> Callable[[float, float], float]:
    def grad_A(x: float, y: float) -> float:
        return 2 * ((h - x) * cos(A) + (k - y) * sin(A)) * ((k - y) * cos(A) - (h - x) * sin(A)) / a ** 2 - 2 * (
                (h - x) * cos(A) + (k - y) * sin(A)) * ((k - y) * cos(A) - (h - x) * sin(A)) / b ** 2

    return grad_A


def diff_h(a: float, b: float, A: float, h: float, k: float) -> Callable[[float, float], float]:
    def grad_h(x: float, y: float) -> float:
        return 2 * ((h - x) * cos(A) + (k - y) * sin(A)) * cos(A) / a ** 2 - 2 * (
                (k - y) * cos(A) - (h - x) * sin(A)) * sin(A) / b ** 2

    return grad_h


def diff_k(a: float, b: float, A: float, h: float, k: float) -> Callable[[float, float], float]:
    def grad_k(x: float, y: float) -> float:
        return 2 * ((k - y) * cos(A) - (h - x) * sin(A)) * cos(A) / b ** 2 + 2 * (
                (h - x) * cos(A) + (k - y) * sin(A)) * sin(A) / a ** 2

    return grad_k


class Ellipse:
    h: float
    k: float
    A: float
    a: float
    b: float

    def __init__(self, h: float, k: float, A: float, a: float, b: float):
        self.h = h
        self.k = k
        self.A = A
        self.a = a
        self.b = b

    def _compute_right_term(self, x: float, y: float) -> float:
        cx = x - self.h
        cy = y - self.k

        u1 = ((cx * cos(self.A)) + (cy * sin(self.A))) ** 2
        u2 = ((cx * sin(self.A)) - (cy * cos(self.A))) ** 2
        d1 = self.a ** 2
        d2 = self.b ** 2

        right = u1 / d1 + u2 / d2
        return right

    def point_related_to_ellipsis(self, x: float, y: float) -> PointRelationship:
        right = self._compute_right_term(x, y)

        match right:
            case _ if right < 1:
                return PointRelationship.INSIDE
            case _ if right == 1:
                return PointRelationship.BELONGS
            case _ if right > 1:
                return PointRelationship.OUTSIDE

    def __str__(self):
        return f"Ellipse(a={self.a}, b={self.b}, A={self.A}, h={self.h}, k={self.k})"


def ellipse_gradient_descent(points: list[tuple[float, float]]) -> Ellipse:
    # state = {"a": 1, "b": 1, "A": 0, "h": 0, "k": 0}
    state = {"a": 5, "b": 1, "A": 0, "h": 0, "k": 1}
    lr = 0.00005
    for _ in range(100000):
        for point in points:
            grads = [
                diff_a(**state),
                diff_b(**state),
                diff_A(**state),
                diff_h(**state),
                diff_k(**state),
            ]
            for grad, v in zip(grads, ["a", "b", "A", "h", "k"]):
                state[v] -= grad(point[0], point[1]) * lr

    return Ellipse(**state)


if __name__ == '__main__':
    e = Ellipse(0, 1, 0, 5, 2)
    print(e.point_related_to_ellipsis(5, 1))

    e = ellipse_gradient_descent(
        [(-5, 1), (5, 1), (0, 0), (0, 2), (-3, 0.2), (3, 0.2), (-3, 1.8), (3, 1.8), (1, 0.02), (-1, 0.02), (1, 1.98),
         (-1, 1.98)])
    print(e)
