from __future__ import annotations
import json
import math
from dataclasses import dataclass
from functools import cmp_to_key

from src.config.config import Config
from src.pipeline import function

@dataclass
class Point3:
    x: float
    y: float
    z: float

    def distance(self, other: Point3):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z

        return math.sqrt(dx**2 + dy**2 + dz**2)



def load_colmap_feature_points(filename: str) -> list[Point3]:
    points = []
    with open(filename, "rt") as f:
        for line in f.readlines():
            if line.startswith("#"): continue

            parts = line.split(" ")
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            point = Point3(x, y, z)
            points.append(point)

    return points

def load_colmap_camera_origins(filename: str) -> list[Point3]:
    points = []
    with open(filename, "rt") as f:
        transforms = json.load(f)
        for frame in transforms["frames"]:
            tm = frame["transform_matrix"]
            x = float(tm[0][3])
            y = float(tm[1][3])
            z = float(tm[2][3])
            point = Point3(x, y, z)
            points.append(point)

    return points


def _estimate_near_far(config: Config):
    colmap_dir = config.options["paths"]["colmap_dir"]

    camera_origins = load_colmap_camera_origins(colmap_dir + "/transforms.json")
    feature_points = load_colmap_feature_points(colmap_dir + "/sparse/0/points3D.txt")

    distances: list[float] = []
    for co in camera_origins:
        for p in feature_points:
            distances.append(co.distance(p))

    distances = sorted(distances)

    near = distances[0]
    skip_outliers = len(distances) - len(distances) // 20
    far = distances[skip_outliers] * 1.2
    data = {"near": near, "far": far}
    with open(colmap_dir + "/nearfar.json", "w") as f:
        json.dump(data, f)



def estimate_near_far(config: Config):
    return function(lambda: _estimate_near_far(config))
