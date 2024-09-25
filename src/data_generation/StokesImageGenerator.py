import math

import numpy as np

from src.load_polarimetric import PolarimetricImage


def angle_between_vectors(vec1, vec2):
    return math.acos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def quaternion_to_rotation_matrix(quaternion: np.array) -> np.array:
    # Extract the values from Q
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def quaternion_multiply(q, r):
    """ Multiply two quaternions q and r. """
    w0, x0, y0, z0 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w1, x1, y1, z1 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ]).T

class StokesImageGenerator:
    def _intensities_to_stokes(self, I000: np.ndarray, I045: np.ndarray, I090: np.ndarray, I135: np.ndarray) -> np.ndarray:
        S0 = (I000 + I045 + I090 + I135) / 2
        S1 = (I000 - I090)
        S2 = (I045 - I135)
        return np.dstack([S0, S1, S2])

    def _get_rotation_correction_matrix(self, pose: np.ndarray) -> np.ndarray:
        ground_normal = np.array([0, -1, 0])
        world_up = np.array([0, 1, 0])
        camera_up = world_up @ pose[0:3, 0:3]
        world_forward = np.array([0, 0, -1])
        camera_forward = world_forward @ pose[0:3, 0:3]
        angle = angle_between_vectors(camera_up, ground_normal)

        q = np.array([
            np.cos(angle / 2),
            camera_forward[0]*np.sin(angle / 2),
            camera_forward[1]*np.sin(angle / 2),
            camera_forward[2]*np.sin(angle / 2),
        ])

        return quaternion_to_rotation_matrix(q)

    def generate_stokes_image(self, pose: np.ndarray, polarimetric_image: PolarimetricImage) -> np.ndarray:
        I000 = polarimetric_image.I0
        I045 = polarimetric_image.I45
        I090 = polarimetric_image.I90
        I135 = polarimetric_image.I135

        raw_stokes_image = self._intensities_to_stokes(I000, I045, I090, I135)

        rotation_correction_matrix = self._get_rotation_correction_matrix(pose)

        stokes_image = np.einsum('ij,hwi->hwj', rotation_correction_matrix, raw_stokes_image)

        return stokes_image



if __name__ == "__main__":
    generator = StokesImageGenerator()

    generator.generate_stokes_image()
