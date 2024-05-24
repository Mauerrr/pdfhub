import torch
from scipy.spatial.transform import Rotation
import numpy as np


def quaternion_to_heading(quaternion_tensor):
    """
    Converts a quaternion to heading angle
    """
    # Convert to numpy array for using scipy's Rotation
    quaternion_np = quaternion_tensor.numpy()
    # Convert quaternion representation to rotation matrices
    rotation_matrices = Rotation.from_quat(quaternion_np).as_matrix()
    # Calculate Euler angle representation corresponding to rotation matrices
    euler_angles = Rotation.from_matrix(rotation_matrices).as_euler('xyz', degrees=True)
    # extract yaw as heading
    heading_tensor = torch.tensor(euler_angles[:, 2], dtype=torch.float64)

    return heading_tensor.unsqueeze(1)


def normalize_angle(angle):
    """
    Map a angle in range [-π, π]
    :param angle: any angle as float
    :return: normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def convert_absolute_to_relative_se2_array(origin, state_se2_array):
    """
        convert x, y, theta from global coordinates to local coordinates
    """
    theta = -origin[-1]
    origin_array = np.array([[origin[0], origin[1], origin[2]]], dtype=np.float64)

    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points_rel = state_se2_array - origin_array
    # TODO: 考虑旋转到底需要不需要？
    points_rel[..., :2] = points_rel[..., :2] @ r.T
    points_rel[:, 2] = normalize_angle(points_rel[:, 2])
    return points_rel

