"""
Transformation Utilities for PARTiaL2GLOBAL
===========================================
Provides functions for 4x4 transformation matrix composition, decomposition,
Euler angle conversions, and Dual Local Centering to prevent float32 precision
loss when registering large cartographic/georeferenced (e.g. UTM) point clouds.
"""

from typing import Tuple, Dict, Any
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def make_transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Constructs a 4x4 homogeneous transformation matrix from a 3x3 rotation matrix
    and a 3-element translation vector.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).flatten()
    return T


def decompose_transformation(T: np.ndarray) -> Dict[str, Any]:
    """
    Decomposes a 4x4 transformation matrix into rotation matrix, translation vector,
    Euler angles (roll, pitch, yaw in degrees and radians), and quaternion (w, x, y, z).
    """
    R_mat = T[:3, :3]
    t_vec = T[:3, 3]

    rot = Rotation.from_matrix(R_mat)
    euler_deg = rot.as_euler('xyz', degrees=True)
    euler_rad = rot.as_euler('xyz', degrees=False)
    quat_xyzw = rot.as_quat()  # [x, y, z, w]
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    return {
        "rotation_matrix": R_mat.tolist(),
        "translation_vector": t_vec.tolist(),
        "euler_angles_deg": {
            "roll": float(euler_deg[0]),
            "pitch": float(euler_deg[1]),
            "yaw": float(euler_deg[2])
        },
        "euler_angles_rad": {
            "roll": float(euler_rad[0]),
            "pitch": float(euler_rad[1]),
            "yaw": float(euler_rad[2])
        },
        "quaternion_wxyz": [float(x) for x in quat_wxyz],
        "translation_magnitude": float(np.linalg.norm(t_vec))
    }


def apply_dual_centering(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies dual centering to both source and target point clouds in-place:
    translates each cloud so its centroid is at the origin (0, 0, 0).
    
    This is essential for large georeferenced/cartographic coordinates (e.g., UTM)
    to eliminate numerical truncation errors in Open3D C++ KDTree and float32 buffers.

    Returns:
        source_centroid: np.ndarray (3,)
        target_centroid: np.ndarray (3,)
    """
    source_centroid = np.asarray(source_pcd.get_center(), dtype=np.float64)
    target_centroid = np.asarray(target_pcd.get_center(), dtype=np.float64)

    source_pcd.translate(-source_centroid)
    target_pcd.translate(-target_centroid)

    return source_centroid, target_centroid


def compute_global_transform(
    local_T: np.ndarray,
    source_centroid: np.ndarray,
    target_centroid: np.ndarray
) -> np.ndarray:
    """
    Converts a local transformation matrix computed on dual-centered point clouds
    back into the global coordinate frame that maps raw source points directly
    to raw target points:
    
        P_target = R * (P_source - C_source) + t_local + C_target
                 = R * P_source + (t_local + C_target - R @ C_source)
    """
    R = local_T[:3, :3]
    t = local_T[:3, 3]

    global_T = np.eye(4, dtype=np.float64)
    global_T[:3, :3] = R
    global_T[:3, 3] = t + target_centroid - (R @ source_centroid)

    return global_T
