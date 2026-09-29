"""
Coarse Registration Module
==========================
Global initial alignment for partial-to-global point cloud registration.
Extracts Fast Point Feature Histograms (FPFH) and computes rigid transformation
using Fast Global Registration (FGR) or RANSAC feature matching.
"""

import time
from typing import Tuple
import numpy as np
import open3d as o3d


def preprocess_point_cloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    is_source: bool = True
) -> o3d.geometry.PointCloud:
    """
    Performs voxel downsampling, statistical outlier removal (for source),
    and robust normal estimation for point-to-plane/G-ICP algorithms.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    if is_source:
        pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Search radius for normals: 4x voxel_size ensures stable local surface planes
    radius_normal = voxel_size * 4.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd_down.orient_normals_to_align_with_direction()

    return pcd_down


def compute_fpfh_features(
    pcd_down: o3d.geometry.PointCloud,
    voxel_size: float
) -> o3d.pipelines.registration.Feature:
    """
    Computes Fast Point Feature Histograms (FPFH) with a search radius
    proportional to voxel size (5x voxel_size).
    """
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def execute_coarse_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    voxel_size: float,
    method: str = "fgr",
    ransac_max_iter: int = 4_000_000
) -> Tuple[np.ndarray, float, float]:
    """
    Executes coarse global alignment.
    
    Args:
        source_down: Downsampled source cloud with normals
        target_down: Downsampled target cloud with normals
        voxel_size: Downsampling voxel size
        method: 'fgr' (Fast Global Registration) or 'ransac'
        ransac_max_iter: Maximum RANSAC iterations

    Returns:
        transformation_matrix: 4x4 np.ndarray
        fitness: Inlier ratio
        inlier_rmse: RMSE of inlier correspondences
    """
    source_fpfh = compute_fpfh_features(source_down, voxel_size)
    target_fpfh = compute_fpfh_features(target_down, voxel_size)

    if method.lower() == "fgr":
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                division_factor=1.4,
                decrease_mu=True,
                maximum_correspondence_distance=voxel_size * 1.5
            )
        )
    else:
        distance_threshold = voxel_size * 1.5
        checkers = [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ]
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=checkers,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_iter, 0.999)
        )

    return result.transformation, float(result.fitness), float(result.inlier_rmse)
