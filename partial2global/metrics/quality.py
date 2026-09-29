"""
Quality Control and Metrology Evaluation Module
================================================
Performs quantitative validation of point cloud alignment:
- Multi-threaded C++ KD-Tree distance calculations
- Overlap ratio (%) estimation
- Inlier Root Mean Square Error (RMSE)
- Comprehensive residual error statistics (min, median, max, percentiles)
- JSON sidecar generation for provenance and metrology auditing
"""

import os
import json
import time
from typing import Dict, Any, Tuple
import numpy as np
import open3d as o3d

from ..core.transforms import decompose_transformation


def evaluate_quality_metrics(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    voxel_size: float
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates alignment quality on the full-resolution clouds.

    Returns:
        metrics_dict: Dictionary with statistics and parameters
        residuals: 1D np.ndarray of residual distances for every source point
    """
    source_transformed = o3d.geometry.PointCloud(source_pcd)
    source_transformed.transform(transformation)

    # Compute OBB for aligned source
    obb = source_transformed.get_oriented_bounding_box()
    obb_dict = {
        "center": np.asarray(obb.center).tolist(),
        "R": np.asarray(obb.R).tolist(),
        "extent": np.asarray(obb.extent).tolist()
    }

    # Fast multi-threaded C++ distance computation
    residuals = np.asarray(source_transformed.compute_point_cloud_distance(target_pcd), dtype=np.float64)

    overlap_threshold = voxel_size * 2.0
    inliers_mask = residuals < overlap_threshold
    inliers = residuals[inliers_mask]

    overlap_pct = (len(inliers) / len(residuals)) * 100.0 if len(residuals) > 0 else 0.0
    inlier_rmse = float(np.sqrt(np.mean(inliers ** 2))) if len(inliers) > 0 else float("nan")

    stats = {
        "total_source_points": int(len(residuals)),
        "inliers_count": int(len(inliers)),
        "overlap_pct": round(float(overlap_pct), 2),
        "overlap_threshold": round(float(overlap_threshold), 4),
        "inlier_rmse": round(float(inlier_rmse), 6),
        "min_error": round(float(np.min(residuals)), 6),
        "q25_error": round(float(np.percentile(residuals, 25)), 6),
        "median_error": round(float(np.percentile(residuals, 50)), 6),
        "mean_error": round(float(np.mean(residuals)), 6),
        "q75_error": round(float(np.percentile(residuals, 75)), 6),
        "q95_error": round(float(np.percentile(residuals, 95)), 6),
        "max_error": round(float(np.max(residuals)), 6),
        "std_error": round(float(np.std(residuals)), 6),
        "obb": obb_dict
    }

    return stats, residuals


def save_sidecar_metadata(
    metadata_path: str,
    stats: Dict[str, Any],
    global_transform: np.ndarray,
    config: Dict[str, Any]
) -> None:
    """
    Saves registration metadata, parameters, quality metrics, and decomposed
    transformation vectors to a standardized JSON sidecar.
    """
    decomposed = decompose_transformation(global_transform)

    metadata = {
        "generator": "PARTiaL2GLOBAL v1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "source_path": config.get("source_path", ""),
            "target_path": config.get("target_path", ""),
            "voxel_size": config.get("voxel_size"),
            "coarse_method": config.get("coarse_method", "fgr"),
            "fine_method": config.get("fine_method", "point_to_plane"),
            "annealing": config.get("annealing", True),
            "ransac_max_iter": config.get("ransac_max_iter", 4_000_000),
            "execution_time_seconds": config.get("execution_time", 0.0)
        },
        "quality_metrics": stats,
        "transformation": {
            "matrix_4x4": global_transform.tolist(),
            "rotation_matrix_3x3": decomposed["rotation_matrix"],
            "translation_vector": decomposed["translation_vector"],
            "euler_angles_deg": decomposed["euler_angles_deg"],
            "quaternion_wxyz": decomposed["quaternion_wxyz"]
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
