"""
Fine Registration Module
========================
High-precision refinement using Point-to-Plane ICP or Generalized ICP (G-ICP).
Incorporates Multi-Scale Gradual Annealing to expand the basin of convergence
and achieve millimeter-level alignment without getting trapped in local minima.
"""

from typing import Tuple, List, Callable, Optional
import numpy as np
import open3d as o3d


def execute_fine_registration(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    voxel_size: float,
    method: str = "point_to_plane",
    anneal: bool = True,
    max_iteration_per_stage: int = 100,
    callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Executes fine registration with gradual multi-scale annealing.

    Args:
        source_down: Downsampled source cloud with normals
        target_down: Downsampled target cloud with normals
        init_transform: 4x4 initial transformation matrix from coarse registration
        voxel_size: Base voxel size
        method: 'point_to_plane' or 'generalized' (G-ICP)
        anneal: If True, uses 3-stage gradual annealing (3.0x -> 1.5x -> 1.0x)
        max_iteration_per_stage: Max ICP iterations per stage
        callback: Optional callback(stage_idx, total_stages, fitness, rmse)

    Returns:
        final_transform: 4x4 refined transformation matrix
        final_fitness: Inlier ratio at tightest stage
        final_rmse: RMSE at tightest stage
    """
    if method == "point_to_plane":
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif method == "generalized":
        estimation = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    else:
        raise ValueError(f"Unknown fine registration method: '{method}'. Choose 'point_to_plane' or 'generalized'.")

    current_transformation = np.array(init_transform, copy=True)
    stages: List[float] = [3.0, 1.5, 1.0] if anneal else [1.5]

    final_fitness = 0.0
    final_rmse = 0.0

    for i, multiplier in enumerate(stages):
        threshold = voxel_size * multiplier
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration_per_stage)

        if method == "point_to_plane":
            result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, threshold, current_transformation,
                estimation, criteria
            )
        else:
            result = o3d.pipelines.registration.registration_generalized_icp(
                source_down, target_down, threshold, current_transformation,
                estimation, criteria
            )

        current_transformation = result.transformation
        final_fitness = float(result.fitness)
        final_rmse = float(result.inlier_rmse)

        if callback is not None:
            callback(i + 1, len(stages), final_fitness, final_rmse)

    return current_transformation, final_fitness, final_rmse
