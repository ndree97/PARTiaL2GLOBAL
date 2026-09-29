"""
PARTiaL2GLOBAL Core Geometric and Transformation Utilities
"""

from .transforms import (
    make_transformation_matrix,
    decompose_transformation,
    compute_global_transform,
    apply_dual_centering
)
from .bounding_box import (
    compute_oriented_bounding_box,
    compute_axis_aligned_bounding_box,
    estimate_optimal_voxel_size
)

__all__ = [
    "make_transformation_matrix",
    "decompose_transformation",
    "compute_global_transform",
    "apply_dual_centering",
    "compute_oriented_bounding_box",
    "compute_axis_aligned_bounding_box",
    "estimate_optimal_voxel_size",
]
