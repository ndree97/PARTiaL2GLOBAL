"""
PARTiaL2GLOBAL
==============
Lightweight Partial-to-Global Point Cloud Registration Suite
with Metrology QC, CLI & Modern PyQt5 GUI.
"""

__version__ = "1.0.0"
__author__ = "PARTiaL2GLOBAL Contributors"

from .registration.pipeline import (
    Partial2GlobalAligner,
    AlignmentResult,
    run_registration_pipeline
)
from .core.transforms import (
    make_transformation_matrix,
    decompose_transformation,
    compute_global_transform,
    apply_dual_centering
)
from .io.point_cloud_io import load_point_cloud, save_aligned_cloud

__all__ = [
    "Partial2GlobalAligner",
    "AlignmentResult",
    "run_registration_pipeline",
    "make_transformation_matrix",
    "decompose_transformation",
    "compute_global_transform",
    "apply_dual_centering",
    "load_point_cloud",
    "save_aligned_cloud",
    "__version__"
]
