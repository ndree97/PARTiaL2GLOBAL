"""
Partial-to-Global Registration Pipeline Orchestrator
===================================================
Orchestrates end-to-end alignment of a smaller partial point cloud (source)
onto a large global reference model (target).
Provides both an object-oriented Partial2GlobalAligner API and a functional
run_registration_pipeline with real-time callbacks.
"""

import os
import time
import copy
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
import numpy as np
import open3d as o3d

from ..core.transforms import apply_dual_centering, compute_global_transform
from ..core.bounding_box import estimate_optimal_voxel_size
from ..io.point_cloud_io import load_point_cloud, save_aligned_cloud
from ..io.e57_handler import check_e57_panoramas_status
from .coarse import preprocess_point_cloud, execute_coarse_registration
from .fine import execute_fine_registration
from ..metrics.quality import evaluate_quality_metrics, save_sidecar_metadata


@dataclass
class AlignmentResult:
    """Encapsulates all outputs and metrics resulting from a registration run."""
    source_path: str
    target_path: str
    aligned_cloud_path: Optional[str]
    sidecar_path: str
    residuals_path: str
    transformation_matrix: np.ndarray
    voxel_size: float
    overlap_pct: float
    inlier_rmse: float
    stats: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    panoramas_status: Optional[Dict[str, Any]] = None
    candidate_cloud_path: str = ""
    raw_source: Optional[Any] = None
    pcd_aligned: Optional[o3d.geometry.PointCloud] = None

    def save_aligned(self, output_path: Optional[str] = None) -> str:
        """
        Saves the aligned point cloud to disk on demand using the in-memory
        raw source representation and transformation matrix.
        """
        target_path = output_path or self.candidate_cloud_path or self.aligned_cloud_path
        if not target_path:
            raise ValueError("No output path specified for saving aligned cloud.")
        ext = os.path.splitext(target_path)[1].lower()
        save_aligned_cloud(
            self.raw_source,
            self.transformation_matrix,
            target_path,
            ext=ext,
            source_path=self.source_path
        )
        self.aligned_cloud_path = target_path
        return target_path

    def __repr__(self) -> str:
        cloud_name = os.path.basename(self.aligned_cloud_path) if self.aligned_cloud_path else "(in memory)"
        return (
            f"AlignmentResult(\n"
            f"  aligned_cloud='{cloud_name}',\n"
            f"  overlap={self.overlap_pct:.2f}%,\n"
            f"  inlier_rmse={self.inlier_rmse:.6f},\n"
            f"  time={self.execution_time:.2f}s\n"
            f")"
        )


def _shorten(name: str, max_len: int = 40) -> str:
    """Shortens a filename to avoid path length limitations while preserving uniqueness."""
    if len(name) <= max_len:
        return name
    h = hashlib.md5(name.encode()).hexdigest()[:6]
    return f"{name[:max_len-10]}_{h}"


def run_registration_pipeline(
    source_path: str,
    target_path: str,
    voxel_size: Optional[float] = None,
    out_dir: Optional[str] = None,
    coarse_method: str = "fgr",
    fine_method: str = "point_to_plane",
    anneal: bool = True,
    ransac_max_iter: int = 4_000_000,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    auto_save_cloud: bool = True
) -> AlignmentResult:
    """
    Executes the full 6-step metrological registration pipeline.

    Args:
        source_path: Path to source point cloud (partial scan)
        target_path: Path to target point cloud (global reference)
        voxel_size: Downsampling voxel size (auto-estimated if None)
        out_dir: Output directory (defaults to source_dir/<name>_to_<name>_out)
        coarse_method: 'fgr' or 'ransac'
        fine_method: 'point_to_plane' or 'generalized'
        anneal: Enable multi-scale annealing in fine registration
        ransac_max_iter: Max iterations if RANSAC is selected
        progress_callback: Callback(step, total_steps, description)
        log_callback: Callback(log_line)
        auto_save_cloud: If True, writes the heavy aligned cloud to disk in step 6.
                         If False, keeps the aligned cloud in-memory and saves only
                         lightweight metadata and residuals, waiting for user confirmation.

    Returns:
        AlignmentResult instance
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def report_progress(step: int, text: str):
        if progress_callback:
            progress_callback(step, 6, text)

    start_total_time = time.time()
    source_dir = os.path.dirname(os.path.abspath(source_path))
    source_base = os.path.splitext(os.path.basename(source_path))[0]
    target_base = os.path.splitext(os.path.basename(target_path))[0]

    short_source = _shorten(source_base)
    short_target = _shorten(target_base)

    if out_dir is None:
        out_dir = os.path.join(source_dir, f"{short_source}_to_{short_target}_out")
    os.makedirs(out_dir, exist_ok=True)

    log(f"--- Starting PARTiaL2GLOBAL Pipeline ---")
    log(f"Source: {source_path}")
    log(f"Target: {target_path}")

    # Inspect panoramas if source is E57
    source_pano_status = None
    if source_path.lower().endswith('.e57'):
        source_pano_status = check_e57_panoramas_status(source_path)
        log(f"[E57 Status] {source_pano_status.get('message', '')}")

    # 1. Load Point Clouds
    report_progress(1, "Loading Point Clouds")
    log("\n[Step 1/6] Loading source and target point clouds...")
    t0 = time.time()
    raw_source, pcd_source, src_ext = load_point_cloud(source_path, load_colors=True, is_target=False)
    log(f"Loaded source: {len(pcd_source.points):,} points in {time.time() - t0:.2f}s")

    # Voxel size estimation
    if voxel_size is None or voxel_size <= 0:
        voxel_size = estimate_optimal_voxel_size(pcd_source)
        log(f"Auto-estimated voxel size: {voxel_size:.4f} m (based on source bounding box)")

    # Load target using memory-safe streaming if large
    t0 = time.time()
    raw_target, pcd_target, tgt_ext = load_point_cloud(
        target_path,
        load_colors=False,
        is_target=True,
        voxel_size=voxel_size
    )
    log(f"Loaded target: {len(pcd_target.points):,} points in {time.time() - t0:.2f}s")

    # Dual Centering (anti-jitter for UTM/cartographic frames)
    source_centroid, target_centroid = apply_dual_centering(pcd_source, pcd_target)
    log(f"Dual centering applied: source_c={np.round(source_centroid, 2)}, target_c={np.round(target_centroid, 2)}")

    # 2. Preprocess Point Clouds
    report_progress(2, "Preprocessing & Normal Estimation")
    log("\n[Step 2/6] Downsampling and estimating normals...")
    t0 = time.time()
    source_down = preprocess_point_cloud(pcd_source, voxel_size, is_source=True)
    target_down = preprocess_point_cloud(pcd_target, voxel_size, is_source=False)
    log(f"Preprocessing completed in {time.time() - t0:.2f}s (source: {len(source_down.points):,}, target: {len(target_down.points):,})")

    # 3. Coarse Alignment
    report_progress(3, f"Coarse Alignment ({coarse_method.upper()})")
    log(f"\n[Step 3/6] Running coarse registration ({coarse_method.upper()})...")
    t0 = time.time()
    coarse_transform, coarse_fitness, coarse_rmse = execute_coarse_registration(
        source_down, target_down, voxel_size,
        method=coarse_method,
        ransac_max_iter=ransac_max_iter
    )
    log(f"Coarse registration finished in {time.time() - t0:.2f}s (fitness={coarse_fitness:.4f}, inlier_rmse={coarse_rmse:.6f})")

    # 4. Fine Alignment
    report_progress(4, f"Fine Refinement ({fine_method.upper()})")
    log(f"\n[Step 4/6] Running fine registration ({fine_method.upper()}) with annealing={anneal}...")
    t0 = time.time()
    final_transform, fine_fitness, fine_rmse = execute_fine_registration(
        source_down, target_down, coarse_transform, voxel_size,
        method=fine_method,
        anneal=anneal
    )
    log(f"Fine registration finished in {time.time() - t0:.2f}s (fitness={fine_fitness:.4f}, inlier_rmse={fine_rmse:.6f})")

    # 5. Quality Control & Residual Analysis (on full clouds)
    report_progress(5, "Quality Control & Residual Analysis")
    log("\n[Step 5/6] Evaluating metrological quality on full clouds...")
    t0 = time.time()
    stats, residuals = evaluate_quality_metrics(pcd_source, pcd_target, final_transform, voxel_size)
    log(f"QC evaluation finished in {time.time() - t0:.2f}s")
    log(f"  Overlap: {stats['overlap_pct']:.2f}% (Threshold: {stats['overlap_threshold']:.4f} m)")
    log(f"  Inlier RMSE: {stats['inlier_rmse']:.6f} m")
    log(f"  Median Error: {stats['median_error']:.6f} m")

    # Convert centered local transform to global transform
    global_transform = compute_global_transform(final_transform, source_centroid, target_centroid)

    # 6. Save Output Cloud & Provenance Metadata
    report_progress(6, "Exporting Aligned Cloud & Sidecar")
    t0 = time.time()

    candidate_cloud_path = os.path.join(out_dir, f"{short_source}_aligned{src_ext}")
    aligned_meta_path = os.path.join(out_dir, f"{short_source}_meta.json")
    aligned_res_path = os.path.join(out_dir, f"{short_source}_residuals.npy")

    # Save residuals array for visualizer
    np.save(aligned_res_path, residuals)

    # In-memory aligned point cloud in global coordinates (for instant 3D inspection)
    pcd_aligned = copy.deepcopy(pcd_source)
    pcd_aligned.transform(final_transform)
    pcd_aligned.translate(target_centroid)

    aligned_cloud_path = None
    if auto_save_cloud:
        log("\n[Step 6/6] Saving outputs to disk...")
        save_aligned_cloud(raw_source, global_transform, candidate_cloud_path, src_ext, source_path=source_path)
        aligned_cloud_path = candidate_cloud_path
        log(f"Aligned point cloud saved to {aligned_cloud_path}")
    else:
        log("\n[Step 6/6] Metadata e residui salvati. Nuvola mantenuta in memoria (in attesa di conferma)...")

    # Save JSON sidecar
    total_duration = time.time() - start_total_time
    config_dict = {
        "source_path": source_path,
        "target_path": target_path,
        "voxel_size": voxel_size,
        "coarse_method": coarse_method,
        "fine_method": fine_method,
        "annealing": anneal,
        "ransac_max_iter": ransac_max_iter,
        "execution_time": total_duration,
        "auto_saved": auto_save_cloud
    }
    save_sidecar_metadata(aligned_meta_path, stats, global_transform, config_dict)
    log(f"Outputs written in {time.time() - t0:.2f}s")
    log(f"Pipeline finished successfully in {total_duration:.2f}s.")

    return AlignmentResult(
        source_path=source_path,
        target_path=target_path,
        aligned_cloud_path=aligned_cloud_path,
        sidecar_path=aligned_meta_path,
        residuals_path=aligned_res_path,
        transformation_matrix=global_transform,
        voxel_size=voxel_size,
        overlap_pct=stats["overlap_pct"],
        inlier_rmse=stats["inlier_rmse"],
        stats=stats,
        execution_time=total_duration,
        panoramas_status=source_pano_status,
        candidate_cloud_path=candidate_cloud_path,
        raw_source=raw_source,
        pcd_aligned=pcd_aligned
    )


class Partial2GlobalAligner:
    """
    Object-oriented interface for Partial-to-Global Point Cloud Registration.
    
    Usage:
        aligner = Partial2GlobalAligner(coarse="fgr", fine="point_to_plane")
        result = aligner.align("source.las", "target.las")
    """

    def __init__(
        self,
        coarse: str = "fgr",
        fine: str = "point_to_plane",
        anneal: bool = True,
        ransac_max_iter: int = 4_000_000,
        voxel_size: Optional[float] = None
    ):
        self.coarse = coarse
        self.fine = fine
        self.anneal = anneal
        self.ransac_max_iter = ransac_max_iter
        self.voxel_size = voxel_size

    def align(
        self,
        source: str,
        target: str,
        out_dir: Optional[str] = None,
        voxel_size: Optional[float] = None,
        auto_save_cloud: bool = True
    ) -> AlignmentResult:
        """Runs the registration pipeline on the specified source and target."""
        v_size = voxel_size if voxel_size is not None else self.voxel_size
        return run_registration_pipeline(
            source_path=source,
            target_path=target,
            voxel_size=v_size,
            out_dir=out_dir,
            coarse_method=self.coarse,
            fine_method=self.fine,
            anneal=self.anneal,
            ransac_max_iter=self.ransac_max_iter,
            auto_save_cloud=auto_save_cloud
        )

