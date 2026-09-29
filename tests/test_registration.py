"""
Tests for PARTiaL2GLOBAL Registration Pipeline
==============================================
Tests transformation math, bounding box estimation, dual local centering,
and synthetic partial-to-global cloud registration with ground-truth verification.
"""

import os
import sys
import shutil
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pytest
except ImportError:
    pytest = None
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from partial2global.core.transforms import (
    make_transformation_matrix,
    decompose_transformation,
    apply_dual_centering,
    compute_global_transform
)
from partial2global.core.bounding_box import estimate_optimal_voxel_size
from partial2global.registration.pipeline import run_registration_pipeline, Partial2GlobalAligner
from partial2global.io.point_cloud_io import save_aligned_cloud


if pytest is not None:
    @pytest.fixture
    def temp_dir():
        d = tempfile.mkdtemp(prefix="p2g_align_test_")
        yield d
        shutil.rmtree(d, ignore_errors=True)


def test_transforms_decomposition():
    """Verify that transformation matrix creation and decomposition are exact inverses."""
    R = Rotation.from_euler("xyz", [12.5, -7.2, 45.0], degrees=True).as_matrix()
    t = np.array([10.5, -3.2, 1.8], dtype=np.float64)
    T = make_transformation_matrix(R, t)

    decomp = decompose_transformation(T)
    assert np.allclose(decomp["rotation_matrix"], R, atol=1e-7)
    assert np.allclose(decomp["translation_vector"], t, atol=1e-7)
    assert np.isclose(decomp["euler_angles_deg"]["roll"], 12.5, atol=1e-5)
    assert np.isclose(decomp["euler_angles_deg"]["pitch"], -7.2, atol=1e-5)
    assert np.isclose(decomp["euler_angles_deg"]["yaw"], 45.0, atol=1e-5)


def test_dual_centering_reconstruction():
    """Verify that dual centering and global reconstruction map points correctly."""
    np.random.seed(42)
    # Simulate points in large coordinates (e.g., UTM 500,000 / 4,500,000)
    center_src = np.array([500120.0, 4500340.0, 150.0])
    center_tgt = np.array([500125.0, 4500342.0, 151.0])

    pts_src = np.random.uniform(-5.0, 5.0, size=(100, 3)) + center_src
    pts_tgt = np.random.uniform(-10.0, 10.0, size=(200, 3)) + center_tgt

    pcd_s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_src))
    pcd_t = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_tgt))

    c_s, c_t = apply_dual_centering(pcd_s, pcd_t)
    assert np.allclose(pcd_s.get_center(), [0, 0, 0], atol=1e-5)
    assert np.allclose(pcd_t.get_center(), [0, 0, 0], atol=1e-5)

    # If local transform is identity, global transform must equal translation between centers
    local_I = np.eye(4)
    global_T = compute_global_transform(local_I, c_s, c_t)
    expected_shift = c_t - c_s
    assert np.allclose(global_T[:3, 3], expected_shift, atol=1e-5)


def test_synthetic_partial_registration(temp_dir):
    """
    Creates a synthetic target mesh/cloud, crops a partial sub-volume as source,
    applies a known rotation and translation in large coordinates, runs registration,
    and checks that the source is accurately aligned onto the target.
    """
    np.random.seed(123)

    # 1. Create a synthetic target model with distinct dimensions (no rotational symmetry)
    mesh = o3d.geometry.TriangleMesh.create_box(width=4.0, height=6.0, depth=10.0)
    mesh.compute_vertex_normals()
    target_pcd = mesh.sample_points_poisson_disk(number_of_points=15_000)

    # Shift target into large coordinates (UTM-like)
    large_offset = np.array([600_000.0, 4_800_000.0, 200.0])
    target_pcd.translate(large_offset)

    # 2. Extract partial source (sub-region: upper section corner, ~25% of target)
    target_points = np.asarray(target_pcd.points)
    mask = (target_points[:, 2] >= 204.0) & (target_points[:, 1] >= 4_800_002.0)
    source_points = np.copy(target_points[mask])
    assert len(source_points) > 1000, "Insufficient source points extracted"

    # 3. Ground truth perturbation: rotation + translation
    rot_gt = Rotation.from_euler("z", 8.0, degrees=True).as_matrix()
    trans_gt = np.array([0.25, -0.20, 0.15])

    source_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_points))
    source_centroid = source_pcd.get_center()
    source_pcd.translate(-source_centroid)
    source_pcd.rotate(rot_gt.T, center=(0, 0, 0))
    source_pcd.translate(source_centroid - trans_gt)

    # Save to disk as PLY
    target_file = os.path.join(temp_dir, "synthetic_target.ply")
    source_file = os.path.join(temp_dir, "synthetic_source.ply")
    o3d.io.write_point_cloud(target_file, target_pcd)
    o3d.io.write_point_cloud(source_file, source_pcd)

    # 4. Run PARTiaL2GLOBAL registration with appropriate voxel size
    result = run_registration_pipeline(
        source_path=source_file,
        target_path=target_file,
        voxel_size=0.08,
        out_dir=temp_dir,
        coarse_method="fgr",
        fine_method="point_to_plane",
        anneal=True
    )

    # 5. Verify results
    assert os.path.exists(result.aligned_cloud_path)
    assert os.path.exists(result.sidecar_path)
    assert os.path.exists(result.residuals_path)

    # Overlap should be high (> 85%)
    assert result.overlap_pct > 85.0, f"Expected overlap > 85%, got {result.overlap_pct:.2f}%"

    # Inlier RMSE should be sub-centimeter
    assert result.inlier_rmse < 0.02, f"Expected inlier RMSE < 0.02m, got {result.inlier_rmse:.6f}m"

    # Verify that aligned cloud aligns with target
    aligned_pcd = o3d.io.read_point_cloud(result.aligned_cloud_path)
    dist = np.asarray(aligned_pcd.compute_point_cloud_distance(target_pcd))
    assert np.median(dist) < 0.02, f"Median residual error after alignment is {np.median(dist):.4f}m"


def test_auto_save_confirmation_and_fast_bounds(temp_dir):
    """
    Verifies that:
    1. Fast bounding box / voxel estimation works without reading full points.
    2. auto_save_cloud=False keeps cloud in memory and does not write to disk until confirmed.
    3. result.save_aligned() writes the aligned cloud correctly on demand.
    """
    from partial2global.core.bounding_box import get_fast_point_cloud_bounds, estimate_optimal_voxel_from_file

    mesh = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=2.0)
    pcd = mesh.sample_points_poisson_disk(number_of_points=1000)
    p_path = os.path.join(temp_dir, "fast_box.ply")
    o3d.io.write_point_cloud(p_path, pcd)

    mins, maxs, count = get_fast_point_cloud_bounds(p_path)
    assert count == 1000
    assert len(mins) == 3 and len(maxs) == 3

    vox, count2 = estimate_optimal_voxel_from_file(p_path)
    assert count2 == 1000
    assert 0.02 <= vox <= 0.50

    # Test registration with auto_save_cloud=False (Confirmation mode)
    target_path = p_path
    source_path = os.path.join(temp_dir, "sub_box.ply")
    pcd_sub = pcd.select_by_index(list(range(500)))
    o3d.io.write_point_cloud(source_path, pcd_sub)

    result = run_registration_pipeline(
        source_path=source_path,
        target_path=target_path,
        voxel_size=vox,
        out_dir=os.path.join(temp_dir, "out_no_save"),
        auto_save_cloud=False
    )

    # Cloud must NOT be automatically saved to disk
    assert result.aligned_cloud_path is None
    assert not os.path.exists(result.candidate_cloud_path)
    assert result.pcd_aligned is not None
    assert len(result.pcd_aligned.points) == 500

    # User confirms and requests saving:
    saved_path = result.save_aligned()
    assert os.path.exists(saved_path)
    assert result.aligned_cloud_path == saved_path


if __name__ == "__main__":
    print("==================================================")
    print(" Running PARTiaL2GLOBAL Test Suite")
    print("==================================================")

    print("[1/4] Testing transforms decomposition...")
    test_transforms_decomposition()
    print("      [OK] PASSED")

    print("[2/4] Testing dual centering reconstruction...")
    test_dual_centering_reconstruction()
    print("      [OK] PASSED")

    print("[3/4] Testing synthetic partial-to-global registration...")
    _tmp = tempfile.mkdtemp(prefix="p2g_test_")
    try:
        test_synthetic_partial_registration(_tmp)
        print("      [OK] PASSED")
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)

    print("[4/4] Testing fast bounds & on-demand saving (no auto-save)...")
    _tmp2 = tempfile.mkdtemp(prefix="p2g_test2_")
    try:
        test_auto_save_confirmation_and_fast_bounds(_tmp2)
        print("      [OK] PASSED")
    finally:
        shutil.rmtree(_tmp2, ignore_errors=True)

    print("==================================================")
    print(" ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


