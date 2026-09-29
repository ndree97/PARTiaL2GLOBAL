"""
PARTiaL2GLOBAL Standalone Command Line Interface (CLI)
======================================================
Usage:
    python cli.py -s partial_scan.las -t master_model.las --coarse fgr --fine point_to_plane --visualize
"""

import sys
import os
import argparse
import time

# Ensure package root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from partial2global.registration.pipeline import run_registration_pipeline
from partial2global.visualization.visualizer import visualize_registration, PYVISTA_AVAILABLE


def parse_args():
    parser = argparse.ArgumentParser(
        description="PARTiaL2GLOBAL: Lightweight Partial-to-Global Point Cloud Registration Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-s", "--source", type=str, required=True,
                        help="Path to source point cloud (partial scan: .las, .laz, .ply, .e57)")
    parser.add_argument("-t", "--target", type=str, required=True,
                        help="Path to target point cloud (global reference: .las, .laz, .ply, .e57)")
    parser.add_argument("-v", "--voxel-size", type=float, default=None,
                        help="Voxel downsampling size in meters. If omitted, estimated automatically from source bounding box.")
    parser.add_argument("-o", "--out-dir", type=str, default=None,
                        help="Output directory. Defaults to <source_dir>/<source_name>_to_<target_name>_out")
    parser.add_argument("--coarse", type=str, choices=["fgr", "ransac"], default="fgr",
                        help="Coarse alignment algorithm: 'fgr' (Fast Global Registration) or 'ransac'")
    parser.add_argument("--fine", type=str, choices=["point_to_plane", "generalized"], default="point_to_plane",
                        help="Fine refinement algorithm: 'point_to_plane' ICP or 'generalized' (G-ICP)")
    parser.add_argument("--no-anneal", action="store_true",
                        help="Disable multi-scale gradual annealing in fine registration")
    parser.add_argument("--ransac-max-iter", type=int, default=4_000_000,
                        help="Maximum RANSAC iterations (when --coarse ransac is used)")
    parser.add_argument("--visualize", action="store_true",
                        help="Launch interactive PyVista 3D residual inspection upon completion")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" PARTiaL2GLOBAL: Partial-to-Global Point Cloud Registration")
    print("=" * 70)

    try:
        result = run_registration_pipeline(
            source_path=args.source,
            target_path=args.target,
            voxel_size=args.voxel_size,
            out_dir=args.out_dir,
            coarse_method=args.coarse,
            fine_method=args.fine,
            anneal=not args.no_anneal,
            ransac_max_iter=args.ransac_max_iter
        )

        print("\n" + "=" * 70)
        print(" METROLOGICAL QUALITY CONTROL REPORT")
        print("=" * 70)
        print(f"• Aligned Output Cloud : {result.aligned_cloud_path}")
        print(f"• Metadata Sidecar     : {result.sidecar_path}")
        print(f"• Overlap Ratio        : {result.overlap_pct:.2f}%")
        print(f"• Inlier RMSE          : {result.inlier_rmse:.6f} m ({result.inlier_rmse * 1000:.2f} mm)")
        print(f"• Median Residual Error: {result.stats.get('median_error', 0.0):.6f} m")
        print(f"• Execution Time       : {result.execution_time:.2f} s")
        print("=" * 70)
        print("4x4 Transformation Matrix:")
        for row in result.transformation_matrix:
            print("  " + "  ".join(f"{val:12.6f}" for val in row))
        print("=" * 70)

        if args.visualize:
            if PYVISTA_AVAILABLE:
                visualize_registration(
                    target_path=args.target,
                    aligned_source_path=result.aligned_cloud_path,
                    residuals_path=result.residuals_path,
                    stats=result.stats
                )
            else:
                print("\n[Warning] PyVista is not installed. Visualization cannot be displayed.")

    except Exception as e:
        print(f"\n[ERROR] Registration failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
