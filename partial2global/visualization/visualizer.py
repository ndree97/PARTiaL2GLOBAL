"""
3D Metrological Visualizer
==========================
Interactive 3D inspection using PyVista with residual distance error heatmaps,
dual source-target overlays, scalar bar controls, and Quality Control metrics HUD.
"""

import os
import sys
from types import ModuleType
from typing import Optional, Dict, Any
import numpy as np

# Workaround for packaging issue in some Conda VTK distributions
sys.modules['vtkmodules.vtkRenderingMatplotlib'] = ModuleType('vtkmodules.vtkRenderingMatplotlib')

try:
    import pyvista as pv
    import vtk
    if hasattr(vtk, 'vtkObject'):
        vtk.vtkObject.GlobalWarningDisplayOff()
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from ..io.point_cloud_io import load_point_cloud
try:
    from .clipping import ClippingManager
    CLIPPING_AVAILABLE = True
except ImportError:
    CLIPPING_AVAILABLE = False


def visualize_registration(
    target_path: str,
    aligned_source_path: Optional[str] = None,
    residuals_path: Optional[str] = None,
    stats: Optional[Dict[str, Any]] = None,
    max_display_points: int = 5_000_000,
    source_pcd: Optional[Any] = None,
    residuals_data: Optional[np.ndarray] = None
) -> None:
    """
    Opens an interactive PyVista 3D window displaying the target reference model
    and the aligned source scan, with local residual error heatmap and interactive
    6-face GPU clipping box.

    Args:
        target_path: Path to target point cloud
        aligned_source_path: Path to aligned source point cloud (optional if source_pcd given)
        residuals_path: Path to .npy residual distances array
        stats: Quality metrics dictionary
        max_display_points: Point downsampling cap for smooth GPU interaction
        source_pcd: In-memory open3d PointCloud of the aligned source (avoids re-reading from disk)
        residuals_data: In-memory numpy array of residual distances
    """
    if not PYVISTA_AVAILABLE:
        print("[Visualizer] PyVista is not installed. 3D visualization skipped.")
        return

    print(f"\n[Visualizer] Preparing 3D inspection...")
    _, pcd_target, _ = load_point_cloud(target_path, load_colors=False, is_target=True, voxel_size=0.05)

    if source_pcd is not None:
        pcd_source = source_pcd
    elif aligned_source_path and os.path.exists(aligned_source_path):
        _, pcd_source, _ = load_point_cloud(aligned_source_path, load_colors=True, is_target=False)
    else:
        raise ValueError("Either aligned_source_path or source_pcd must be provided.")

    pts_target = np.asarray(pcd_target.points)
    pts_source = np.asarray(pcd_source.points)

    # Subsample if point count exceeds GPU rendering limit
    if len(pts_target) > max_display_points:
        idx = np.random.choice(len(pts_target), max_display_points, replace=False)
        pts_target = pts_target[idx]

    # Load residuals if available
    residuals = residuals_data
    if residuals is None and residuals_path and os.path.exists(residuals_path):
        try:
            residuals = np.load(residuals_path)
        except Exception:
            pass

    # Create PyVista PolyData
    cloud_target = pv.PolyData(pts_target)
    cloud_source = pv.PolyData(pts_source)

    # Initialize PyVista Plotter
    plotter = pv.Plotter(window_size=[1400, 900], title="PARTiaL2GLOBAL: 3D Metrological Inspection & Clipping")
    plotter.set_background("#0f172a", top="#1e293b")  # Modern slate gradient
    plotter.show_grid(color="#334155")

    # Add Target (subtle gray)
    plotter.add_mesh(
        cloud_target,
        color="#94a3b8",
        point_size=2.0,
        render_points_as_spheres=True,
        opacity=0.45,
        name="Target Reference",
        label="Target Reference"
    )

    # Add Source with Heatmap or Distinct Color
    if residuals is not None and len(residuals) == len(pts_source):
        cloud_source["Residual Error (m)"] = residuals
        clim_max = float(np.percentile(residuals, 95)) if len(residuals) > 0 else 0.05
        plotter.add_mesh(
            cloud_source,
            scalars="Residual Error (m)",
            cmap="turbo",
            clim=[0.0, max(clim_max, 0.01)],
            point_size=3.5,
            render_points_as_spheres=True,
            scalar_bar_args={
                "title": "Residual Error [m]",
                "color": "#ffffff",
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.25,
                "width": 0.05,
                "height": 0.5
            },
            name="Aligned Source (Residual Error)",
            label="Aligned Source (Residual Error)"
        )
    else:
        plotter.add_mesh(
            cloud_source,
            color="#38bdf8",
            point_size=3.5,
            render_points_as_spheres=True,
            name="Aligned Source",
            label="Aligned Source"
        )

    # Wire Interactive 3D GPU Clipping Box
    clip_mgr = None
    if CLIPPING_AVAILABLE:
        try:
            clip_mgr = ClippingManager(
                plotter=plotter,
                target_actor_names=["Target Reference", "Aligned Source (Residual Error)", "Aligned Source"],
                mesh=cloud_source,
                name="AlignedSource",
                allow_crop=False
            )
            plotter.add_key_event("c", clip_mgr.toggle_clipping)
            plotter.add_key_event("C", clip_mgr.toggle_clipping)
            plotter.add_key_event("x", clip_mgr.toggle_clipping)
            plotter.add_key_event("X", clip_mgr.toggle_clipping)
            plotter.add_key_event("r", clip_mgr.reset)
            plotter.add_key_event("R", clip_mgr.reset)
            plotter.add_key_event("b", clip_mgr.reset)
            plotter.add_key_event("B", clip_mgr.reset)
            clip_mgr._update_hud()
        except Exception as e:
            print(f"[Visualizer] ClippingManager warning: {e}")

    # Add Metrological HUD
    hud_text = "PARTiaL2GLOBAL Metrological Inspection\n"
    if stats:
        overlap = stats.get("overlap_pct", 0.0)
        rmse = stats.get("inlier_rmse", 0.0)
        median = stats.get("median_error", 0.0)
        rmse_fmt = f"{rmse*1000:.1f} mm" if rmse < 0.1 else f"{rmse:.4f} m"
        med_fmt = f"{median*1000:.1f} mm" if median < 0.1 else f"{median:.4f} m"
        hud_text += f"Overlap: {overlap:.1f}%\nInlier RMSE: {rmse_fmt}\nMedian Error: {med_fmt}\n"
    else:
        hud_text += f"Source Points: {len(pts_source):,}\nTarget Points: {len(pts_target):,}\n"

    hud_text += "---------------------------------------\n"
    hud_text += "[C] o [X] : Attiva / Disattiva Clipping Box\n"
    hud_text += "[R] o [B] : Ripristina Limiti Clipping Box\n"
    hud_text += "Trascina le 6 facce del box per sezionare"

    plotter.add_text(
        hud_text,
        position="upper_left",
        font_size=10,
        color="#f8fafc",
        font="courier",
        shadow=True
    )

    plotter.add_axes()
    plotter.add_legend(bcolor="#1e293b", border=True, size=(0.25, 0.12), loc="upper right")
    plotter.show()

