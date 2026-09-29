"""
Universal Point Cloud I/O Adapter
=================================
Unified loading and saving for LAS, LAZ, PLY, and E57 files.
Includes:
- Memory-safe streaming reader and progressive voxel downsampling for huge target point clouds
- Full preservation of georeferenced headers, scale, offsets, classification, and extra dimensions
- Consistent RGB color normalization [0, 1] for Open3D and uint8 [0, 255] for disk
- Rototranslation of spherical panoramas for E57
"""

import os
import copy
import time
from typing import Tuple, Optional, Any
import numpy as np
import open3d as o3d
import laspy

from .e57_handler import write_e57_point_cloud, is_e57_available


def load_point_cloud(
    filepath: str,
    load_colors: bool = True,
    is_target: bool = False,
    voxel_size: Optional[float] = None,
    chunk_capacity: int = 5_000_000
) -> Tuple[Any, o3d.geometry.PointCloud, str]:
    """
    Loads a point cloud from LAS, LAZ, PLY, or E57 format.
    
    For large target point clouds (is_target=True), employs streaming downsampling
    where available to keep memory footprint bounded.

    Returns:
        raw_obj: Native raw object (LasData, Open3D PointCloud, or raw dict)
        pcd: open3d.geometry.PointCloud with coordinates and normalized colors
        ext: file extension in lowercase (.las, .laz, .ply, .e57)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Point cloud file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    raw_obj = None

    if ext in ['.las', '.laz']:
        if is_target and voxel_size is not None and voxel_size > 0:
            # Memory-safe streaming chunked downsampler for large target clouds:
            # Prevents reading hundreds of millions of points into memory at once
            t0 = time.time()
            with laspy.open(filepath) as reader:
                total_pts = getattr(reader.header, 'point_count', 0)
                downsampled_parts = []
                for chunk in reader.chunk_iterator(chunk_capacity):
                    pts_chunk = np.column_stack((chunk.x, chunk.y, chunk.z))
                    chunk_pcd = o3d.geometry.PointCloud()
                    chunk_pcd.points = o3d.utility.Vector3dVector(pts_chunk)
                    chunk_down = chunk_pcd.voxel_down_sample(voxel_size)
                    downsampled_parts.append(chunk_down)

                pcd = o3d.geometry.PointCloud()
                for part in downsampled_parts:
                    pcd += part
                if len(downsampled_parts) > 1:
                    pcd = pcd.voxel_down_sample(voxel_size)
            raw_obj = None
        else:
            las = laspy.read(filepath)
            if not is_target:
                raw_obj = las
            points = np.column_stack((las.x, las.y, las.z))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            if load_colors and not is_target and hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                try:
                    r = np.asarray(las.red, dtype=np.float32)
                    g = np.asarray(las.green, dtype=np.float32)
                    b = np.asarray(las.blue, dtype=np.float32)
                    if len(r) > 0:
                        max_val = max(float(r.max()), float(g.max()), float(b.max()))
                        denom = 65535.0 if max_val > 255.0 else 255.0
                        colors = np.column_stack((r / denom, g / denom, b / denom))
                        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
                        del colors
                except Exception:
                    pass

            if is_target and voxel_size is not None and voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size)

    elif ext == '.ply':
        pcd = o3d.io.read_point_cloud(filepath)
        if not is_target:
            raw_obj = copy.deepcopy(pcd)
        if not load_colors or is_target:
            if pcd.has_colors():
                pcd.colors = o3d.utility.Vector3dVector()
        if is_target and voxel_size is not None and voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)

    elif ext == '.e57':
        if not is_e57_available():
            raise ImportError("pye57 is required to read E57 point clouds.")
        import pye57
        e57 = pye57.E57(filepath, mode='r')
        try:
            header = e57.get_header(0)
            total_points = header.point_count

            if is_target:
                # Target streaming chunked reader to prevent RAM exhaustion
                fields = ['cartesianX', 'cartesianY', 'cartesianZ']
                buf_cap = min(chunk_capacity, max(total_points, 1000))
                data, buffers = e57.make_buffers(fields, buf_cap)
                reader = header.points.reader(buffers)

                downsampled_parts = []
                read_pts_total = 0
                use_voxel = voxel_size if (voxel_size is not None and voxel_size > 0) else 0.05

                while True:
                    n_read = reader.read()
                    if n_read == 0:
                        break
                    read_pts_total += n_read

                    pts_chunk = np.column_stack((
                        data['cartesianX'][:n_read],
                        data['cartesianY'][:n_read],
                        data['cartesianZ'][:n_read]
                    ))

                    chunk_pcd = o3d.geometry.PointCloud()
                    chunk_pcd.points = o3d.utility.Vector3dVector(pts_chunk)
                    chunk_down = chunk_pcd.voxel_down_sample(use_voxel)
                    downsampled_parts.append(chunk_down)

                reader.close()

                pcd = o3d.geometry.PointCloud()
                for part in downsampled_parts:
                    pcd += part

                if len(downsampled_parts) > 1:
                    pcd = pcd.voxel_down_sample(use_voxel)

                raw_obj = None
            else:
                try:
                    raw_obj = e57.read_scan_raw(0, ignore_unsupported_fields=True)
                except Exception:
                    raw_obj = e57.read_scan(0, ignore_missing_fields=True)

                points = np.vstack((raw_obj["cartesianX"], raw_obj["cartesianY"], raw_obj["cartesianZ"])).T
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                if load_colors and "colorRed" in raw_obj and "colorGreen" in raw_obj and "colorBlue" in raw_obj:
                    r, g, b = raw_obj["colorRed"], raw_obj["colorGreen"], raw_obj["colorBlue"]
                    if len(r) > 0:
                        max_val = max(np.max(r), np.max(g), np.max(b))
                        denom = 255.0 if max_val <= 255 else 65535.0
                        colors = np.vstack((r, g, b)).T / denom
                        try:
                            pcd.colors = o3d.utility.Vector3dVector(colors)
                        except MemoryError:
                            pass
                        del colors
        finally:
            e57.close()

    else:
        raise ValueError(f"Unsupported point cloud format: '{ext}'. Supported formats: .las, .laz, .ply, .e57")

    return raw_obj, pcd, ext


def save_aligned_cloud(
    raw_obj: Any,
    transformation: np.ndarray,
    output_path: str,
    ext: Optional[str] = None,
    source_path: Optional[str] = None
) -> None:
    """
    Applies the 4x4 transformation matrix to point coordinates and writes the aligned
    cloud to disk, preserving headers, classification codes, intensities, and camera poses.
    """
    if ext is None:
        ext = os.path.splitext(output_path)[1].lower()
    else:
        ext = ext.lower()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Fallback: re-read source raw object if None
    if raw_obj is None and source_path and os.path.exists(source_path):
        if ext in ['.las', '.laz']:
            raw_obj = laspy.read(source_path)
        elif ext == '.ply':
            raw_obj = o3d.io.read_point_cloud(source_path)
        elif ext == '.e57':
            if is_e57_available():
                import pye57
                with pye57.E57(source_path, mode='r') as e57_fallback:
                    try:
                        raw_obj = e57_fallback.read_scan_raw(0, ignore_unsupported_fields=True)
                    except Exception:
                        raw_obj = e57_fallback.read_scan(0, ignore_missing_fields=True)

    if ext in ['.las', '.laz']:
        if hasattr(raw_obj, 'x'):
            original_points = np.vstack((raw_obj.x, raw_obj.y, raw_obj.z)).T
            hom_points = np.hstack((original_points, np.ones((len(original_points), 1))))
            transformed_points = (hom_points @ transformation.T)[:, :3]

            new_header = copy.deepcopy(raw_obj.header)
            new_header.offsets = np.min(transformed_points, axis=0)
            new_header.mins = np.min(transformed_points, axis=0)
            new_header.maxs = np.max(transformed_points, axis=0)

            new_las = laspy.LasData(new_header)
            for dim_name in new_header.point_format.dimension_names:
                new_las[dim_name] = raw_obj[dim_name]

            new_las.x = transformed_points[:, 0]
            new_las.y = transformed_points[:, 1]
            new_las.z = transformed_points[:, 2]
            new_las.write(output_path)
        else:
            if isinstance(raw_obj, dict):
                original_points = np.vstack((raw_obj["cartesianX"], raw_obj["cartesianY"], raw_obj["cartesianZ"])).T
            elif hasattr(raw_obj, 'points'):
                original_points = np.asarray(raw_obj.points)
            else:
                raise ValueError("No valid point coordinates found in raw_obj to write LAS.")
            hom_points = np.hstack((original_points, np.ones((len(original_points), 1))))
            transformed_points = (hom_points @ transformation.T)[:, :3]
            hdr = laspy.LasHeader(point_format=3, version="1.4")
            hdr.offsets = np.min(transformed_points, axis=0)
            hdr.scales = [0.001, 0.001, 0.001]
            new_las = laspy.LasData(hdr)
            new_las.x = transformed_points[:, 0]
            new_las.y = transformed_points[:, 1]
            new_las.z = transformed_points[:, 2]
            new_las.write(output_path)

    elif ext == '.ply':
        if hasattr(raw_obj, 'transform'):
            pcd_transformed = copy.deepcopy(raw_obj)
            pcd_transformed.transform(transformation)
        else:
            if isinstance(raw_obj, dict):
                original_points = np.vstack((raw_obj["cartesianX"], raw_obj["cartesianY"], raw_obj["cartesianZ"])).T
            elif hasattr(raw_obj, 'x'):
                original_points = np.vstack((raw_obj.x, raw_obj.y, raw_obj.z)).T
            elif hasattr(raw_obj, 'points'):
                original_points = np.asarray(raw_obj.points)
            else:
                raise ValueError("No valid point coordinates found in raw_obj to write PLY.")
            hom_points = np.hstack((original_points, np.ones((len(original_points), 1))))
            transformed_points = (hom_points @ transformation.T)[:, :3]
            pcd_transformed = o3d.geometry.PointCloud()
            pcd_transformed.points = o3d.utility.Vector3dVector(transformed_points)
        o3d.io.write_point_cloud(output_path, pcd_transformed)

    elif ext == '.e57':
        colors = None
        intensities = None

        if isinstance(raw_obj, dict):
            original_points = np.vstack((raw_obj["cartesianX"], raw_obj["cartesianY"], raw_obj["cartesianZ"])).T
            if "colorRed" in raw_obj and "colorGreen" in raw_obj and "colorBlue" in raw_obj:
                r, g, b = raw_obj["colorRed"], raw_obj["colorGreen"], raw_obj["colorBlue"]
                if len(r) > 0:
                    max_val = max(np.max(r), np.max(g), np.max(b))
                    denom = 255.0 if max_val <= 255 else 65535.0
                    colors = np.vstack((r, g, b)).T / denom
            if "intensity" in raw_obj:
                intensities = np.asarray(raw_obj["intensity"])
        elif hasattr(raw_obj, 'x'):
            original_points = np.vstack((raw_obj.x, raw_obj.y, raw_obj.z)).T
            if hasattr(raw_obj, 'red') and hasattr(raw_obj, 'green') and hasattr(raw_obj, 'blue'):
                r, g, b = np.asarray(raw_obj.red), np.asarray(raw_obj.green), np.asarray(raw_obj.blue)
                if len(r) > 0:
                    max_val = max(np.max(r), np.max(g), np.max(b))
                    denom = 255.0 if max_val <= 255 else 65535.0
                    colors = np.vstack((r, g, b)).T / denom
            if hasattr(raw_obj, 'intensity'):
                intensities = np.asarray(raw_obj.intensity)
        elif hasattr(raw_obj, 'points'):
            original_points = np.asarray(raw_obj.points)
            if hasattr(raw_obj, 'has_colors') and raw_obj.has_colors():
                colors = np.asarray(raw_obj.colors)
        else:
            raise ValueError("No valid point coordinates found in raw_obj to write E57.")

        hom_points = np.hstack((original_points, np.ones((len(original_points), 1))))
        transformed_points = (hom_points @ transformation.T)[:, :3]

        src_e57 = source_path if (source_path and source_path.lower().endswith('.e57')) else None
        write_e57_point_cloud(
            file_path=output_path,
            points=transformed_points,
            colors=colors,
            intensity=intensities,
            source_e57_for_panoramas=src_e57,
            transform_matrix=transformation
        )
    else:
        raise ValueError(f"Unsupported format for writing: {ext}")
