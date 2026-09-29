"""
E57 Point Cloud and Spherical Panoramas Handler
================================================
Handles reading, writing, and transforming E57 files with full preservation of:
- High-precision 64-bit cartesian coordinates (X, Y, Z)
- RGB colors and intensity / reflectance scalar fields
- 360-degree spherical panoramic imagery (images2D)
- Scanner poses and camera rototranslation during point cloud registration
- Windows anti-file-lock safe handles
"""

import os
import math
from typing import Dict, Optional, Any
import numpy as np

try:
    import pye57
    from pye57 import libe57
    from pye57.utils import copy_node, get_node
    E57_AVAILABLE = True
except ImportError:
    E57_AVAILABLE = False


def is_e57_available() -> bool:
    """Returns True if pye57 is installed and functional in the Python environment."""
    return E57_AVAILABLE


def _quaternion_to_matrix(quaternion_wxyz):
    w, x, y, z = quaternion_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _matrix_to_quaternion(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return [qw, qx, qy, qz]


def copy_images2d(
    source_e57: Any,
    target_e57: Any,
    transform_matrix: Optional[np.ndarray] = None
) -> int:
    """
    Copies all 2D panoramic images (images2D) from source_e57 to target_e57,
    applying the 4x4 rigid registration transformation matrix to camera positions
    and rotation quaternions.
    """
    if not E57_AVAILABLE:
        return 0

    try:
        if not source_e57.root.isDefined("images2D"):
            return 0

        images = source_e57.root["images2D"]
        count = len(images)
        if count == 0:
            return 0

        if not target_e57.root.isDefined("images2D"):
            target_e57.root.set("images2D", libe57.VectorNode(target_e57.image_file, False))

        target_images = target_e57.root["images2D"]
        copied = 0

        R_align = np.asarray(transform_matrix[:3, :3], dtype=np.float64) if transform_matrix is not None else None
        t_align = np.asarray(transform_matrix[:3, 3], dtype=np.float64) if transform_matrix is not None else None

        for i in range(count):
            try:
                src_img = images[i]
                has_pose = src_img.isDefined("pose")
                pos_trans = None
                quat_trans = None
                has_trans = False
                has_rot = False

                if has_pose:
                    pose_src = src_img["pose"]
                    has_trans = pose_src.isDefined("translation")
                    has_rot = pose_src.isDefined("rotation")

                    if has_trans:
                        t_node = pose_src["translation"]
                        pos = np.array([t_node["x"].value(), t_node["y"].value(), t_node["z"].value()], dtype=np.float64)
                        pos_trans = (R_align @ pos + t_align) if (R_align is not None or t_align is not None) else pos

                    if has_rot:
                        r_node = pose_src["rotation"]
                        quat = [r_node["w"].value(), r_node["x"].value(), r_node["y"].value(), r_node["z"].value()]
                        if R_align is not None:
                            R_cam = _quaternion_to_matrix(quat)
                            R_cam_trans = R_align @ R_cam
                            quat_trans = _matrix_to_quaternion(R_cam_trans)
                        else:
                            quat_trans = quat

                if has_pose and (R_align is not None or t_align is not None):
                    dest_img = target_e57.image_file
                    copied_image = libe57.StructureNode(dest_img)
                    blob_pairs = []
                    for child_idx in range(src_img.childCount()):
                        in_child = get_node(src_img, child_idx)
                        c_name = in_child.elementName()
                        if c_name == "pose":
                            new_pose_node = libe57.StructureNode(dest_img)
                            if has_trans and pos_trans is not None:
                                t_struct = libe57.StructureNode(dest_img)
                                t_struct.set("x", libe57.FloatNode(dest_img, float(pos_trans[0])))
                                t_struct.set("y", libe57.FloatNode(dest_img, float(pos_trans[1])))
                                t_struct.set("z", libe57.FloatNode(dest_img, float(pos_trans[2])))
                                new_pose_node.set("translation", t_struct)
                            if has_rot and quat_trans is not None:
                                r_struct = libe57.StructureNode(dest_img)
                                r_struct.set("w", libe57.FloatNode(dest_img, float(quat_trans[0])))
                                r_struct.set("x", libe57.FloatNode(dest_img, float(quat_trans[1])))
                                r_struct.set("y", libe57.FloatNode(dest_img, float(quat_trans[2])))
                                r_struct.set("z", libe57.FloatNode(dest_img, float(quat_trans[3])))
                                new_pose_node.set("rotation", r_struct)
                            copied_image.set("pose", new_pose_node)
                        else:
                            out_child, _, b_pairs = copy_node(in_child, dest_img)
                            copied_image.set(c_name, out_child)
                            blob_pairs.extend(b_pairs)
                else:
                    copied_image, _compressed_pairs, blob_pairs = copy_node(src_img, target_e57.image_file)

                target_images.append(copied_image)
                for pair in blob_pairs:
                    data = pair["in"].read_buffer()
                    pair["out"].write(data, 0, pair["in"].byteCount())
                copied += 1
            except Exception as img_err:
                print(f"[E57 Handler] Warning: panoramic image {i} copy failed: {img_err}")

        return copied
    except Exception as e:
        print(f"[E57 Handler] Warning during copy_images2d: {e}")
        return 0


def copy_images2d_from_file(
    source_e57_path: str,
    target_e57: Any,
    transform_matrix: Optional[np.ndarray] = None
) -> int:
    """Reads source_e57_path, copies all 2D panoramas into target_e57, then closes the file safely."""
    if not E57_AVAILABLE or not source_e57_path or not os.path.exists(source_e57_path):
        return 0

    if not source_e57_path.lower().endswith(".e57"):
        return 0

    src = None
    try:
        src = pye57.E57(source_e57_path, mode="r")
        return copy_images2d(src, target_e57, transform_matrix=transform_matrix)
    except Exception as e:
        print(f"[E57 Handler] Could not extract panoramas from '{source_e57_path}': {e}")
        return 0
    finally:
        if src is not None:
            try:
                src.close()
            except Exception:
                pass


def write_e57_point_cloud(
    file_path: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    source_e57_for_panoramas: Optional[str] = None,
    extra_fields: Optional[Dict[str, np.ndarray]] = None,
    transform_matrix: Optional[np.ndarray] = None,
    **kwargs
) -> None:
    """
    Writes a point cloud to an E57 file preserving:
    - 64-bit cartesian coordinates
    - RGB colors (uint8)
    - Intensity values (float32)
    - Spherical 360-degree panoramas (images2D) copied & rototranslated from source
    """
    if not E57_AVAILABLE:
        raise ImportError("pye57 is not installed in the active Python environment.")

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Invalid point shape: {pts.shape}, expected (N, 3)")

    data_dict = {
        "cartesianX": pts[:, 0],
        "cartesianY": pts[:, 1],
        "cartesianZ": pts[:, 2]
    }

    if colors is not None and len(colors) == len(pts):
        c = np.asarray(colors)
        if np.max(c) <= 1.0:
            c8 = np.clip(c * 255.0, 0, 255).astype(np.uint8)
        elif np.max(c) > 255:
            c8 = np.clip(c / 257.0, 0, 255).astype(np.uint8)
        else:
            c8 = np.clip(c, 0, 255).astype(np.uint8)

        data_dict["colorRed"] = c8[:, 0]
        data_dict["colorGreen"] = c8[:, 1]
        data_dict["colorBlue"] = c8[:, 2]

    if intensity is not None and len(intensity) == len(pts):
        data_dict["intensity"] = np.asarray(intensity, dtype=np.float32)

    if extra_fields:
        for k, v in extra_fields.items():
            if k not in data_dict and len(v) == len(pts):
                data_dict[k] = v

    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    dst_e57 = pye57.E57(file_path, mode="w")
    try:
        if source_e57_for_panoramas and os.path.exists(source_e57_for_panoramas):
            copy_images2d_from_file(
                source_e57_for_panoramas,
                dst_e57,
                transform_matrix=transform_matrix
            )

        dst_e57.write_scan_raw(data_dict)
    finally:
        dst_e57.close()


def read_e57_point_cloud(file_path: str) -> Dict[str, Any]:
    """Reads 3D points, colors and intensity from an E57 scan."""
    if not E57_AVAILABLE:
        raise ImportError("pye57 is not installed in the active Python environment.")
    e57 = pye57.E57(file_path, mode="r")
    try:
        raw = e57.read_scan_raw(0, ignore_unsupported_fields=True)
        pts = np.vstack((raw["cartesianX"], raw["cartesianY"], raw["cartesianZ"])).T
        colors = None
        if "colorRed" in raw and "colorGreen" in raw and "colorBlue" in raw:
            r, g, b = raw["colorRed"], raw["colorGreen"], raw["colorBlue"]
            if len(r) > 0:
                max_v = max(np.max(r), np.max(g), np.max(b))
                denom = 65535.0 if max_v > 255 else 255.0
                colors = np.vstack((r, g, b)).T / denom
        intensity = raw.get("intensity")
        return {
            "points": pts,
            "colors": colors,
            "intensity": intensity,
            "raw": raw
        }
    finally:
        e57.close()


def check_e57_panoramas_status(file_path: str) -> Dict[str, Any]:
    """
    Checks if an E57 file contains panoramic images (images2D) and validates
    whether they possess valid 3D camera pose metadata (translation and rotation).
    """
    result = {
        "is_e57": False,
        "total_images": 0,
        "has_panoramas": False,
        "poses_count": 0,
        "has_panopose": False,
        "status_type": "none",
        "message": ""
    }

    if not file_path or not os.path.exists(file_path):
        result["message"] = "File not found or unspecified."
        return result

    ext = os.path.splitext(file_path)[1].lower()
    if ext != ".e57":
        result["is_e57"] = False
        result["message"] = f"Format {ext.upper()} does not support spherical 360-degree panoramas."
        return result

    result["is_e57"] = True

    if not E57_AVAILABLE:
        result["status_type"] = "info"
        result["message"] = "pye57 is not available to inspect panoramas."
        return result

    try:
        with pye57.E57(file_path, mode="r") as e57:
            if not e57.root.isDefined("images2D"):
                result["has_panoramas"] = False
                result["status_type"] = "info"
                result["message"] = "No panoramic images (images2D) embedded in E57 file."
                return result

            images = e57.root["images2D"]
            count = len(images)
            result["total_images"] = count
            result["has_panoramas"] = (count > 0)

            if count == 0:
                result["status_type"] = "info"
                result["message"] = "No panoramic images embedded in E57 file."
                return result

            poses_count = 0
            for i in range(count):
                try:
                    img = images[i]
                    if img.isDefined("pose"):
                        p = img["pose"]
                        if p.isDefined("translation") and p.isDefined("rotation"):
                            poses_count += 1
                except Exception:
                    pass

            result["poses_count"] = poses_count
            result["has_panopose"] = (poses_count > 0)

            if poses_count == count:
                result["status_type"] = "ok"
                result["message"] = f"OK: {count} spherical panoramas found with valid camera poses (panopose)."
            elif poses_count == 0:
                result["status_type"] = "warning"
                result["message"] = f"Warning: {count} panoramas found but ZERO camera poses (coordinates missing)."
            else:
                result["status_type"] = "warning"
                result["message"] = f"Warning: Only {poses_count}/{count} panoramas have valid camera poses."

            return result
    except Exception as e:
        result["status_type"] = "error"
        result["message"] = f"Error inspecting E57 panoramas: {e}"
        return result
