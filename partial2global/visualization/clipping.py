"""
PARTiaL2GLOBAL Interactive Clipping Box Manager
===============================================
Gestore del box interattivo di ritaglio e sezionamento 3D OBB per PyVista/VTK.
Permette il sezionamento tramite click-and-drag diretto delle 6 facce (senza frecce),
con aggiornamento in tempo reale dei piani di clipping hardware GPU (vtkPlane).
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import numpy as np
try:
    import pyvista as pv
    import vtk
    if hasattr(vtk, 'vtkObject'):
        vtk.vtkObject.GlobalWarningDisplayOff()
except ImportError:
    pv = None
    vtk = None

try:
    from ..core.models import OBBData
except (ImportError, ValueError):
    try:
        from core.models import OBBData
    except ImportError:
        @dataclass
        class OBBData:
            mesh: Optional[object]
            center: np.ndarray
            axes: np.ndarray
            half_extents: np.ndarray


class ClippingManager:
    """
    Gestione clipping box OBB con trascinamento diretto e fluido delle 6 facce senza frecce.
    
    Caratteristiche:
    - Selezione e trascinamento delle facce con ray-casting 3D mouse -> asse normale della faccia.
    - Nessun widget con frecce/maniglie da dover allineare o trascinare.
    - Piani di taglio GPU (vtkPlane) agganciati direttamente ai mapper degli attori target.
    - Limitazione (clamp) automatica per impedire inversioni o sconfinamenti oltre l'OBB originale.
    - Scorciatoie da tastiera integrate: [C] Attiva/Disattiva, [R] Reset limiti, [?] Aiuto.
    """

    MIN_HALF_EXTENT = 0.02  # metri minimi per asse (evita inversioni della mesh)
    RENDER_INTERVAL_SEC = 1.0 / 60.0  # limitatore 60 FPS per fluidità assoluta
    PICK_TOLERANCE = 0.005

    def __init__(
        self,
        plotter,
        obb_data: Optional[OBBData] = None,
        target_actor_names: Optional[Union[str, List[str]]] = None,
        mesh: Optional[object] = None,
        name: str = "Cloud",
        on_disable_planes_cb=None,
        on_reset_button_cb=None,
        on_clip_applied_cb=None,
        allow_crop: bool = False,
    ):
        self.plotter = plotter
        self.obb_data_init = obb_data
        self.mesh = mesh
        self.default_name = name
        self.allow_crop = allow_crop

        if target_actor_names is not None:
            if isinstance(target_actor_names, str):
                self.target_actor_names = [target_actor_names]
            else:
                self.target_actor_names = list(target_actor_names)
        else:
            self.target_actor_names = [name] if name else None

        self.on_disable_planes_cb = on_disable_planes_cb
        self.on_reset_button_cb = on_reset_button_cb
        self.on_clip_applied_cb = on_clip_applied_cb

        self.clip_active = False
        self.clipping_planes: List[object] = []
        self.obb: Optional[OBBData] = None

        self.dragging = False
        self.active_idx: Optional[int] = None
        self.active_cell_id: Optional[int] = None

        self.box: Optional[pv.PolyData] = None
        self.outline: Optional[pv.PolyData] = None
        self.faccia_evidenziata: Optional[pv.UnstructuredGrid] = None

        self.box_actor = None
        self.outline_actor = None
        self.faccia_actor = None
        self.picker = None
        self.prev_interactor_style = None

        self.base_obb_center = None
        self.base_obb_axes = None
        self.base_obb_half_extents = None

        # Stato raycasting per il drag
        self.drag_start_center = None
        self.drag_start_half_extents = None
        self.drag_axis_origin = None
        self.drag_axis_dir = None
        self.drag_last_delta = 0.0

        self._last_render_time = 0.0
        self.obs_press = None
        self.obs_move = None
        self.obs_release = None
        self.hud_actor = None

    @property
    def is_clipping(self) -> bool:
        """Compatibilità con la precedente API."""
        return self.clip_active

    # ------------------------------------------------------------------
    # Risoluzione attori target
    # ------------------------------------------------------------------
    def _get_target_actors(self):
        """Restituisce gli attori della scena sui quali applicare i piani di taglio GPU."""
        actors_dict = getattr(self.plotter, "actors", {}) or {}
        # Supporto se plotter usa renderer.actors
        if not actors_dict and hasattr(self.plotter, "renderer"):
            actors_dict = getattr(self.plotter.renderer, "actors", {}) or {}

        targets = []
        if self.target_actor_names:
            for name in self.target_actor_names:
                act = actors_dict.get(name)
                if act and hasattr(act, "GetMapper") and act.GetMapper():
                    targets.append(act)

        # Se non trovati per nome esplicito, cerca tutti gli attori nuvola/mesh escludendo i widget
        if not targets:
            excluded = {
                "CustomClipBox", "CustomClipOutline", "CustomClipFace",
                "Automatic OBB", "Target Cloud OBB", "HelpText", "ClipBoxHUD",
                "bounding_box", "BoundingBox"
            }
            for name, act in actors_dict.items():
                if name in excluded or name.startswith("viewcube") or name.startswith("__"):
                    continue
                if hasattr(act, "GetMapper") and act.GetMapper():
                    targets.append(act)

        return targets

    # ------------------------------------------------------------------
    # Utility numeriche e ortonormalizzazione
    # ------------------------------------------------------------------
    def _orthonormalize_axes(self, axes: np.ndarray) -> np.ndarray:
        """Rende gli assi una base ortonormale destrorsa per evitare skew o distorsioni."""
        arr = np.asarray(axes, dtype=np.float64)
        if arr.shape != (3, 3):
            return np.eye(3, dtype=np.float64)

        rows = []
        for i in range(3):
            v = np.asarray(arr[i], dtype=np.float64)
            n = float(np.linalg.norm(v))
            if not np.isfinite(n) or n < 1e-12:
                return np.eye(3, dtype=np.float64)
            rows.append(v / n)
        arr = np.vstack(rows)

        try:
            u, _, vh = np.linalg.svd(arr, full_matrices=False)
            ortho = u @ vh
            if np.linalg.det(ortho) < 0.0:
                ortho[2] *= -1.0
            for i in range(3):
                ortho[i] /= max(float(np.linalg.norm(ortho[i])), 1e-12)
            return ortho.astype(np.float64)
        except Exception:
            return np.eye(3, dtype=np.float64)

    def _sanitize_half_extents(self, half_extents: np.ndarray) -> np.ndarray:
        e = np.asarray(half_extents, dtype=np.float64).reshape(3)
        e = np.nan_to_num(e, nan=self.MIN_HALF_EXTENT, posinf=self.MIN_HALF_EXTENT, neginf=self.MIN_HALF_EXTENT)
        return np.maximum(np.abs(e), self.MIN_HALF_EXTENT)

    def _safe_render(self, force: bool = False):
        now = time.perf_counter()
        if force or (now - self._last_render_time) >= self.RENDER_INTERVAL_SEC:
            try:
                self.plotter.render()
            except Exception:
                pass
            self._last_render_time = now

    def _safe_remove_actor(self, name: str):
        try:
            self.plotter.remove_actor(name, render=False)
        except TypeError:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Geometria OBB a 6 celle/facce
    # ------------------------------------------------------------------
    def reconstruct_obb_mesh(self) -> pv.PolyData:
        """
        Costruisce una mesh solida OBB a 6 celle corrispondenti alle facce:
        0=-axis0, 1=+axis0, 2=-axis1, 3=+axis1, 4=-axis2, 5=+axis2.
        """
        self.obb.axes = self._orthonormalize_axes(self.obb.axes)
        self.obb.half_extents = self._sanitize_half_extents(self.obb.half_extents)

        c = np.asarray(self.obb.center, dtype=np.float64)
        a = self.obb.axes
        e = self.obb.half_extents

        local_signs = np.array(
            [
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
            ],
            dtype=np.float64,
        )
        points = c + (local_signs * e) @ a

        # Facce con winding coerente per normali esterne
        faces = np.array(
            [
                4, 0, 4, 7, 3,  # 0: -axis0
                4, 1, 2, 6, 5,  # 1: +axis0
                4, 0, 1, 5, 4,  # 2: -axis1
                4, 3, 7, 6, 2,  # 3: +axis1
                4, 0, 3, 2, 1,  # 4: -axis2
                4, 4, 5, 6, 7,  # 5: +axis2
            ],
            dtype=np.int64,
        )
        mesh = pv.PolyData(points, faces)
        try:
            mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        except Exception:
            pass
        return mesh

    def aggiornaPianiGpu(self):
        """Aggiorna le origini e le normali dei 6 piani di ritaglio vtkPlane."""
        if not self.obb or not self.clipping_planes:
            return

        self.obb.axes = self._orthonormalize_axes(self.obb.axes)
        self.obb.half_extents = self._sanitize_half_extents(self.obb.half_extents)

        axes = self.obb.axes
        center = self.obb.center
        half = self.obb.half_extents

        normals = [
            axes[0], -axes[0],
            axes[1], -axes[1],
            axes[2], -axes[2],
        ]
        origins = [
            center - axes[0] * half[0],
            center + axes[0] * half[0],
            center - axes[1] * half[1],
            center + axes[1] * half[1],
            center - axes[2] * half[2],
            center + axes[2] * half[2],
        ]

        for i in range(6):
            self.clipping_planes[i].SetNormal(normals[i])
            self.clipping_planes[i].SetOrigin(origins[i])

    def reapply_clipping_planes(self):
        """Riapplica i 6 piani di ritaglio VTK GPU agli attori target.
        Utile quando gli attori vengono rigenerati o sovrascritti (es. cambio modalità di colore)."""
        if not self.clipping_planes:
            return

        target_actors = self._get_target_actors()
        for act in target_actors:
            mapper = act.GetMapper()
            if mapper:
                mapper.RemoveAllClippingPlanes()
                if self.clip_active:
                    for plane in self.clipping_planes:
                        mapper.AddClippingPlane(plane)

        if self.clip_active:
            self.aggiornaPianiGpu()

    def _update_hud(self):
        """Aggiorna il banner HUD a video che indica lo stato della ClipBox."""
        try:
            if self.clip_active:
                if self.allow_crop:
                    msg = "[C] ClipBox: ATTIVA (trascina facce) | [K] / [S] ✂️ Applica Ritaglio ROI | [R] Reset | [?] Aiuto"
                else:
                    msg = "[C] ClipBox: ATTIVA (trascina facce) | [R] Reset | [?] Aiuto"
            else:
                msg = "[C] ClipBox: DISATTIVA | [?] Aiuto"
            if self.hud_actor is None:
                self.hud_actor = self.plotter.add_text(
                    msg,
                    position=(15, 14),
                    font_size=9,
                    color="#38bdf8" if self.clip_active else "#94a3b8",
                    font="courier",
                    shadow=True,
                    name="ClipBoxHUD"
                )
            else:
                if hasattr(self.hud_actor, "SetInput"):
                    self.hud_actor.SetInput(msg)
                prop = self.hud_actor.GetTextProperty()
                if prop:
                    rgb = (0.22, 0.74, 0.97) if self.clip_active else (0.58, 0.64, 0.72)
                    prop.SetColor(*rgb)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Attivazione e Disattivazione
    # ------------------------------------------------------------------
    def enable(self, obb_data: Optional[OBBData] = None):
        """Abilita la ClipBox OBB e il ritaglio GPU."""
        if self.clip_active:
            return

        if self.on_disable_planes_cb:
            try:
                self.on_disable_planes_cb()
            except Exception:
                pass

        # Determina la OBB di riferimento
        source_obb = obb_data or self.obb or self.obb_data_init
        if source_obb is None:
            pts = None
            if self.mesh is not None and hasattr(self.mesh, "points"):
                pts = np.asarray(self.mesh.points, dtype=np.float64)
            elif self.plotter is not None:
                for act in self._get_target_actors():
                    mapper = act.GetMapper()
                    if mapper and hasattr(mapper, "GetInput") and mapper.GetInput():
                        inp = mapper.GetInput()
                        if hasattr(inp, "GetPoints") and inp.GetPoints():
                            import vtk.util.numpy_support as vtk_np
                            pts = vtk_np.vtk_to_numpy(inp.GetPoints().GetData())
                            break

            if pts is not None and len(pts) > 0:
                try:
                    from ..core.bounding_box import computeobb
                except (ImportError, ValueError):
                    try:
                        from core.bounding_box import computeobb
                    except ImportError:
                        computeobb = None

                if computeobb:
                    try:
                        source_obb = computeobb(pts)
                    except Exception:
                        source_obb = None

            if source_obb is None:
                bounds = None
                if pts is not None and len(pts) > 0:
                    bounds = [
                        np.min(pts[:, 0]), np.max(pts[:, 0]),
                        np.min(pts[:, 1]), np.max(pts[:, 1]),
                        np.min(pts[:, 2]), np.max(pts[:, 2])
                    ]
                elif hasattr(self.plotter, "bounds"):
                    bounds = self.plotter.bounds
                else:
                    bounds = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

                c = np.array([
                    (bounds[0] + bounds[1]) / 2.0,
                    (bounds[2] + bounds[3]) / 2.0,
                    (bounds[4] + bounds[5]) / 2.0
                ], dtype=np.float64)
                h = np.array([
                    max((bounds[1] - bounds[0]) / 2.0, self.MIN_HALF_EXTENT),
                    max((bounds[3] - bounds[2]) / 2.0, self.MIN_HALF_EXTENT),
                    max((bounds[5] - bounds[4]) / 2.0, self.MIN_HALF_EXTENT)
                ], dtype=np.float64)
                source_obb = OBBData(mesh=None, center=c, axes=np.eye(3, dtype=np.float64), half_extents=h)

        if source_obb is None:
            return

        self.obb = OBBData(
            mesh=getattr(source_obb, "mesh", None),
            center=np.asarray(source_obb.center, dtype=np.float64).copy(),
            axes=self._orthonormalize_axes(np.asarray(source_obb.axes, dtype=np.float64).copy()),
            half_extents=self._sanitize_half_extents(source_obb.half_extents),
        )

        if self.base_obb_center is None:
            self.base_obb_center = np.copy(self.obb.center)
            self.base_obb_axes = self._orthonormalize_axes(np.copy(self.obb.axes))
            self.base_obb_half_extents = self._sanitize_half_extents(self.obb.half_extents)

        # Inizializza i piani di ritaglio VTK GPU
        if not self.clipping_planes:
            self.clipping_planes = [vtk.vtkPlane() for _ in range(6)]

        self.clip_active = True
        self.reapply_clipping_planes()

        # Costruisce la geometria interattiva del box
        self.box = self.reconstruct_obb_mesh()
        self.box_actor = self.plotter.add_mesh(
            self.box,
            color="white",
            opacity=0.001,
            name="CustomClipBox",
            reset_camera=False,
            render=False,
            pickable=True,
        )

        self.outline = self.box.extract_all_edges()
        self.outline_actor = self.plotter.add_mesh(
            self.outline,
            color="#00e5ff",
            line_width=2.5,
            name="CustomClipOutline",
            reset_camera=False,
            render=False,
            pickable=False,
        )

        self.faccia_evidenziata = self.box.extract_cells(0)
        self.faccia_actor = self.plotter.add_mesh(
            self.faccia_evidenziata,
            color="red",
            opacity=0.0,
            name="CustomClipFace",
            reset_camera=False,
            render=False,
            pickable=False,
        )

        # Nasconde OBB statico duplicato se presente nella scena
        actors_dict = getattr(self.plotter, "actors", {}) or {}
        for obb_name in ("Automatic OBB", "Target Cloud OBB"):
            if obb_name in actors_dict and hasattr(actors_dict[obb_name], "SetVisibility"):
                actors_dict[obb_name].SetVisibility(False)

        # Picker per intercettare il click esclusivo sulla faccia del box
        if self.picker is None:
            self.picker = vtk.vtkCellPicker()
            self.picker.SetTolerance(self.PICK_TOLERANCE)
            self.picker.PickFromListOn()
        else:
            self.picker.InitializePickList()
        self.picker.AddPickList(self.box_actor)

        # Observer eventi mouse con priorità elevata
        interactor = getattr(getattr(self.plotter, "iren", None), "interactor", None)
        if interactor is not None:
            self.obs_press = interactor.AddObserver("LeftButtonPressEvent", self.onPressClip, 100.0)
            self.obs_move = interactor.AddObserver("MouseMoveEvent", self.onMoveClip, 100.0)
            self.obs_release = interactor.AddObserver("LeftButtonReleaseEvent", self.onReleaseClip, 100.0)

        self.clip_active = True
        self._update_hud()
        self._safe_render(force=True)

    def disable(self):
        """Disattiva la ClipBox e rimuove il ritaglio GPU."""
        if not self.clip_active:
            return

        interactor = getattr(getattr(self.plotter, "iren", None), "interactor", None)
        if interactor is not None:
            for obs_name in ("obs_press", "obs_move", "obs_release"):
                obs_id = getattr(self, obs_name, None)
                if obs_id is not None:
                    try:
                        interactor.RemoveObserver(obs_id)
                    except Exception:
                        pass
                    setattr(self, obs_name, None)

        self._restore_interactor_style()
        self._clear_drag_state()

        self._safe_remove_actor("CustomClipBox")
        self._safe_remove_actor("CustomClipOutline")
        self._safe_remove_actor("CustomClipFace")

        # Rimuove piani di ritaglio dai mapper
        for act in self._get_target_actors():
            mapper = act.GetMapper()
            if mapper:
                mapper.RemoveAllClippingPlanes()

        # Ripristina visibilità dell'OBB statico se presente
        actors_dict = getattr(self.plotter, "actors", {}) or {}
        for obb_name in ("Automatic OBB", "Target Cloud OBB"):
            if obb_name in actors_dict and hasattr(actors_dict[obb_name], "SetVisibility"):
                actors_dict[obb_name].SetVisibility(True)

        self.clip_active = False
        self._update_hud()
        self._safe_render(force=True)

    def toggle_clipping(self):
        """Attiva o disattiva la ClipBox."""
        if self.clip_active:
            self.disable()
        else:
            self.enable()

    def reset(self):
        """Ripristina la ClipBox ai limiti geometrici originali dell'intera nuvola."""
        if self.base_obb_center is not None and self.base_obb_half_extents is not None:
            self.obb.center = np.copy(self.base_obb_center)
            self.obb.axes = self._orthonormalize_axes(np.copy(self.base_obb_axes))
            self.obb.half_extents = self._sanitize_half_extents(self.base_obb_half_extents)
            self.aggiornaPianiGpu()

            if self.clip_active and self.box is not None:
                nuova_mesh = self.reconstruct_obb_mesh()
                self.box.copy_from(nuova_mesh)
                if self.outline is not None:
                    self.outline.copy_from(nuova_mesh.extract_all_edges())
                if self.faccia_actor is not None and hasattr(self.faccia_actor, "GetProperty"):
                    self.faccia_actor.GetProperty().SetOpacity(0.0)

        if self.on_reset_button_cb:
            try:
                self.on_reset_button_cb()
            except Exception:
                pass

        self._safe_render(force=True)

    def _clear_drag_state(self):
        self.dragging = False
        self.active_idx = None
        self.active_cell_id = None
        self.drag_start_center = None
        self.drag_start_half_extents = None
        self.drag_axis_origin = None
        self.drag_axis_dir = None
        self.drag_last_delta = 0.0

    def _restore_interactor_style(self):
        interactor = getattr(getattr(self.plotter, "iren", None), "interactor", None)
        if interactor is not None:
            if self.prev_interactor_style is not None:
                interactor.SetInteractorStyle(self.prev_interactor_style)
            else:
                interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.prev_interactor_style = None

    # ------------------------------------------------------------------
    # Ray casting / Trascinamento vincolato su asse normale
    # ------------------------------------------------------------------
    def _event_position(self) -> Tuple[int, int]:
        try:
            return self.plotter.iren.get_event_position()
        except Exception:
            try:
                return self.plotter.iren.interactor.GetEventPosition()
            except Exception:
                return (0, 0)

    def _display_to_world(self, x: float, y: float, z: float) -> Optional[np.ndarray]:
        renderer = self.plotter.renderer
        renderer.SetDisplayPoint(float(x), float(y), float(z))
        renderer.DisplayToWorld()
        wp = np.asarray(renderer.GetWorldPoint(), dtype=np.float64)
        if wp.shape[0] < 4 or abs(wp[3]) < 1e-12:
            return None
        p = wp[:3] / wp[3]
        if not np.all(np.isfinite(p)):
            return None
        return p

    def _mouse_ray(self, display_pos: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x, y = display_pos
        p0 = self._display_to_world(x, y, 0.0)
        p1 = self._display_to_world(x, y, 1.0)
        if p0 is None or p1 is None:
            return None
        d = p1 - p0
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            return None
        return p0, d / n

    def _axis_delta_from_mouse(self, display_pos: Tuple[int, int]) -> float:
        """Calcola lo spostamento proiettato lungo l'asse normale della faccia tramite intersezione raggi 3D."""
        if self.drag_axis_origin is None or self.drag_axis_dir is None:
            return self.drag_last_delta

        ray = self._mouse_ray(display_pos)
        if ray is None:
            return self.drag_last_delta

        ray_origin, ray_dir = ray
        axis_origin = self.drag_axis_origin
        axis_dir = self.drag_axis_dir

        w0 = axis_origin - ray_origin
        b = float(np.dot(axis_dir, ray_dir))
        d = float(np.dot(axis_dir, w0))
        e = float(np.dot(ray_dir, w0))
        denom = 1.0 - b * b

        if abs(denom) < 1e-8:
            return self.drag_last_delta

        t = (b * e - d) / denom
        if not np.isfinite(t):
            return self.drag_last_delta

        self.drag_last_delta = float(t)
        return self.drag_last_delta

    def _clamp_delta_to_limits(self, axis_idx: int, is_positive: bool, delta_dist: float) -> float:
        """Impedisce semi-estensioni negative e impedisce di espandere oltre l'OBB originaria."""
        if self.drag_start_center is None or self.drag_start_half_extents is None:
            return 0.0

        start_half = float(self.drag_start_half_extents[axis_idx])
        min_delta = 2.0 * (self.MIN_HALF_EXTENT - start_half)
        delta_dist = max(float(delta_dist), min_delta)

        if (
            self.base_obb_center is None or
            self.base_obb_axes is None or
            self.base_obb_half_extents is None
        ):
            return delta_dist

        axis = self.base_obb_axes[axis_idx]
        base_half = float(self.base_obb_half_extents[axis_idx])
        start_center_local = float(np.dot(self.drag_start_center - self.base_obb_center, axis))
        start_min = start_center_local - start_half
        start_max = start_center_local + start_half

        if is_positive:
            max_delta = base_half - start_max
        else:
            max_delta = start_min + base_half

        return min(delta_dist, max_delta)

    def _apply_drag_delta(self, delta_dist: float):
        if self.active_idx is None or self.drag_axis_dir is None:
            return

        axis_idx = self.active_idx // 2
        outward_normal = self.drag_axis_dir

        new_half = self.drag_start_half_extents.copy()
        new_center = self.drag_start_center.copy()
        new_half[axis_idx] = max(
            self.MIN_HALF_EXTENT,
            float(self.drag_start_half_extents[axis_idx]) + delta_dist * 0.5,
        )
        new_center = new_center + outward_normal * (delta_dist * 0.5)

        self.obb.half_extents = self._sanitize_half_extents(new_half)
        self.obb.center = np.asarray(new_center, dtype=np.float64)
        self.obb.axes = self._orthonormalize_axes(self.obb.axes)

        nuova_mesh = self.reconstruct_obb_mesh()
        self.box.copy_from(nuova_mesh)
        self.outline.copy_from(nuova_mesh.extract_all_edges())

        if self.faccia_evidenziata is not None and self.active_idx is not None:
            self.faccia_evidenziata.copy_from(self.box.extract_cells(int(self.active_idx)))

        self.aggiornaPianiGpu()
        self._safe_render(force=False)

    # ------------------------------------------------------------------
    # Handler Interazione Mouse
    # ------------------------------------------------------------------
    def onPressClip(self, obj, event):
        if not self.box_actor or self.box is None:
            return

        click_pos = self._event_position()
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)

        if self.picker.GetActor() != self.box_actor:
            return

        cell_id = int(self.picker.GetCellId())
        if cell_id < 0:
            return

        if 0 <= cell_id <= 5:
            self.active_idx = cell_id
        else:
            normal = np.asarray(self.box.cell_normals[cell_id], dtype=np.float64)
            axes = self.obb.axes
            outward_normals = [-axes[0], axes[0], -axes[1], axes[1], -axes[2], axes[2]]
            dots = [float(np.dot(normal, n)) for n in outward_normals]
            self.active_idx = int(np.argmax(dots))

        self.active_cell_id = self.active_idx
        axis_idx = self.active_idx // 2
        is_positive = (self.active_idx % 2) != 0
        outward_normal = self.obb.axes[axis_idx] if is_positive else -self.obb.axes[axis_idx]
        outward_normal = outward_normal / max(float(np.linalg.norm(outward_normal)), 1e-12)

        pick_pos = np.asarray(self.picker.GetPickPosition(), dtype=np.float64)
        if not np.all(np.isfinite(pick_pos)):
            pick_pos = self.obb.center + outward_normal * self.obb.half_extents[axis_idx]

        self.dragging = True
        self.drag_start_center = self.obb.center.copy()
        self.drag_start_half_extents = self.obb.half_extents.copy()
        self.drag_axis_origin = pick_pos.copy()
        self.drag_axis_dir = outward_normal.copy()
        self.drag_last_delta = 0.0

        # Evidenzia la faccia selezionata in rosso
        self.faccia_evidenziata.copy_from(self.box.extract_cells(int(self.active_idx)))
        if self.faccia_actor is not None and hasattr(self.faccia_actor, "GetProperty"):
            self.faccia_actor.GetProperty().SetOpacity(0.45)

        # Blocca l'interactor di rotazione telecamera durante il trascinamento della faccia
        interactor = getattr(getattr(self.plotter, "iren", None), "interactor", None)
        if interactor is not None:
            self.prev_interactor_style = interactor.GetInteractorStyle()
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleUser())

        self._safe_render(force=True)

    def onMoveClip(self, obj, event):
        if not self.dragging or self.active_idx is None:
            return

        click_pos = self._event_position()
        axis_idx = self.active_idx // 2
        is_positive = (self.active_idx % 2) != 0

        delta_dist = self._axis_delta_from_mouse(click_pos)
        delta_dist = self._clamp_delta_to_limits(axis_idx, is_positive, delta_dist)

        self._apply_drag_delta(delta_dist)

    def onReleaseClip(self, obj, event):
        if not self.dragging:
            return

        self.dragging = False
        if self.faccia_actor is not None and hasattr(self.faccia_actor, "GetProperty"):
            self.faccia_actor.GetProperty().SetOpacity(0.0)

        self._restore_interactor_style()
        self._safe_render(force=True)
        self._clear_drag_state()

        if self.on_clip_applied_cb:
            try:
                self.on_clip_applied_cb()
            except Exception:
                pass
