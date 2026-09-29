"""
GUI Background Worker Threads
=============================
Executes the registration pipeline in a background QThread to ensure
the PyQt5 user interface remains completely responsive during heavy computations.
"""

from typing import Optional
from PyQt5.QtCore import QThread, pyqtSignal

from ..registration.pipeline import run_registration_pipeline, AlignmentResult


class AlignmentWorker(QThread):
    """Background worker thread for running point cloud registration."""

    progress_signal = pyqtSignal(int, int, str)  # (current_step, total_steps, description)
    log_signal = pyqtSignal(str)                 # log text line
    finished_signal = pyqtSignal(object)         # AlignmentResult
    error_signal = pyqtSignal(str)               # exception message

    def __init__(
        self,
        source_path: str,
        target_path: str,
        voxel_size: Optional[float] = None,
        out_dir: Optional[str] = None,
        coarse_method: str = "fgr",
        fine_method: str = "point_to_plane",
        anneal: bool = True,
        ransac_max_iter: int = 4_000_000,
        auto_save_cloud: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.source_path = source_path
        self.target_path = target_path
        self.voxel_size = voxel_size
        self.out_dir = out_dir
        self.coarse_method = coarse_method
        self.fine_method = fine_method
        self.anneal = anneal
        self.ransac_max_iter = ransac_max_iter
        self.auto_save_cloud = auto_save_cloud

    def run(self):
        try:
            result = run_registration_pipeline(
                source_path=self.source_path,
                target_path=self.target_path,
                voxel_size=self.voxel_size,
                out_dir=self.out_dir,
                coarse_method=self.coarse_method,
                fine_method=self.fine_method,
                anneal=self.anneal,
                ransac_max_iter=self.ransac_max_iter,
                progress_callback=self._on_progress,
                log_callback=self._on_log,
                auto_save_cloud=self.auto_save_cloud
            )
            self.finished_signal.emit(result)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log_signal.emit(f"\n[ERROR] Pipeline aborted: {e}\n{tb}")
            self.error_signal.emit(str(e))

    def _on_progress(self, step: int, total: int, text: str):
        self.progress_signal.emit(step, total, text)

    def _on_log(self, message: str):
        self.log_signal.emit(message)
