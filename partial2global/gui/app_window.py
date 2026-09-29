"""
PARTiaL2GLOBAL Main Application Window
======================================
PyQt5 graphical user interface for partial-to-global point cloud registration,
real-time metric visualization, and 3D residual inspection.
"""

import os
import sys
import time
import shutil
import subprocess
from typing import Optional
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QDoubleSpinBox,
    QComboBox, QCheckBox, QGroupBox, QProgressBar, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QFrame, QHeaderView, QMessageBox,
    QScrollArea, QSplashScreen
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QClipboard, QIcon, QPixmap

from .styles import MODERN_STYLESHEET, METRIC_CARD_STYLE
from .workers import AlignmentWorker
from ..registration.pipeline import AlignmentResult
from ..core.bounding_box import estimate_optimal_voxel_size, estimate_optimal_voxel_from_file
from ..io.point_cloud_io import load_point_cloud, save_aligned_cloud
from ..visualization.visualizer import visualize_registration, PYVISTA_AVAILABLE


class Partial2GlobalApp(QMainWindow):
    """Main window for PARTiaL2GLOBAL desktop application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PARTiaL2GLOBAL: Partial-to-Global Point Cloud Registration Suite")
        self.resize(1100, 920)
        self.setMinimumSize(850, 700)

        self.logo_path = os.path.join(os.path.dirname(__file__), "resources", "logo.png")
        if os.path.exists(self.logo_path):
            self.setWindowIcon(QIcon(self.logo_path))

        self.last_result: Optional[AlignmentResult] = None
        self.worker: Optional[AlignmentWorker] = None

        self.init_ui()
        self.setStyleSheet(MODERN_STYLESHEET + METRIC_CARD_STYLE)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Header Bar
        header_layout = QHBoxLayout()
        header_layout.setSpacing(12)

        # Logo Icon in Header
        if hasattr(self, 'logo_path') and os.path.exists(self.logo_path):
            lbl_logo = QLabel()
            pix = QPixmap(self.logo_path).scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl_logo.setPixmap(pix)
            lbl_logo.setFixedSize(48, 48)
            lbl_logo.setStyleSheet("border-radius: 8px; background-color: #0f172a; padding: 2px;")
            header_layout.addWidget(lbl_logo)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        lbl_title = QLabel("PARTiaL2GLOBAL")
        lbl_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #1e3a8a;")
        lbl_sub = QLabel("Lightweight Partial-to-Global Point Cloud Registration & Metrology Suite")
        lbl_sub.setStyleSheet("font-size: 12px; color: #64748b;")

        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_sub)
        header_layout.addLayout(title_box)
        header_layout.addStretch()

        badge_version = QLabel("v1.0.0 Stable")
        badge_version.setStyleSheet(
            "background-color: #dbeafe; color: #1e40af; font-size: 11px; "
            "font-weight: 700; padding: 4px 10px; border-radius: 12px; border: 1px solid #bfdbfe;"
        )
        header_layout.addWidget(badge_version)
        main_layout.addLayout(header_layout)

        # Scroll Area for main content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll, 1)

        # 2. Input Clouds Card
        grp_inputs = QGroupBox("1. File Selezionati (Source e Target)")
        vbox_inputs = QVBoxLayout(grp_inputs)
        vbox_inputs.setSpacing(10)

        # Source Cloud Row
        row_src = QHBoxLayout()
        lbl_src = QLabel("<b>Source Scan (Parziale):</b>")
        lbl_src.setFixedWidth(170)
        self.txt_source = QLineEdit()
        self.txt_source.setPlaceholderText("Seleziona il rilievo parziale (.las, .laz, .ply, .e57)...")
        self.btn_src = QPushButton("Sfoglia...")
        self.btn_src.clicked.connect(self.browse_source)
        row_src.addWidget(lbl_src)
        row_src.addWidget(self.txt_source, 1)
        row_src.addWidget(self.btn_src)
        vbox_inputs.addLayout(row_src)

        # Target Cloud Row
        row_tgt = QHBoxLayout()
        lbl_tgt = QLabel("<b>Target Model (Riferimento):</b>")
        lbl_tgt.setFixedWidth(170)
        self.txt_target = QLineEdit()
        self.txt_target.setPlaceholderText("Seleziona il modello globale di riferimento (.las, .laz, .ply, .e57)...")
        self.btn_tgt = QPushButton("Sfoglia...")
        self.btn_tgt.clicked.connect(self.browse_target)
        row_tgt.addWidget(lbl_tgt)
        row_tgt.addWidget(self.txt_target, 1)
        row_tgt.addWidget(self.btn_tgt)
        vbox_inputs.addLayout(row_tgt)

        layout.addWidget(grp_inputs)

        # 3. Parameters Card
        grp_params = QGroupBox("2. Parametri di Allineamento e Metrologia")
        grid_params = QVBoxLayout(grp_params)
        grid_params.setSpacing(10)

        row_p1 = QHBoxLayout()
        lbl_vox = QLabel("Voxel Size (m):")
        self.spin_voxel = QDoubleSpinBox()
        self.spin_voxel.setRange(0.001, 10.0)
        self.spin_voxel.setSingleStep(0.01)
        self.spin_voxel.setDecimals(4)
        self.spin_voxel.setValue(0.05)
        self.btn_auto_voxel = QPushButton("⚡ Auto-Calcola")
        self.btn_auto_voxel.setToolTip("Stima il voxel ottimale in base alla diagonale della Bounding Box del Source")
        self.btn_auto_voxel.clicked.connect(self.auto_calculate_voxel)

        lbl_coarse = QLabel("Coarse Method:")
        self.combo_coarse = QComboBox()
        self.combo_coarse.addItems(["fgr (Fast Global Registration)", "ransac (Stochastic RANSAC)"])

        lbl_fine = QLabel("Fine Method:")
        self.combo_fine = QComboBox()
        self.combo_fine.addItems(["point_to_plane (Point-to-Plane ICP)", "generalized (G-ICP)"])

        row_p1.addWidget(lbl_vox)
        row_p1.addWidget(self.spin_voxel)
        row_p1.addWidget(self.btn_auto_voxel)
        row_p1.addSpacing(15)
        row_p1.addWidget(lbl_coarse)
        row_p1.addWidget(self.combo_coarse)
        row_p1.addSpacing(15)
        row_p1.addWidget(lbl_fine)
        row_p1.addWidget(self.combo_fine)
        grid_params.addLayout(row_p1)

        row_p2 = QHBoxLayout()
        self.chk_anneal = QCheckBox("Gradual Annealing Multi-Scala (3.0x -> 1.5x -> 1.0x voxel)")
        self.chk_anneal.setChecked(True)
        self.chk_anneal.setToolTip("Raffina gradualmente la soglia ICP per evitare minimi locali e convergere con precisione sub-centimetrica")

        lbl_out = QLabel("Cartella Output:")
        self.txt_outdir = QLineEdit()
        self.txt_outdir.setPlaceholderText("(Predefinita: cartella del file source)")
        self.btn_outdir = QPushButton("Sfoglia...")
        self.btn_outdir.clicked.connect(self.browse_outdir)

        row_p2.addWidget(self.chk_anneal)
        row_p2.addSpacing(20)
        row_p2.addWidget(lbl_out)
        row_p2.addWidget(self.txt_outdir, 1)
        row_p2.addWidget(self.btn_outdir)
        grid_params.addLayout(row_p2)

        layout.addWidget(grp_params)

        # 4. Action & Progress Section
        grp_action = QGroupBox("3. Esecuzione Registrazione")
        vbox_action = QVBoxLayout(grp_action)
        vbox_action.setSpacing(10)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ Avvia Allineamento Metrologico")
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.clicked.connect(self.start_alignment)
        btn_row.addWidget(self.btn_run)
        btn_row.addStretch()

        vbox_action.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 6)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("In attesa... (%v/%m)")
        vbox_action.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Pronto per l'elaborazione.")
        self.lbl_status.setStyleSheet("font-weight: 600; color: #475569;")
        vbox_action.addWidget(self.lbl_status)

        layout.addWidget(grp_action)

        # 5. Results & Metrics Card
        self.grp_results = QGroupBox("4. Risultati && Controllo Qualità (QC Gate)")
        vbox_results = QVBoxLayout(self.grp_results)
        vbox_results.setSpacing(12)

        # Metric KPI cards
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)

        self.card_overlap = self.create_metric_card("Overlap", "--", "Soglia inlier: --")
        self.card_rmse = self.create_metric_card("Inlier RMSE", "--", "Precisione metrologica")
        self.card_median = self.create_metric_card("Errore Mediano", "--", "Distribuzione residui")
        self.card_time = self.create_metric_card("Tempo Totale", "--", "Secondi di calcolo")

        kpi_row.addWidget(self.card_overlap)
        kpi_row.addWidget(self.card_rmse)
        kpi_row.addWidget(self.card_median)
        kpi_row.addWidget(self.card_time)
        vbox_results.addLayout(kpi_row)

        # Transformation Matrix Table
        lbl_mat = QLabel("<b>Matrice di Rototraslazione 4 × 4 (Coordinate Globali):</b>")
        vbox_results.addWidget(lbl_mat)

        self.table_matrix = QTableWidget(4, 4)
        self.table_matrix.setFixedHeight(120)
        self.table_matrix.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_matrix.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_matrix.setHorizontalHeaderLabels(["R0", "R1", "R2", "Tx / Ty / Tz"])
        self.table_matrix.setVerticalHeaderLabels(["X", "Y", "Z", "Homogeneous"])
        for r in range(4):
            for c in range(4):
                item = QTableWidgetItem("1.000000" if r == c else "0.000000")
                item.setTextAlignment(Qt.AlignCenter)
                self.table_matrix.setItem(r, c, item)
        vbox_results.addWidget(self.table_matrix)

        # Aligned File Path Indicator
        row_file = QHBoxLayout()
        lbl_file = QLabel("<b>Nuvola Allineata:</b>")
        self.txt_aligned_path = QLineEdit()
        self.txt_aligned_path.setReadOnly(True)
        self.txt_aligned_path.setPlaceholderText("La nuvola allineata viene salvata automaticamente qui...")
        self.txt_aligned_path.setStyleSheet("background-color: #f8fafc; font-weight: 600; color: #0f172a;")
        row_file.addWidget(lbl_file)
        row_file.addWidget(self.txt_aligned_path, 1)
        vbox_results.addLayout(row_file)

        # Action Buttons for results
        row_res_act = QHBoxLayout()
        self.btn_inspect_3d = QPushButton("👁 Ispeziona nel 3D (PyVista)")
        self.btn_inspect_3d.setObjectName("successButton")
        self.btn_inspect_3d.clicked.connect(self.inspect_3d)
        self.btn_inspect_3d.setVisible(False)  # Comparirà solo al termine dell'allineamento

        self.btn_save_as = QPushButton("💾 Salva Nuvola Con Nome...")
        self.btn_save_as.clicked.connect(self.save_cloud_as)
        self.btn_save_as.setVisible(False)

        self.btn_open_folder = QPushButton("📁 Apri Cartella Output")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setVisible(False)

        self.btn_copy_matrix = QPushButton("📋 Copia Matrice 4 × 4")
        self.btn_copy_matrix.clicked.connect(self.copy_matrix)
        self.btn_copy_matrix.setVisible(False)

        row_res_act.addWidget(self.btn_inspect_3d)
        row_res_act.addWidget(self.btn_save_as)
        row_res_act.addWidget(self.btn_open_folder)
        row_res_act.addWidget(self.btn_copy_matrix)
        row_res_act.addStretch()
        vbox_results.addLayout(row_res_act)

        layout.addWidget(self.grp_results)

        # 6. Console Log Output
        grp_log = QGroupBox("5. Log di Elaborazione in Tempo Reale")
        vbox_log = QVBoxLayout(grp_log)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(140)
        vbox_log.addWidget(self.txt_log)
        layout.addWidget(grp_log)

    def create_metric_card(self, title: str, value: str, subtext: str) -> QFrame:
        card = QFrame()
        card.setObjectName("metricCard")
        vbox = QVBoxLayout(card)
        vbox.setContentsMargins(10, 8, 10, 8)
        vbox.setSpacing(2)

        lbl_t = QLabel(title)
        lbl_t.setObjectName("metricTitle")
        lbl_v = QLabel(value)
        lbl_v.setObjectName("metricValue")
        lbl_s = QLabel(subtext)
        lbl_s.setObjectName("metricSub")

        vbox.addWidget(lbl_t)
        vbox.addWidget(lbl_v)
        vbox.addWidget(lbl_s)
        card.lbl_val = lbl_v
        card.lbl_sub = lbl_s
        return card

    def browse_source(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleziona Nuvola Source (Parziale)", "",
            "Point Clouds (*.las *.laz *.ply *.e57);;LAS/LAZ (*.las *.laz);;E57 (*.e57);;PLY (*.ply)"
        )
        if f:
            self.txt_source.setText(f)
            self.auto_calculate_voxel()

    def browse_target(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Seleziona Nuvola Target (Riferimento Globale)", "",
            "Point Clouds (*.las *.laz *.ply *.e57);;LAS/LAZ (*.las *.laz);;E57 (*.e57);;PLY (*.ply)"
        )
        if f:
            self.txt_target.setText(f)

    def browse_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Seleziona Cartella di Output")
        if d:
            self.txt_outdir.setText(d)

    def set_ui_busy(self, busy: bool):
        """Disabilita o abilita tutti i controlli interattivi durante l'esecuzione di operazioni."""
        self.btn_run.setEnabled(not busy)
        self.btn_src.setEnabled(not busy)
        self.btn_tgt.setEnabled(not busy)
        self.btn_auto_voxel.setEnabled(not busy)
        self.btn_outdir.setEnabled(not busy)
        self.txt_source.setEnabled(not busy)
        self.txt_target.setEnabled(not busy)
        self.spin_voxel.setEnabled(not busy)
        self.combo_coarse.setEnabled(not busy)
        self.combo_fine.setEnabled(not busy)
        self.chk_anneal.setEnabled(not busy)
        self.txt_outdir.setEnabled(not busy)

        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.btn_inspect_3d.setEnabled(False)
            if hasattr(self, 'btn_save_as'):
                self.btn_save_as.setEnabled(False)
            self.btn_copy_matrix.setEnabled(False)
            self.btn_open_folder.setEnabled(False)
        else:
            while QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()
            if self.last_result:
                self.btn_inspect_3d.setEnabled(True)
                if hasattr(self, 'btn_save_as'):
                    self.btn_save_as.setEnabled(True)
                self.btn_copy_matrix.setEnabled(True)
                self.btn_open_folder.setEnabled(True)

    def auto_calculate_voxel(self):
        src_path = self.txt_source.text().strip()
        if not src_path or not os.path.exists(src_path):
            return
        try:
            self.lbl_status.setText("Stima voxel in corso...")
            vox, count = estimate_optimal_voxel_from_file(src_path)
            self.spin_voxel.setValue(vox)
            self.lbl_status.setText(f"Voxel ottimale stimato: {vox:.4f} m ({count:,} punti)")
            self.log(f"[Auto-Voxel] Stimato istantaneamente {vox:.4f} m ({count:,} punti) dagli header del file.")
        except Exception as e:
            # Fallback in case of non-standard file
            try:
                _, pcd, _ = load_point_cloud(src_path, load_colors=False)
                vox = estimate_optimal_voxel_size(pcd)
                self.spin_voxel.setValue(vox)
                self.lbl_status.setText(f"Voxel stimato: {vox:.4f} m ({len(pcd.points):,} punti)")
                self.log(f"[Auto-Voxel] Stimato {vox:.4f} m (fallback: {len(pcd.points):,} punti).")
            except Exception as e_fallback:
                self.log(f"[Auto-Voxel] Avviso: {e_fallback}")

    def log(self, text: str):
        self.txt_log.appendPlainText(text)
        self.txt_log.verticalScrollBar().setValue(self.txt_log.verticalScrollBar().maximum())

    def start_alignment(self):
        source = self.txt_source.text().strip()
        target = self.txt_target.text().strip()

        if not source or not os.path.exists(source):
            QMessageBox.warning(self, "Attenzione", "Specificare un file Source valido ed esistente.")
            return
        if not target or not os.path.exists(target):
            QMessageBox.warning(self, "Attenzione", "Specificare un file Target valido ed esistente.")
            return

        voxel = self.spin_voxel.value()
        outdir = self.txt_outdir.text().strip() or None
        coarse = "fgr" if "fgr" in self.combo_coarse.currentText() else "ransac"
        fine = "point_to_plane" if "point_to_plane" in self.combo_fine.currentText() else "generalized"
        anneal = self.chk_anneal.isChecked()

        # Nascondi i tasti di ispezione ed esportazione durante l'elaborazione
        self.btn_inspect_3d.setVisible(False)
        self.btn_save_as.setVisible(False)
        self.btn_copy_matrix.setVisible(False)
        self.btn_open_folder.setVisible(False)
        self.txt_aligned_path.clear()

        # Blocca tutti i pulsanti e gli input per evitare operazioni concorrenti
        self.set_ui_busy(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Avvio pipeline di allineamento...")
        self.txt_log.clear()

        self.worker = AlignmentWorker(
            source_path=source,
            target_path=target,
            voxel_size=voxel,
            out_dir=outdir,
            coarse_method=coarse,
            fine_method=fine,
            anneal=anneal
        )
        self.worker.progress_signal.connect(self.on_worker_progress)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_worker_finished)
        self.worker.error_signal.connect(self.on_worker_error)
        self.worker.start()

    def on_worker_progress(self, step: int, total: int, text: str):
        self.progress_bar.setValue(step)
        self.progress_bar.setFormat(f"Passo {step}/{total}: {text}")
        self.lbl_status.setText(f"Elaborazione in corso: {text}...")

    def on_worker_finished(self, result: AlignmentResult):
        self.last_result = result
        self.set_ui_busy(False)
        self.progress_bar.setValue(6)
        self.progress_bar.setFormat("✓ Allineamento Completato!")
        self.lbl_status.setText("Allineamento completato con successo.")

        # Update KPI cards
        self.card_overlap.lbl_val.setText(f"{result.overlap_pct:.1f}%")
        self.card_overlap.lbl_sub.setText(f"Soglia: {result.stats.get('overlap_threshold', 0):.3f} m")

        rmse = result.inlier_rmse
        rmse_str = f"{rmse * 1000:.1f} mm" if rmse < 0.1 else f"{rmse:.4f} m"
        self.card_rmse.lbl_val.setText(rmse_str)

        med = result.stats.get("median_error", 0.0)
        med_str = f"{med * 1000:.1f} mm" if med < 0.1 else f"{med:.4f} m"
        self.card_median.lbl_val.setText(med_str)

        self.card_time.lbl_val.setText(f"{result.execution_time:.1f} s")

        # Fill table with 4x4 matrix
        mat = result.transformation_matrix
        for r in range(4):
            for c in range(4):
                self.table_matrix.item(r, c).setText(f"{mat[r, c]:.6f}")

        # Mostra pulsanti di azione
        self.btn_inspect_3d.setVisible(True)
        self.btn_inspect_3d.setEnabled(True)
        self.btn_save_as.setVisible(True)
        self.btn_save_as.setEnabled(True)
        self.btn_open_folder.setVisible(True)
        self.btn_open_folder.setEnabled(True)
        self.btn_copy_matrix.setVisible(True)
        self.btn_copy_matrix.setEnabled(True)

        if result.aligned_cloud_path:
            self.txt_aligned_path.setText(result.aligned_cloud_path)
            QMessageBox.information(
                self,
                "Allineamento Completato",
                f"Allineamento completato con successo in {result.execution_time:.2f} secondi!\n\n"
                f"• Sovrapposizione (Overlap): {result.overlap_pct:.2f}%\n"
                f"• Inlier RMSE: {rmse_str}\n"
                f"• Errore Mediano: {med_str}\n\n"
                f"File salvato in:\n{result.aligned_cloud_path}"
            )
        else:
            self.txt_aligned_path.setText("")
            self.txt_aligned_path.setPlaceholderText("(Nuvola in memoria RAM - Non ancora salvata su disco)")

            # Chiedi conferma all'utente: NON salvare in automatico!
            filename_hint = os.path.basename(result.candidate_cloud_path)
            msg = (
                f"Allineamento completato con successo in {result.execution_time:.2f} secondi!\n\n"
                f"• Sovrapposizione (Overlap): {result.overlap_pct:.2f}%\n"
                f"• Inlier RMSE: {rmse_str}\n"
                f"• Errore Mediano: {med_str}\n\n"
                f"Vuoi salvare su disco la nuvola allineata adesso?\n"
                f"(File proposto: {filename_hint})"
            )
            reply = QMessageBox.question(
                self,
                "Conferma Salvataggio Nuvola",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.save_aligned_to_disk(result.candidate_cloud_path)
            else:
                self.log("[Info] Nuvola mantenuta in memoria RAM. Puoi ispezionarla in 3D o salvarla con 'Salva Nuvola Con Nome...'.")

    def save_aligned_to_disk(self, target_path: str):
        """Salva la nuvola allineata su disco con indicatore di progresso."""
        if not self.last_result:
            return
        self.set_ui_busy(True)
        self.lbl_status.setText("Salvataggio nuvola allineata su disco in corso...")
        QApplication.processEvents()
        try:
            saved_file = self.last_result.save_aligned(target_path)
            self.txt_aligned_path.setText(saved_file)
            self.lbl_status.setText("Nuvola allineata salvata con successo.")
            self.log(f"[Salvataggio] Nuvola allineata esportata in: {saved_file}")
            QMessageBox.information(
                self,
                "Salvataggio Riuscito",
                f"La nuvola allineata è stata salvata con successo in:\n\n{saved_file}"
            )
        except Exception as e:
            self.log(f"[Errore Salvataggio] {e}")
            QMessageBox.critical(self, "Errore Salvataggio", f"Impossibile salvare il file: {e}")
        finally:
            self.set_ui_busy(False)

    def on_worker_error(self, err_msg: str):
        self.set_ui_busy(False)
        self.btn_inspect_3d.setVisible(False)
        self.btn_save_as.setVisible(False)
        self.btn_copy_matrix.setVisible(False)
        self.btn_open_folder.setVisible(False)
        self.progress_bar.setFormat("❌ Errore durante l'elaborazione")
        self.lbl_status.setText("Errore.")
        QMessageBox.critical(self, "Errore di Registrazione", f"Si è verificato un errore:\n\n{err_msg}")

    def inspect_3d(self):
        if not self.last_result:
            return
        if not PYVISTA_AVAILABLE:
            QMessageBox.warning(self, "PyVista non installato", "Il modulo PyVista è richiesto per la visualizzazione 3D.")
            return

        try:
            self.lbl_status.setText("Apertura visualizzatore 3D...")
            QApplication.processEvents()
            visualize_registration(
                target_path=self.last_result.target_path,
                aligned_source_path=self.last_result.aligned_cloud_path,
                residuals_path=self.last_result.residuals_path,
                stats=self.last_result.stats,
                source_pcd=self.last_result.pcd_aligned
            )
            self.lbl_status.setText("Pronto.")
        except Exception as e:
            QMessageBox.critical(self, "Errore Visualizzatore 3D", f"Impossibile aprire il visualizzatore: {e}")

    def copy_matrix(self):
        if not self.last_result:
            return
        mat = self.last_result.transformation_matrix
        mat_str = "\n".join(["\t".join([f"{val:.8f}" for val in row]) for row in mat])
        clipboard = QApplication.clipboard()
        clipboard.setText(mat_str)
        QMessageBox.information(self, "Copiato", "Matrice 4x4 copiata negli appunti in formato tabulare!")

    def save_cloud_as(self):
        """Consente di salvare o esportare una copia della nuvola allineata in una posizione/formato personalizzato."""
        if not self.last_result:
            QMessageBox.warning(self, "Attenzione", "Nessuna nuvola allineata disponibile da salvare.")
            return

        suggested = self.last_result.aligned_cloud_path or self.last_result.candidate_cloud_path
        target_file, _ = QFileDialog.getSaveFileName(
            self,
            "Salva Nuvola Allineata Con Nome",
            suggested,
            "Point Clouds (*.las *.laz *.ply *.e57);;LAS/LAZ (*.las *.laz);;PLY (*.ply);;E57 (*.e57)"
        )
        if not target_file:
            return

        self.save_aligned_to_disk(target_file)

    def open_output_folder(self):
        if not self.last_result:
            return
        path = self.last_result.aligned_cloud_path or self.last_result.candidate_cloud_path or self.last_result.sidecar_path
        folder = os.path.dirname(os.path.abspath(path))
        if os.path.exists(folder):
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Splash Screen con logo ufficiale all'apertura dell'applicazione
    logo_path = os.path.join(os.path.dirname(__file__), "resources", "logo.png")
    splash = None
    if os.path.exists(logo_path):
        pixmap = QPixmap(logo_path).scaled(420, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.show()
        splash.showMessage(
            "   PARTiaL2GLOBAL: Avvio suite di registrazione metrologica...",
            Qt.AlignBottom | Qt.AlignLeft,
            Qt.white
        )
        app.processEvents()
        time.sleep(1.0)

    window = Partial2GlobalApp()
    window.show()

    if splash:
        splash.finish(window)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
