"""
Real-time 3D 6-DOF visualization for Polaris Vega multi-tracker poses.

Left sidebar:  ROM checkboxes + femur correction toggle + connect button
Center:        3D OpenGL view with coordinate frames per tracker
Bottom:        Live stats (Hz, position, quality per tracker)

Usage:
    python3 vega_viz_3d.py
"""

import sys
import os
import glob
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl

from vega_discover import discover_vega, VegaNotFoundError
from sksurgerynditracker.nditracker import NDITracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROM_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "roms"))
FEMUR_ROM_BASENAME = "BBT-110017Rev1-FemurTracker-SPH.rom"
FEMUR_SPH_X_CORRECTION_MM = 8.770 - (-2.127)  # 10.897

DEFAULT_ROMS = {FEMUR_ROM_BASENAME, "BBT-TrackerA-Gray_Polaris.rom"}

TRACKER_COLORS = [
    (1.0, 0.4, 0.4),
    (0.4, 0.7, 1.0),
    (0.4, 1.0, 0.5),
    (1.0, 0.8, 0.3),
    (0.7, 0.4, 1.0),
    (1.0, 0.5, 0.8),
]


def rom_label(path):
    return os.path.splitext(os.path.basename(path))[0]


def apply_x_correction(T, dx):
    Tc = T.copy()
    Tc[:3, 3] += dx * T[:3, 0]
    return Tc


def euler_zyx_deg(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees([rz, ry, rx])


# ---------------------------------------------------------------------------
# 3D tracker frame (three RGB axis lines + origin dot)
# ---------------------------------------------------------------------------

class TrackerFrame:
    AXIS_LEN = 80.0  # mm

    def __init__(self, view, index):
        self.view = view
        self.items = []
        self.axis_lines = []

        axis_colors = [(1, 0.2, 0.2, 1), (0.2, 1, 0.2, 1), (0.3, 0.5, 1, 1)]
        for c in axis_colors:
            line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3)), color=c, width=3, antialias=True,
            )
            line.setVisible(False)
            view.addItem(line)
            self.axis_lines.append(line)
            self.items.append(line)

        tc = TRACKER_COLORS[index % len(TRACKER_COLORS)]
        self.dot = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=np.array([[tc[0], tc[1], tc[2], 1.0]]),
            size=10,
        )
        self.dot.setVisible(False)
        view.addItem(self.dot)
        self.items.append(self.dot)

    def update(self, T, visible):
        if not visible:
            for it in self.items:
                it.setVisible(False)
            return

        pos = T[:3, 3]
        R = T[:3, :3]

        for i, line in enumerate(self.axis_lines):
            end = pos + self.AXIS_LEN * R[:, i]
            line.setData(pos=np.array([pos, end]))
            line.setVisible(True)

        self.dot.setData(pos=pos.reshape(1, 3))
        self.dot.setVisible(True)

    def remove(self):
        for it in self.items:
            self.view.removeItem(it)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class VegaVizWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polaris Vega — 6-DOF Tracker Visualization")
        self.resize(1280, 800)

        self.tracker = None
        self.tracker_frames = []
        self.active_roms = []
        self.x_corrections = []
        self.hz_stamps = []

        self._build_ui()

        self.poll_timer = QtCore.QTimer()
        self.poll_timer.timeout.connect(self._poll)

    # ---- UI construction ----

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # -- Left sidebar --
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(270)
        sidebar.setStyleSheet(
            "QWidget { background: #252525; }"
            "QLabel { color: #ccc; }"
            "QCheckBox { color: #bbb; spacing: 4px; }"
            "QPushButton { background: #3a3a3a; color: #ddd; border: 1px solid #555;"
            "  padding: 6px; border-radius: 3px; }"
            "QPushButton:hover { background: #4a4a4a; }"
        )
        sb_lay = QtWidgets.QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(8, 8, 8, 8)

        title = QtWidgets.QLabel("Tracker ROMs")
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #eee;")
        sb_lay.addWidget(title)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll_w = QtWidgets.QWidget()
        self.rom_lay = QtWidgets.QVBoxLayout(scroll_w)
        self.rom_lay.setSpacing(2)
        self.rom_lay.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(scroll_w)

        self.rom_cbs = {}
        roms = sorted(glob.glob(os.path.join(ROM_DIR, "*.rom")))
        for rp in roms:
            bn = os.path.basename(rp)
            cb = QtWidgets.QCheckBox(rom_label(rp))
            cb.setToolTip(rp)
            cb.setChecked(bn in DEFAULT_ROMS)
            self.rom_lay.addWidget(cb)
            self.rom_cbs[rp] = cb
        self.rom_lay.addStretch()
        sb_lay.addWidget(scroll, stretch=1)

        sb_lay.addSpacing(6)
        self.corr_cb = QtWidgets.QCheckBox("Apply femur X correction")
        self.corr_cb.setToolTip("+10.897 mm along tool X (SPH ROM with flat disc markers)")
        sb_lay.addWidget(self.corr_cb)

        self.ekf_cb = QtWidgets.QCheckBox("EKF smoothing")
        self.ekf_cb.setToolTip("Kalman filter: smooths jitter and bridges brief tracking dropouts")
        sb_lay.addWidget(self.ekf_cb)

        sb_lay.addSpacing(10)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setStyleSheet(
            "QPushButton { font-size: 13px; padding: 8px; }"
        )
        self.connect_btn.clicked.connect(self._toggle_connection)
        sb_lay.addWidget(self.connect_btn)

        root.addWidget(sidebar)

        # -- Right: 3D + stats --
        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        self.glw = gl.GLViewWidget()
        self.glw.setBackgroundColor(25, 25, 30)
        self.glw.setCameraPosition(distance=2000, elevation=25, azimuth=-50)
        right_lay.addWidget(self.glw, stretch=1)

        # Floor grid
        grid = gl.GLGridItem()
        grid.setSize(4000, 4000)
        grid.setSpacing(200, 200)
        grid.setColor((60, 60, 65, 100))
        self.glw.addItem(grid)

        # Origin frame ("camera")
        origin_len = 120
        for vec, col in [
            ([origin_len, 0, 0], (1, 0.3, 0.3, 0.6)),
            ([0, origin_len, 0], (0.3, 1, 0.3, 0.6)),
            ([0, 0, origin_len], (0.3, 0.5, 1, 0.6)),
        ]:
            self.glw.addItem(gl.GLLinePlotItem(
                pos=np.array([[0, 0, 0], vec], dtype=float),
                color=col, width=2, antialias=True,
            ))

        # Origin label
        try:
            self._camera_label = gl.GLTextItem(
                pos=np.array([0, 0, 30], dtype=float),
                text="camera", color=(180, 180, 180, 200),
            )
            self.glw.addItem(self._camera_label)
        except Exception:
            pass

        # Bottom stats bar
        self.stats = QtWidgets.QLabel("Not connected")
        self.stats.setStyleSheet(
            "QLabel { background: #1a1a1e; color: #aaa; padding: 6px;"
            "  font-family: 'Menlo', 'Courier', monospace; font-size: 12px; }"
        )
        self.stats.setMinimumHeight(70)
        self.stats.setMaximumHeight(120)
        self.stats.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        right_lay.addWidget(self.stats)

        root.addWidget(right, stretch=1)

    # ---- Connection management ----

    def _selected_roms(self):
        return [p for p, cb in self.rom_cbs.items() if cb.isChecked()]

    def _toggle_connection(self):
        if self.tracker:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        roms = self._selected_roms()
        if not roms:
            self.stats.setText("Select at least one ROM.")
            return

        for cb in self.rom_cbs.values():
            cb.setEnabled(False)
        self.corr_cb.setEnabled(False)
        self.ekf_cb.setEnabled(False)
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Connecting...")
        QtWidgets.QApplication.processEvents()

        try:
            info = discover_vega(verbose=False)
        except VegaNotFoundError as e:
            self.stats.setText(f"Discovery failed:\n{e}")
            self._unlock_sidebar()
            return

        try:
            settings = info.as_settings(roms)
            self.tracker = NDITracker(settings)
            self.tracker.start_tracking()
        except Exception as e:
            self.stats.setText(f"Connection failed:\n{e}")
            self.tracker = None
            self._unlock_sidebar()
            return

        self.active_roms = roms
        self.hz_stamps = []

        # X corrections
        apply_corr = self.corr_cb.isChecked()
        self.x_corrections = []
        for rp in roms:
            if apply_corr and os.path.basename(rp) == FEMUR_ROM_BASENAME:
                self.x_corrections.append(FEMUR_SPH_X_CORRECTION_MM)
            else:
                self.x_corrections.append(0.0)

        # EKF filters
        self.ekf_filters = None
        if self.ekf_cb.isChecked():
            from pose_ekf import PoseEKF
            self.ekf_filters = [PoseEKF(max_misses=60) for _ in roms]

        # Create 3D frames
        for tf in self.tracker_frames:
            tf.remove()
        self.tracker_frames = [TrackerFrame(self.glw, i) for i in range(len(roms))]

        # Create 3D labels for each tracker
        self._tracker_labels = []
        for i, rp in enumerate(roms):
            try:
                tc = TRACKER_COLORS[i % len(TRACKER_COLORS)]
                lbl = gl.GLTextItem(
                    pos=np.array([0, 0, 0], dtype=float),
                    text=rom_label(rp),
                    color=tuple(int(c * 255) for c in tc) + (220,),
                )
                lbl.setVisible(False)
                self.glw.addItem(lbl)
                self._tracker_labels.append(lbl)
            except Exception:
                self._tracker_labels.append(None)

        self.connect_btn.setText("Disconnect")
        self.connect_btn.setEnabled(True)
        self.stats.setText(f"Connected to {info.ip}:{info.port}")
        self.poll_timer.start(16)  # ~60 fps timer

    def _disconnect(self):
        self.poll_timer.stop()
        if self.tracker:
            try:
                self.tracker.stop_tracking()
                self.tracker.close()
            except Exception:
                pass
            self.tracker = None

        for tf in self.tracker_frames:
            tf.remove()
        self.tracker_frames = []

        for lbl in getattr(self, "_tracker_labels", []):
            if lbl:
                self.glw.removeItem(lbl)
        self._tracker_labels = []

        self._unlock_sidebar()
        self.stats.setText("Disconnected")

    def _unlock_sidebar(self):
        for cb in self.rom_cbs.values():
            cb.setEnabled(True)
        self.corr_cb.setEnabled(True)
        self.ekf_cb.setEnabled(True)
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)

    # ---- Tracking loop ----

    def _poll(self):
        if not self.tracker:
            return

        try:
            _, _, _, tracking, quality = self.tracker.get_frame()
        except Exception as e:
            self.stats.setText(f"Tracking error: {e}")
            return

        now = time.monotonic()
        self.hz_stamps.append(now)
        self.hz_stamps = [t for t in self.hz_stamps if now - t < 1.0]
        hz = len(self.hz_stamps)

        lines = [f"  Hz: {hz}"]

        for i, rp in enumerate(self.active_roms):
            T = tracking[i]
            q = quality[i]
            raw_vis = not np.isnan(T[0, 0])
            label = rom_label(rp)

            # Apply X correction before EKF (so EKF sees corrected coords)
            if raw_vis:
                dx = self.x_corrections[i]
                if dx != 0:
                    T = apply_x_correction(T, dx)

            # Apply EKF if enabled
            vis = raw_vis
            ekf_predicted = False
            if self.ekf_filters:
                T_filt, valid = self.ekf_filters[i].process(T, raw_vis, now)
                if valid:
                    T = T_filt
                    if not raw_vis:
                        ekf_predicted = True
                    vis = True
                else:
                    vis = False

            if vis:
                self.tracker_frames[i].update(T, True)

                if i < len(self._tracker_labels) and self._tracker_labels[i]:
                    self._tracker_labels[i].setData(
                        pos=T[:3, 3] + np.array([0, 0, 25], dtype=float)
                    )
                    self._tracker_labels[i].setVisible(True)

                x, y, z = T[0, 3], T[1, 3], T[2, 3]
                status = "EKF PRED " if ekf_predicted else "TRACKING "
                lines.append(
                    f"  {label:<40s}  {status} "
                    f"({x:7.1f}, {y:7.1f}, {z:7.1f}) mm   "
                    f"q={float(q):.4f}"
                )
            else:
                self.tracker_frames[i].update(None, False)
                if i < len(self._tracker_labels) and self._tracker_labels[i]:
                    self._tracker_labels[i].setVisible(False)
                lines.append(f"  {label:<40s}  NOT VISIBLE")

        self.stats.setText("\n".join(lines))

    # ---- Cleanup ----

    def closeEvent(self, event):
        self._disconnect()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(40, 40, 42))
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(200, 200, 200))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(28, 28, 30))
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(40, 40, 42))
    pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(200, 200, 200))
    pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(50, 50, 52))
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(200, 200, 200))
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(70, 110, 200))
    app.setPalette(pal)

    win = VegaVizWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
