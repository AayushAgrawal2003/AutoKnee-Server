"""
Unified IR tracking + camera-to-KUKA transform node.

Single ROS 2 node that handles the full pipeline:
  1. Discovers and connects to the Polaris Vega camera
  2. Tracks all 3 ROM tools (femur, tibia, end-effector) with EKF smoothing
  3. Subscribes to /ee_marker_pos for the EE tracker ROM origin in KUKA base frame
     (this is KUKA FK + constant CAD offset, published by upstream KUKA node)
  4. One-shot calibration: computes T_kuka_cam from initial /ee_marker_pos vs Polaris EE
  5. Publishes all tracker 6-DOF poses in KUKA base frame
  6. Publishes registration drift (live comparison of transformed Polaris vs /ee_marker_pos)

Calibration math:
    T_kuka_cam = T_kuka_rom_initial @ inv(T_cam_rom_initial)

Continuous output:
    T_kuka_tracker = T_kuka_cam @ T_cam_tracker   (for femur, tibia, EE)

Drift (published every frame, logged every 2s):
    predicted = T_kuka_cam @ T_cam_rom_live        (frozen transform x live Polaris)
    actual    = latest /ee_marker_pos reading       (live KUKA FK + offset)
    drift_m   = ||predicted.position - actual.position||

Subscriptions:
    /ee_marker_pos    PoseStamped   ISTAR ROM origin in KUKA base frame (meters)

Publications:
    /kuka_frame/pose_ee           PoseStamped   EE tracker in KUKA base frame
    /kuka_frame/bone_pose_femur   PoseStamped   Femur tracker in KUKA base frame
    /kuka_frame/bone_pose_tibia   PoseStamped   Tibia tracker in KUKA base frame
    /kuka_frame/drift             Float64       Translational drift magnitude (meters)

Usage:
    ros2 run ir_tracking ir_tracking_node
    ros2 run ir_tracking ir_tracking_node --ros-args -p hz:=50.0 -p vega_ip:=169.254.9.239
"""

import os
import sys
import time
import numpy as np

from ir_tracking.vega_discover import discover_vega, VegaNotFoundError
from ir_tracking.pose_ekf import PoseEKF
from sksurgerynditracker.nditracker import NDITracker

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from ament_index_python.packages import get_package_share_directory


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROM_DIR = os.path.join(get_package_share_directory('ir_tracking'), 'roms')

# (rom_filename, output_topic, x_correction_mm)
# Index 0 = femur, 1 = tibia, 2 = EE (ISTAR)
TRACKERS = [
    ("BBT-110017Rev1-FemurTracker-SPH.rom", "/kuka_frame/bone_pose_femur", 8.770 - (-2.127)),
    ("BBT-TrackerA-Gray_Polaris.rom",       "/kuka_frame/bone_pose_tibia", 0.0),
    ("ISTAR-APPLE01.rom",                   "/kuka_frame/pose_ee",         0.0),
]

EE_INDEX = 2  # ISTAR is the 3rd tracker
MM_TO_M = 0.001


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def quaternion_to_rotation_matrix(x, y, z, w):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def pose_msg_to_matrix(msg):
    """Convert a PoseStamped to a 4x4 homogeneous transform."""
    p = msg.pose
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    T[0, 3] = p.position.x
    T[1, 3] = p.position.y
    T[2, 3] = p.position.z
    return T


def matrix_to_pose_stamped(T, frame_id, stamp):
    """Build a PoseStamped from a 4x4 homogeneous transform."""
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    qx, qy, qz, qw = rotation_matrix_to_quaternion(T[:3, :3])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def invert_transform(T):
    """SE(3) inverse via R^T."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_x_correction(T, dx_mm):
    """Shift tool origin along its local X axis by dx_mm."""
    Tc = T.copy()
    Tc[:3, 3] += dx_mm * T[:3, 0]
    return Tc


def rotation_matrix_to_euler_zyx(R):
    """Extract ZYX Euler angles (degrees) from a 3x3 rotation matrix."""
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
# Node
# ---------------------------------------------------------------------------

class IRTrackingNode(Node):

    def __init__(self):
        super().__init__('ir_tracking_node')

        # ── ROS 2 Parameters ──────────────────────────────────────────────
        self.declare_parameter('hz', 50.0)
        self.declare_parameter('vega_ip', '')

        hz = self.get_parameter('hz').value
        vega_ip = self.get_parameter('vega_ip').value
        if not vega_ip:
            vega_ip = None

        # ── Resolve ROM files ────────────────────────────────────────────
        self.romfiles = []
        self.topics = []
        self.x_corrections = []
        for rom_name, topic, dx in TRACKERS:
            rom_path = os.path.join(ROM_DIR, rom_name)
            if not os.path.isfile(rom_path):
                self.get_logger().fatal(f"ROM not found: {rom_path}")
                raise FileNotFoundError(rom_path)
            self.romfiles.append(rom_path)
            self.topics.append(topic)
            self.x_corrections.append(dx)
            corr = f"  (X correction: +{dx:.3f}mm)" if dx else ""
            self.get_logger().info(f"ROM: {rom_name}{corr}  ->  {topic}")

        # ── Discover and connect to Polaris Vega ─────────────────────────
        self.get_logger().info("Discovering Polaris Vega...")
        try:
            self.vega_info = discover_vega(known_ip=vega_ip, verbose=True)
        except VegaNotFoundError as e:
            self.get_logger().fatal(str(e))
            raise

        settings = self.vega_info.as_settings(self.romfiles)
        self.get_logger().info(
            f"Connecting to {self.vega_info.ip}:{self.vega_info.port}...")
        self.ndi_tracker = NDITracker(settings)
        self.ndi_tracker.start_tracking()
        self.get_logger().info("Polaris Vega tracking started.")

        # ── EKF filters (one per tracker) ────────────────────────────────
        self.ekf_filters = [PoseEKF(max_misses=60) for _ in self.romfiles]
        self._last_frame_num = None

        # ── Calibration state ────────────────────────────────────────────
        self.calibrated = False
        self.T_kuka_cam = None
        self._latest_istar_global = None      # from /ee_marker_pos subscription
        self._latest_cam_ee_matrix = None      # from Polaris (4x4, meters)
        self._drift_m = None

        # ── ROS subscription: ISTAR ROM origin in KUKA base frame ────────
        self.create_subscription(
            PoseStamped, '/ee_marker_pos', self._istar_global_cb, 10)
        self.get_logger().info("Subscribed to /ee_marker_pos (KUKA EE + CAD offset)")

        # ── ROS publishers ───────────────────────────────────────────────
        self.pose_pubs = []
        for topic in self.topics:
            self.pose_pubs.append(self.create_publisher(PoseStamped, topic, 10))

        self.drift_pub = self.create_publisher(Float64, '/kuka_frame/drift', 10)

        # ── Timers ───────────────────────────────────────────────────────
        self.create_timer(1.0 / hz, self._poll_and_publish)
        self.create_timer(2.0, self._log_status)

        self.get_logger().info(
            f"Running at {hz:.0f} Hz.  Waiting for /ee_marker_pos to calibrate...")

    # ── /ee_marker_pos callback ───────────────────────────────────────────

    def _istar_global_cb(self, msg):
        self._latest_istar_global = msg

    # ── Main loop: poll Vega, calibrate, transform, publish ──────────────

    def _poll_and_publish(self):
        # ── 1. Poll Polaris Vega for raw tracker poses ───────────────────
        try:
            _, _, framenumbers, tracking, quality = self.ndi_tracker.get_frame()
        except Exception as e:
            self.get_logger().warn(
                f"get_frame error: {e}", throttle_duration_sec=2.0)
            return

        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()

        cur_frame = framenumbers[0] if len(framenumbers) > 0 else None
        new_frame = (cur_frame != self._last_frame_num)
        self._last_frame_num = cur_frame

        # ── 2. EKF-filter each tracker, apply corrections ────────────────
        #    Store filtered 4x4 transforms in meters
        cam_poses = [None] * len(self.romfiles)
        for i in range(len(self.romfiles)):
            T = tracking[i]
            raw_vis = not np.isnan(T[0, 0])

            if raw_vis and self.x_corrections[i] != 0:
                T = apply_x_correction(T, self.x_corrections[i])

            vis_for_ekf = raw_vis and new_frame
            T_filt, valid = self.ekf_filters[i].process(T, vis_for_ekf, now)
            if not valid:
                continue

            # Convert mm -> m
            T_m = T_filt.copy()
            T_m[:3, 3] *= MM_TO_M
            cam_poses[i] = T_m

        # Keep latest EE pose for calibration / drift
        if cam_poses[EE_INDEX] is not None:
            self._latest_cam_ee_matrix = cam_poses[EE_INDEX]

        # ── 3. Calibrate (one-shot) ──────────────────────────────────────
        if not self.calibrated:
            if (self._latest_istar_global is not None
                    and self._latest_cam_ee_matrix is not None):
                T_kuka_rom = pose_msg_to_matrix(self._latest_istar_global)
                T_cam_rom = self._latest_cam_ee_matrix
                self.T_kuka_cam = T_kuka_rom @ invert_transform(T_cam_rom)
                self.calibrated = True

                t = self.T_kuka_cam[:3, 3]
                rz, ry, rx = rotation_matrix_to_euler_zyx(
                    self.T_kuka_cam[:3, :3])
                self.get_logger().info("=== CALIBRATION COMPLETE ===")
                self.get_logger().info(
                    f"  T_kuka_cam translation: "
                    f"({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")
                self.get_logger().info(
                    f"  T_kuka_cam rotation (ZYX): "
                    f"({rz:.2f}, {ry:.2f}, {rx:.2f}) deg")
            else:
                return  # nothing to publish yet

        # ── 4. Transform all poses to KUKA base frame and publish ────────
        for i in range(len(self.romfiles)):
            if cam_poses[i] is None:
                continue
            T_kuka = self.T_kuka_cam @ cam_poses[i]
            out_msg = matrix_to_pose_stamped(T_kuka, 'lbr_link_0', stamp)
            self.pose_pubs[i].publish(out_msg)

        # ── 5. Drift computation and publishing ──────────────────────────
        if (self._latest_istar_global is not None
                and cam_poses[EE_INDEX] is not None):
            # Predicted: transform live Polaris EE using frozen calibration
            T_predicted = self.T_kuka_cam @ cam_poses[EE_INDEX]
            # Actual: latest /ee_marker_pos (KUKA FK + CAD offset)
            T_actual = pose_msg_to_matrix(self._latest_istar_global)

            drift_vec = T_predicted[:3, 3] - T_actual[:3, 3]
            self._drift_m = float(np.linalg.norm(drift_vec))

            msg = Float64()
            msg.data = self._drift_m
            self.drift_pub.publish(msg)

    # ── Periodic status log ──────────────────────────────────────────────

    def _log_status(self):
        if not self.calibrated:
            has_istar = self._latest_istar_global is not None
            has_polaris = self._latest_cam_ee_matrix is not None
            self.get_logger().info(
                f"Waiting to calibrate...  "
                f"/ee_marker_pos: {'OK' if has_istar else 'waiting'}  "
                f"Polaris EE: {'OK' if has_polaris else 'waiting'}")
            return
        if self._drift_m is not None:
            self.get_logger().info(
                f"Drift: {self._drift_m * 1000.0:.3f} mm")

    # ── Cleanup ──────────────────────────────────────────────────────────

    def destroy_node(self):
        if hasattr(self, 'ndi_tracker') and self.ndi_tracker:
            try:
                self.ndi_tracker.stop_tracking()
                self.ndi_tracker.close()
            except Exception:
                pass
            self.ndi_tracker = None
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)

    try:
        node = IRTrackingNode()
    except (FileNotFoundError, VegaNotFoundError):
        rclpy.shutdown()
        sys.exit(1)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.get_logger().info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
