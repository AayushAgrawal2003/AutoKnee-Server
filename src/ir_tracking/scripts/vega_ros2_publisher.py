"""
ROS 2 publisher for EKF-filtered Polaris Vega tracker poses.

Publishes geometry_msgs/PoseStamped for three trackers:
  /bone_pose_femur  — BBT-110017Rev1-FemurTracker-SPH (with +10.897mm X correction)
  /bone_pose_tibia  — BBT-TrackerA-Gray_Polaris
  /pose_ee          — ISTAR-APPLE01

All poses are EKF-filtered for jitter smoothing and dropout bridging.
Positions are published in meters, orientations as quaternions.
The EKF also interpolates between camera frames when the publish rate
exceeds the Vega's native frame rate (~20-60 Hz).

Usage:
    python3 vega_ros2_publisher.py
    python3 vega_ros2_publisher.py --hz 100
    python3 vega_ros2_publisher.py --hz 30 --ip 169.254.9.239
"""

import sys
import os
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vega_discover import discover_vega, VegaNotFoundError
from pose_ekf import PoseEKF
from sksurgerynditracker.nditracker import NDITracker

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# ---------------------------------------------------------------------------
# ROM paths and per-tracker config
# ---------------------------------------------------------------------------
ROM_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# (rom_filename, ros_topic, x_correction_mm)
TRACKERS = [
    ("BBT-110017Rev1-FemurTracker-SPH.rom", "/bone_pose_femur", 8.770 - (-2.127)),
    ("BBT-TrackerA-Gray_Polaris.rom",       "/bone_pose_tibia", 0.0),
    ("ISTAR-APPLE01.rom",                   "/pose_ee",         0.0),
]

MM_TO_M = 0.001


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def apply_x_correction(T, dx_mm):
    """Shift tool origin along its local X axis by dx_mm."""
    Tc = T.copy()
    Tc[:3, 3] += dx_mm * T[:3, 0]
    return Tc


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


# ---------------------------------------------------------------------------
# ROS 2 Node
# ---------------------------------------------------------------------------

class VegaPosePublisher(Node):

    def __init__(self, hz, vega_ip=None):
        super().__init__('vega_pose_publisher')

        # Resolve ROM files
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
            self.get_logger().info(f"  ROM: {rom_name}{corr}")

        # Publishers
        self.pubs = []
        for topic in self.topics:
            self.pubs.append(self.create_publisher(PoseStamped, topic, 10))
            self.get_logger().info(f"  Topic: {topic}")

        # Discover and connect
        self.get_logger().info("Discovering Polaris Vega...")
        try:
            self.vega_info = discover_vega(
                known_ip=vega_ip, verbose=True)
        except VegaNotFoundError as e:
            self.get_logger().fatal(str(e))
            raise

        settings = self.vega_info.as_settings(self.romfiles)
        self.get_logger().info(
            f"Connecting to {self.vega_info.ip}:{self.vega_info.port}...")
        self.tracker = NDITracker(settings)
        self.tracker.start_tracking()
        self.get_logger().info("Tracking started.")

        # EKF filters (one per tracker)
        self.ekf_filters = [PoseEKF(max_misses=60) for _ in self.romfiles]

        # Frame deduplication — avoid double-updating EKF on duplicate camera frames
        self._last_frame_num = None

        # Timer
        self.timer = self.create_timer(1.0 / hz, self._timer_cb)
        self.get_logger().info(
            f"Publishing {len(self.romfiles)} trackers at {hz} Hz  [EKF enabled]")

    def _timer_cb(self):
        try:
            _, _, framenumbers, tracking, quality = self.tracker.get_frame()
        except Exception as e:
            self.get_logger().warn(f"get_frame error: {e}", throttle_duration_sec=2.0)
            return

        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()

        # Detect duplicate camera frame (poll rate > camera rate)
        cur_frame = framenumbers[0] if len(framenumbers) > 0 else None
        new_frame = (cur_frame != self._last_frame_num)
        self._last_frame_num = cur_frame

        for i in range(len(self.romfiles)):
            T = tracking[i]
            raw_vis = not np.isnan(T[0, 0])

            # Apply X correction before EKF
            if raw_vis and self.x_corrections[i] != 0:
                T = apply_x_correction(T, self.x_corrections[i])

            # On duplicate frames: predict-only (don't double-update EKF)
            vis_for_ekf = raw_vis and new_frame

            T_filt, valid = self.ekf_filters[i].process(T, vis_for_ekf, now)

            if not valid:
                continue

            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'polaris_vega'

            # Position: mm -> m
            msg.pose.position.x = float(T_filt[0, 3]) * MM_TO_M
            msg.pose.position.y = float(T_filt[1, 3]) * MM_TO_M
            msg.pose.position.z = float(T_filt[2, 3]) * MM_TO_M

            # Orientation: rotation matrix -> quaternion
            qx, qy, qz, qw = rotation_matrix_to_quaternion(T_filt[:3, :3])
            msg.pose.orientation.x = float(qx)
            msg.pose.orientation.y = float(qy)
            msg.pose.orientation.z = float(qz)
            msg.pose.orientation.w = float(qw)

            self.pubs[i].publish(msg)

    def destroy_node(self):
        if hasattr(self, 'tracker') and self.tracker:
            try:
                self.tracker.stop_tracking()
                self.tracker.close()
            except Exception:
                pass
            self.tracker = None
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publish EKF-filtered Polaris Vega poses to ROS 2.",
    )
    parser.add_argument(
        "--hz", type=float, default=50.0,
        help="Publishing rate in Hz (default: 50)",
    )
    parser.add_argument(
        "--ip", default=None,
        help="Known Vega IP (skips discovery; e.g. 169.254.9.239)",
    )

    # parse_known_args so ROS2 remapping args pass through
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args if ros_args else None)

    try:
        node = VegaPosePublisher(hz=args.hz, vega_ip=args.ip)
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
