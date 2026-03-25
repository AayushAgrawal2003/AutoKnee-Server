#!/usr/bin/env python3
"""
Waypoint Visualizer Node — publish saved joint waypoints as RViz markers.

Loads waypoints.npy (N,7 joint angles in radians), computes FK for each
waypoint using the live TF tree, and publishes:
  - /waypoint_markers  (MarkerArray): numbered EE spheres + trajectory line
  - /waypoint_poses    (PoseArray):   EE poses for MoveIt visualization

Designed to run alongside the existing scan.launch.py — just add
  visualize_waypoints:=true
and it lights up automatically in your existing RViz.

Usage:
  ros2 run scan_and_merge waypoint_visualizer --ros-args \
    -p waypoints_file:=~/scan_output/waypoints.npy

  # Or via launch (see scan.launch.py additions):
  ros2 launch scan_and_merge scan.launch.py visualize_waypoints:=true
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA

from tf2_ros import Buffer, TransformListener

import numpy as np
import os
import time
import threading


# ── KUKA LBR Med 7 ──
JOINT_NAMES = [
    "lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4",
    "lbr_A5", "lbr_A6", "lbr_A7",
]
BASE_FRAME = "lbr_link_0"
EE_FRAME = "lbr_link_ee"
CAMERA_FRAME = "camera_depth_optical_frame"

# Simplified FK from KUKA Med 7 URDF (same as replay_waypoints.py)
JOINT_ORIGINS = [
    (0.0, 0.0, 0.1475),
    (0.0, -0.0105, 0.1925),
    (0.0, 0.0105, 0.2075),
    (0.0, 0.0105, 0.1925),
    (0.0, -0.0105, 0.2075),
    (0.0, -0.0707, 0.1925),
    (0.0, 0.0707, 0.091),
]
JOINT_AXES = [
    (0, 0, 1),
    (0, 1, 0),
    (0, 0, 1),
    (0, -1, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 0, 1),
]
EE_OFFSET = np.array([0.0, 0.0, 0.189])

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


class WaypointVisualizerNode(Node):
    def __init__(self):
        super().__init__("waypoint_visualizer")
        self.get_logger().info("=== Waypoint Visualizer Starting ===")

        # ── Parameters ──
        self.declare_parameter("waypoints_file", "~/scan_output/waypoints.npy")
        self.declare_parameter("publish_rate", 1.0)
        self.declare_parameter("show_camera_frame", True)
        self.declare_parameter("use_tf_fk", True)  # True = use TF tree, False = manual FK

        self.cb_group = ReentrantCallbackGroup()

        # ── TF (for live FK via robot_state_publisher) ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Publishers ──
        self.marker_pub = self.create_publisher(
            MarkerArray, "/waypoint_markers", LATCHED_QOS
        )
        self.pose_pub = self.create_publisher(
            PoseArray, "/waypoint_poses", LATCHED_QOS
        )

        # ── Joint state publisher (to drive robot_state_publisher for TF FK) ──
        self.joint_pub = self.create_publisher(
            JointState, "/lbr/joint_states_viz", 10
        )

        # ── Main thread ──
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()

    def _main(self):
        time.sleep(1.0)

        wp_path = os.path.expanduser(
            self.get_parameter("waypoints_file").value
        )
        rate = self.get_parameter("publish_rate").value
        show_camera = self.get_parameter("show_camera_frame").value
        use_tf = self.get_parameter("use_tf_fk").value

        if not wp_path or not os.path.exists(wp_path):
            self.get_logger().error(f"Waypoints file not found: {wp_path}")
            return

        waypoints = np.load(wp_path)
        n = len(waypoints)
        self.get_logger().info(
            f"  Loaded {n} waypoints from {wp_path}"
        )

        # Print waypoints
        for i, wp in enumerate(waypoints):
            deg = [f"{np.degrees(j):.1f}°" for j in wp]
            self.get_logger().info(f"  WP {i}: {deg}")

        # ── Compute EE positions ──
        ee_positions = []    # (N, 3) in base frame
        ee_transforms = []   # (N, 4, 4) full transform

        if use_tf:
            ee_positions, ee_transforms = self._fk_via_tf(waypoints)
        
        # Fall back to manual FK if TF didn't work
        if not ee_positions:
            self.get_logger().info("  Using manual FK (URDF approximate)")
            for wp in waypoints:
                T = self._manual_fk(wp)
                ee_positions.append(T[:3, 3].copy())
                ee_transforms.append(T.copy())

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  Waypoint EE positions (in {BASE_FRAME}):\n"
            + "\n".join(
                f"  WP {i}: [{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]"
                for i, p in enumerate(ee_positions)
            )
            + f"\n{'='*60}"
        )

        # ── Build and publish markers ──
        markers = self._build_markers(ee_positions, ee_transforms, show_camera)
        poses = self._build_pose_array(ee_transforms)

        self.marker_pub.publish(markers)
        self.pose_pub.publish(poses)

        self.get_logger().info(
            f"  Published to RViz:\n"
            f"    /waypoint_markers  - MarkerArray ({len(markers.markers)} markers)\n"
            f"    /waypoint_poses    - PoseArray ({n} poses)\n"
            f"\n"
            f"  In RViz, add:\n"
            f"    MarkerArray  → topic: /waypoint_markers\n"
            f"    PoseArray    → topic: /waypoint_poses"
        )

        # Re-publish at fixed rate (latched topics + periodic for reliability)
        while rclpy.ok():
            markers.markers = [
                self._update_stamp(m) for m in markers.markers
            ]
            poses.header.stamp = self.get_clock().now().to_msg()
            self.marker_pub.publish(markers)
            self.pose_pub.publish(poses)
            time.sleep(1.0 / rate)

    # ──────────────────────────────────────────────────────────────
    # FK via TF tree (accurate, uses actual URDF)
    # ──────────────────────────────────────────────────────────────
    def _fk_via_tf(self, waypoints):
        """Publish each waypoint's joint state, wait for TF, read EE pose."""
        # Check if robot_state_publisher is alive by trying a TF lookup
        self.get_logger().info("  Attempting FK via TF tree...")
        time.sleep(1.0)

        try:
            self.tf_buffer.lookup_transform(
                BASE_FRAME, "lbr_link_7",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3.0),
            )
        except Exception:
            self.get_logger().warn(
                "  TF tree not available (robot not running?). "
                "Falling back to manual FK."
            )
            return [], []

        ee_positions = []
        ee_transforms = []

        for i, wp in enumerate(waypoints):
            # Publish joint state to move the TF tree
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(JOINT_NAMES)
            js.position = [float(j) for j in wp]
            self.joint_pub.publish(js)

            # Give TF time to update
            time.sleep(0.3)

            try:
                tf = self.tf_buffer.lookup_transform(
                    BASE_FRAME, EE_FRAME,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=2.0),
                )
                t = tf.transform
                pos = np.array([t.translation.x, t.translation.y, t.translation.z])
                quat = np.array([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
                T = np.eye(4)
                T[:3, :3] = self._quat_to_matrix(quat)
                T[:3, 3] = pos
                ee_positions.append(pos)
                ee_transforms.append(T)
            except Exception as e:
                self.get_logger().warn(f"  TF lookup failed for WP {i}: {e}")
                # Fall back to manual FK for this one
                T = self._manual_fk(wp)
                ee_positions.append(T[:3, 3].copy())
                ee_transforms.append(T.copy())

        return ee_positions, ee_transforms

    # ──────────────────────────────────────────────────────────────
    # Manual FK (approximate, from URDF DH-like parameters)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _manual_fk(q):
        """Compute EE transform using simplified URDF-based FK."""
        T = np.eye(4)
        for i in range(7):
            t = np.eye(4)
            t[:3, 3] = JOINT_ORIGINS[i]
            T = T @ t

            r = np.eye(4)
            r[:3, :3] = WaypointVisualizerNode._rot_axis(JOINT_AXES[i], q[i])
            T = T @ r

        t_ee = np.eye(4)
        t_ee[:3, 3] = EE_OFFSET
        T = T @ t_ee
        return T

    @staticmethod
    def _rot_axis(axis, angle):
        ax = np.array(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = ax
        return np.array([
            [c + x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
            [y*x*(1-c)+z*s,   c + y*y*(1-c), y*z*(1-c)-x*s],
            [z*x*(1-c)-y*s,   z*y*(1-c)+x*s, c + z*z*(1-c)],
        ])

    @staticmethod
    def _quat_to_matrix(q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])

    # ──────────────────────────────────────────────────────────────
    # Marker Building
    # ──────────────────────────────────────────────────────────────
    def _build_markers(self, positions, transforms, show_camera):
        markers = MarkerArray()
        n = len(positions)
        stamp = self.get_clock().now().to_msg()

        # ── Delete all previous markers first ──
        delete_all = Marker()
        delete_all.header.frame_id = BASE_FRAME
        delete_all.header.stamp = stamp
        delete_all.ns = "delete"
        delete_all.id = 0
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        # ── Waypoint spheres (green→red gradient) ──
        for i, pos in enumerate(positions):
            t_frac = i / max(n - 1, 1)

            sphere = Marker()
            sphere.header.frame_id = BASE_FRAME
            sphere.header.stamp = stamp
            sphere.ns = "waypoints"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2])
            )
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.04
            sphere.scale.y = 0.04
            sphere.scale.z = 0.04
            sphere.color = ColorRGBA(
                r=t_frac, g=1.0 - t_frac, b=0.2, a=1.0
            )
            sphere.lifetime.sec = 0  # persistent
            markers.markers.append(sphere)

            # ── Waypoint number label ──
            txt = Marker()
            txt.header.frame_id = BASE_FRAME
            txt.header.stamp = stamp
            txt.ns = "labels"
            txt.id = i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position = Point(
                x=float(pos[0]),
                y=float(pos[1]),
                z=float(pos[2]) + 0.04,
            )
            txt.scale.z = 0.035
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text = f"WP {i}"
            markers.markers.append(txt)

            # ── EE Z-axis arrow (tool direction) ──
            R = transforms[i][:3, :3]
            arrow = Marker()
            arrow.header.frame_id = BASE_FRAME
            arrow.header.stamp = stamp
            arrow.ns = "tool_axis"
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x = 0.008  # shaft diameter
            arrow.scale.y = 0.014  # head diameter
            arrow.scale.z = 0.010  # head length
            arrow.color = ColorRGBA(
                r=t_frac, g=1.0 - t_frac, b=0.2, a=1.0
            )

            origin = pos
            tip = origin + R[:, 2] * 0.12  # Z-axis, 12cm long
            arrow.points.append(Point(
                x=float(origin[0]), y=float(origin[1]), z=float(origin[2])
            ))
            arrow.points.append(Point(
                x=float(tip[0]), y=float(tip[1]), z=float(tip[2])
            ))
            markers.markers.append(arrow)

        # ── Trajectory line (orange) ──
        if n > 1:
            line = Marker()
            line.header.frame_id = BASE_FRAME
            line.header.stamp = stamp
            line.ns = "trajectory"
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.012
            line.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
            for pos in positions:
                line.points.append(Point(
                    x=float(pos[0]), y=float(pos[1]), z=float(pos[2])
                ))
            markers.markers.append(line)

            # Dashed return line (reverse, faded)
            ret = Marker()
            ret.header.frame_id = BASE_FRAME
            ret.header.stamp = stamp
            ret.ns = "trajectory_return"
            ret.id = 0
            ret.type = Marker.LINE_STRIP
            ret.action = Marker.ADD
            ret.scale.x = 0.008
            ret.color = ColorRGBA(r=0.5, g=0.5, b=1.0, a=0.7)
            for pos in reversed(positions):
                ret.points.append(Point(
                    x=float(pos[0]), y=float(pos[1]), z=float(pos[2])
                ))
            markers.markers.append(ret)

        # ── Robot arm links at each waypoint (ghost visualization) ──
        for i in range(n):
            t_frac = i / max(n - 1, 1)
            link_pts = self._fk_link_positions(
                [float(j) for j in np.load(
                    os.path.expanduser(
                        self.get_parameter("waypoints_file").value
                    )
                )[i]]
            )

            arm = Marker()
            arm.header.frame_id = BASE_FRAME
            arm.header.stamp = stamp
            arm.ns = "arm_ghost"
            arm.id = i
            arm.type = Marker.LINE_STRIP
            arm.action = Marker.ADD
            arm.scale.x = 0.014
            arm.color = ColorRGBA(
                r=t_frac, g=1.0 - t_frac, b=0.3, a=0.5
            )
            for pt in link_pts:
                arm.points.append(Point(
                    x=float(pt[0]), y=float(pt[1]), z=float(pt[2])
                ))
            markers.markers.append(arm)

        # ── Origin frame ──
        for axis_idx, (color, axis_vec) in enumerate([
            (ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8), [0.1, 0, 0]),
            (ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8), [0, 0.1, 0]),
            (ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8), [0, 0, 0.1]),
        ]):
            ax = Marker()
            ax.header.frame_id = BASE_FRAME
            ax.header.stamp = stamp
            ax.ns = "origin_frame"
            ax.id = axis_idx
            ax.type = Marker.ARROW
            ax.action = Marker.ADD
            ax.scale.x = 0.010
            ax.scale.y = 0.016
            ax.scale.z = 0.016
            ax.color = color
            ax.points.append(Point(x=0.0, y=0.0, z=0.0))
            ax.points.append(Point(
                x=float(axis_vec[0]),
                y=float(axis_vec[1]),
                z=float(axis_vec[2]),
            ))
            markers.markers.append(ax)

        return markers

    @staticmethod
    def _fk_link_positions(q):
        """Return all link positions for ghost arm visualization."""
        positions = [np.array([0, 0, 0])]
        T = np.eye(4)
        for i in range(7):
            t = np.eye(4)
            t[:3, 3] = JOINT_ORIGINS[i]
            T = T @ t
            r = np.eye(4)
            r[:3, :3] = WaypointVisualizerNode._rot_axis(JOINT_AXES[i], q[i])
            T = T @ r
            positions.append(T[:3, 3].copy())
        t_ee = np.eye(4)
        t_ee[:3, 3] = EE_OFFSET
        T = T @ t_ee
        positions.append(T[:3, 3].copy())
        return positions

    def _build_pose_array(self, transforms):
        msg = PoseArray()
        msg.header.frame_id = BASE_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        for T in transforms:
            pose = Pose()
            pose.position = Point(
                x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3])
            )
            q = self._matrix_to_quat(T[:3, :3])
            pose.orientation = Quaternion(
                x=float(q[0]), y=float(q[1]),
                z=float(q[2]), w=float(q[3])
            )
            msg.poses.append(pose)
        return msg

    @staticmethod
    def _matrix_to_quat(R):
        """Rotation matrix to quaternion [x, y, z, w]."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
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
        return np.array([x, y, z, w])

    def _update_stamp(self, marker):
        marker.header.stamp = self.get_clock().now().to_msg()
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = WaypointVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()