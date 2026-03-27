#!/usr/bin/env python3
"""
Multi-Orientation Utilities

Shared helpers for detect_and_merge_node and bone_cloud_mover.
No ROS dependencies in pure-math functions; ROS helpers clearly separated.
"""

import numpy as np
import threading
import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import time 


# ──────────────────────────────────────────────────────────────────────
# SE(3) / Quaternion helpers  (pure numpy)
# ──────────────────────────────────────────────────────────────────────

def quat_to_rotation_matrix(q):
    """Quaternion [x, y, z, w] -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def pose_msg_to_matrix(msg):
    """geometry_msgs/PoseStamped -> 4x4 homogeneous transform."""
    p = msg.pose
    x, y, z, w = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(4)
    x, y, z, w = x/n, y/n, z/n, w/n
    T = np.eye(4)
    T[:3, :3] = quat_to_rotation_matrix([x, y, z, w])
    T[0, 3] = p.position.x
    T[1, 3] = p.position.y
    T[2, 3] = p.position.z
    return T


def invert_transform(T):
    """SE(3) inverse via R^T."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def average_poses(poses):
    """Average a list of 4x4 poses (Wahba SVD for rotation, mean for translation)."""
    if not poses:
        return None
    translations = np.array([T[:3, 3] for T in poses])
    avg_t = translations.mean(axis=0)

    M = np.zeros((3, 3))
    for T in poses:
        M += T[:3, :3]
    M /= len(poses)
    U, _, Vt = np.linalg.svd(M)
    d = np.linalg.det(U @ Vt)
    avg_R = U @ np.diag([1.0, 1.0, d]) @ Vt

    T_avg = np.eye(4)
    T_avg[:3, :3] = avg_R
    T_avg[:3, 3] = avg_t
    return T_avg


def parse_target_classes(cls_str):
    """Parse target classes from string param. Returns list of ints or []."""
    import json as _json
    if not cls_str or cls_str.strip() == "":
        return []
    cls_str = cls_str.strip()
    if cls_str.startswith("["):
        try:
            return [int(c) for c in _json.loads(cls_str)]
        except _json.JSONDecodeError:
            pass
    try:
        return [int(c.strip()) for c in cls_str.split(",") if c.strip()]
    except ValueError:
        return []


# ──────────────────────────────────────────────────────────────────────
# PointCloud2 conversion  (needs sensor_msgs but no node)
# ──────────────────────────────────────────────────────────────────────

def numpy_to_pc2(points, colors, frame_id, stamp=None):
    """(N,3) points + optional (N,3) RGB [0-1] colors -> PointCloud2."""
    n = len(points)
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = 1
    msg.width = n

    if n == 0:
        return msg

    pts = points.astype(np.float32)

    if colors is not None and len(colors) == n:
        cols = np.clip(colors * 255, 0, 255).astype(np.uint8)
        rgb_uint32 = (cols[:, 0].astype(np.uint32) << 16 |
                      cols[:, 1].astype(np.uint32) << 8 |
                      cols[:, 2].astype(np.uint32))
    else:
        rgb_uint32 = np.full(n, (200 << 16) | (200 << 8) | 200, dtype=np.uint32)

    rgb_float = rgb_uint32.view(np.float32)
    data = np.column_stack([pts, rgb_float.reshape(-1, 1)])

    msg.fields = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.data = data.tobytes()
    msg.is_dense = True
    return msg


# ──────────────────────────────────────────────────────────────────────
# MultiOrientHelper — independent node for commands + bone poses
# ──────────────────────────────────────────────────────────────────────

class MultiOrientHelper:
    """
    Runs its OWN ROS node + SingleThreadedExecutor on a daemon thread.
    Handles both operator commands and bone-pose collection, fully
    independent of the main node's executor (which breaks under Humble
    after spin_until_future_complete calls from a daemon thread).

    Usage:
        helper = MultiOrientHelper()
        T = helper.avg_pose("femur", n_samples=50, timeout=5.0)
        cmd = helper.wait_command("keep", "discard")
        helper.clear_poses()
        helper.shutdown()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._commands = []
        self._poses = {"tibia": [], "femur": []}
        self._pose_max = 200

        self._node = rclpy.create_node("_mo_helper")

        # Command subscription
        self._node.create_subscription(
            String, "/multi_orient/command", self._cmd_cb, 10)

        # Bone pose subscriptions
        from geometry_msgs.msg import PoseStamped as _PS
        self._node.create_subscription(
            _PS, "/kuka_frame/bone_pose_tibia",
            lambda msg: self._pose_cb(msg, "tibia"), 10)
        self._node.create_subscription(
            _PS, "/kuka_frame/bone_pose_femur",
            lambda msg: self._pose_cb(msg, "femur"), 10)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        try:
            self._executor.spin()
        except Exception:
            pass

    # ── Command handling ──────────────────────────────────────────────

    def _cmd_cb(self, msg):
        raw = msg.data
        cmd = raw.strip().strip('"').strip("'").strip().lower()
        with self._lock:
            self._commands.append(cmd)

    def wait_command(self, *valid_commands, poll_interval=0.3):
        """Block until a message matching one of valid_commands arrives."""
        valid = set(valid_commands)
        with self._lock:
            self._commands.clear()
        while True:
            with self._lock:
                if self._commands:
                    cmd = self._commands.pop(0)
                    if cmd in valid:
                        return cmd
            time.sleep(poll_interval)

    def drain_commands(self):
        with self._lock:
            self._commands.clear()

    # ── Bone pose handling ────────────────────────────────────────────

    def _pose_cb(self, msg, bone):
        T = pose_msg_to_matrix(msg)
        with self._lock:
            buf = self._poses[bone]
            buf.append(T)
            if len(buf) > self._pose_max:
                buf.pop(0)

    def clear_poses(self, bone=None):
        """Clear pose buffer for one or all bones."""
        with self._lock:
            if bone:
                self._poses[bone].clear()
            else:
                for b in self._poses:
                    self._poses[b].clear()

    def avg_pose(self, bone, n_samples=50, timeout=5.0):
        """Average the last n_samples poses, waiting up to timeout seconds."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if len(self._poses[bone]) >= n_samples:
                    break
            time.sleep(0.05)

        with self._lock:
            poses = list(self._poses[bone][-n_samples:])

        if not poses:
            return None
        return average_poses(poses)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def shutdown(self):
        self._executor.shutdown()
        self._node.destroy_node()
