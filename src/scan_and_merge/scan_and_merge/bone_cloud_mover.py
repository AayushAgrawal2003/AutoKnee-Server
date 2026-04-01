#!/usr/bin/env python3
"""
Bone Cloud Mover — tracks reference point clouds with IR markers.

Two modes of operation:

1. BEFORE calibration (anchor+delta):
   Latches onto the ICP-aligned model clouds from /bone_model/*,
   locks the tracker pose at reception time, and publishes clouds
   transformed by the tracker's relative motion.

2. AFTER calibration (direct):
   Receives T_ref_to_tracker (4x4) on /calibration/* and model-frame
   reference points on /model_frame/*. Publishes live-tracked clouds
   as /tracked/{bone} = T_tracker @ T_ref_to_tracker applied to
   model-frame points.

Topic summary:
    Subscribes:
        /bone_model/{bone}             PointCloud2   ICP-aligned model (pre-cal fallback)
        /model_frame/{bone}            PointCloud2   Model in its own local frame
        /calibration/ref_to_tracker_*  Float64MultiArray  Calibration 4x4 matrix
        /kuka_frame/bone_pose_*        PoseStamped   Live IR tracker pose

    Publishes:
        /tracked/{bone}                PointCloud2   Live-tracked bone model in base frame
"""

import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Float64MultiArray


FRAME_ID = 'lbr_link_0'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_to_matrix(msg):
    p = msg.pose
    x, y, z, w = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        return np.eye(4)
    x, y, z, w = x/n, y/n, z/n, w/n
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),     1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x),     1 - 2*(x*x+y*y)],
    ])
    T[0, 3], T[1, 3], T[2, 3] = p.position.x, p.position.y, p.position.z
    return T


def _invert_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _pc2_to_numpy(msg):
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    ps = msg.point_step
    raw = np.frombuffer(msg.data, np.uint8).reshape(n, ps)
    pts = np.frombuffer(raw[:, 0:12].tobytes(), np.float32).reshape(n, 3).copy()
    if ps >= 16:
        rgb = np.frombuffer(raw[:, 12:16].tobytes(), np.uint32).reshape(n)
        cols = np.column_stack([
            ((rgb >> 16) & 0xFF).astype(np.uint8),
            ((rgb >> 8) & 0xFF).astype(np.uint8),
            (rgb & 0xFF).astype(np.uint8),
        ])
    else:
        cols = np.full((n, 3), 200, np.uint8)
    return pts, cols


def _numpy_to_pc2(points, colors, frame_id, stamp):
    n = len(points)
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = 1
    msg.width = n
    if n == 0:
        return msg
    pts = points.astype(np.float32)
    if colors is not None and len(colors) == n:
        c = colors.astype(np.uint8)
        rgb = (c[:, 0].astype(np.uint32) << 16 |
               c[:, 1].astype(np.uint32) << 8 |
               c[:, 2].astype(np.uint32))
    else:
        rgb = np.full(n, 0xC8C8C8, np.uint32)
    data = np.column_stack([pts, rgb.view(np.float32).reshape(-1, 1)])
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.data = data.tobytes()
    msg.is_dense = True
    return msg


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class BoneCloudMover(Node):

    BONES = {
        'femur': '/kuka_frame/bone_pose_femur',
        'tibia': '/kuka_frame/bone_pose_tibia',
    }

    def __init__(self):
        super().__init__('bone_cloud_mover')

        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('ref_to_tracker_tibia', '')
        self.declare_parameter('ref_to_tracker_femur', '')

        rate = self.get_parameter('publish_rate').value

        latch_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Per-bone state
        self._tracker_pose = {}      # bone -> 4x4 (live)
        self._calibration = {}       # bone -> 4x4 T_ref_to_tracker
        self._model_pts = {}         # bone -> (pts, cols) in model frame
        self._anchor_clouds = {}     # bone -> {pts, cols, initial_pose}  (pre-cal fallback)

        # Output publishers — live-tracked bone model in base frame
        self._pub = {}
        for bone in self.BONES:
            self._pub[bone] = self.create_publisher(
                PointCloud2, f'/tracked/{bone}', 10)

        # Load calibrations from file if provided at launch
        for bone in self.BONES:
            path = self.get_parameter(f'ref_to_tracker_{bone}').value
            if path:
                path = os.path.expanduser(path)
                if os.path.exists(path):
                    self._calibration[bone] = np.load(path)
                    self.get_logger().info(f'[{bone}] Loaded calibration from {path}')

        # ── Subscriptions ──

        # Live tracker poses
        for bone, topic in self.BONES.items():
            self.create_subscription(
                PoseStamped, topic,
                lambda msg, b=bone: self._pose_cb(msg, b), 10)

        # Live calibration updates (from solver)
        for bone in self.BONES:
            self.create_subscription(
                Float64MultiArray, f'/calibration/ref_to_tracker_{bone}',
                lambda msg, b=bone: self._cal_cb(msg, b), 10)

        # Model-frame reference clouds — published by solver after calibration
        for bone in self.BONES:
            self.create_subscription(
                PointCloud2, f'/model_frame/{bone}',
                lambda msg, b=bone: self._model_cb(msg, b), latch_qos)

        # Pre-calibration fallback: ICP-aligned model clouds in base frame
        for bone in self.BONES:
            self.create_subscription(
                PointCloud2, f'/bone_model/{bone}',
                lambda msg, b=bone: self._ref_cloud_cb(msg, b), latch_qos)

        self.create_timer(1.0 / rate, self._publish_loop)
        self.get_logger().info(
            f'Bone Cloud Mover running at {rate:.0f} Hz — '
            f'waiting for tracker + calibration data...')

    # ── Callbacks ─────────────────────────────────────────────────────

    def _pose_cb(self, msg, bone):
        self._tracker_pose[bone] = _pose_to_matrix(msg)

    def _cal_cb(self, msg, bone):
        if len(msg.data) != 16:
            return
        self._calibration[bone] = np.array(msg.data).reshape(4, 4)
        self.get_logger().info(f'[{bone}] Calibration received — direct tracking active')

    def _model_cb(self, msg, bone):
        pts, cols = _pc2_to_numpy(msg)
        if len(pts) > 0:
            self._model_pts[bone] = (pts, cols)
            self.get_logger().info(f'[{bone}] Model reference received ({len(pts)} pts)')

    def _ref_cloud_cb(self, msg, bone):
        """Pre-calibration fallback: store ICP-aligned base-frame cloud."""
        if bone in self._calibration:
            return  # already calibrated, ignore
        pts, cols = _pc2_to_numpy(msg)
        if len(pts) == 0:
            return
        T = self._tracker_pose.get(bone)
        if T is None:
            return
        self._anchor_clouds[bone] = {
            'pts': pts, 'cols': cols, 'initial_pose': T.copy(),
        }
        self.get_logger().info(f'[{bone}] Anchor cloud locked ({len(pts)} pts)')

    # ── Publish loop ──────────────────────────────────────────────────

    def _publish_loop(self):
        stamp = self.get_clock().now().to_msg()

        for bone in self.BONES:
            T_tracker = self._tracker_pose.get(bone)
            if T_tracker is None:
                continue

            # MODE 1: Calibrated — model pts × (T_tracker @ T_ref_to_tracker)
            if bone in self._calibration and bone in self._model_pts:
                pts, cols = self._model_pts[bone]
                T = T_tracker @ self._calibration[bone]
                out = (T[:3, :3] @ pts.T).T + T[:3, 3]
                self._pub[bone].publish(
                    _numpy_to_pc2(out, cols, FRAME_ID, stamp))
                continue

            # MODE 2: Pre-calibration fallback — anchor + delta
            anchor = self._anchor_clouds.get(bone)
            if anchor is not None:
                T_delta = T_tracker @ _invert_T(anchor['initial_pose'])
                out = (T_delta[:3, :3] @ anchor['pts'].T).T + T_delta[:3, 3]
                self._pub[bone].publish(
                    _numpy_to_pc2(out, anchor['cols'], FRAME_ID, stamp))


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = BoneCloudMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
