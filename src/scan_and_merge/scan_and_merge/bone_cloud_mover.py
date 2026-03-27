#!/usr/bin/env python3
"""
Bone Cloud Mover — rigid-body tracking of aligned point clouds.

After ICP alignment, the detect_and_merge_node publishes bone point clouds
once (with transient_local durability) in lbr_link_0 frame. This node
subscribes to those clouds AND to the live bone-tracker poses from the IR
tracking system. Once both are available for a bone, it locks the initial
tracker pose and continuously re-publishes the clouds transformed by the
tracker's relative motion.

The result: scanned/reference point clouds move in real time with the
physical bone as tracked by the Polaris Vega IR camera.

Subscriptions:
    /kuka_frame/bone_pose_femur   PoseStamped   Live femur tracker pose
    /kuka_frame/bone_pose_tibia   PoseStamped   Live tibia tracker pose
    /registered/femur             PointCloud2   ICP-aligned femur scan  (latched)
    /registered/tibia             PointCloud2   ICP-aligned tibia scan  (latched)
    /reference/femur              PointCloud2   Reference femur model   (latched)
    /reference/tibia              PointCloud2   Reference tibia model   (latched)

Publications:
    /tracked/registered/femur     PointCloud2   Femur scan following tracker
    /tracked/registered/tibia     PointCloud2   Tibia scan following tracker
    /tracked/reference/femur      PointCloud2   Reference femur following tracker
    /tracked/reference/tibia      PointCloud2   Reference tibia following tracker
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


FRAME_ID = 'lbr_link_0'


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _pose_msg_to_matrix(msg):
    """PoseStamped -> 4x4 homogeneous transform."""
    p = msg.pose
    x, y, z, w = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(4)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])
    T[0, 3] = p.position.x
    T[1, 3] = p.position.y
    T[2, 3] = p.position.z
    return T


def _invert_transform(T):
    """SE(3) inverse via R^T."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ---------------------------------------------------------------------------
# PointCloud2 conversion
# ---------------------------------------------------------------------------

def _pc2_to_numpy(msg):
    """Parse XYZRGB PointCloud2 -> (N,3) float32 points, (N,3) uint8 colors."""
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    point_step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, point_step)

    points = np.frombuffer(raw[:, 0:12].tobytes(), dtype=np.float32).reshape(n, 3).copy()

    if point_step >= 16:
        rgb_uint32 = np.frombuffer(raw[:, 12:16].tobytes(), dtype=np.uint32).reshape(n)
        r = ((rgb_uint32 >> 16) & 0xFF).astype(np.uint8)
        g = ((rgb_uint32 >> 8) & 0xFF).astype(np.uint8)
        b = (rgb_uint32 & 0xFF).astype(np.uint8)
        colors = np.column_stack([r, g, b])
    else:
        colors = np.full((n, 3), 200, dtype=np.uint8)

    return points, colors


def _numpy_to_pc2(points, colors, frame_id, stamp):
    """Build XYZRGB PointCloud2 from numpy arrays."""
    n = len(points)
    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = n

    if n == 0:
        return msg

    pts = points.astype(np.float32)

    if colors is not None and len(colors) == n:
        cols = colors.astype(np.uint8)
        rgb_uint32 = (cols[:, 0].astype(np.uint32) << 16 |
                      cols[:, 1].astype(np.uint32) << 8 |
                      cols[:, 2].astype(np.uint32))
    else:
        rgb_uint32 = np.full(n, (200 << 16) | (200 << 8) | 200, dtype=np.uint32)

    rgb_float = rgb_uint32.view(np.float32)
    data = np.column_stack([pts, rgb_float.reshape(-1, 1)])

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

    BONES = [
        {
            'name': 'femur',
            'pose_topic': '/kuka_frame/bone_pose_femur',
            'cloud_topics': ['/registered/femur', '/reference/femur'],
            'out_topics': ['/tracked/registered/femur', '/tracked/reference/femur'],
        },
        {
            'name': 'tibia',
            'pose_topic': '/kuka_frame/bone_pose_tibia',
            'cloud_topics': ['/registered/tibia', '/reference/tibia'],
            'out_topics': ['/tracked/registered/tibia', '/tracked/reference/tibia'],
        },
    ]

    def __init__(self):
        super().__init__('bone_cloud_mover')

        self.declare_parameter('publish_rate', 20.0)
        rate = self.get_parameter('publish_rate').value

        self._bones = {}

        latch_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        for cfg in self.BONES:
            bone = cfg['name']
            self._bones[bone] = {
                'current_pose': None,
                # Per-cloud-topic tracking: each locks independently
                # topic -> {points, colors, initial_pose, locked}
                'cloud_slots': {},
                'publishers': {},      # input_topic -> Publisher
            }

            # Pose subscriber (volatile — live stream)
            self.create_subscription(
                PoseStamped, cfg['pose_topic'],
                lambda msg, b=bone: self._pose_cb(msg, b), 10)

            # Cloud subscribers (transient_local — latched from detect node)
            for in_topic, out_topic in zip(cfg['cloud_topics'], cfg['out_topics']):
                self._bones[bone]['cloud_slots'][in_topic] = {
                    'points': None,
                    'colors': None,
                    'initial_pose': None,
                    'locked': False,
                }
                self.create_subscription(
                    PointCloud2, in_topic,
                    lambda msg, b=bone, t=in_topic: self._cloud_cb(msg, b, t),
                    latch_qos)
                self._bones[bone]['publishers'][in_topic] = \
                    self.create_publisher(PointCloud2, out_topic, 10)

        self.create_timer(1.0 / rate, self._publish_loop)
        self.get_logger().info(
            f'Bone Cloud Mover running at {rate:.0f} Hz. '
            f'Waiting for tracker poses and aligned point clouds...')

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _pose_cb(self, msg, bone):
        b = self._bones[bone]
        T = _pose_msg_to_matrix(msg)
        b['current_pose'] = T
        # Lock any cloud slots that have data but were waiting for a pose
        for topic, slot in b['cloud_slots'].items():
            if not slot['locked'] and slot['points'] is not None:
                self._try_lock(bone, topic)

    def _cloud_cb(self, msg, bone, topic):
        b = self._bones[bone]
        slot = b['cloud_slots'][topic]

        # Allow re-receiving a cloud even after lock (e.g. republish)
        if slot['locked']:
            return

        points, colors = _pc2_to_numpy(msg)
        if len(points) == 0:
            return

        slot['points'] = points
        slot['colors'] = colors
        self.get_logger().info(f'[{bone}] Received {topic} ({len(points)} pts)')
        self._try_lock(bone, topic)

    def _try_lock(self, bone, topic):
        """Lock a single cloud slot when we have its data + a pose."""
        b = self._bones[bone]
        slot = b['cloud_slots'][topic]
        if slot['locked'] or slot['points'] is None or b['current_pose'] is None:
            return
        slot['initial_pose'] = b['current_pose'].copy()
        slot['locked'] = True
        self.get_logger().info(
            f'[{bone}] LOCKED {topic} — {len(slot["points"])} pts, '
            f'tracking active')

    # ── Publish loop ──────────────────────────────────────────────────────

    def _publish_loop(self):
        stamp = self.get_clock().now().to_msg()

        for bone_name, b in self._bones.items():
            if b['current_pose'] is None:
                continue

            for topic, slot in b['cloud_slots'].items():
                if not slot['locked']:
                    continue

                T_delta = b['current_pose'] @ _invert_transform(slot['initial_pose'])
                pts = slot['points']
                cols = slot['colors']

                pts_transformed = (T_delta[:3, :3] @ pts.T).T + T_delta[:3, 3]

                msg = _numpy_to_pc2(pts_transformed, cols, FRAME_ID, stamp)
                b['publishers'][topic].publish(msg)


# ---------------------------------------------------------------------------
# Entry point
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
