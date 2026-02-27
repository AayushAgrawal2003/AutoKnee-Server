#!/usr/bin/env python3
"""
Simple point cloud publisher for debugging.

Loads a .ply or .npy bone cloud from ~/detect_output/ and publishes
it as PointCloud2 at 1 Hz. Use this to verify the cloud shows up
in the right place in RViz.

Usage:
  # Publish the raw merged left bone in base frame:
  ros2 run scan_and_merge cloud_publisher --ros-args \
    -p file:=~/detect_output/bone_left_raw_20260227_161610.ply \
    -p frame:=lbr_link_0

  # Or try camera frame:
  ros2 run scan_and_merge cloud_publisher --ros-args \
    -p file:=~/detect_output/bone_left_raw_20260227_161610.ply \
    -p frame:=camera_depth_optical_frame
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import os
import glob


class CloudPublisher(Node):
    def __init__(self):
        super().__init__("cloud_publisher")

        self.declare_parameter("file", "")
        self.declare_parameter("frame", "lbr_link_0")
        self.declare_parameter("topic", "/debug/cloud")

        file_path = os.path.expanduser(
            self.get_parameter("file").get_parameter_value().string_value
        )
        frame_id = self.get_parameter("frame").get_parameter_value().string_value
        topic = self.get_parameter("topic").get_parameter_value().string_value

        # Auto-find latest if no file specified
        if not file_path or not os.path.exists(file_path):
            detect_dir = os.path.expanduser("~/detect_output")
            candidates = sorted(glob.glob(os.path.join(detect_dir, "bone_*_raw_*.ply")))
            if candidates:
                file_path = candidates[-1]
                self.get_logger().info(f"Auto-found: {file_path}")
            else:
                self.get_logger().error("No file found! Pass -p file:=<path>")
                return

        self.get_logger().info(f"Loading: {file_path}")
        self.get_logger().info(f"Frame:   {frame_id}")
        self.get_logger().info(f"Topic:   {topic}")

        # Load points + colors
        points, colors = self._load_cloud(file_path)
        if points is None:
            self.get_logger().error("Failed to load cloud!")
            return

        self.get_logger().info(f"Points:  {len(points)}")
        self.get_logger().info(
            f"Range X: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]"
        )
        self.get_logger().info(
            f"Range Y: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]"
        )
        self.get_logger().info(
            f"Range Z: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]"
        )

        # Build message
        self._msg = self._make_pc2(points, colors, frame_id)
        self.get_logger().info(
            f"PC2 msg: width={self._msg.width}, "
            f"point_step={self._msg.point_step}, "
            f"data_len={len(self._msg.data)}"
        )

        # Publisher + timer
        self._pub = self.create_publisher(PointCloud2, topic, 10)
        self._timer = self.create_timer(1.0, self._publish)
        self.get_logger().info("Publishing at 1 Hz...")

    def _publish(self):
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._msg)

    def _load_cloud(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext == ".ply":
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(path)
                pts = np.asarray(pcd.points)
                cols = np.asarray(pcd.colors) if pcd.has_colors() else None
                return pts, cols
            except Exception as e:
                self.get_logger().error(f"open3d load failed: {e}")
                return None, None

        elif ext == ".npy":
            try:
                data = np.load(path, allow_pickle=True).item()
                pts = data["points"]
                cols = data.get("colors", None)
                return pts, cols
            except Exception as e:
                self.get_logger().error(f"npy load failed: {e}")
                return None, None

        else:
            self.get_logger().error(f"Unknown format: {ext}")
            return None, None

    @staticmethod
    def _make_pc2(points, colors, frame_id):
        n = len(points)
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

        msg = PointCloud2()
        msg.header = Header()
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = n
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


def main(args=None):
    rclpy.init(args=args)
    node = CloudPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()