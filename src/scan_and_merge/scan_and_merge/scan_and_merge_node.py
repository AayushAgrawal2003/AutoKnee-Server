#!/usr/bin/env python3
"""
Point Cloud Scanning & Merging Node for KUKA LBR Med 7 + Intel RealSense

Workflow:
  1. TEACH: Manually guide the robot, press ENTER to record joint positions
  2. SCAN:  Execute trajectory through waypoints:
           - Continuous 2D RGB frames saved during motion (own thread)
           - At each waypoint: robot settles, TF is captured, then point cloud
             is captured.  The transform is stored WITH the cloud so merge
             uses the exact pose the robot had when stationary.
  3. MERGE: Transform the 5 waypoint clouds to base frame using their
            stored transforms and merge into a single cloud.

Depth filter: all clouds filtered to MAX_DEPTH_M along camera Z before save/merge.

Usage:
  ros2 run scan_and_merge scan_and_merge_node
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState, PointCloud2, Image
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    JointConstraint,
)

import numpy as np
import struct
import threading
import time
import os
import json
import cv2
from cv_bridge import CvBridge
from datetime import datetime

from tf2_ros import Buffer, TransformListener
import tf2_sensor_msgs  # noqa: F401

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARN] open3d not found. Merged cloud saved as .npy only.")


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4",
    "lbr_A5", "lbr_A6", "lbr_A7",
]

ROBOT_NAME = "lbr"
PLANNING_GROUP = "arm"
BASE_FRAME = f"{ROBOT_NAME}_link_0"
CAMERA_FRAME = "camera_depth_optical_frame"

JOINT_STATE_TOPIC = "/lbr/joint_states"
POINTCLOUD_TOPIC = "camera_arm/camera/depth/color/points"
IMAGE_TOPIC = "/camera_arm/camera/color/image_rect_raw"

NUM_WAYPOINTS = 5
VELOCITY_SCALING = 0.1
SETTLE_TIME = 1.5
PLANNING_TIME = 10.0

MAX_DEPTH_M = 1.0
CONTINUOUS_IMAGE_HZ = 2.0

OUTPUT_DIR = os.path.expanduser("~/scan_output")


class ScanAndMergeNode(Node):
    def __init__(self):
        super().__init__("scan_and_merge_node")
        self.get_logger().info("=== Scan & Merge Node Starting ===")

        # ── Parameters ──
        self.declare_parameter("load_waypoints", "")
        self.declare_parameter("velocity_scaling", VELOCITY_SCALING)
        self.declare_parameter("scan", True)
        self.declare_parameter("max_depth_m", MAX_DEPTH_M)
        self.declare_parameter("continuous_image_hz", CONTINUOUS_IMAGE_HZ)
        self.declare_parameter("settle_time", SETTLE_TIME)

        self.cb_group = ReentrantCallbackGroup()

        # ── State ──
        self.recorded_waypoints = []
        self.latest_joint_state = None
        self.latest_pointcloud = None
        self.latest_image = None
        self.latest_image_stamp = None     # track freshness
        self.cv_bridge = CvBridge()
        self.continuous_capture_enabled = False
        self.image_count = 0
        self.cloud_count = 0

        # For merging — only forward-pass waypoint clouds
        self.waypoint_clouds = []

        # ── Thread lock for image access ──
        self._image_lock = threading.Lock()

        # ── TF ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ──
        self.joint_sub = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )
        self.pc_sub = self.create_subscription(
            PointCloud2, POINTCLOUD_TOPIC, self._pointcloud_cb, 5,
            callback_group=self.cb_group,
        )
        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self._image_cb, 5,
            callback_group=self.cb_group,
        )

        # ── MoveIt2 ──
        self.move_group_client = ActionClient(
            self, MoveGroup, "/lbr/move_action",
            callback_group=self.cb_group,
        )

        # ── Output dirs ──
        self.img_dir = os.path.join(OUTPUT_DIR, "images")
        self.cloud_dir = os.path.join(OUTPUT_DIR, "clouds")
        for d in [OUTPUT_DIR, self.img_dir, self.cloud_dir]:
            os.makedirs(d, exist_ok=True)

        # ── Continuous capture thread (independent of executor) ──
        self._cont_hz = self.get_parameter(
            "continuous_image_hz"
        ).get_parameter_value().double_value
        if self._cont_hz > 0.0:
            self._cont_thread = threading.Thread(
                target=self._continuous_capture_loop, daemon=True
            )
            self._cont_thread.start()
            self.get_logger().info(
                f"  Continuous 2D capture thread: {self._cont_hz:.1f} Hz"
            )
        else:
            self.get_logger().info("  Continuous 2D capture: disabled")

        # ── Main workflow thread ──
        self._thread = threading.Thread(target=self._main_workflow, daemon=True)
        self._thread.start()

    # ──────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────
    def _joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def _pointcloud_cb(self, msg: PointCloud2):
        self.latest_pointcloud = msg

    def _image_cb(self, msg: Image):
        with self._image_lock:
            self.latest_image = msg
            self.latest_image_stamp = time.time()

    # ──────────────────────────────────────────────────────────────────
    # Continuous 2D Capture — runs in its OWN thread, not the executor
    # ──────────────────────────────────────────────────────────────────
    def _continuous_capture_loop(self):
        """
        Independent thread that saves RGB frames at a fixed rate.
        Not blocked by spin_until_future_complete in the executor.
        """
        period = 1.0 / self._cont_hz
        while rclpy.ok():
            if self.continuous_capture_enabled:
                self._save_rgb(f"continuous_{self.image_count:04d}")
            time.sleep(period)

    # ──────────────────────────────────────────────────────────────────
    # Main Workflow
    # ──────────────────────────────────────────────────────────────────
    def _main_workflow(self):
        time.sleep(2.0)

        load_path = self.get_parameter("load_waypoints").get_parameter_value().string_value
        velocity = self.get_parameter("velocity_scaling").get_parameter_value().double_value
        do_scan = self.get_parameter("scan").get_parameter_value().bool_value
        self.max_depth = self.get_parameter("max_depth_m").get_parameter_value().double_value
        self.settle_time = self.get_parameter("settle_time").get_parameter_value().double_value

        self.get_logger().info(f"  Depth filter: {self.max_depth:.2f} m")
        self.get_logger().info(f"  Settle time:  {self.settle_time:.1f} s")

        # ── Phase 1: Teach or Load ──
        if load_path and os.path.exists(load_path):
            self.recorded_waypoints = np.load(load_path).tolist()
            n = len(self.recorded_waypoints)
            self.get_logger().info(
                f"\n{'='*60}\n  LOADED {n} waypoints from {load_path}\n{'='*60}"
            )
            for i, wp in enumerate(self.recorded_waypoints):
                deg = [f"{np.degrees(j):.1f}°" for j in wp]
                self.get_logger().info(f"  WP {i+1}: {deg}")
        else:
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"  PHASE 1: TEACH MODE\n"
                f"  Move robot, press ENTER to record ({NUM_WAYPOINTS} waypoints).\n"
                f"  Type 'q' to finish early.\n"
                f"{'='*60}"
            )
            while len(self.recorded_waypoints) < NUM_WAYPOINTS:
                user_input = input(
                    f"\n  [{len(self.recorded_waypoints)+1}/{NUM_WAYPOINTS}] "
                    f"Press ENTER to record (q to finish): "
                )
                if user_input.strip().lower() == "q":
                    break
                if self.latest_joint_state is None:
                    self.get_logger().warn("No joint state received yet!")
                    continue
                joints = self._extract_joint_positions(self.latest_joint_state)
                if joints is None:
                    continue
                self.recorded_waypoints.append(joints)
                self.get_logger().info(
                    f"  Recorded WP {len(self.recorded_waypoints)}: "
                    f"{[f'{np.degrees(j):.1f}°' for j in joints]}"
                )

            wp_path = os.path.join(OUTPUT_DIR, "waypoints.npy")
            np.save(wp_path, np.array(self.recorded_waypoints))
            self.get_logger().info(f"  Waypoints saved to {wp_path}")

        if len(self.recorded_waypoints) < 2:
            self.get_logger().error("Need at least 2 waypoints. Exiting.")
            return

        # ── Phase 2: Execute & Capture ──
        input("\n  Press ENTER to start trajectory...")

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  PHASE 2: {'SCAN' if do_scan else 'REPLAY'}\n"
            f"  {len(self.recorded_waypoints)} waypoints, forward + reverse\n"
            f"  Velocity: {velocity}  |  Depth: {self.max_depth:.1f}m\n"
            f"  Settle: {self.settle_time:.1f}s  |  TF captured at each waypoint\n"
            f"{'='*60}"
        )

        # Forward pass — these clouds are used for merging
        self._execute_scan_pass(
            self.recorded_waypoints, "fwd", capture=do_scan,
            velocity=velocity, store_for_merge=True,
        )

        # Reverse pass — continuous images only, no clouds for merging
        self._execute_scan_pass(
            list(reversed(self.recorded_waypoints)), "rev", capture=do_scan,
            velocity=velocity, store_for_merge=False,
        )

        # ── Phase 3: Merge (forward-pass clouds only) ──
        if do_scan and self.waypoint_clouds:
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"  PHASE 3: MERGE ({len(self.waypoint_clouds)} waypoint clouds)\n"
                f"  Using stored transforms (captured when robot was stationary)\n"
                f"{'='*60}"
            )
            self._merge_clouds()

        self._save_manifest()

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  DONE\n"
            f"  Clouds: {self.cloud_count}  |  Images: {self.image_count}\n"
            f"  Merge clouds: {len(self.waypoint_clouds)}\n"
            f"  Output: {OUTPUT_DIR}\n"
            f"{'='*60}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Joint Extraction
    # ──────────────────────────────────────────────────────────────────
    def _extract_joint_positions(self, msg: JointState):
        try:
            return [msg.position[list(msg.name).index(j)] for j in JOINT_NAMES]
        except ValueError as e:
            self.get_logger().error(f"Joint not found: {e} (available: {msg.name})")
            return None

    # ──────────────────────────────────────────────────────────────────
    # TF Capture — called when robot is stationary
    # ──────────────────────────────────────────────────────────────────
    def _capture_current_transform(self):
        """
        Grab camera->base transform while robot is stationary.
        Returns (rotation_3x3, translation_3) or (None, None).
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAMERA_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3.0),
            )
            t = tf.transform
            trans = np.array([t.translation.x, t.translation.y, t.translation.z])
            quat = np.array([
                t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
            ])
            rot = self._quat_to_rotation_matrix(quat)

            self.get_logger().info(
                f"    TF captured: t=[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]"
            )
            return rot, trans

        except Exception as e:
            self.get_logger().error(f"    TF capture failed: {e}")
            return None, None

    # ──────────────────────────────────────────────────────────────────
    # Trajectory Execution
    # ──────────────────────────────────────────────────────────────────
    def _execute_scan_pass(self, waypoints, direction, capture, velocity,
                           store_for_merge=False):
        for i, target_joints in enumerate(waypoints):
            self.get_logger().info(
                f"\n  [{direction}] Waypoint {i+1}/{len(waypoints)}..."
            )

            self.continuous_capture_enabled = True
            success = self._move_to_joint_target(target_joints, velocity)
            self.continuous_capture_enabled = False

            if not success:
                self.get_logger().warn(f"  Failed WP {i+1}, skipping.")
                continue

            if capture:
                self.get_logger().info(
                    f"  Settling {self.settle_time}s (robot stationary)..."
                )
                time.sleep(self.settle_time)

                # Capture TF while stationary
                self.get_logger().info("  Capturing transform...")
                rot, trans = self._capture_current_transform()

                # Capture cloud + image
                self._capture_at_waypoint(
                    target_joints, f"{direction}_{i}",
                    rotation=rot, translation=trans,
                    store_for_merge=store_for_merge,
                )

    def _move_to_joint_target(self, target_joints, velocity=VELOCITY_SCALING) -> bool:
        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup action server not available!")
            return False

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = PLANNING_TIME
        req.max_velocity_scaling_factor = velocity
        req.max_acceleration_scaling_factor = velocity

        constraints = Constraints()
        for jname, jval in zip(JOINT_NAMES, target_joints):
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = jval
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints.append(constraints)

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        goal_handle = future.result()

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("  Goal rejected.")
            return False

        self.get_logger().info("  Executing...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)

        result = result_future.result()
        if result is None:
            self.get_logger().error("  Timed out.")
            return False

        if result.result.error_code.val == 1:
            self.get_logger().info("  Done.")
            return True
        else:
            self.get_logger().error(f"  MoveIt error: {result.result.error_code.val}")
            return False

    # ──────────────────────────────────────────────────────────────────
    # Capture at Waypoint
    # ──────────────────────────────────────────────────────────────────
    def _capture_at_waypoint(self, joints, label, rotation=None, translation=None,
                             store_for_merge=False):
        """Capture depth-filtered point cloud and RGB at a waypoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # ── Wait for a fresh image frame (up to 1s) ──
        self._wait_for_fresh_image(timeout=1.0)

        # ── Point Cloud ──
        if self.latest_pointcloud is not None:
            cloud_msg = self.latest_pointcloud
            points_cam, colors = self._pointcloud2_to_numpy(cloud_msg)
            n_raw = len(points_cam)

            points_cam, colors = self._apply_depth_filter(points_cam, colors)
            n_filt = len(points_cam)

            self.get_logger().info(
                f"  Cloud '{label}': {n_raw} -> {n_filt} pts "
                f"(depth <= {self.max_depth:.1f}m)"
            )

            if n_filt > 0:
                cloud_file = os.path.join(
                    self.cloud_dir, f"cloud_{label}_{timestamp}.npy"
                )
                save_dict = {
                    "points": points_cam.astype(np.float32),
                    "joints": np.array(joints, dtype=np.float32),
                }
                if colors is not None:
                    save_dict["colors"] = colors.astype(np.float32)
                if rotation is not None and translation is not None:
                    save_dict["rotation"] = rotation.astype(np.float64)
                    save_dict["translation"] = translation.astype(np.float64)
                np.save(cloud_file, save_dict)
                self.cloud_count += 1

                self.get_logger().info(f"  Saved: {os.path.basename(cloud_file)}")

                if store_for_merge and rotation is not None and translation is not None:
                    self.waypoint_clouds.append({
                        "joints": joints,
                        "points_cam": points_cam,
                        "colors": colors,
                        "label": label,
                        "rotation": rotation,
                        "translation": translation,
                    })
                    self.get_logger().info(
                        f"  Stored for merge: '{label}' "
                        f"({len(self.waypoint_clouds)} total)"
                    )
                elif store_for_merge:
                    self.get_logger().warn(
                        f"  No transform for '{label}', NOT stored for merge."
                    )
            else:
                self.get_logger().warn(f"  No points after filter for '{label}'.")
        else:
            self.get_logger().warn("  No point cloud available!")

        # ── RGB at waypoint ──
        self._save_rgb(f"waypoint_{label}")

    # ──────────────────────────────────────────────────────────────────
    # Wait for fresh image
    # ──────────────────────────────────────────────────────────────────
    def _wait_for_fresh_image(self, timeout=1.0):
        """
        Wait until we receive an image newer than 'now'.
        Ensures we don't save a stale frame from before settling.
        """
        start = time.time()
        # Mark current time — we want an image stamped AFTER this
        threshold = time.time()

        while time.time() - start < timeout:
            with self._image_lock:
                if (self.latest_image_stamp is not None
                        and self.latest_image_stamp > threshold):
                    return True
            time.sleep(0.05)

        self.get_logger().warn("  Timed out waiting for fresh image frame.")
        return False

    # ──────────────────────────────────────────────────────────────────
    # Depth Filter
    # ──────────────────────────────────────────────────────────────────
    def _apply_depth_filter(self, points, colors=None):
        if len(points) == 0:
            return points, colors
        mask = (points[:, 2] > 0.0) & (points[:, 2] <= self.max_depth)
        return points[mask], (colors[mask] if colors is not None else None)

    # ──────────────────────────────────────────────────────────────────
    # RGB Saving
    # ──────────────────────────────────────────────────────────────────
    def _save_rgb(self, label):
        """Save current RGB image as PNG with joint metadata sidecar."""
        with self._image_lock:
            image_msg = self.latest_image

        if image_msg is None:
            self.get_logger().warn(f"  No RGB image available for '{label}'")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
            filename = f"{label}_{timestamp}.png"
            cv2.imwrite(os.path.join(self.img_dir, filename), cv_img)
            self.image_count += 1
            self.get_logger().info(f"  RGB: {filename}")
        except Exception as e:
            self.get_logger().warn(f"  RGB save failed: {e}")
            return

        # Joint state sidecar
        if self.latest_joint_state is not None:
            joints = self._extract_joint_positions(self.latest_joint_state)
            if joints:
                np.save(
                    os.path.join(self.img_dir, f"{label}_{timestamp}_joints.npy"),
                    np.array(joints, dtype=np.float32),
                )

    # ──────────────────────────────────────────────────────────────────
    # Cloud Merging — uses stored transforms, no TF lookup
    # ──────────────────────────────────────────────────────────────────
    def _merge_clouds(self):
        """
        Transform waypoint clouds to base frame using transforms
        captured when the robot was stationary. No TF lookup needed.
        """
        all_points = []
        all_colors = []

        for entry in self.waypoint_clouds:
            label = entry["label"]
            pts = entry["points_cam"]
            cols = entry["colors"]
            rot = entry["rotation"]
            trans = entry["translation"]

            self.get_logger().info(
                f"  Transforming '{label}': {len(pts)} pts  "
                f"t=[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]"
            )

            transformed = (rot @ pts.T).T + trans
            all_points.append(transformed)
            if cols is not None:
                all_colors.append(cols)

        if not all_points:
            self.get_logger().error("  No clouds to merge!")
            return

        merged_pts = np.vstack(all_points)
        merged_cols = np.vstack(all_colors) if all_colors else None

        self.get_logger().info(f"  Merged: {len(merged_pts)} points total")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # .npy (always)
        npy_path = os.path.join(OUTPUT_DIR, f"merged_{timestamp}.npy")
        save_dict = {"points": merged_pts.astype(np.float32)}
        if merged_cols is not None:
            save_dict["colors"] = merged_cols.astype(np.float32)
        np.save(npy_path, save_dict)
        self.get_logger().info(f"  Saved: {npy_path}")

        # .ply (if open3d)
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged_pts)
            if merged_cols is not None and len(merged_cols) == len(merged_pts):
                pcd.colors = o3d.utility.Vector3dVector(merged_cols)

            ply_full = os.path.join(OUTPUT_DIR, f"merged_full_{timestamp}.ply")
            o3d.io.write_point_cloud(ply_full, pcd)
            self.get_logger().info(f"  Full: {ply_full}")

            voxel_size = 0.002
            pcd_down = pcd.voxel_down_sample(voxel_size)
            ply_down = os.path.join(OUTPUT_DIR, f"merged_down_{timestamp}.ply")
            o3d.io.write_point_cloud(ply_down, pcd_down)
            self.get_logger().info(
                f"  Downsampled ({voxel_size*1000:.0f}mm): "
                f"{len(pcd_down.points)} pts -> {ply_down}"
            )

    # ──────────────────────────────────────────────────────────────────
    # Manifest
    # ──────────────────────────────────────────────────────────────────
    def _save_manifest(self):
        manifest = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_depth_m": self.max_depth,
                "settle_time": self.settle_time,
                "velocity_scaling": self.get_parameter(
                    "velocity_scaling"
                ).get_parameter_value().double_value,
                "base_frame": BASE_FRAME,
                "camera_frame": CAMERA_FRAME,
            },
            "waypoints": [
                {"index": i, "joints_rad": wp}
                for i, wp in enumerate(self.recorded_waypoints)
            ],
            "clouds_saved": self.cloud_count,
            "clouds_for_merge": len(self.waypoint_clouds),
            "images_saved": self.image_count,
            "merge_labels": [e["label"] for e in self.waypoint_clouds],
            "alignment_method": "TF captured at waypoint (robot stationary)",
        }
        path = os.path.join(OUTPUT_DIR, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.get_logger().info(f"  Manifest: {path}")

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _pointcloud2_to_numpy(msg: PointCloud2):
        field_offsets = {f.name: f.offset for f in msg.fields}
        has_rgb = "rgb" in field_offsets
        data = bytes(msg.data)
        points = []
        colors = []

        for i in range(msg.width * msg.height):
            off = i * msg.point_step
            x = struct.unpack_from("f", data, off + field_offsets["x"])[0]
            y = struct.unpack_from("f", data, off + field_offsets["y"])[0]
            z = struct.unpack_from("f", data, off + field_offsets["z"])[0]

            if np.isnan(x) or np.isnan(y) or np.isnan(z) or z <= 0.0:
                continue
            points.append([x, y, z])

            if has_rgb:
                rgb_f = struct.unpack_from("f", data, off + field_offsets["rgb"])[0]
                rgb_i = struct.unpack("I", struct.pack("f", rgb_f))[0]
                colors.append([
                    ((rgb_i >> 16) & 0xFF) / 255.0,
                    ((rgb_i >> 8) & 0xFF) / 255.0,
                    (rgb_i & 0xFF) / 255.0,
                ])

        points = np.array(points, dtype=np.float64) if points else np.empty((0, 3))
        colors = np.array(colors, dtype=np.float64) if colors else None
        return points, colors

    @staticmethod
    def _quat_to_rotation_matrix(q):
        x, y, z, w = q
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])


def main(args=None):
    rclpy.init(args=args)
    node = ScanAndMergeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()