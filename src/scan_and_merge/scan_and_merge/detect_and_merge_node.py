#!/usr/bin/env python3
"""
YOLO-Filtered Point Cloud Scanning & Merging Node
for KUKA LBR Med 7 + Intel RealSense

Workflow:
  1. LOAD waypoints from .npy file (same format as scan_and_merge_node)
  2. MOVE to each waypoint via MoveIt2
  3. At each waypoint:
     a. Capture aligned RGB + Depth
     b. Run YOLOv8 detection on RGB
     c. For EACH detection instance, mask depth separately
     d. Back-project each instance → separate point cloud (camera frame)
     e. Capture camera→base TF (robot stationary)
  4. MERGE: produces per-instance merged clouds AND a combined cloud

The node is self-contained — it does NOT depend on scan_and_merge_node.
It reads the same waypoints.npy file and drives the robot itself.

Usage:
  ros2 run scan_and_merge detect_and_merge_node --ros-args \
    -p load_waypoints:=~/scan_output/waypoints.npy \
    -p weights:=~/weights/best.pt \
    -p target_classes:="[0, 1]"            # empty = all classes
    -p confidence:=0.5 \
    -p velocity_scaling:=0.1


ros2 launch scan_and_merge scan.launch.py   run_detect:=true scan_node:=false   weights:=$HOME/scan_output/best.pt   use_seg_mask:=true   confidence:=0.8   load_waypoints:=$HOME/scan_output/waypoints.npy


"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from sensor_msgs.msg import JointState, Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header, String, Float64MultiArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    JointConstraint,
)

import numpy as np
import threading
import time
import os
import json
import cv2
from cv_bridge import CvBridge
from datetime import datetime
from collections import defaultdict

from tf2_ros import Buffer, TransformListener
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[ERROR] ultralytics not installed. pip install ultralytics")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARN] open3d not found. Merged cloud saved as .npy only.")

from scan_and_merge.cloud_denoise import denoise_per_bone_pipeline
from scan_and_merge.icp_registration import register_bone
from scan_and_merge.multi_orientation_solver import solve_with_outlier_rejection
from scan_and_merge.mo_utils import (
    quat_to_rotation_matrix, numpy_to_pc2,
    parse_target_classes, MultiOrientHelper, invert_transform,
)


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
IMAGE_TOPIC = "/camera_arm/camera/color/image_rect_raw"
DEPTH_TOPIC = "/camera_arm/camera/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera_arm/camera/aligned_depth_to_color/camera_info"

VELOCITY_SCALING = 0.1
SETTLE_TIME = 1.5
PLANNING_TIME = 10.0
MAX_DEPTH_M = 1.0

OUTPUT_DIR = os.path.expanduser("~/detect_output")


class DetectAndMergeNode(Node):
    def __init__(self):
        super().__init__("detect_and_merge_node")
        self.get_logger().info("=== Detect & Merge Node Starting ===")

        # ── Parameters ──
        self.declare_parameter("load_waypoints", "")
        self.declare_parameter("weights", "")
        self.declare_parameter("target_classes", "",
                              ParameterDescriptor(
                                  description="Comma-sep class IDs or JSON list (empty=all)",
                                  type=ParameterType.PARAMETER_STRING,
                              ))
        self.declare_parameter("confidence", 0.5)
        self.declare_parameter("velocity_scaling", VELOCITY_SCALING)
        self.declare_parameter("max_depth_m", MAX_DEPTH_M)
        self.declare_parameter("settle_time", SETTLE_TIME)
        self.declare_parameter("use_seg_mask", False)

        # Denoise params
        self.declare_parameter("denoise", True)
        self.declare_parameter("cross_cloud_dist", 0.005)   # meters
        self.declare_parameter("cross_cloud_min_views", 2)
        self.declare_parameter("sor_k", 20)
        self.declare_parameter("sor_std", 2.0)
        self.declare_parameter("radius_filter", 0.01)       # meters
        self.declare_parameter("radius_min_neighbors", 5)
        self.declare_parameter("denoise_voxel_size", 0.001)  # meters
        self.declare_parameter("smooth_k", 10)
        self.declare_parameter("smooth_iterations", 1)

        # Registration params
        pkg_share = get_package_share_directory("scan_and_merge")
        self.declare_parameter("register", True)
        self.declare_parameter("tibia_reference",
            os.path.join(pkg_share, "resource", "tibia.ply"))

        self.declare_parameter("icp_coarse_method", "fpfh")
        self.declare_parameter("icp_voxel_size", 0.002)

        """
        General Good registration saved here for scans
        ~/detect_output/tibia_icp_T_ref2base_20260227_201636.npy
        ~/detect_output/femur_icp_T_ref2base_20260227_201636.npy
        
        For Full bone
        
        
        """

        self.declare_parameter("tibia_init_transform", "")  # path to .npy 4x4
        self.declare_parameter("femur_init_transform", "")  # path to .npy 4x4

        # self.declare_parameter("tibia_init_transform", "~/detect_output/tibia_icp_T_ref2base_20260227_202247.npy")  # path to .npy 4x4
        # self.declare_parameter("femur_init_transform", "~/detect_output/femur_icp_T_ref2base_20260227_202247.npy")  # path to .npy 4x4

        # Multi-orientation calibration params
        self.declare_parameter("multi_orientation", False)
        self.declare_parameter("n_orientations", 3)
        self.declare_parameter("tracker_avg_samples", 50)

        self.cb_group = ReentrantCallbackGroup()

        # ── State ──
        self.latest_joint_state = None
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_camera_info = None
        self.latest_rgb_stamp = None
        self.latest_depth_stamp = None
        self.cv_bridge = CvBridge()
        self.waypoint_clouds = []  # per-instance entries
        self.denoise_results = {}  # populated after denoise, used for registration

        self._image_lock = threading.Lock()

        # ── TF ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ──
        self.joint_sub = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )
        self.rgb_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self._rgb_cb, 5,
            callback_group=self.cb_group,
        )
        self.depth_sub = self.create_subscription(
            Image, DEPTH_TOPIC, self._depth_cb, 5,
            callback_group=self.cb_group,
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._caminfo_cb, 5,
            callback_group=self.cb_group,
        )

        # ── Multi-orientation helper (own node + executor — commands + bone poses) ──
        self._mo_helper = MultiOrientHelper()
        self._mo_status_pub = self.create_publisher(String, "/multi_orient/status", 10)

        # ── MoveIt2 ──
        self.move_group_client = ActionClient(
            self, MoveGroup, "/lbr/move_action",
            callback_group=self.cb_group,
        )

        # ── PointCloud2 Publishers (transient_local so bone_cloud_mover can latch) ──
        latch_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # /bone_scan/*  — filtered+denoised camera scan clouds (what the robot saw)
        self.pub_scan_tibia = self.create_publisher(PointCloud2, "/bone_scan/tibia", latch_qos)
        self.pub_scan_femur = self.create_publisher(PointCloud2, "/bone_scan/femur", latch_qos)
        # /bone_model/* — ICP-aligned reference PLY (model moved to match scan)
        self.pub_model_tibia = self.create_publisher(PointCloud2, "/bone_model/tibia", latch_qos)
        self.pub_model_femur = self.create_publisher(PointCloud2, "/bone_model/femur", latch_qos)

        # Messages to latch-publish (set after registration)
        self._registered_msgs = {}  # populated after ICP
        self._last_icp_metrics = {}  # bone_name -> {fitness, rmse, T_icp}

        # Calibration publishers (send T_ref_to_tracker to bone_cloud_mover live)
        self._cal_pub_tibia = self.create_publisher(
            Float64MultiArray, "/calibration/ref_to_tracker_tibia", 10)
        self._cal_pub_femur = self.create_publisher(
            Float64MultiArray, "/calibration/ref_to_tracker_femur", 10)

        # /model_frame/* — reference PLY in its own model frame (bone_cloud_mover uses this
        #                   + calibration to do live tracking)
        self.pub_model_frame_tibia = self.create_publisher(PointCloud2, "/model_frame/tibia", latch_qos)
        self.pub_model_frame_femur = self.create_publisher(PointCloud2, "/model_frame/femur", latch_qos)

        # ── Output dirs ──
        self.det_dir = os.path.join(OUTPUT_DIR, "detections")
        self.cloud_dir = os.path.join(OUTPUT_DIR, "clouds")
        for d in [OUTPUT_DIR, self.det_dir, self.cloud_dir]:
            os.makedirs(d, exist_ok=True)

        # ── Main workflow thread ──
        self._thread = threading.Thread(target=self._main_workflow, daemon=True)
        self._thread.start()

    # ──────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────
    def _joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def _rgb_cb(self, msg: Image):
        with self._image_lock:
            self.latest_rgb = msg
            self.latest_rgb_stamp = time.time()

    def _depth_cb(self, msg: Image):
        with self._image_lock:
            self.latest_depth = msg
            self.latest_depth_stamp = time.time()

    def _caminfo_cb(self, msg: CameraInfo):
        self.latest_camera_info = msg

    def _publish_mo_status(self, text):
        msg = String()
        msg.data = text
        self._mo_status_pub.publish(msg)
        self.get_logger().info(f"  [multi_orient] {text}")

    def _avg_bone_pose(self, bone, n_samples=50, timeout=5.0):
        """Average the last n_samples tracker poses for a bone."""
        T = self._mo_helper.avg_pose(bone, n_samples, timeout)
        if T is None:
            self.get_logger().warn(f"  No tracker poses for {bone}")
        return T

    def _wait_for_mo_command(self, valid_commands, prompt_msg):
        """Block until a valid command arrives on /multi_orient/command."""
        self._publish_mo_status(prompt_msg)
        cmd = self._mo_helper.wait_command(*valid_commands)
        self.get_logger().info(f"  [multi_orient] Got command: '{cmd}'")
        return cmd

    # ──────────────────────────────────────────────────────────────────
    # Main Workflow
    # ──────────────────────────────────────────────────────────────────
    def _main_workflow(self):
        time.sleep(2.0)

        # ── Read params ──
        load_path = os.path.expanduser(
            self.get_parameter("load_waypoints").get_parameter_value().string_value
        )
        weights_path = os.path.expanduser(
            self.get_parameter("weights").get_parameter_value().string_value
        )
        target_cls_str = self.get_parameter(
            "target_classes"
        ).get_parameter_value().string_value
        self.confidence = self.get_parameter(
            "confidence"
        ).get_parameter_value().double_value
        velocity = self.get_parameter(
            "velocity_scaling"
        ).get_parameter_value().double_value
        self.max_depth = self.get_parameter(
            "max_depth_m"
        ).get_parameter_value().double_value
        self.settle_time = self.get_parameter(
            "settle_time"
        ).get_parameter_value().double_value
        self.use_seg_mask = self.get_parameter(
            "use_seg_mask"
        ).get_parameter_value().bool_value

        # Denoise params
        self.do_denoise = self.get_parameter("denoise").get_parameter_value().bool_value
        self.cross_cloud_dist = self.get_parameter("cross_cloud_dist").get_parameter_value().double_value
        self.cross_cloud_min_views = self.get_parameter("cross_cloud_min_views").get_parameter_value().integer_value
        self.sor_k = self.get_parameter("sor_k").get_parameter_value().integer_value
        self.sor_std = self.get_parameter("sor_std").get_parameter_value().double_value
        self.radius_filter = self.get_parameter("radius_filter").get_parameter_value().double_value
        self.radius_min_neighbors = self.get_parameter("radius_min_neighbors").get_parameter_value().integer_value
        self.denoise_voxel_size = self.get_parameter("denoise_voxel_size").get_parameter_value().double_value
        self.smooth_k = self.get_parameter("smooth_k").get_parameter_value().integer_value
        self.smooth_iters = self.get_parameter("smooth_iterations").get_parameter_value().integer_value

        # Registration params
        self.do_register = self.get_parameter("register").get_parameter_value().bool_value
        pkg_share = get_package_share_directory("scan_and_merge")
        self.tibia_ref_path = self._resolve_ref_path(
            self.get_parameter("tibia_reference").get_parameter_value().string_value, pkg_share)
        self.femur_ref_path = self._resolve_ref_path(
            self.get_parameter("femur_reference").get_parameter_value().string_value, pkg_share)
        self.icp_coarse = self.get_parameter("icp_coarse_method").get_parameter_value().string_value
        self.icp_voxel = self.get_parameter("icp_voxel_size").get_parameter_value().double_value
        self.tibia_init_T_path = os.path.expanduser(
            self.get_parameter("tibia_init_transform").get_parameter_value().string_value
        )
        self.femur_init_T_path = os.path.expanduser(
            self.get_parameter("femur_init_transform").get_parameter_value().string_value
        )

        # Multi-orientation params
        self.multi_orientation = self.get_parameter("multi_orientation").get_parameter_value().bool_value
        self.n_orientations = self.get_parameter("n_orientations").get_parameter_value().integer_value
        self.tracker_avg_samples = self.get_parameter("tracker_avg_samples").get_parameter_value().integer_value

        # ── Parse target classes ──
        self.target_classes = parse_target_classes(target_cls_str)
        if self.target_classes:
            self.get_logger().info(f"  Filtering classes: {self.target_classes}")
        else:
            self.get_logger().info("  Using ALL detected classes")

        # ── Load YOLO ──
        if not HAS_YOLO:
            self.get_logger().error("ultralytics not installed. Exiting.")
            return
        if not weights_path or not os.path.exists(weights_path):
            self.get_logger().error(f"Weights not found: {weights_path}")
            return

        self.get_logger().info(f"  Loading YOLO: {weights_path}")
        self.model = YOLO(weights_path)
        self.get_logger().info("  YOLO loaded OK")

        # ── Load waypoints ──
        if not load_path or not os.path.exists(load_path):
            self.get_logger().error(f"Waypoints not found: {load_path}")
            return

        waypoints = np.load(load_path).tolist()
        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  LOADED {len(waypoints)} waypoints from {load_path}\n"
            f"  Weights: {os.path.basename(weights_path)}\n"
            f"  Confidence: {self.confidence}  |  Depth: {self.max_depth:.1f}m\n"
            f"  Seg masks: {self.use_seg_mask}\n"
            f"{'='*60}"
        )

        if len(waypoints) < 1:
            self.get_logger().error("Need at least 1 waypoint. Exiting.")
            return

        # ── Wait for camera info ──
        self.get_logger().info("  Waiting for camera intrinsics...")
        t0 = time.time()
        while self.latest_camera_info is None and time.time() - t0 < 15.0:
            time.sleep(0.2)
        if self.latest_camera_info is None:
            self.get_logger().error("No CameraInfo received! Exiting.")
            return
        self._parse_intrinsics(self.latest_camera_info)
        self.get_logger().info(
            f"  Intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} "
            f"cx={self.cx:.1f} cy={self.cy:.1f}"
        )

        # ── Execute ──
        if self.multi_orientation:
            self._run_multi_orientation_workflow(waypoints, velocity, weights_path)
        else:
            self._run_single_orientation_workflow(waypoints, velocity, weights_path)

    # ──────────────────────────────────────────────────────────────────
    # Single-Orientation Workflow (original behavior)
    # ──────────────────────────────────────────────────────────────────
    def _run_single_orientation_workflow(self, waypoints, velocity, weights_path):
        """Original pipeline: scan -> merge -> denoise -> register -> publish."""
        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  PHASE: DETECT & CAPTURE\n"
            f"  {len(waypoints)} waypoints, velocity={velocity}\n"
            f"{'='*60}"
        )

        for i, target_joints in enumerate(waypoints):
            self.get_logger().info(
                f"\n  ── Waypoint {i+1}/{len(waypoints)} ──"
            )

            success = self._move_to_joint_target(target_joints, velocity)
            if not success:
                self.get_logger().warn(f"  Failed WP {i+1}, skipping.")
                continue

            self.get_logger().info(
                f"  Settling {self.settle_time:.1f}s..."
            )
            time.sleep(self.settle_time)

            rot, trans = self._capture_current_transform()
            self._detect_and_capture(i, rot, trans)

        if self.waypoint_clouds:
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"  MERGE ({len(self.waypoint_clouds)} instance clouds)\n"
                f"{'='*60}"
            )
            self._merge_clouds()
        else:
            self.get_logger().warn("No clouds to merge!")

        self._save_manifest(waypoints, weights_path)

        if self.do_register and self.denoise_results:
            self._run_registration()

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"  DONE — {len(self.waypoint_clouds)} instance clouds merged\n"
            f"  Output: {OUTPUT_DIR}\n"
            f"{'='*60}"
        )

        if self._registered_msgs:
            self._publish_registered()
            self.get_logger().info(
                "  Published registered clouds (transient_local, one-shot). "
                "bone_cloud_mover will handle live tracking."
            )

    # ──────────────────────────────────────────────────────────────────
    # Multi-Orientation Workflow
    # ──────────────────────────────────────────────────────────────────
    def _run_multi_orientation_workflow(self, waypoints, velocity, weights_path):
        """
        Multi-orientation calibration workflow.

        For each orientation:
          1. Run full scan -> YOLO -> merge -> denoise -> ICP pipeline
          2. Ask user to keep/discard the registration result
          3. If kept, store (T_icp, T_tracker, fitness, rmse) per bone
          4. Wait for user to signal "ready" (bones repositioned)
        After all orientations or user says "solve":
          5. Run least-squares solver per bone
          6. Save calibration files
          7. Publish calibrated reference clouds

        User commands via /multi_orient/command topic:
          "keep"    — accept current registration for least-squares
          "discard" — reject current registration, re-try or skip
          "ready"   — bones have been moved to next position, start next scan
          "solve"   — skip remaining orientations and run least-squares now
        """
        bone_ref_map = {
            "bone_left":  "tibia",
            "bone_right": "femur",
        }

        # Per-bone orientation data collectors
        orientation_data = {
            "tibia": [],
            "femur": [],
        }

        # Accumulated denoised clouds per bone across orientations.
        # Each entry: {"points_base": (N,3), "colors": (N,3)|None,
        #              "T_tracker": 4x4, "orientation_idx": int}
        # After each orientation we REPLACE with the single merged result,
        # so at most one entry per bone at any time.
        accumulated_clouds = {"bone_left": [], "bone_right": []}

        self._publish_mo_status(
            f"MULTI-ORIENTATION MODE: {self.n_orientations} orientations planned. "
            f"Commands: publish to /multi_orient/command"
        )

        orientation_idx = 0
        done = False

        while orientation_idx < self.n_orientations and not done:
            self._publish_mo_status(
                f"=== ORIENTATION {orientation_idx+1}/{self.n_orientations} ==="
            )

            # ── Clear state for this orientation ──
            self.waypoint_clouds = []
            self.denoise_results = {}
            self._registered_msgs = {}

            # ── Capture tracker poses BEFORE scanning (average over N frames) ──
            self.get_logger().info("  Averaging tracker poses...")
            T_tracker_tibia = self._avg_bone_pose("tibia", self.tracker_avg_samples)
            T_tracker_femur = self._avg_bone_pose("femur", self.tracker_avg_samples)

            if T_tracker_tibia is not None:
                self.get_logger().info(
                    f"  Tibia tracker: t=[{T_tracker_tibia[0,3]:.4f}, "
                    f"{T_tracker_tibia[1,3]:.4f}, {T_tracker_tibia[2,3]:.4f}]"
                )
            if T_tracker_femur is not None:
                self.get_logger().info(
                    f"  Femur tracker: t=[{T_tracker_femur[0,3]:.4f}, "
                    f"{T_tracker_femur[1,3]:.4f}, {T_tracker_femur[2,3]:.4f}]"
                )

            # ── Run the full scan+detect+merge+denoise+ICP pipeline ──
            self.get_logger().info(
                f"\n{'='*60}\n"
                f"  PHASE: DETECT & CAPTURE (orientation {orientation_idx+1})\n"
                f"  {len(waypoints)} waypoints, velocity={velocity}\n"
                f"{'='*60}"
            )

            for i, target_joints in enumerate(waypoints):
                self.get_logger().info(f"\n  -- Waypoint {i+1}/{len(waypoints)} --")

                success = self._move_to_joint_target(target_joints, velocity)
                if not success:
                    self.get_logger().warn(f"  Failed WP {i+1}, skipping.")
                    continue

                self.get_logger().info(f"  Settling {self.settle_time:.1f}s...")
                time.sleep(self.settle_time)

                rot, trans = self._capture_current_transform()
                self._detect_and_capture(i, rot, trans)

            # ── Inject accumulated clouds from prior orientations ──
            if orientation_idx > 0:
                bone_tracker_pairs = [
                    ("bone_left", T_tracker_tibia),
                    ("bone_right", T_tracker_femur),
                ]
                for bone_id, current_T_tracker in bone_tracker_pairs:
                    if current_T_tracker is None:
                        continue
                    for prev in accumulated_clouds[bone_id]:
                        # T_delta moves points from old bone position to current
                        T_delta = current_T_tracker @ invert_transform(prev["T_tracker"])
                        old_pts = prev["points_base"]
                        transformed = (T_delta[:3, :3] @ old_pts.T).T + T_delta[:3, 3]
                        # Inject as synthetic waypoint entry (identity TF — already in base)
                        self.waypoint_clouds.append({
                            "label": f"accum_ori{prev['orientation_idx']}_{bone_id}",
                            "bone_id": bone_id,
                            "class_name": "accumulated",
                            "wp_idx": -1,
                            "points_cam": transformed,
                            "colors": prev["colors"],
                            "rotation": np.eye(3),
                            "translation": np.zeros(3),
                        })
                        self.get_logger().info(
                            f"  Injected {len(old_pts)} accumulated pts for {bone_id} "
                            f"from orientation {prev['orientation_idx']+1}"
                        )

            # ── Merge + denoise ──
            if self.waypoint_clouds:
                self.get_logger().info(
                    f"\n{'='*60}\n"
                    f"  MERGE ({len(self.waypoint_clouds)} instance clouds)\n"
                    f"{'='*60}"
                )
                self._merge_clouds()
            else:
                self.get_logger().warn("No clouds to merge for this orientation!")

            # ── Update accumulated clouds with merged+denoised result ──
            # Replace prior accumulated entries with the single combined cloud
            # (which now includes old accumulated + new scan data, all denoised).
            for bone_id, tracker_T in [("bone_left", T_tracker_tibia),
                                        ("bone_right", T_tracker_femur)]:
                if bone_id in self.denoise_results and tracker_T is not None:
                    pts = self.denoise_results[bone_id]["points"]
                    cols = self.denoise_results[bone_id]["colors"]
                    accumulated_clouds[bone_id] = [{
                        "points_base": pts.copy(),
                        "colors": cols.copy() if cols is not None else None,
                        "T_tracker": tracker_T.copy(),
                        "orientation_idx": orientation_idx,
                    }]
                    self.get_logger().info(
                        f"  Accumulated cloud updated for {bone_id}: "
                        f"{len(pts)} denoised pts at orientation {orientation_idx+1}"
                    )

            # ── ICP Registration ──
            icp_results = {}
            if self.do_register and self.denoise_results:
                self._run_registration()
                icp_results = dict(self._registered_msgs)

            # ── Publish so user can see result in RViz ──
            if self._registered_msgs:
                self._publish_registered()

            # ── Ask user: which bones to keep? ──
            cmd = self._wait_for_mo_command(
                ["both", "femur", "tibia", "neither"],
                f"Orientation {orientation_idx+1} done. "
                f"Publish 'both', 'femur', 'tibia', or 'neither' to /multi_orient/command"
            )

            accept_tibia = cmd in ("both", "tibia")
            accept_femur = cmd in ("both", "femur")

            bone_tracker_map = {
                "tibia": T_tracker_tibia,
                "femur": T_tracker_femur,
            }
            accept_map = {"tibia": accept_tibia, "femur": accept_femur}

            for bone_name in ["tibia", "femur"]:
                if not accept_map[bone_name]:
                    continue

                T_tracker = bone_tracker_map.get(bone_name)
                if T_tracker is None:
                    self.get_logger().warn(
                        f"  No tracker pose for {bone_name}, skipping"
                    )
                    continue

                if bone_name not in self._last_icp_metrics:
                    self.get_logger().warn(
                        f"  No ICP result for {bone_name}, skipping"
                    )
                    continue

                metrics = self._last_icp_metrics[bone_name]
                orientation_data[bone_name].append({
                    "T_tracker": T_tracker.copy(),
                    "T_icp": metrics["T_icp"].copy(),
                    "fitness": metrics["fitness"],
                    "rmse": metrics["rmse"],
                    "orientation_idx": orientation_idx,
                })
                self.get_logger().info(
                    f"  Stored orientation {orientation_idx+1} for {bone_name} "
                    f"(fitness={metrics['fitness']:.4f}, rmse={metrics['rmse']:.6f})"
                )

            self._publish_mo_status(
                f"Orientation {orientation_idx+1}: "
                f"tibia={'KEPT' if accept_tibia else 'DISCARDED'}, "
                f"femur={'KEPT' if accept_femur else 'DISCARDED'}. "
                f"Tibia: {len(orientation_data['tibia'])} stored, "
                f"Femur: {len(orientation_data['femur'])} stored."
            )

            orientation_idx += 1

            # ── If not last orientation, wait for user to reposition bones ──
            if orientation_idx < self.n_orientations:
                cmd = self._wait_for_mo_command(
                    ["ready", "solve"],
                    f"Reposition bones, then publish 'ready'. "
                    f"Or publish 'solve' to finish early."
                )
                if cmd == "solve":
                    self._publish_mo_status("Early solve requested.")
                    done = True

                # Clear pose buffer so we get fresh poses for next orientation
                self._mo_helper.clear_poses()

        # ── Run the least-squares solver ──
        self._run_multi_orient_solve(orientation_data)

    def _run_multi_orient_solve(self, orientation_data):
        """Run the least-squares solver and save results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_id = BASE_FRAME
        report = {}

        for bone_name in ["tibia", "femur"]:
            oris = orientation_data[bone_name]

            if len(oris) < 1:
                self.get_logger().warn(
                    f"  No orientation data for {bone_name}, skipping solve"
                )
                continue

            self._publish_mo_status(
                f"Solving {bone_name} with {len(oris)} orientations..."
            )

            try:
                result = solve_with_outlier_rejection(oris)
            except Exception as e:
                self.get_logger().error(f"  Solver failed for {bone_name}: {e}")
                continue

            T_ref_to_tracker = result["T_ref_to_tracker"]
            val = result["validation"]

            self.get_logger().info(
                f"\n{'='*60}\n"
                f"  {bone_name.upper()} MULTI-ORIENT RESULT\n"
                f"  Orientations used: {result['n_orientations_used']}/{len(oris)}\n"
                f"  Rejected: {result['rejected_indices']}\n"
                f"  Mean translation residual: {val['mean_trans_mm']:.2f} mm\n"
                f"  Mean rotation residual: {val['mean_rot_deg']:.2f} deg\n"
                f"{'='*60}"
            )

            for i, r in enumerate(val["residuals"]):
                flag = " ** OUTLIER" if i in [
                    ri for ri in result["rejected_indices"]
                ] else ""
                self.get_logger().info(
                    f"    Orientation {i+1}: "
                    f"trans={r['trans_mm']:.2f}mm, rot={r['rot_deg']:.2f}deg{flag}"
                )

            # Save calibration
            cal_path = os.path.join(
                OUTPUT_DIR, f"T_ref_to_tracker_{bone_name}_{timestamp}.npy"
            )
            np.save(cal_path, T_ref_to_tracker)
            self.get_logger().info(f"  Saved calibration: {cal_path}")

            # Publish calibration to bone_cloud_mover for immediate use
            cal_msg = Float64MultiArray()
            cal_msg.data = T_ref_to_tracker.flatten().tolist()
            if bone_name == "tibia":
                self._cal_pub_tibia.publish(cal_msg)
            else:
                self._cal_pub_femur.publish(cal_msg)
            self.get_logger().info(
                f"  Published live calibration for {bone_name} to bone_cloud_mover"
            )

            # Save orientation data
            ori_path = os.path.join(
                OUTPUT_DIR, f"multi_orient_{bone_name}_{timestamp}.npy"
            )
            np.save(ori_path, oris, allow_pickle=True)

            report[bone_name] = {
                "n_orientations": len(oris),
                "n_used": result["n_orientations_used"],
                "rejected": result["rejected_indices"],
                "mean_trans_mm": val["mean_trans_mm"],
                "mean_rot_deg": val["mean_rot_deg"],
                "residuals": val["residuals"],
                "calibration_file": cal_path,
            }

            # Publish model-frame reference points for bone_cloud_mover
            ref_path = self.tibia_ref_path if bone_name == "tibia" else self.femur_ref_path
            if ref_path and os.path.exists(ref_path):
                try:
                    from scan_and_merge.icp_registration import load_reference_mesh
                    ref_pcd = load_reference_mesh(ref_path, voxel_size=self.icp_voxel)
                    ref_pts = np.asarray(ref_pcd.points)
                    ref_cols = np.asarray(ref_pcd.colors) if ref_pcd.has_colors() else None

                    # Publish model-frame points (bone_cloud_mover applies T_tracker @ T_cal)
                    model_msg = numpy_to_pc2(ref_pts, ref_cols, "model_frame")
                    if bone_name == "tibia":
                        self.pub_model_frame_tibia.publish(model_msg)
                    else:
                        self.pub_model_frame_femur.publish(model_msg)
                    self.get_logger().info(
                        f"  Published model-frame reference ({len(ref_pts)} pts) for {bone_name}"
                    )
                except Exception as e:
                    self.get_logger().error(
                        f"  Failed to publish model reference for {bone_name}: {e}"
                    )

        # Save report
        report_path = os.path.join(
            OUTPUT_DIR, f"multi_orient_report_{timestamp}.json"
        )
        # Convert numpy types for JSON
        def _json_safe(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        safe_report = json.loads(json.dumps(report, default=_json_safe))
        with open(report_path, "w") as f:
            json.dump(safe_report, f, indent=2)
        self.get_logger().info(f"  Report saved: {report_path}")

        self._publish_mo_status(
            f"MULTI-ORIENTATION CALIBRATION COMPLETE. "
            f"Calibration files saved to {OUTPUT_DIR}"
        )

    # ──────────────────────────────────────────────────────────────────
    # YOLO Detection — per-instance depth masking & back-projection
    # ──────────────────────────────────────────────────────────────────
    def _detect_and_capture(self, wp_idx, rotation, translation):
        """
        At the current waypoint:
          1. Grab fresh RGB + aligned depth
          2. Run YOLO on RGB
          3. Expect exactly 2 detections (2 bones)
          4. Sort by bbox x-center → left bone / right bone
          5. Mask depth, back-project, store with bone_id for cross-waypoint merge
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label = f"wp_{wp_idx}"

        # ── Wait for fresh frames ──
        self._wait_for_fresh_frames(timeout=2.0)

        with self._image_lock:
            rgb_msg = self.latest_rgb
            depth_msg = self.latest_depth

        if rgb_msg is None or depth_msg is None:
            self.get_logger().warn(f"  Missing RGB or depth at {label}")
            return

        # Convert
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_m = depth.astype(np.float32) / 1000.0

        self.get_logger().info(
            f"  RGB: {rgb.shape}  Depth: {depth_m.shape}  "
            f"range: [{depth_m[depth_m > 0].min():.3f}, {depth_m.max():.3f}]m"
        )

        # ── YOLO inference ──
        results = self.model(rgb, conf=self.confidence, verbose=False)[0]

        boxes = results.boxes
        n_det = len(boxes)
        self.get_logger().info(f"  YOLO: {n_det} detections")

        if n_det == 0:
            self.get_logger().info(f"  No detections at {label}, skipping.")
            return

        # ── Save annotated image ──
        annotated = results.plot()
        det_path = os.path.join(self.det_dir, f"det_{label}_{timestamp}.png")
        cv2.imwrite(det_path, annotated)

        # ── Filter to target classes ──
        filtered = []
        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if self.target_classes and cls_id not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x_center = (x1 + x2) / 2.0
            filtered.append({
                "idx": j,
                "box": box,
                "cls_id": cls_id,
                "cls_name": results.names[cls_id],
                "conf": float(box.conf[0]),
                "x_center": x_center,
                "xyxy": (x1, y1, x2, y2),
            })

        if len(filtered) < 2:
            self.get_logger().warn(
                f"  Expected 2 bones, got {len(filtered)} detections at {label}. "
                f"Skipping this waypoint."
            )
            return

        if len(filtered) > 2:
            # Take the 2 highest-confidence detections
            filtered.sort(key=lambda d: d["conf"], reverse=True)
            filtered = filtered[:2]
            self.get_logger().info(
                f"  Got {n_det} detections, using top 2 by confidence"
            )

        # ── Sort by x_center: left bone first, right bone second ──
        filtered.sort(key=lambda d: d["x_center"])
        bone_names = ["bone_left", "bone_right"]

        for bone_idx, (det, bone_name) in enumerate(zip(filtered, bone_names)):
            j = det["idx"]
            cls_name = det["cls_name"]
            conf = det["conf"]
            x1, y1, x2, y2 = det["xyxy"]
            inst_label = f"{label}_{bone_name}"

            # ── Build per-instance mask ──
            inst_mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)

            if self.use_seg_mask and results.masks is not None:
                seg_mask = results.masks.data[j].cpu().numpy()
                seg_mask_resized = cv2.resize(
                    seg_mask, (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                inst_mask[seg_mask_resized > 0.5] = 255
                self.get_logger().info(
                    f"    {bone_name}: {cls_name} ({conf:.2f}) "
                    f"x_center={det['x_center']:.0f} [seg mask]"
                )
            else:
                inst_mask[y1:y2, x1:x2] = 255
                self.get_logger().info(
                    f"    {bone_name}: {cls_name} ({conf:.2f}) "
                    f"box=[{x1},{y1},{x2},{y2}] x_center={det['x_center']:.0f}"
                )

            # ── Mask depth for this bone ──
            inst_depth = np.where(inst_mask > 0, depth_m, 0.0)
            inst_depth[inst_depth > self.max_depth] = 0.0

            n_valid = np.count_nonzero(inst_depth > 0)
            if n_valid == 0:
                self.get_logger().warn(f"    {inst_label}: no valid depth, skipping")
                continue

            # ── Back-project this bone ──
            points_cam, colors = self._backproject(inst_depth, rgb, inst_mask)
            self.get_logger().info(
                f"    {inst_label}: {len(points_cam)} points"
            )

            # ── Save per-instance cloud ──
            cloud_file = os.path.join(
                self.cloud_dir, f"cloud_{inst_label}_{timestamp}.npy"
            )
            save_dict = {"points": points_cam.astype(np.float32)}
            if colors is not None:
                save_dict["colors"] = colors.astype(np.float32)
            if rotation is not None and translation is not None:
                save_dict["rotation"] = rotation.astype(np.float64)
                save_dict["translation"] = translation.astype(np.float64)
            np.save(cloud_file, save_dict)

            # ── Store for merge, keyed by bone_name ──
            if rotation is not None and translation is not None:
                self.waypoint_clouds.append({
                    "label": inst_label,
                    "bone_id": bone_name,   # "bone_left" or "bone_right"
                    "class_name": cls_name,
                    "wp_idx": wp_idx,
                    "points_cam": points_cam,
                    "colors": colors,
                    "rotation": rotation,
                    "translation": translation,
                })

        self.get_logger().info(
            f"  {label}: 2 bones identified (left/right by x-position)  "
            f"({len(self.waypoint_clouds)} total stored)"
        )

    # ──────────────────────────────────────────────────────────────────
    # Back-projection: masked depth → 3D points
    # ──────────────────────────────────────────────────────────────────
    def _backproject(self, depth_m, rgb, mask):
        """
        Back-project depth image to 3D points in camera_depth_optical_frame.

        Since we use aligned_depth_to_color, the depth pixels are already
        registered to the color image — same resolution and pixel grid.
        So intrinsics from the aligned depth camera_info apply directly.

        Returns:
            points: (N, 3) float64 in meters, camera optical frame
            colors: (N, 3) float64 in [0, 1] RGB
        """
        h, w = depth_m.shape[:2]

        # Pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Valid mask: detected + positive depth
        valid = (mask > 0) & (depth_m > 0.0)
        z = depth_m[valid]
        u = u[valid].astype(np.float64)
        v = v[valid].astype(np.float64)

        # Pinhole back-projection
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        points = np.stack([x, y, z], axis=-1)

        # Colors from RGB (OpenCV is BGR)
        rgb_float = rgb[:, :, ::-1].astype(np.float64) / 255.0  # BGR→RGB
        colors = rgb_float[valid]

        return points, colors

    def _parse_intrinsics(self, info: CameraInfo):
        """Extract fx, fy, cx, cy from camera info K matrix."""
        K = np.array(info.k).reshape(3, 3)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    # ──────────────────────────────────────────────────────────────────
    # Wait for fresh frames
    # ──────────────────────────────────────────────────────────────────
    def _wait_for_fresh_frames(self, timeout=2.0):
        """Wait until both RGB and depth are newer than 'now'."""
        threshold = time.time()
        start = time.time()
        while time.time() - start < timeout:
            with self._image_lock:
                rgb_ok = (self.latest_rgb_stamp is not None
                          and self.latest_rgb_stamp > threshold)
                depth_ok = (self.latest_depth_stamp is not None
                            and self.latest_depth_stamp > threshold)
                if rgb_ok and depth_ok:
                    return True
            time.sleep(0.05)
        self.get_logger().warn("  Timed out waiting for fresh RGB+Depth.")
        return False

    # ──────────────────────────────────────────────────────────────────
    # TF Capture
    # ──────────────────────────────────────────────────────────────────
    def _capture_current_transform(self):
        """Grab camera→base TF while robot is stationary."""
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
            rot = quat_to_rotation_matrix(quat)
            self.get_logger().info(
                f"    TF: t=[{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]"
            )
            return rot, trans
        except Exception as e:
            self.get_logger().error(f"    TF failed: {e}")
            return None, None

    # ──────────────────────────────────────────────────────────────────
    # Motion
    # ──────────────────────────────────────────────────────────────────
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
            self.get_logger().info("  Arrived.")
            return True
        else:
            self.get_logger().error(
                f"  MoveIt error: {result.result.error_code.val}"
            )
            return False

    # ──────────────────────────────────────────────────────────────────
    # Cloud Merging — per-instance + combined
    # ──────────────────────────────────────────────────────────────────
    def _merge_clouds(self):
        """
        Merge point clouds per bone across all waypoints.

        If denoise is enabled, runs the full pipeline:
          1. Cross-cloud consistency filter (remove points only seen in 1 view)
          2. Statistical outlier removal
          3. Radius outlier removal
          4. Voxel downsample
          5. Laplacian smoothing

        Produces:
          - bone_left_<ts>.ply / bone_right_<ts>.ply (raw merged)
          - bone_left_clean_<ts>.ply / bone_right_clean_<ts>.ply (denoised)
          - detected_merged_clean_<ts>.ply (both bones denoised + combined)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Save raw merged per bone first ──
        by_bone = defaultdict(list)
        for entry in self.waypoint_clouds:
            by_bone[entry["bone_id"]].append(entry)

        for bone_id in ["bone_left", "bone_right"]:
            entries = by_bone.get(bone_id, [])
            if not entries:
                continue

            bone_pts = []
            bone_cols = []
            for entry in entries:
                pts = entry["points_cam"]
                cols = entry["colors"]
                rot = entry["rotation"]
                trans = entry["translation"]
                transformed = (rot @ pts.T).T + trans
                bone_pts.append(transformed)
                if cols is not None:
                    bone_cols.append(cols)

            merged = np.vstack(bone_pts)
            merged_c = np.vstack(bone_cols) if bone_cols else None

            self.get_logger().info(
                f"  {bone_id} raw: {len(merged)} pts from {len(entries)} views"
            )

            # Save raw
            if HAS_OPEN3D:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(merged)
                if merged_c is not None:
                    pcd.colors = o3d.utility.Vector3dVector(merged_c)
                raw_path = os.path.join(OUTPUT_DIR, f"{bone_id}_raw_{timestamp}.ply")
                o3d.io.write_point_cloud(raw_path, pcd)
                self.get_logger().info(f"  Saved raw: {raw_path}")

        # ── Denoise ──
        if self.do_denoise:
            self.get_logger().info(
                f"\n  ── Denoising Pipeline ──\n"
                f"  Cross-cloud: dist={self.cross_cloud_dist}m, min_views={self.cross_cloud_min_views}\n"
                f"  SOR: k={self.sor_k}, std={self.sor_std}\n"
                f"  Radius: r={self.radius_filter}m, min_n={self.radius_min_neighbors}\n"
                f"  Voxel: {self.denoise_voxel_size*1000:.1f}mm\n"
                f"  Smooth: k={self.smooth_k}, iters={self.smooth_iters}"
            )

            results = denoise_per_bone_pipeline(
                self.waypoint_clouds,
                cross_dist=self.cross_cloud_dist,
                min_views=self.cross_cloud_min_views,
                sor_k=self.sor_k,
                sor_std=self.sor_std,
                radius=self.radius_filter,
                min_neighbors=self.radius_min_neighbors,
                voxel_size=self.denoise_voxel_size,
                smooth_k=self.smooth_k,
                smooth_iters=self.smooth_iters,
                verbose=True,
            )

            # Save denoised per-bone
            for bone_id in ["bone_left", "bone_right"]:
                if bone_id not in results:
                    continue
                pts = results[bone_id]["points"]
                cols = results[bone_id]["colors"]

                self.get_logger().info(f"  {bone_id} clean: {len(pts)} pts")

                npy_path = os.path.join(OUTPUT_DIR, f"{bone_id}_clean_{timestamp}.npy")
                save_dict = {"points": pts.astype(np.float32)}
                if cols is not None:
                    save_dict["colors"] = cols.astype(np.float32)
                np.save(npy_path, save_dict)

                if HAS_OPEN3D:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    if cols is not None and len(cols) == len(pts):
                        pcd.colors = o3d.utility.Vector3dVector(cols)
                    ply_path = os.path.join(OUTPUT_DIR, f"{bone_id}_clean_{timestamp}.ply")
                    o3d.io.write_point_cloud(ply_path, pcd)
                    self.get_logger().info(f"  Saved: {ply_path}")

            # Store for registration phase
            self.denoise_results = results

            # Save combined denoised
            if "combined" in results:
                pts = results["combined"]["points"]
                cols = results["combined"]["colors"]

                self.get_logger().info(f"  Combined clean: {len(pts)} pts")

                npy_path = os.path.join(OUTPUT_DIR, f"detected_merged_clean_{timestamp}.npy")
                save_dict = {"points": pts.astype(np.float32)}
                if cols is not None:
                    save_dict["colors"] = cols.astype(np.float32)
                np.save(npy_path, save_dict)

                if HAS_OPEN3D:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    if cols is not None and len(cols) == len(pts):
                        pcd.colors = o3d.utility.Vector3dVector(cols)
                    ply_path = os.path.join(OUTPUT_DIR, f"detected_merged_clean_{timestamp}.ply")
                    o3d.io.write_point_cloud(ply_path, pcd)
                    self.get_logger().info(f"  Saved: {ply_path}")

        else:
            # No denoise — just merge and save as before
            all_points = []
            all_colors = []
            for bone_id in ["bone_left", "bone_right"]:
                for entry in by_bone.get(bone_id, []):
                    pts = entry["points_cam"]
                    rot = entry["rotation"]
                    trans = entry["translation"]
                    transformed = (rot @ pts.T).T + trans
                    all_points.append(transformed)
                    if entry["colors"] is not None:
                        all_colors.append(entry["colors"])

            if all_points:
                merged_pts = np.vstack(all_points)
                merged_cols = np.vstack(all_colors) if all_colors else None
                npy_path = os.path.join(OUTPUT_DIR, f"detected_merged_{timestamp}.npy")
                np.save(npy_path, {"points": merged_pts.astype(np.float32),
                                    "colors": merged_cols.astype(np.float32) if merged_cols is not None else None})
                if HAS_OPEN3D:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(merged_pts)
                    if merged_cols is not None:
                        pcd.colors = o3d.utility.Vector3dVector(merged_cols)
                    ply_path = os.path.join(OUTPUT_DIR, f"detected_merged_{timestamp}.ply")
                    o3d.io.write_point_cloud(ply_path, pcd)
                    self.get_logger().info(f"  Saved: {ply_path}")

    # ──────────────────────────────────────────────────────────────────
    # ICP Registration
    # ──────────────────────────────────────────────────────────────────
    def _run_registration(self):
        """
        Register reference models to the scanned bones.

        ICP direction: reference (source) → scan (target)
        Result: T maps reference model frame → lbr_link_0 (base frame)

        Publishes:
          /registered/tibia, /registered/femur  — denoised scan clouds (base frame)
          /reference/tibia, /reference/femur    — reference models aligned into base frame
        """
        frame_id = BASE_FRAME
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        bone_ref_map = {
            "bone_left":  ("tibia", self.tibia_ref_path),
            "bone_right": ("femur", self.femur_ref_path),
        }

        for bone_id, (bone_name, ref_path) in bone_ref_map.items():
            if bone_id not in self.denoise_results:
                self.get_logger().warn(f"  No denoised cloud for {bone_id}, skipping registration")
                continue

            if not ref_path or not os.path.exists(ref_path):
                self.get_logger().warn(f"  Reference PLY not found for {bone_name}: {ref_path}")
                continue

            pts = self.denoise_results[bone_id]["points"]   # already in lbr_link_0
            cols = self.denoise_results[bone_id]["colors"]

            self.get_logger().info(
                f"\n{'='*60}\n"
                f"  REGISTERING {bone_name.upper()} ({bone_id})\n"
                f"  Scan: {len(pts)} pts (in {frame_id}) → Ref: {ref_path}\n"
                f"{'='*60}"
            )

            # Load initial transform if provided
            init_T = None
            init_T_path = self.tibia_init_T_path if bone_name == "tibia" else self.femur_init_T_path
            if init_T_path and os.path.exists(init_T_path):
                init_T = np.load(init_T_path)
                self.get_logger().info(f"  Using init transform: {init_T_path}")

            try:
                result = register_bone(
                    pts, cols, ref_path,
                    coarse_method=self.icp_coarse,
                    voxel_size=self.icp_voxel,
                    init_transform=init_T,
                )
            except Exception as e:
                self.get_logger().error(f"  Registration failed for {bone_name}: {e}")
                continue

            self.get_logger().info(
                f"  {bone_name}: fitness={result['fitness']:.4f}, "
                f"RMSE={result['rmse']:.6f}"
            )

            # T maps reference model → base frame
            T_ref_to_base = result["transform"]

            # aligned_ref_points = reference model now in base frame
            aligned_ref_pts = result["aligned_ref_points"]
            aligned_ref_cols = result["aligned_ref_colors"]

            # scan_points already in base frame (from denoise pipeline)
            scan_pts = result["scan_points"]
            scan_cols = result["scan_colors"]

            self.get_logger().info(
                f"  {bone_name} scan in base: "
                f"x=[{scan_pts[:,0].min():.3f},{scan_pts[:,0].max():.3f}] "
                f"y=[{scan_pts[:,1].min():.3f},{scan_pts[:,1].max():.3f}] "
                f"z=[{scan_pts[:,2].min():.3f},{scan_pts[:,2].max():.3f}]"
            )
            self.get_logger().info(
                f"  {bone_name} ref in base: "
                f"x=[{aligned_ref_pts[:,0].min():.3f},{aligned_ref_pts[:,0].max():.3f}] "
                f"y=[{aligned_ref_pts[:,1].min():.3f},{aligned_ref_pts[:,1].max():.3f}] "
                f"z=[{aligned_ref_pts[:,2].min():.3f},{aligned_ref_pts[:,2].max():.3f}]"
            )

            # Store metrics for multi-orientation solver
            self._last_icp_metrics[bone_name] = {
                "fitness": result["fitness"],
                "rmse": result["rmse"],
                "T_icp": T_ref_to_base.copy(),
            }

            # Save transform + aligned reference PLY
            np.save(
                os.path.join(OUTPUT_DIR, f"{bone_name}_icp_T_ref2base_{timestamp}.npy"),
                T_ref_to_base,
            )
            if HAS_OPEN3D:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(aligned_ref_pts)
                if aligned_ref_cols is not None:
                    pcd.colors = o3d.utility.Vector3dVector(aligned_ref_cols)
                ref_path_out = os.path.join(OUTPUT_DIR, f"{bone_name}_ref_in_base_{timestamp}.ply")
                o3d.io.write_point_cloud(ref_path_out, pcd)
                self.get_logger().info(f"  Saved: {ref_path_out}")

            # PointCloud2 messages — all in lbr_link_0
            scan_msg = numpy_to_pc2(scan_pts, scan_cols, frame_id)
            ref_msg = numpy_to_pc2(aligned_ref_pts, aligned_ref_cols, frame_id)

            if bone_name == "tibia":
                self._registered_msgs["scan_tibia"] = scan_msg
                self._registered_msgs["model_tibia"] = ref_msg
            else:
                self._registered_msgs["scan_femur"] = scan_msg
                self._registered_msgs["model_femur"] = ref_msg

            self.get_logger().info(
                f"  {bone_name}: publishing /bone_scan + /bone_model in {frame_id}"
            )

    # ──────────────────────────────────────────────────────────────────
    # Periodic Publish (latch for RViz)
    # ──────────────────────────────────────────────────────────────────
    def _publish_registered(self):
        """Publish scan + ICP-model clouds (one-shot, transient_local)."""
        now = self.get_clock().now().to_msg()

        for key, pub in [
            ("scan_tibia",  self.pub_scan_tibia),
            ("scan_femur",  self.pub_scan_femur),
            ("model_tibia", self.pub_model_tibia),
            ("model_femur", self.pub_model_femur),
        ]:
            if key in self._registered_msgs:
                msg = self._registered_msgs[key]
                msg.header.stamp = now
                pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────
    # Manifest
    # ──────────────────────────────────────────────────────────────────
    def _save_manifest(self, waypoints, weights_path):
        # Summarize per bone
        bone_summary = defaultdict(int)
        for e in self.waypoint_clouds:
            bone_summary[e["bone_id"]] += 1

        manifest = {
            "timestamp": datetime.now().isoformat(),
            "node": "detect_and_merge_node",
            "config": {
                "weights": weights_path,
                "confidence": self.confidence,
                "target_classes": self.target_classes,
                "use_seg_mask": self.use_seg_mask,
                "max_depth_m": self.max_depth,
                "settle_time": self.settle_time,
                "base_frame": BASE_FRAME,
                "camera_frame": CAMERA_FRAME,
            },
            "waypoints": len(waypoints),
            "total_instance_clouds": len(self.waypoint_clouds),
            "clouds_per_bone": dict(bone_summary),
            "instance_labels": [e["label"] for e in self.waypoint_clouds],
        }
        path = os.path.join(OUTPUT_DIR, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        self.get_logger().info(f"  Manifest: {path}")

    @staticmethod
    def _resolve_ref_path(path, pkg_share):
        """Resolve a reference PLY path: absolute, ~, or relative to pkg share."""
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(pkg_share, path)

    def _extract_joint_positions(self, msg: JointState):
        try:
            return [msg.position[list(msg.name).index(j)] for j in JOINT_NAMES]
        except ValueError as e:
            self.get_logger().error(f"Joint not found: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = DetectAndMergeNode()
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