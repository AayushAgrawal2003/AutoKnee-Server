#!/usr/bin/env python3
"""
Bone Peak Navigation Node (self-contained)

Subscribes to tracked bone point clouds, finds the highest-Z point on each,
and executes robot motion directly via MoveIt IK + MoveGroup.

The computed marker pose represents the desired drill frame pose.  Since
"drill" is a static-TF child of lbr_link_ee (not in the URDF), we convert
the drill target to an lbr_link_ee target before sending to IK.

Modes (via /bone_peak_nav/command):
  femur     - go to the highest Z point on the femur
  tibia     - go to the highest Z point on the tibia
  oscillate - alternate between femur and tibia peaks
  stop      - cancel oscillation

Usage:
  ros2 run bone_peak_nav bone_peak_nav_node
"""

import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    JointConstraint,
)
from moveit_msgs.srv import GetPositionIK
from tf2_ros import Buffer, TransformListener


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
EE_LINK = f"{ROBOT_NAME}_link_ee"
PLANNING_TIME = 10.0


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def pc2_to_xyz(msg):
    """Extract Nx3 float32 XYZ array from a PointCloud2 message."""
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), np.float32)
    ps = msg.point_step
    raw = np.frombuffer(msg.data, np.uint8).reshape(n, ps)
    return np.frombuffer(raw[:, 0:12].tobytes(), np.float32).reshape(n, 3).copy()


def quat_to_mat(q):
    """Quaternion [x, y, z, w] → 4x4 homogeneous matrix."""
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


def mat_to_quat(T):
    """4x4 homogeneous matrix → quaternion [x, y, z, w]."""
    R = T[:3, :3]
    trace = np.trace(R)
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
    return np.array([x, y, z, w])


def invert_T(T):
    """Invert a 4x4 homogeneous transform."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def tf_to_matrix(tf_msg):
    """Convert a TransformStamped.transform to a 4x4 matrix."""
    t = tf_msg.translation
    q = tf_msg.rotation
    T = quat_to_mat([q.x, q.y, q.z, q.w])
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


def compute_peak_pose(points, normal_radius=0.01, standoff=0.03):
    """Find highest-Z point, compute approach pose with standoff.

    Returns (position, quaternion) for the desired drill frame pose.
    Drill z-axis points into the surface, rotated 180 deg about z.
    Returns (None, None) if the cloud is empty.
    """
    if len(points) == 0:
        return None, None

    idx = int(np.argmax(points[:, 2]))
    peak = points[idx].copy()

    dists = np.linalg.norm(points - peak, axis=1)
    nearby = points[dists < normal_radius]

    if len(nearby) < 3:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        centered = nearby - nearby.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normal = Vt[-1]
        if normal[2] < 0:
            normal = -normal

    position = peak + standoff * normal

    # Drill z-axis = -normal (into bone), rotated 180 deg about z
    z_axis = -normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([-x_axis, -y_axis, z_axis])
    T = np.eye(4)
    T[:3, :3] = R
    quat = mat_to_quat(T)

    return position, quat


# ──────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────

class BonePeakNavNode(Node):
    def __init__(self):
        super().__init__("bone_peak_nav_node")
        self.get_logger().info("Bone Peak Nav starting...")

        self.declare_parameter("standoff_distance", 0)
        self.declare_parameter("normal_radius", 0.01)
        self.declare_parameter("velocity_scaling", 0.5  )

        self.standoff = self.get_parameter("standoff_distance").value
        self.normal_radius = self.get_parameter("normal_radius").value
        self.velocity = self.get_parameter("velocity_scaling").value

        self.cb_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # State
        self._femur_points = None
        self._tibia_points = None
        self._mode = "idle"           # idle | femur | tibia | oscillate
        self._osc_next = "femur"
        self._motion_in_progress = False
        self.latest_joint_state = None

        # TF — needed to look up the static lbr_link_ee → drill transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(
            PointCloud2, "/tracked/femur", self._femur_cb, 10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            PointCloud2, "/tracked/tibia", self._tibia_cb, 10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            String, "/bone_peak_nav/command", self._command_cb, 10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            JointState, "/lbr/joint_states", self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )

        # ── Publishers ────────────────────────────────────────────────
        latch_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.status_pub = self.create_publisher(String, "/bone_peak_nav/status", 10)
        self.femur_peak_pub = self.create_publisher(
            PoseStamped, "/bone_peak_nav/femur_peak", latch_qos,
        )
        self.tibia_peak_pub = self.create_publisher(
            PoseStamped, "/bone_peak_nav/tibia_peak", latch_qos,
        )
        self.ee_target_pub = self.create_publisher(
            PoseStamped, "/bone_peak_nav/ee_target", latch_qos,
        )

        # ── MoveIt clients ───────────────────────────────────────────
        self.move_group_client = ActionClient(
            self, MoveGroup, "/lbr/move_action",
            callback_group=self.cb_group,
        )
        self._ik_client = self.create_client(
            GetPositionIK, "/lbr/compute_ik",
            callback_group=self.cb_group,
        )

        self._publish_status("Ready. Waiting for tracked bone clouds...")

    # ── Helpers ───────────────────────────────────────────────────────

    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f"[status] {text}")

    def _make_pose_stamped(self, pos, quat):
        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(pos[0])
        ps.pose.position.y = float(pos[1])
        ps.pose.position.z = float(pos[2])
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        return ps

    # ── Callbacks ─────────────────────────────────────────────────────

    def _femur_cb(self, msg):
        pts = pc2_to_xyz(msg)
        if len(pts) > 0:
            with self._lock:
                self._femur_points = pts

    def _tibia_cb(self, msg):
        pts = pc2_to_xyz(msg)
        if len(pts) > 0:
            with self._lock:
                self._tibia_points = pts

    def _joint_state_cb(self, msg):
        self.latest_joint_state = msg

    def _command_cb(self, msg):
        cmd = msg.data.strip().lower()

        if cmd == "femur":
            self._mode = "idle"
            self._motion_in_progress = False
            self._send_to_bone("femur")

        elif cmd == "tibia":
            self._mode = "idle"
            self._motion_in_progress = False
            self._send_to_bone("tibia")

        elif cmd == "oscillate":
            self._mode = "oscillate"
            self._osc_next = "femur"
            self._motion_in_progress = False
            self._publish_status("Oscillation started — heading to femur peak")
            self._send_to_bone("femur")

        elif cmd == "stop":
            self._mode = "idle"
            self._motion_in_progress = False
            self._publish_status("Stopped")

        else:
            self._publish_status(f"Unknown command: {cmd}")

    # ── Motion ────────────────────────────────────────────────────────

    def _send_to_bone(self, bone):
        with self._lock:
            points = self._femur_points if bone == "femur" else self._tibia_points

        if points is None:
            self._publish_status(f"No {bone} cloud received yet!")
            return

        drill_pos, drill_quat = compute_peak_pose(
            points, self.normal_radius, self.standoff,
        )
        if drill_pos is None:
            self._publish_status(f"{bone} cloud is empty!")
            return

        # Publish drill-frame target for visualisation (the marker)
        drill_ps = self._make_pose_stamped(drill_pos, drill_quat)
        if bone == "femur":
            self.femur_peak_pub.publish(drill_ps)
        else:
            self.tibia_peak_pub.publish(drill_ps)

        self._publish_status(f"Planning {bone} peak motion...")
        self._execute_motion(bone, drill_pos, drill_quat)

    def _execute_motion(self, bone, drill_pos, drill_quat):
        """Convert desired drill pose → lbr_link_ee pose, solve IK, execute."""

        # ── drill → lbr_link_ee conversion ────────────────────────────
        # "drill" is a static TF child of lbr_link_ee (not in the URDF).
        # MoveIt only knows about lbr_link_ee, so we must compute where
        # lbr_link_ee needs to be for drill to land at the target.
        #
        # Relationship:  T_base_drill = T_base_ee * T_ee_drill
        # We want:       T_base_ee    = T_base_drill * inv(T_ee_drill)
        #
        # lookup_transform(target="lbr_link_ee", source="drill")
        #   returns T that maps p_drill → p_ee, i.e. T_ee_drill  (what we need to invert... wait no)
        #
        # Actually: lookup_transform(target, source) gives the transform
        # that takes coordinates FROM source INTO target.
        # So lookup_transform("lbr_link_ee", "drill") = T_ee←drill
        #   meaning  p_ee = T * p_drill
        #   this IS  T_ee_drill
        #
        # T_base_ee = T_base_drill * inv(T_ee_drill)

        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                EE_LINK, "drill",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
        except Exception as e:
            self._publish_status(f"TF {EE_LINK}←drill lookup failed: {e}")
            return

        T_ee_drill = tf_to_matrix(tf_stamped.transform)

        # Build T_base_drill (the desired drill pose as 4x4)
        T_base_drill = quat_to_mat(drill_quat)
        T_base_drill[0, 3] = drill_pos[0]
        T_base_drill[1, 3] = drill_pos[1]
        T_base_drill[2, 3] = drill_pos[2]

        # Compute T_base_ee = T_base_drill * inv(T_ee_drill)
        T_base_ee = T_base_drill @ invert_T(T_ee_drill)

        ee_pos = T_base_ee[:3, 3]
        ee_quat = mat_to_quat(T_base_ee)

        self.get_logger().info(
            f"Drill target: pos={drill_pos}, quat={drill_quat}\n"
            f"EE target:    pos={ee_pos}, quat={ee_quat}"
        )

        # Publish EE target for debugging visualisation
        self.ee_target_pub.publish(self._make_pose_stamped(ee_pos, ee_quat))

        # ── IK (solve for lbr_link_ee) ────────────────────────────────
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self._publish_status("IK service not available!")
            return

        if self.latest_joint_state is None:
            self._publish_status("No joint state received yet!")
            return

        joint_map = dict(zip(
            self.latest_joint_state.name,
            self.latest_joint_state.position,
        ))

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = PLANNING_GROUP
        ik_req.ik_request.avoid_collisions = False
        ik_req.ik_request.timeout.sec = 5
        ik_req.ik_request.pose_stamped = self._make_pose_stamped(ee_pos, ee_quat)

        seed_js = JointState()
        seed_js.name = JOINT_NAMES
        seed_js.position = [float(joint_map.get(j, 0.0)) for j in JOINT_NAMES]
        ik_req.ik_request.robot_state.joint_state = seed_js

        self._publish_status("Computing IK...")

        ik_future = self._ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, ik_future, timeout_sec=10.0)
        ik_result = ik_future.result()

        if ik_result is None or ik_result.error_code.val != 1:
            ec = ik_result.error_code.val if ik_result else "timeout"
            error_names = {
                -31: "NO_IK_SOLUTION (pose may be unreachable)",
                -12: "INVALID_GOAL_CONSTRAINTS",
                -10: "START_STATE_IN_COLLISION",
                -4: "PLANNING_FAILED",
            }
            self._publish_status(f"IK failed: {error_names.get(ec, ec)}")
            self._on_motion_done(success=False)
            return

        ik_joint_map = dict(zip(
            ik_result.solution.joint_state.name,
            ik_result.solution.joint_state.position,
        ))

        # ── MoveGroup ─────────────────────────────────────────────────
        self._publish_status("IK solved, executing motion...")

        goal_constraints = Constraints()
        for jname in JOINT_NAMES:
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(ik_joint_map.get(jname, joint_map.get(jname, 0.0)))
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = PLANNING_TIME
        req.max_velocity_scaling_factor = self.velocity
        req.max_acceleration_scaling_factor = self.velocity
        req.goal_constraints.append(goal_constraints)

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self._publish_status("MoveGroup action server not available!")
            self._on_motion_done(success=False)
            return

        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        goal_handle = future.result()

        if not goal_handle or not goal_handle.accepted:
            self._publish_status("Goal rejected by MoveGroup!")
            self._on_motion_done(success=False)
            return

        self._publish_status("Executing motion...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        result = result_future.result()

        if result is None or result.result.error_code.val != 1:
            ec = result.result.error_code.val if result else "timeout"
            self._publish_status(f"Motion failed (error={ec})")
            self._on_motion_done(success=False)
            return

        self._publish_status("Motion complete!")
        self._on_motion_done(success=True)

    def _on_motion_done(self, success):
        """Handle oscillation continuation after a motion finishes."""
        if self._mode != "oscillate":
            self._motion_in_progress = False
            return

        if not success:
            self._mode = "idle"
            self._motion_in_progress = False
            self._publish_status("Oscillation stopped — motion failed")
            return

        self._osc_next = "tibia" if self._osc_next == "femur" else "femur"
        self._publish_status(f"Heading to {self._osc_next} peak")
        self._send_to_bone(self._osc_next)


# ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BonePeakNavNode()

    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor()
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
