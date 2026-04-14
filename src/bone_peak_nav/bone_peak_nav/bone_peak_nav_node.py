#!/usr/bin/env python3
"""
Bone Peak Navigation Node — MoveIt plan-and-execute.

Locks the highest-Z point on each tracked bone from the first cloud,
then on command plans an IK-solved motion to hold the drill directly
above that point, with the drill frame aligned parallel to the world
Z axis (tip pointing down).

Commands on /bone_peak_nav/command:
  femur      plan+execute to the locked femur peak
  tibia      plan+execute to the locked tibia peak
  oscillate  alternate femur → tibia → femur …
  stop       cancel the in-flight goal (if any)

Drill alignment logic mirrors bone_scan_trajectory.normal_to_drill_quat
with normal = [0, 0, 1]: drill-Z = -normal = world-down.
"""

import threading
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import String, Bool
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


JOINT_NAMES = [
    "lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4",
    "lbr_A5", "lbr_A6", "lbr_A7",
]
ROBOT_NAME = "lbr"
PLANNING_GROUP = "arm"
BASE_FRAME = f"{ROBOT_NAME}_link_0"
EE_LINK = f"{ROBOT_NAME}_link_ee"
DRILL_FRAME = "drill"


# ──────────────────────────────────────────────────────────────────────
# Async helpers
# ──────────────────────────────────────────────────────────────────────

def wait_for_future(future, timeout_sec=10.0):
    """Poll a future until done or timeout.

    Safe to call from the worker thread because the main
    MultiThreadedExecutor is spinning the node and will fulfil the
    future in the background. Do NOT use rclpy.spin_until_future_complete
    here — that would spin the node from a second thread and race with
    the main executor for callbacks.
    """
    end = time.monotonic() + timeout_sec
    while time.monotonic() < end:
        if future.done():
            return future.result()
        time.sleep(0.02)
    return None


# ──────────────────────────────────────────────────────────────────────
# Math helpers
# ──────────────────────────────────────────────────────────────────────

def pc2_to_xyz(msg):
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3), np.float32)
    ps = msg.point_step
    raw = np.frombuffer(msg.data, np.uint8).reshape(n, ps)
    return np.frombuffer(raw[:, 0:12].tobytes(), np.float32).reshape(n, 3).copy()


def quat_to_mat(q):
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
    R = T[:3, :3]
    tr = np.trace(R)
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


def invert_T(T):
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def tf_to_matrix(tf_msg):
    t = tf_msg.translation
    q = tf_msg.rotation
    T = quat_to_mat([q.x, q.y, q.z, q.w])
    T[0, 3] = t.x
    T[1, 3] = t.y
    T[2, 3] = t.z
    return T


def normal_to_drill_quat(normal):
    """Drill-frame quaternion from a surface normal (mirror of
    bone_scan_trajectory.normal_to_drill_quat).

    Drill-Z axis points into the surface (= -normal), with the other
    axes flipped to match the rest of the stack.
    """
    z_axis = -np.asarray(normal, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([-x_axis, -y_axis, z_axis])
    T = np.eye(4)
    T[:3, :3] = R
    return mat_to_quat(T)


# ──────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────
MODE_IDLE = "idle"
MODE_FEMUR = "femur"
MODE_TIBIA = "tibia"
MODE_OSCILLATE = "oscillate"
TRACKING_MODES = (MODE_FEMUR, MODE_TIBIA, MODE_OSCILLATE)


# ──────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────

class BonePeakNavNode(Node):
    def __init__(self):
        super().__init__("bone_peak_nav_node")
        self.get_logger().info("Bone Peak Nav (plan+execute) starting...")

        self.declare_parameter("standoff_distance", 0.03)
        self.declare_parameter("velocity_scaling", 0.1)
        self.declare_parameter("planning_time", 10.0)
        self.declare_parameter("num_planning_attempts", 5)
        self.declare_parameter("tracking_rate_hz", 1.0)
        self.declare_parameter("movement_threshold_m", 0.001)
        self.declare_parameter("pause_topic", "/pedal_press")

        self.standoff = float(self.get_parameter("standoff_distance").value)
        self.velocity = float(self.get_parameter("velocity_scaling").value)
        self.planning_time = float(self.get_parameter("planning_time").value)
        self.num_attempts = int(self.get_parameter("num_planning_attempts").value)
        self.tracking_period = 1.0 / max(
            float(self.get_parameter("tracking_rate_hz").value), 0.05)
        self.move_threshold = float(self.get_parameter("movement_threshold_m").value)

        # Last successfully executed target per bone (world frame).
        # Used to skip replanning when the peak hasn't moved more than
        # move_threshold between ticks.
        self._last_target = {}

        self.cb_group = ReentrantCallbackGroup()
        self._cloud_lock = threading.Lock()

        # Cloud + peak state
        self._femur_points = None
        self._tibia_points = None
        self._femur_peak_idx = None
        self._tibia_peak_idx = None

        # Joint-state cache (IK seed)
        self._latest_joint_state = None

        # Mode state + worker synchronization.
        # A new command sets _mode and notifies _mode_cv; if a MoveGroup
        # goal is in flight, it is cancelled so the worker can re-plan.
        # _paused gates execution when the pedal/bool topic is True.
        self._mode_cv = threading.Condition()
        self._mode = MODE_IDLE
        self._osc_next = "femur"
        self._active_goal_handle = None
        self._shutdown = False
        self._paused = False

        # Fixed drill-down orientation (drill-Z along world -Z).
        self._drill_quat_global_z = normal_to_drill_quat([0.0, 0.0, 1.0])

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=True,
        )

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
        pause_topic = str(self.get_parameter("pause_topic").value)
        self.create_subscription(
            Bool, pause_topic, self._pause_cb, 10,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            JointState, f"/{ROBOT_NAME}/joint_states", self._joint_state_cb, 10,
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

        # ── MoveIt clients ────────────────────────────────────────────
        self._move_group_client = ActionClient(
            self, MoveGroup, f"/{ROBOT_NAME}/move_action",
            callback_group=self.cb_group,
        )
        self._ik_client = self.create_client(
            GetPositionIK, f"/{ROBOT_NAME}/compute_ik",
            callback_group=self.cb_group,
        )

        # ── Peak-pose publish timer (for RViz visualization) ──────────
        self.create_timer(0.5, self._publish_peaks_timer,
                          callback_group=self.cb_group)

        # Persistent worker — wakes on mode changes and plans+executes.
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="bone_peak_worker",
        )
        self._worker_thread.start()

        self._publish_status("Ready — publish femur/tibia/oscillate to /bone_peak_nav/command.")

    # ── Small helpers ─────────────────────────────────────────────────

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

    # ── Subscriber callbacks ──────────────────────────────────────────

    def _femur_cb(self, msg):
        pts = pc2_to_xyz(msg)
        if len(pts) == 0:
            return
        with self._cloud_lock:
            self._femur_points = pts
            if self._femur_peak_idx is None:
                self._femur_peak_idx = int(np.argmax(pts[:, 2]))
                self.get_logger().info(
                    f"Locked femur peak at index {self._femur_peak_idx}: {pts[self._femur_peak_idx]}"
                )

    def _tibia_cb(self, msg):
        pts = pc2_to_xyz(msg)
        if len(pts) == 0:
            return
        with self._cloud_lock:
            self._tibia_points = pts
            if self._tibia_peak_idx is None:
                self._tibia_peak_idx = int(np.argmax(pts[:, 2]))
                self.get_logger().info(
                    f"Locked tibia peak at index {self._tibia_peak_idx}: {pts[self._tibia_peak_idx]}"
                )

    def _joint_state_cb(self, msg):
        self._latest_joint_state = msg

    def _get_locked_world_peak(self, bone):
        with self._cloud_lock:
            idx = self._femur_peak_idx if bone == "femur" else self._tibia_peak_idx
            pts = self._femur_points if bone == "femur" else self._tibia_points
        if idx is None or pts is None or idx >= len(pts):
            return None
        return pts[idx].copy()

    def _publish_peaks_timer(self):
        for bone, pub in [("femur", self.femur_peak_pub),
                          ("tibia", self.tibia_peak_pub)]:
            peak_w = self._get_locked_world_peak(bone)
            if peak_w is None:
                continue
            target_pos = peak_w + np.array([0.0, 0.0, self.standoff])
            pub.publish(self._make_pose_stamped(target_pos, self._drill_quat_global_z))

    # ── Command dispatch ──────────────────────────────────────────────

    def _command_cb(self, msg):
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"[cmd received] '{cmd}'")

        if cmd == "stop":
            with self._mode_cv:
                self._mode = MODE_IDLE
                self._mode_cv.notify_all()
            self._cancel_active_goal()
            self._publish_status("Stopped")
            return

        if cmd not in TRACKING_MODES:
            self._publish_status(f"Unknown command: {cmd}")
            return

        # Set mode + kick worker. Clear last-target so the first tick
        # of a freshly-issued command always plans, then subsequent
        # ticks gate on the movement threshold. If a goal is in flight,
        # cancel it so the worker picks up the new mode immediately.
        with self._mode_cv:
            self._mode = cmd
            self._last_target.clear()
            if cmd == MODE_OSCILLATE:
                self._osc_next = "femur"
            self._mode_cv.notify_all()
        self._cancel_active_goal()

    def _cancel_active_goal(self):
        gh = self._active_goal_handle
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception:
                pass

    def _pause_cb(self, msg):
        """Bool topic: True pauses the tracking cycle, False resumes.

        Paused = worker is gated before planning the next motion; any
        goal already in flight is cancelled so the arm halts immediately.
        Mode is preserved so it resumes the same bone on release.
        """
        new_paused = bool(msg.data)
        if new_paused == self._paused:
            return
        with self._mode_cv:
            self._paused = new_paused
            self._mode_cv.notify_all()
        if new_paused:
            self._cancel_active_goal()
            self._publish_status("Paused (pedal).")
        else:
            self._publish_status("Resumed.")

    # ── Persistent worker loop ────────────────────────────────────────

    def _worker_loop(self):
        """Continuous tracking loop, responsive to mode changes.

        Idle:       wait on _mode_cv.
        Femur/Tibia: every tracking_period, replan iff the locked peak
                    moved more than move_threshold since the last
                    executed target.
        Oscillate:  same threshold gating, alternating bones per
                    successful motion.
        A new command cancels the in-flight goal and loops back to
        pick up the new _mode on the next iteration.
        """
        while not self._shutdown and rclpy.ok():
            with self._mode_cv:
                # Block while idle OR paused. Both conditions release
                # on _mode_cv.notify_all() (command, pedal, shutdown).
                while ((self._mode == MODE_IDLE or self._paused)
                       and not self._shutdown and rclpy.ok()):
                    self._mode_cv.wait(timeout=1.0)
                mode = self._mode
                bone = self._osc_next if mode == MODE_OSCILLATE else mode

            if self._shutdown or not rclpy.ok():
                return

            peak_w = self._get_locked_world_peak(bone)
            if peak_w is None:
                self._sleep_interruptible(self.tracking_period, mode)
                continue

            last = self._last_target.get(bone)
            if last is not None and np.linalg.norm(peak_w - last) < self.move_threshold:
                # Peak hasn't moved enough — skip this tick silently.
                self._sleep_interruptible(self.tracking_period, mode)
                continue

            ok = self._plan_and_execute(bone)
            if ok:
                self._last_target[bone] = peak_w.copy()

            with self._mode_cv:
                if self._mode != mode:
                    # Overridden during execution — re-enter the loop.
                    continue
                if mode == MODE_OSCILLATE and ok:
                    self._osc_next = "tibia" if bone == "femur" else "femur"

            self._sleep_interruptible(self.tracking_period, mode)

    def _sleep_interruptible(self, seconds, current_mode):
        """Sleep up to `seconds`, waking early on mode change or pause."""
        end = time.monotonic() + seconds
        with self._mode_cv:
            while (not self._shutdown
                   and self._mode == current_mode
                   and not self._paused
                   and time.monotonic() < end):
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                self._mode_cv.wait(timeout=remaining)

    # ── Core: plan + execute one target ──────────────────────────────

    def _plan_and_execute(self, bone):
        peak_w = self._get_locked_world_peak(bone)
        if peak_w is None:
            self._publish_status(f"No locked peak for {bone} yet.")
            return False

        drill_pos = peak_w + np.array([0.0, 0.0, self.standoff])
        drill_quat = self._drill_quat_global_z

        # Convert drill-frame target → lbr_link_ee target.
        ee_pose = self._drill_to_ee(drill_pos, drill_quat)
        if ee_pose is None:
            self._publish_status("TF drill→ee lookup failed")
            return False
        ee_pos, ee_quat = ee_pose

        self.ee_target_pub.publish(self._make_pose_stamped(ee_pos, ee_quat))
        self._publish_status(f"[{bone}] solving IK…")

        joint_targets = self._solve_ik(ee_pos, ee_quat)
        if joint_targets is None:
            self._publish_status(f"[{bone}] IK failed")
            return False

        self._publish_status(f"[{bone}] planning + executing…")
        ok = self._send_joint_goal(joint_targets)
        if ok:
            self._publish_status(f"[{bone}] done.")
        else:
            self._publish_status(f"[{bone}] motion failed (or superseded)")
        return ok

    def _drill_to_ee(self, drill_pos, drill_quat):
        """Turn a desired drill pose in lbr_link_0 into the ee pose."""
        try:
            tf_ee_drill = self.tf_buffer.lookup_transform(
                EE_LINK, DRILL_FRAME, rclpy.time.Time(),
                rclpy.duration.Duration(seconds=2.0),
            )
        except Exception as e:
            self.get_logger().error(f"TF {EE_LINK} ← {DRILL_FRAME} failed: {e}")
            return None

        T_ee_drill = tf_to_matrix(tf_ee_drill.transform)
        T_base_drill = quat_to_mat(drill_quat)
        T_base_drill[:3, 3] = drill_pos
        T_base_ee = T_base_drill @ invert_T(T_ee_drill)
        return T_base_ee[:3, 3], mat_to_quat(T_base_ee)

    def _solve_ik(self, ee_pos, ee_quat):
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("compute_ik service unavailable")
            return None
        if self._latest_joint_state is None:
            self.get_logger().error("no joint state received yet — cannot seed IK")
            return None

        joint_map = dict(zip(
            self._latest_joint_state.name,
            self._latest_joint_state.position,
        ))
        seed = JointState()
        seed.name = JOINT_NAMES
        seed.position = [float(joint_map.get(j, 0.0)) for j in JOINT_NAMES]

        req = GetPositionIK.Request()
        req.ik_request.group_name = PLANNING_GROUP
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 5
        req.ik_request.robot_state.joint_state = seed
        req.ik_request.pose_stamped = self._make_pose_stamped(ee_pos, ee_quat)

        fut = self._ik_client.call_async(req)
        result = wait_for_future(fut, timeout_sec=10.0)
        if result is None or result.error_code.val != 1:
            ec = result.error_code.val if result else "timeout"
            self.get_logger().error(f"IK failed (error {ec})")
            return None

        solved = dict(zip(
            result.solution.joint_state.name,
            result.solution.joint_state.position,
        ))
        return [float(solved.get(j, joint_map.get(j, 0.0))) for j in JOINT_NAMES]

    def _send_joint_goal(self, joint_targets):
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("move_action not available")
            return False

        constraints = Constraints()
        for jname, val in zip(JOINT_NAMES, joint_targets):
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        plan_req = MotionPlanRequest()
        plan_req.group_name = PLANNING_GROUP
        plan_req.num_planning_attempts = self.num_attempts
        plan_req.allowed_planning_time = self.planning_time
        plan_req.max_velocity_scaling_factor = self.velocity
        plan_req.max_acceleration_scaling_factor = self.velocity
        plan_req.goal_constraints.append(constraints)

        goal = MoveGroup.Goal()
        goal.request = plan_req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        send_fut = self._move_group_client.send_goal_async(goal)
        gh = wait_for_future(send_fut, timeout_sec=10.0)
        if gh is None or not gh.accepted:
            self.get_logger().error("MoveGroup rejected goal")
            return False

        self._active_goal_handle = gh
        try:
            res_fut = gh.get_result_async()
            res = wait_for_future(res_fut, timeout_sec=120.0)
            if res is None:
                self.get_logger().error("MoveGroup result timeout")
                return False
            return res.result.error_code.val == 1
        finally:
            self._active_goal_handle = None


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
        node._shutdown = True
        with node._mode_cv:
            node._mode_cv.notify_all()
        node._cancel_active_goal()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
