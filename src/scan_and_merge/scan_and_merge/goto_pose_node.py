#!/usr/bin/env python3
"""
Go-To-Pose Node for KUKA LBR Med 7

Subscribes to a PoseStamped topic. As soon as a pose arrives, the node
starts continuously tracking it with the same plan-and-execute worker
pattern as bone_peak_nav: every tracking_period it replans iff the
desired drill pose moved more than movement_threshold_m since the last
executed target.

If a new pose arrives while a motion is in flight, the current goal is
cancelled ("new goal received, still tracking old" is published as a
status), and the worker picks up the new target on its next tick.

Topics:
  Subscribed:
    <input_topic> (geometry_msgs/PoseStamped) - goal pose for the drill frame
    /goto_pose/command (std_msgs/String) - "stop" to halt tracking
    /lbr/joint_states (sensor_msgs/JointState) - IK seed

  Published:
    /goto_pose/target (geometry_msgs/PoseStamped) - latest received goal
    /goto_pose/ee_target (geometry_msgs/PoseStamped) - computed EE target
    /goto_pose/status (std_msgs/String) - status messages
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import JointState
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

import numpy as np

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
DRILL_FRAME = "drill"


def wait_for_future(future, timeout_sec=10.0):
    """Poll a future until done or timeout.

    Safe to call from the worker thread: the main MultiThreadedExecutor
    is spinning the node and fulfils futures in the background.
    """
    end = time.monotonic() + timeout_sec
    while time.monotonic() < end:
        if future.done():
            return future.result()
        time.sleep(0.02)
    return None


def quat_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def quat_slerp(q0, q1, t):
    """Spherical linear interpolation between two unit quaternions [x,y,z,w]."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    # Take the shorter arc.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    # Nearly colinear → linear interp + renormalize.
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def rotation_matrix_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
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


MODE_IDLE = "idle"
MODE_TRACK = "track"


class GoToPoseNode(Node):
    def __init__(self):
        super().__init__("goto_pose_node")
        self.get_logger().info("GoToPose Node starting...")

        # Params — match bone_peak_nav so both nodes share defaults.
        self.declare_parameter("input_topic", "/surgical_plan/probe_pose")
        self.declare_parameter("velocity_scaling", 0.1)
        self.declare_parameter("planning_time", 10.0)
        self.declare_parameter("num_planning_attempts", 5)
        self.declare_parameter("tracking_rate_hz", 1.0)
        self.declare_parameter("movement_threshold_m", 0.001)
        # Option A: EMA on translation + SLERP on quaternion. Lower
        # alpha = heavier smoothing (more lag). 1.0 disables the filter.
        self.declare_parameter("pose_filter_alpha", 0.2)
        # Option F: after a motion completes, measure the drill TF and
        # re-plan with an offset if the tip error exceeds this. Zero
        # disables closed-loop correction.
        self.declare_parameter("tip_error_correction_m", 0.0005)
        self.declare_parameter("tip_error_correction_max_iters", 2)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.velocity = float(self.get_parameter("velocity_scaling").value)
        self.planning_time = float(self.get_parameter("planning_time").value)
        self.num_attempts = int(self.get_parameter("num_planning_attempts").value)
        self.tracking_period = 1.0 / max(
            float(self.get_parameter("tracking_rate_hz").value), 0.05)
        self.move_threshold = float(self.get_parameter("movement_threshold_m").value)
        self.pose_filter_alpha = float(self.get_parameter("pose_filter_alpha").value)
        self.tip_err_thresh = float(self.get_parameter("tip_error_correction_m").value)
        self.tip_err_max_iters = int(self.get_parameter("tip_error_correction_max_iters").value)

        self.cb_group = ReentrantCallbackGroup()

        # Pose + joint state caches.
        # _active_goal_pose is what the worker is currently tracking.
        # _pending_goal_pose is the latest pose received while the
        # worker was already tracking — it is promoted to active only
        # when `go` is published on /goto_pose/command.
        self._active_goal_pose = None   # geometry_msgs/PoseStamped
        self._pending_goal_pose = None
        self._latest_joint_state = None
        self._pose_lock = threading.Lock()

        # Last successfully executed drill-frame target (world/base position).
        # None means "never executed" so the first tick always plans.
        self._last_target = None

        # Option A: EMA-filtered drill pose in BASE_FRAME. Reset on
        # every new active goal so the filter starts clean.
        self._filtered_pos = None   # np.array shape (3,)
        self._filtered_quat = None  # np.array shape (4,) [x,y,z,w]

        # Option D: cached joint vector from the last successful IK
        # solve. Used as the seed on subsequent IK calls so the arm
        # does not jump configurations between ticks.
        self._last_solved_joints = None

        # Mode state + worker synchronization. Mirrors bone_peak_nav:
        # a new pose sets _mode=track and notifies _mode_cv; if a goal
        # is in flight, it is cancelled so the worker re-plans.
        self._mode_cv = threading.Condition()
        self._mode = MODE_IDLE
        self._active_goal_handle = None
        self._shutdown = False

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=True,
        )

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped, self.input_topic, self._goal_pose_cb, 10,
            callback_group=self.cb_group,
        )
        self.command_sub = self.create_subscription(
            String, "/goto_pose/command", self._command_cb, 10,
            callback_group=self.cb_group,
        )
        self.joint_sub = self.create_subscription(
            JointState, f"/{ROBOT_NAME}/joint_states", self._joint_state_cb, 10,
            callback_group=self.cb_group,
        )

        # Publishers
        latch_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.target_pub = self.create_publisher(PoseStamped, "/goto_pose/target", latch_qos)
        self.ee_target_pub = self.create_publisher(PoseStamped, "/goto_pose/ee_target", latch_qos)
        self.status_pub = self.create_publisher(String, "/goto_pose/status", 10)

        # MoveIt2 action client
        self.move_group_client = ActionClient(
            self, MoveGroup, f"/{ROBOT_NAME}/move_action",
            callback_group=self.cb_group,
        )

        # IK service client
        self._ik_client = self.create_client(
            GetPositionIK, f"/{ROBOT_NAME}/compute_ik",
            callback_group=self.cb_group,
        )

        # Persistent worker — wakes on mode changes and plans+executes.
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="goto_pose_worker",
        )
        self._worker_thread.start()

        self._publish_status("Ready. Waiting for goal pose...")
        self.get_logger().info(f"Subscribed to {self.input_topic} for goal poses")
        self.get_logger().info("Publish 'stop' to /goto_pose/command to halt tracking")

    # ── Small helpers ─────────────────────────────────────────────────

    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f"[status] {text}")

    # ── Subscriber callbacks ──────────────────────────────────────────

    def _goal_pose_cb(self, msg: PoseStamped):
        # Always republish for visualization.
        self.target_pub.publish(msg)

        pos = msg.pose.position

        with self._pose_lock:
            already_tracking = self._active_goal_pose is not None
            if already_tracking:
                # Buffer as pending — the worker keeps tracking the old
                # goal (which still follows its source frame via TF).
                # A `go` command on /goto_pose/command promotes pending
                # → active.
                self._pending_goal_pose = msg
            else:
                self._active_goal_pose = msg

        if already_tracking:
            self._publish_status(
                f"New goal received [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}] — "
                f"still tracking old. Publish 'go' to /goto_pose/command to switch."
            )
            return

        # First goal — kick the worker into tracking mode.
        with self._mode_cv:
            self._mode = MODE_TRACK
            self._last_target = None
            self._filtered_pos = None
            self._filtered_quat = None
            self._last_solved_joints = None
            self._mode_cv.notify_all()
        self._publish_status(
            f"Tracking new goal [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}] in {msg.header.frame_id}"
        )

    def _joint_state_cb(self, msg: JointState):
        self._latest_joint_state = msg

    def _command_cb(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == "stop":
            with self._mode_cv:
                self._mode = MODE_IDLE
                self._last_target = None
                self._filtered_pos = None
                self._filtered_quat = None
                self._last_solved_joints = None
                self._mode_cv.notify_all()
            with self._pose_lock:
                self._active_goal_pose = None
                self._pending_goal_pose = None
            self._cancel_active_goal()
            self._publish_status("Stopped")
            return

        if cmd == "go":
            with self._pose_lock:
                if self._pending_goal_pose is None:
                    if self._active_goal_pose is None:
                        self._publish_status("No goal pose received yet!")
                        return
                    self._publish_status("No pending goal — already tracking the active one.")
                    return
                # Promote pending → active.
                self._active_goal_pose = self._pending_goal_pose
                self._pending_goal_pose = None
                promoted = self._active_goal_pose

            # Force an immediate replan on the new active target and
            # reset the EMA filter so it starts clean on the new pose.
            with self._mode_cv:
                self._mode = MODE_TRACK
                self._last_target = None
                self._filtered_pos = None
                self._filtered_quat = None
                self._mode_cv.notify_all()
            self._cancel_active_goal()

            pos = promoted.pose.position
            self._publish_status(
                f"Switched to new goal [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]"
            )
            return

        self._publish_status(f"Unknown command: {cmd}")

    def _cancel_active_goal(self):
        gh = self._active_goal_handle
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception:
                pass

    # ── Persistent worker loop ────────────────────────────────────────

    def _worker_loop(self):
        """Continuous tracking loop, responsive to new poses.

        Idle:  wait on _mode_cv.
        Track: every tracking_period, replan iff the latest pose moved
               more than move_threshold since the last executed target.
        A new pose cancels the in-flight goal; _last_target is cleared
        so the next tick always plans.
        """
        while not self._shutdown and rclpy.ok():
            with self._mode_cv:
                while (self._mode == MODE_IDLE
                       and not self._shutdown and rclpy.ok()):
                    self._mode_cv.wait(timeout=1.0)
                mode = self._mode

            if self._shutdown or not rclpy.ok():
                return

            with self._pose_lock:
                goal_pose = self._active_goal_pose
            if goal_pose is None:
                self._sleep_interruptible(self.tracking_period, mode)
                continue

            # Resolve the desired drill pose in the base frame.
            drill_base = self._goal_to_base(goal_pose)
            if drill_base is None:
                self._sleep_interruptible(self.tracking_period, mode)
                continue
            raw_pos, raw_quat = drill_base

            # Option A: EMA on translation + SLERP on quaternion. The
            # filter state is reset whenever a new active goal is
            # installed, so the first tick after a switch seeds from
            # the raw pose.
            alpha = max(0.0, min(1.0, self.pose_filter_alpha))
            if self._filtered_pos is None or self._filtered_quat is None:
                self._filtered_pos = raw_pos.copy()
                self._filtered_quat = raw_quat.copy()
            else:
                self._filtered_pos = (
                    alpha * raw_pos + (1.0 - alpha) * self._filtered_pos
                )
                self._filtered_quat = quat_slerp(
                    self._filtered_quat, raw_quat, alpha
                )
            drill_pos = self._filtered_pos.copy()
            drill_quat = self._filtered_quat.copy()

            # Gate on movement threshold (translation only — matches
            # bone_peak_nav semantics and is robust to quaternion noise).
            if (self._last_target is not None
                    and np.linalg.norm(drill_pos - self._last_target) < self.move_threshold):
                self._sleep_interruptible(self.tracking_period, mode)
                continue

            ok = self._plan_and_execute(drill_pos, drill_quat)
            if ok:
                self._last_target = drill_pos.copy()

            with self._mode_cv:
                if self._mode != mode:
                    # Overridden during execution — re-enter the loop.
                    continue

            self._sleep_interruptible(self.tracking_period, mode)

    def _sleep_interruptible(self, seconds, current_mode):
        """Sleep up to `seconds`, waking early on mode change."""
        end = time.monotonic() + seconds
        with self._mode_cv:
            while (not self._shutdown
                   and self._mode == current_mode
                   and time.monotonic() < end):
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                self._mode_cv.wait(timeout=remaining)

    # ── Goal → base-frame drill pose ──────────────────────────────────

    def _goal_to_base(self, goal_pose: PoseStamped):
        """Return (drill_pos, drill_quat) in BASE_FRAME for a goal pose,
        or None on TF failure."""
        goal_pos = np.array([
            goal_pose.pose.position.x,
            goal_pose.pose.position.y,
            goal_pose.pose.position.z,
        ])
        goal_quat = np.array([
            goal_pose.pose.orientation.x,
            goal_pose.pose.orientation.y,
            goal_pose.pose.orientation.z,
            goal_pose.pose.orientation.w,
        ])

        goal_frame = goal_pose.header.frame_id
        if goal_frame and goal_frame != BASE_FRAME:
            try:
                tf = self.tf_buffer.lookup_transform(
                    BASE_FRAME, goal_frame,
                    rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)
                )
            except Exception as e:
                self._publish_status(f"TF lookup failed: {e}")
                return None
            t = tf.transform.translation
            q = tf.transform.rotation
            R = quat_to_rotation_matrix([q.x, q.y, q.z, q.w])
            goal_pos = R @ goal_pos + np.array([t.x, t.y, t.z])
            R_goal = quat_to_rotation_matrix(goal_quat)
            goal_quat = rotation_matrix_to_quat(R @ R_goal)

        return goal_pos, goal_quat

    # ── Core: plan + execute one target ───────────────────────────────

    def _plan_and_execute(self, drill_pos, drill_quat):
        """Plan and execute a motion that places the drill frame at
        drill_pos / drill_quat in BASE_FRAME.

        Option F: after each successful motion, measure the actual
        drill TF and, if the residual exceeds tip_error_correction_m,
        re-plan with the desired target offset by the measured error.
        Runs up to tip_error_correction_max_iters correction passes.
        """
        target_drill_pos = np.asarray(drill_pos, dtype=float)
        # The drill-frame target pos we'll feed to IK. Starts equal to
        # the desired pos, then gets offset by measured errors.
        commanded_pos = target_drill_pos.copy()
        last_ok = False

        max_iters = max(1, 1 + self.tip_err_max_iters)
        for iteration in range(max_iters):
            ee_pose = self._drill_to_ee(commanded_pos, drill_quat)
            if ee_pose is None:
                return False
            ee_pos, ee_quat = ee_pose

            self.ee_target_pub.publish(self._make_pose_stamped(ee_pos, ee_quat))

            joint_targets = self._solve_ik(ee_pos, ee_quat)
            if joint_targets is None:
                self._publish_status("IK failed")
                return last_ok

            if iteration == 0:
                self._publish_status("Planning + executing…")
            else:
                self._publish_status(
                    f"Correction pass {iteration}: replanning to close residual"
                )

            ok = self._send_joint_goal(joint_targets)
            last_ok = ok
            if not ok:
                self._publish_status("Motion failed (or superseded)")
                return False

            # Residual check: actual drill TF vs desired drill_pos.
            if self.tip_err_thresh <= 0.0 or iteration >= self.tip_err_max_iters:
                break
            actual = self._lookup_drill_world_pos()
            if actual is None:
                break
            err_vec = target_drill_pos - actual
            err = float(np.linalg.norm(err_vec))
            self.get_logger().info(
                f"Tip residual after pass {iteration}: {err*1000:.3f} mm"
            )
            if err < self.tip_err_thresh:
                break
            # Push the commanded drill target in the direction of the
            # measured error so the next plan over-reaches by exactly
            # the residual. Bounded accumulation — the IK seed cache
            # (Option D) keeps us near the same config.
            commanded_pos = commanded_pos + err_vec

        self._publish_status("Motion complete.")
        return last_ok

    def _drill_to_ee(self, drill_pos, drill_quat):
        """Turn a desired drill pose in BASE_FRAME into the matching
        lbr_link_ee pose. Returns (ee_pos, ee_quat) or None on TF fail."""
        try:
            tf_drill_ee = self.tf_buffer.lookup_transform(
                DRILL_FRAME, EE_LINK,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)
            )
        except Exception as e:
            self._publish_status(f"TF lookup drill→{EE_LINK} failed: {e}")
            return None

        dt = tf_drill_ee.transform.translation
        dq = tf_drill_ee.transform.rotation
        R_drill_ee = quat_to_rotation_matrix([dq.x, dq.y, dq.z, dq.w])
        R_goal_mat = quat_to_rotation_matrix(drill_quat)

        ee_pos = R_goal_mat @ np.array([dt.x, dt.y, dt.z]) + drill_pos
        ee_quat = rotation_matrix_to_quat(R_goal_mat @ R_drill_ee)
        return ee_pos, ee_quat

    def _lookup_drill_world_pos(self):
        """Return the actual drill origin in BASE_FRAME, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, DRILL_FRAME, rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2),
            )
        except Exception:
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z])

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
        # Option D: prefer the last successfully solved joints as the
        # IK seed — this keeps the arm in the same configuration
        # across ticks and prevents shoulder/elbow flips. Fall back to
        # the live joint state on the very first solve.
        seed = JointState()
        seed.name = JOINT_NAMES
        if self._last_solved_joints is not None:
            seed.position = [float(v) for v in self._last_solved_joints]
        else:
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
        joints = [float(solved.get(j, joint_map.get(j, 0.0))) for j in JOINT_NAMES]
        # Cache for the next IK seed (Option D).
        self._last_solved_joints = joints[:]
        return joints

    def _send_joint_goal(self, joint_targets):
        if not self.move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("move_action not available")
            return False

        # Option C: tight joint tolerance (~0.057°) so MoveIt drives the
        # arm to the IK solution rather than a 0.57° neighborhood.
        constraints = Constraints()
        for jname, val in zip(JOINT_NAMES, joint_targets):
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(val)
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
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

        send_fut = self.move_group_client.send_goal_async(goal)
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


def main(args=None):
    rclpy.init(args=args)
    node = GoToPoseNode()

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
