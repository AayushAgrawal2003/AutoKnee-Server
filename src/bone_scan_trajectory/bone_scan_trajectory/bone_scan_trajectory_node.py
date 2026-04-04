#!/usr/bin/env python3
"""
Bone Scan Trajectory Node

Generates and executes surface-normal scanning trajectories over tracked
bone point clouds.  The drill end-effector follows a boustrophedon
(serpentine) raster path over the top surface of a bone, oriented normal
to the surface at each waypoint with a configurable standoff distance.

When IK fails for a desired orientation the node automatically searches
nearby orientations (rotations around the surface normal, then small tilts)
and picks the closest reachable one.

Commands (via /bone_scan_trajectory/command):
  scan_femur  — generate trajectory over the femur and publish for review
  scan_tibia  — generate trajectory over the tibia and publish for review
  execute     — start executing the most recently planned trajectory
  stop        — abort the current scan

Parameters (all SI — meters, radians, Hz):
  standoff_distance  (float, default 0.01)  — height above surface [m]
  grid_resolution    (float, default 0.005) — raster cell size [m]
  surface_depth      (float, default 0.01)  — Z-band for top surface [m]
  normal_radius      (float, default 0.008) — neighbourhood for normals [m]
  velocity_scaling   (float, default 0.1)   — MoveIt velocity scaling [0–1]

Usage:
  ros2 run bone_scan_trajectory bone_scan_trajectory_node
  ros2 run bone_scan_trajectory bone_scan_trajectory_node \
       --ros-args -p standoff_distance:=0.015 -p grid_resolution:=0.003
"""

import collections
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from sensor_msgs.msg import PointCloud2, JointState
from std_msgs.msg import String, Int32
from geometry_msgs.msg import PoseStamped, PoseArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    PlanningOptions,
    Constraints,
    JointConstraint,
)
from moveit_msgs.srv import GetPositionIK
from tf2_ros import Buffer, TransformListener

from scipy.spatial import KDTree

# ──────────────────────────────────────────────────────────────────────
# Robot configuration (must match bone_peak_nav / goto_pose)
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
# Linear-algebra helpers (same conventions as bone_peak_nav)
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


def rodrigues(axis, angle):
    """Rotation matrix from axis-angle (Rodrigues formula)."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ──────────────────────────────────────────────────────────────────────
# Surface extraction & trajectory generation
# ──────────────────────────────────────────────────────────────────────

def extract_top_surface(points, depth):
    """Keep only points within `depth` metres of the highest Z."""
    z_max = np.max(points[:, 2])
    mask = points[:, 2] > (z_max - depth)
    return points[mask]


def estimate_normals(points, radius):
    """Estimate outward (positive-Z) surface normals via local SVD."""
    tree = KDTree(points)
    normals = np.zeros_like(points)
    for i, pt in enumerate(points):
        idxs = tree.query_ball_point(pt, radius)
        if len(idxs) < 3:
            normals[i] = [0.0, 0.0, 1.0]
            continue
        local = points[idxs]
        centered = local - local.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        n = Vt[-1]
        if n[2] < 0:
            n = -n
        normals[i] = n / np.linalg.norm(n)
    return normals


def normal_to_drill_quat(normal):
    """Build drill-frame quaternion from a surface normal.

    Drill z-axis points into the surface (= -normal), rotated 180 deg
    about z (same convention as bone_peak_nav).
    """
    z_axis = -normal / np.linalg.norm(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(ref, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([-x_axis, -y_axis, z_axis])
    T = np.eye(4)
    T[:3, :3] = R
    return mat_to_quat(T)


def generate_scan_trajectory(points, grid_res, surface_depth, normal_radius, standoff):
    """Return list of (position, quaternion, normal) waypoints.

    1. Extract top surface
    2. Estimate normals
    3. Project onto XY grid
    4. Walk grid in boustrophedon order
    5. For each grid cell, compute drill pose at standoff above surface
    """
    surface = extract_top_surface(points, surface_depth)
    if len(surface) < 10:
        return []

    normals = estimate_normals(surface, normal_radius)

    xy = surface[:, :2]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)

    xs = np.arange(x_min, x_max + grid_res, grid_res)
    ys = np.arange(y_min, y_max + grid_res, grid_res)

    tree = KDTree(xy)
    max_dist = grid_res * 1.5  # discard cells outside bone boundary

    # Primary sweep along Y, step in X — serpentine (boustrophedon)
    waypoints = []
    for j, x in enumerate(xs):
        col_ys = ys if j % 2 == 0 else ys[::-1]
        for y in col_ys:
            dist, idx = tree.query([x, y])
            if dist > max_dist:
                continue
            pt = surface[idx]
            n = normals[idx]
            position = pt + standoff * n
            quat = normal_to_drill_quat(n)
            waypoints.append((position.copy(), quat.copy(), n.copy()))

    return waypoints


# ──────────────────────────────────────────────────────────────────────
# IK orientation fallback
# ──────────────────────────────────────────────────────────────────────

def generate_candidate_orientations(normal, base_quat):
    """Yield (quaternion, deviation) candidates — fast, few iterations.

    1. Ideal orientation (deviation=0)
    2. 6 spins around surface normal (60-deg steps) — free DOF
    3. 4 tilts (10 deg) in cardinal directions — quick fallback
    """
    # 0) Ideal
    yield base_quat, 0

    T_base = quat_to_mat(base_quat)

    # 1) Spin around normal axis
    for angle_deg in (60, 120, 180, 240, 300):
        R_spin = rodrigues(normal, np.radians(angle_deg))
        T_cand = np.eye(4)
        T_cand[:3, :3] = R_spin @ T_base[:3, :3]
        yield mat_to_quat(T_cand), angle_deg

    # 2) Small tilts in 4 cardinal directions
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    tilt = np.radians(10)
    for axis in (u, -u, v, -v):
        tilted_normal = rodrigues(axis, tilt) @ normal
        yield normal_to_drill_quat(tilted_normal), 10


# ──────────────────────────────────────────────────────────────────────
# Node
# ──────────────────────────────────────────────────────────────────────

class BoneScanTrajectoryNode(Node):
    def __init__(self):
        super().__init__("bone_scan_trajectory_node")
        self.get_logger().info("Bone Scan Trajectory starting...")

        self.declare_parameter("standoff_distance", 0.01)
        self.declare_parameter("grid_resolution", 0.005)
        self.declare_parameter("surface_depth", 0.01)
        self.declare_parameter("normal_radius", 0.008)
        self.declare_parameter("velocity_scaling", 0.1)

        self.standoff = self.get_parameter("standoff_distance").value
        self.grid_res = self.get_parameter("grid_resolution").value
        self.surface_depth = self.get_parameter("surface_depth").value
        self.normal_radius = self.get_parameter("normal_radius").value
        self.velocity = self.get_parameter("velocity_scaling").value

        self.cb_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()

        # State
        self._femur_points = None
        self._tibia_points = None
        self.latest_joint_state = None
        self._busy = False          # True while planning or executing
        self._stop_requested = False
        # Planning results: list of (joint_map, drill_pos, drill_quat) per waypoint
        self._solved_trajectory = None

        # TF
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
            String, "/bone_scan_trajectory/command", self._command_cb, 10,
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
        self.status_pub = self.create_publisher(
            String, "/bone_scan_trajectory/status", 10,
        )
        self.trajectory_pub = self.create_publisher(
            PoseArray, "/bone_scan_trajectory/planned_path", latch_qos,
        )
        self.current_target_pub = self.create_publisher(
            PoseStamped, "/bone_scan_trajectory/current_target", latch_qos,
        )
        self.progress_pub = self.create_publisher(
            Int32, "/bone_scan_trajectory/progress", 10,
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

        self._publish_status("Ready. Send scan_femur or scan_tibia.")

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

    def _wait_future(self, future, timeout_sec):
        """Wait for a future without spinning — the executor handles that."""
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout=timeout_sec):
            return None
        return future.result()

    def _publish_trajectory(self, waypoints):
        """Publish the full planned path as PoseArray for visualisation."""
        pa = PoseArray()
        pa.header.frame_id = BASE_FRAME
        pa.header.stamp = self.get_clock().now().to_msg()
        for pos, quat, _normal in waypoints:
            ps = self._make_pose_stamped(pos, quat)
            pa.poses.append(ps.pose)
        self.trajectory_pub.publish(pa)

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

        if cmd == "stop":
            self._stop_requested = True
            self._publish_status("Stop requested — finishing current waypoint...")
            return

        if cmd in ("scan_femur", "scan_tibia"):
            if self._busy:
                self._publish_status("Busy! Send 'stop' first.")
                return
            bone = cmd.split("_")[1]
            # Run in a thread so we don't block the executor — IK calls
            # need the executor free to process service responses.
            self._busy = True
            threading.Thread(
                target=self._plan_scan, args=(bone,), daemon=True,
            ).start()

        elif cmd == "execute":
            if self._busy:
                self._publish_status("Busy! Send 'stop' first.")
                return
            if self._solved_trajectory is None:
                self._publish_status("No trajectory planned! Send scan_femur or scan_tibia first.")
                return
            self._busy = True
            threading.Thread(
                target=self._execute_scan, daemon=True,
            ).start()

        else:
            self._publish_status(f"Unknown command: {cmd}")

    # ── Planning (trajectory generation + IK for every waypoint) ──────

    def _plan_scan(self, bone):
        try:
            self._plan_scan_impl(bone)
        finally:
            self._busy = False

    def _plan_scan_impl(self, bone):
        with self._lock:
            points = self._femur_points if bone == "femur" else self._tibia_points

        if points is None:
            self._publish_status(f"No {bone} cloud received yet!")
            return

        self._publish_status(f"Generating {bone} scan trajectory...")
        self.get_logger().info(f"Depth = {self.surface_depth}")

        waypoints = generate_scan_trajectory(
            points, self.grid_res, self.surface_depth,
            self.normal_radius, self.standoff,
        )

        if not waypoints:
            self._publish_status("Could not generate trajectory — too few surface points")
            self._solved_trajectory = None
            return

        self._publish_status(
            f"{len(waypoints)} waypoints generated. Solving IK for all..."
        )
        self._publish_trajectory(waypoints)

        # Look up drill→EE transform once
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                EE_LINK, "drill",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
        except Exception as e:
            self._publish_status(f"TF {EE_LINK}←drill lookup failed: {e}")
            self._solved_trajectory = None
            return

        T_ee_drill = tf_to_matrix(tf_stamped.transform)

        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self._publish_status("IK service not available!")
            self._solved_trajectory = None
            return

        if self.latest_joint_state is None:
            self._publish_status("No joint state received yet!")
            self._solved_trajectory = None
            return

        # Use current joints as first seed, then chain solutions
        seed_map = dict(zip(
            self.latest_joint_state.name,
            self.latest_joint_state.position,
        ))

        solved = []
        skipped = 0

        for i, (drill_pos, drill_quat, normal) in enumerate(waypoints):
            if (i + 1) % 20 == 0 or i == len(waypoints) - 1:
                self._publish_status(
                    f"IK solving... {i+1}/{len(waypoints)} "
                    f"({skipped} unreachable so far)"
                )

            joint_map = self._solve_waypoint_ik(
                drill_pos, drill_quat, normal, T_ee_drill, seed_map,
            )

            if joint_map is not None:
                solved.append((joint_map, drill_pos, drill_quat))
                seed_map = joint_map  # chain: next IK seeds from this solution
            else:
                skipped += 1

        self._solved_trajectory = solved
        self._publish_status(
            f"Planning done: {len(solved)} reachable, {skipped} skipped. "
            f"Review in RViz, then send 'execute'."
        )

    def _solve_waypoint_ik(self, drill_pos, drill_quat, normal, T_ee_drill, seed_map):
        """Try IK with orientation fallback. Returns joint_map or None."""
        for cand_quat, _deviation in generate_candidate_orientations(normal, drill_quat):
            ee_pos, ee_quat = self._drill_to_ee(drill_pos, cand_quat, T_ee_drill)
            result = self._try_ik(ee_pos, ee_quat, seed_map)
            if result is not None:
                return result
        return None

    # ── Execution (just plays back pre-solved joint configs) ──────────

    def _execute_scan(self):
        try:
            self._execute_scan_impl()
        finally:
            self._busy = False

    def _execute_scan_impl(self):
        trajectory = self._solved_trajectory
        self._stop_requested = False

        total = len(trajectory)
        self._publish_status(f"Executing {total} pre-solved waypoints...")

        executed = 0

        for i, (joint_map, drill_pos, drill_quat) in enumerate(trajectory):
            if self._stop_requested:
                self._publish_status(
                    f"Scan stopped at waypoint {i}/{total} ({executed} executed)"
                )
                break

            prog = Int32()
            prog.data = i
            self.progress_pub.publish(prog)
            self.current_target_pub.publish(self._make_pose_stamped(drill_pos, drill_quat))

            if self._move_to_joints(joint_map, i, total):
                executed += 1
            else:
                self.get_logger().warn(f"Waypoint {i+1}/{total}: motion failed, continuing")
        else:
            self._publish_status(f"Scan complete! {executed}/{total} waypoints executed.")

        self._stop_requested = False

    # ── IK / motion helpers ───────────────────────────────────────────

    def _drill_to_ee(self, drill_pos, drill_quat, T_ee_drill):
        """Convert drill-frame pose to lbr_link_ee pose."""
        T_base_drill = quat_to_mat(drill_quat)
        T_base_drill[0, 3] = drill_pos[0]
        T_base_drill[1, 3] = drill_pos[1]
        T_base_drill[2, 3] = drill_pos[2]

        T_base_ee = T_base_drill @ invert_T(T_ee_drill)
        return T_base_ee[:3, 3], mat_to_quat(T_base_ee)

    def _try_ik(self, ee_pos, ee_quat, seed_map):
        """Call IK service. Returns joint map on success, None on failure."""
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = PLANNING_GROUP
        ik_req.ik_request.avoid_collisions = False
        ik_req.ik_request.timeout.sec = 1

        ik_req.ik_request.pose_stamped = self._make_pose_stamped(ee_pos, ee_quat)

        seed_js = JointState()
        seed_js.name = JOINT_NAMES
        seed_js.position = [float(seed_map.get(j, 0.0)) for j in JOINT_NAMES]
        ik_req.ik_request.robot_state.joint_state = seed_js

        future = self._ik_client.call_async(ik_req)
        result = self._wait_future(future, timeout_sec=3.0)

        if result is None or result.error_code.val != 1:
            return None

        return dict(zip(
            result.solution.joint_state.name,
            result.solution.joint_state.position,
        ))

    def _move_to_joints(self, joint_map, idx, total):
        """Execute a MoveGroup motion to a pre-solved joint configuration."""
        goal_constraints = Constraints()
        for jname in JOINT_NAMES:
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(joint_map.get(jname, 0.0))
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            goal_constraints.joint_constraints.append(jc)

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.num_planning_attempts = 3
        req.allowed_planning_time = PLANNING_TIME
        req.max_velocity_scaling_factor = self.velocity
        req.max_acceleration_scaling_factor = self.velocity
        req.goal_constraints.append(goal_constraints)

        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2

        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveGroup server unavailable")
            return False

        future = self.move_group_client.send_goal_async(goal)
        goal_handle = self._wait_future(future, timeout_sec=15.0)

        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn(f"Waypoint {idx+1}/{total}: goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future, timeout_sec=60.0)

        if result is None or result.result.error_code.val != 1:
            ec = result.result.error_code.val if result else "timeout"
            self.get_logger().warn(f"Waypoint {idx+1}/{total}: motion failed ({ec})")
            return False

        self.get_logger().info(f"Waypoint {idx+1}/{total}: done")
        return True


# ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = BoneScanTrajectoryNode()

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
