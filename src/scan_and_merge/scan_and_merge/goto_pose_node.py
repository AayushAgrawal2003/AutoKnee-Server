#!/usr/bin/env python3
"""
Go-To-Pose Node for KUKA LBR Med 7

Subscribes to a PoseStamped topic and allows the user to send the end effector
to that position via a Foxglove button. The drill frame axes are matched
to the goal pose axes.

Topics:
  Subscribed:
    /goal_pose_input (geometry_msgs/PoseStamped) - input goal pose
    /goto_pose/command (std_msgs/String) - "go" command from Foxglove button

  Published:
    /goto_pose/target (geometry_msgs/PoseStamped) - current target pose (for visualization)
    /goto_pose/status (std_msgs/String) - status messages

Usage:
  ros2 run scan_and_merge goto_pose_node

"""

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
import threading

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

VELOCITY_SCALING = 0.1
PLANNING_TIME = 10.0


def quat_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


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


class GoToPoseNode(Node):
    def __init__(self):
        super().__init__("goto_pose_node")
        self.get_logger().info("GoToPose Node starting...")

        self.declare_parameter("input_topic", "/surgical_plan/probe_pose")
        self.declare_parameter("velocity_scaling", VELOCITY_SCALING)

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.velocity = self.get_parameter("velocity_scaling").get_parameter_value().double_value

        self.cb_group = ReentrantCallbackGroup()

        # State
        self.latest_goal_pose = None
        self.latest_joint_state = None
        self._lock = threading.Lock()

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
            JointState, "/lbr/joint_states", self._joint_state_cb, 10,
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
            self, MoveGroup, "/lbr/move_action",
            callback_group=self.cb_group,
        )

        # IK service client
        self._ik_client = self.create_client(
            GetPositionIK, "/lbr/compute_ik",
            callback_group=self.cb_group,
        )

        self._publish_status("Ready. Waiting for goal pose...")
        self.get_logger().info(f"Subscribed to {self.input_topic} for goal poses")
        self.get_logger().info("Send 'go' to /goto_pose/command to execute motion")

    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f"[status] {text}")

    def _goal_pose_cb(self, msg: PoseStamped):
        with self._lock:
            self.latest_goal_pose = msg

        # Republish for visualization
        self.target_pub.publish(msg)

        pos = msg.pose.position
        self._publish_status(
            f"Received goal: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}] in {msg.header.frame_id}"
        )

    def _joint_state_cb(self, msg: JointState):
        self.latest_joint_state = msg

    def _command_cb(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == "go":
            self._execute_goto()
        else:
            self._publish_status(f"Unknown command: {cmd}")

    def _execute_goto(self):
        with self._lock:
            goal_pose = self.latest_goal_pose

        if goal_pose is None:
            self._publish_status("No goal pose received yet!")
            return

        self._publish_status("Planning motion to goal...")

        # Get goal position
        goal_pos = np.array([
            goal_pose.pose.position.x,
            goal_pose.pose.position.y,
            goal_pose.pose.position.z,
        ])

        # Get goal orientation
        goal_quat = np.array([
            goal_pose.pose.orientation.x,
            goal_pose.pose.orientation.y,
            goal_pose.pose.orientation.z,
            goal_pose.pose.orientation.w,
        ])

        self.get_logger().info(f"Goal position: {goal_pos}")
        self.get_logger().info(f"Goal quaternion: {goal_quat}")

        # Transform goal to base frame if needed
        goal_frame = goal_pose.header.frame_id
        if goal_frame and goal_frame != BASE_FRAME:
            try:
                tf = self.tf_buffer.lookup_transform(
                    BASE_FRAME, goal_frame,
                    rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)
                )
                # Transform position
                t = tf.transform.translation
                q = tf.transform.rotation
                R = quat_to_rotation_matrix([q.x, q.y, q.z, q.w])
                goal_pos = R @ goal_pos + np.array([t.x, t.y, t.z])

                # Transform orientation
                R_goal = quat_to_rotation_matrix(goal_quat)
                R_transformed = R @ R_goal
                goal_quat = rotation_matrix_to_quat(R_transformed)

                self.get_logger().info(f"Transformed goal to {BASE_FRAME}")
            except Exception as e:
                self._publish_status(f"TF lookup failed: {e}")
                return

        # goal_pos/goal_quat is where we want the drill frame to be.
        # Compute where lbr_link_ee must be so drill lands there.
        # T_base_ee = T_base_drill * T_drill_ee
        try:
            tf_drill_ee = self.tf_buffer.lookup_transform(
                'drill', EE_LINK,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=2.0)
            )
        except Exception as e:
            self._publish_status(f"TF lookup drill→{EE_LINK} failed: {e}")
            return

        dt = tf_drill_ee.transform.translation
        dq = tf_drill_ee.transform.rotation
        R_drill_ee = quat_to_rotation_matrix([dq.x, dq.y, dq.z, dq.w])
        R_goal_mat = quat_to_rotation_matrix(goal_quat)

        goal_pos = R_goal_mat @ np.array([dt.x, dt.y, dt.z]) + goal_pos
        goal_quat = rotation_matrix_to_quat(R_goal_mat @ R_drill_ee)

        self.get_logger().info(f"Computed EE pose: pos={goal_pos}, quat={goal_quat}")

        # Call IK to get joint angles
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            self._publish_status("IK service not available!")
            return

        # Get current joint state for seed
        if self.latest_joint_state is None:
            self._publish_status("No joint state received yet!")
            return

        joint_map = dict(zip(
            self.latest_joint_state.name,
            self.latest_joint_state.position
        ))

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = PLANNING_GROUP
        ik_req.ik_request.avoid_collisions = True
        ik_req.ik_request.timeout.sec = 5

        # Build pose stamped for IK
        ps = PoseStamped()
        ps.header.frame_id = BASE_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(goal_pos[0])
        ps.pose.position.y = float(goal_pos[1])
        ps.pose.position.z = float(goal_pos[2])
        ps.pose.orientation.x = float(goal_quat[0])
        ps.pose.orientation.y = float(goal_quat[1])
        ps.pose.orientation.z = float(goal_quat[2])
        ps.pose.orientation.w = float(goal_quat[3])
        ik_req.ik_request.pose_stamped = ps

        # Publish the computed EE target pose for visualization
        self.ee_target_pub.publish(ps)
        self.get_logger().info("Published computed EE target to /goto_pose/ee_target")

        self.get_logger().info(
            f"IK request pose in {BASE_FRAME}:\n"
            f"  position: [{goal_pos[0]:.4f}, {goal_pos[1]:.4f}, {goal_pos[2]:.4f}]\n"
            f"  orientation: [{goal_quat[0]:.4f}, {goal_quat[1]:.4f}, {goal_quat[2]:.4f}, {goal_quat[3]:.4f}]"
        )

        # Seed state
        from sensor_msgs.msg import JointState as SensorJointState
        seed_js = SensorJointState()
        seed_js.name = JOINT_NAMES
        seed_js.position = [float(joint_map.get(j, 0.0)) for j in JOINT_NAMES]
        ik_req.ik_request.robot_state.joint_state = seed_js

        self._publish_status("Computing IK...")

        ik_future = self._ik_client.call_async(ik_req)
        rclpy.spin_until_future_complete(self, ik_future, timeout_sec=10.0)
        ik_result = ik_future.result()

        if ik_result is None or ik_result.error_code.val != 1:
            ec = ik_result.error_code.val if ik_result else "timeout"
            # Common error codes: -31 = NO_IK_SOLUTION, -12 = INVALID_GOAL_CONSTRAINTS
            error_names = {
                -31: "NO_IK_SOLUTION (pose may be unreachable)",
                -12: "INVALID_GOAL_CONSTRAINTS",
                -10: "START_STATE_IN_COLLISION",
                -4: "PLANNING_FAILED",
            }
            error_msg = error_names.get(ec, f"error code {ec}")
            self._publish_status(f"IK failed: {error_msg}")
            self.get_logger().error(
                f"IK failed with error {ec}. The requested pose may be:\n"
                f"  - Outside robot workspace\n"
                f"  - In collision\n"
                f"  - Requiring joint limits to be exceeded\n"
                f"Try a different goal pose or orientation."
            )
            return

        # Extract solved joint positions
        ik_joint_map = dict(
            zip(ik_result.solution.joint_state.name,
                ik_result.solution.joint_state.position)
        )

        self._publish_status("IK solved, executing motion...")

        # Build MoveIt goal with joint constraints
        goal_constraints = Constraints()
        for jname in JOINT_NAMES:
            jval = ik_joint_map.get(jname, joint_map.get(jname, 0.0))
            jc = JointConstraint()
            jc.joint_name = jname
            jc.position = float(jval)
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
            return

        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        goal_handle = future.result()

        if not goal_handle or not goal_handle.accepted:
            self._publish_status("Goal rejected by MoveGroup!")
            return

        self._publish_status("Executing motion...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        result = result_future.result()

        if result is None or result.result.error_code.val != 1:
            ec = result.result.error_code.val if result else "timeout"
            self._publish_status(f"Motion failed (error={ec})")
            return

        self._publish_status("Motion complete! EE at goal position.")


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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
