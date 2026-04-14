#!/usr/bin/env python3
"""
Hand-Guiding Toggle for KUKA LBR Med 7 via lbr_bringup / lbr_fri_ros2_stack

SPACE = toggle hand-guiding (compliant) ↔ position hold (stiff)
Q     = quit (returns to stiff first)

═══════════════════════════════════════════════════════════════════
HOW IT WORKS
═══════════════════════════════════════════════════════════════════

The lbr_fri_ros2_stack uses ros2_control. Stiffness is set on the
Sunrise/Java side and CANNOT be changed at runtime through FRI.

So for hand-guiding you need TWO things:

  1. SUNRISE SIDE: Your LBRServer.java must start in joint impedance
     mode with ZERO stiffness:

       JointImpedanceControlMode mode =
           new JointImpedanceControlMode(0, 0, 0, 0, 0, 0, 0);

     This makes the robot gravity-compensated (compliant) at the
     FRI level. The robot will be freely movable by hand UNLESS
     a ros2_control controller is actively commanding positions.

  2. ROS2 SIDE (this node): We toggle by activating/deactivating
     the position trajectory controller:

     - STIFF:     position_trajectory_controller ACTIVE
                  (commands current joint pos -> robot holds still)
     - COMPLIANT: position_trajectory_controller DEACTIVATED
                  (no commands sent -> zero-stiffness takes over
                   -> robot is freely movable)

═══════════════════════════════════════════════════════════════════
IMPORTANT: If your Sunrise app uses position control mode (not
impedance), deactivating the controller will cause the robot to
hold its last commanded position -- it will NOT become compliant.
You MUST use impedance mode with zero stiffness on the Sunrise side.
═══════════════════════════════════════════════════════════════════

Usage:
  python3 hand_guide_toggle.py

  # Or with custom controller name:
  python3 hand_guide_toggle.py --ros-args \
    -p controller_name:=joint_trajectory_controller
"""

import rclpy
from rclpy.node import Node
from controller_manager_msgs.srv import SwitchController, ListControllers
from sensor_msgs.msg import JointState

import sys
import time
import termios
import tty
import select
import threading
import numpy as np


DEFAULT_CONTROLLER = "joint_trajectory_controller"

JOINT_NAMES = [
    "lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4",
    "lbr_A5", "lbr_A6", "lbr_A7",
]


def get_keypress(timeout=0.1):
    """Read a single keypress, draining any auto-repeat duplicates."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = sys.stdin.read(1)
        # Drain any buffered repeats so we only get one event per press
        while True:
            r2, _, _ = select.select([sys.stdin], [], [], 0.02)
            if r2:
                sys.stdin.read(1)
            else:
                break
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class HandGuideToggle(Node):
    def __init__(self):
        super().__init__("hand_guide_toggle")

        self.declare_parameter("controller_name", DEFAULT_CONTROLLER)
        self.declare_parameter("robot_name", "lbr")

        self.ctrl_name = (
            self.get_parameter("controller_name")
            .get_parameter_value().string_value
        )
        robot_name = (
            self.get_parameter("robot_name")
            .get_parameter_value().string_value
        )

        self._switch_client = self.create_client(
            SwitchController,
            f"/{robot_name}/controller_manager/switch_controller",
        )
        self._list_client = self.create_client(
            ListControllers,
            f"/{robot_name}/controller_manager/list_controllers",
        )

        self.latest_joints = None
        self.joint_sub = self.create_subscription(
            JointState, f"/{robot_name}/joint_states",
            self._joint_cb, 10,
        )

        self.compliant = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _joint_cb(self, msg):
        self.latest_joints = msg

    def _switch_controller(self, activate=None, deactivate=None):
        if not self._switch_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("switch_controller service not available!")
            return False

        req = SwitchController.Request()
        req.activate_controllers = activate or []
        req.deactivate_controllers = deactivate or []
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True

        future = self._switch_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()

        if result is None:
            self.get_logger().error("switch_controller timed out")
            return False
        if not result.ok:
            self.get_logger().error("switch_controller returned not ok")
            return False
        return True

    def _list_controllers(self):
        if not self._list_client.wait_for_service(timeout_sec=3.0):
            return []
        future = self._list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        return result.controller if result else []

    def _get_joint_str(self):
        if self.latest_joints is None:
            return "(no joint state)"
        jmap = dict(zip(self.latest_joints.name, self.latest_joints.position))
        vals = [jmap.get(j, 0.0) for j in JOINT_NAMES]
        return "[" + ", ".join(f"{np.degrees(v):+7.2f}" for v in vals) + "]"

    def _enter_compliant(self):
        self.get_logger().info("  Deactivating controller -> COMPLIANT...")
        ok = self._switch_controller(deactivate=[self.ctrl_name])
        if ok:
            self.compliant = True
            self.get_logger().info(
                f"  State: COMPLIANT (hand-guide)\n"
                f"  Joints: {self._get_joint_str()}"
            )
        else:
            self.get_logger().error("  Failed to deactivate controller!")

    def _enter_stiff(self):
        self.get_logger().info("  Activating controller -> STIFF...")
        ok = self._switch_controller(activate=[self.ctrl_name])
        if ok:
            self.compliant = False
            self.get_logger().info(
                f"  State: STIFF (position hold)\n"
                f"  Joints: {self._get_joint_str()}"
            )
        else:
            self.get_logger().error("  Failed to activate controller!")

    def _loop(self):
        time.sleep(2.0)

        # Show what's running
        controllers = self._list_controllers()
        if controllers:
            self.get_logger().info("  Controllers:")
            for c in controllers:
                self.get_logger().info(f"    {c.name} [{c.state}]")

        # Wait for joint states
        t0 = time.time()
        while self.latest_joints is None and time.time() - t0 < 10.0:
            time.sleep(0.2)

        self.get_logger().info(
            f"\n{'='*56}\n"
            f"  HAND-GUIDE TOGGLE\n"
            f"  Controller: {self.ctrl_name}\n"
            f"  SPACE = toggle compliant / stiff\n"
            f"  Q     = quit\n"
            f"{'='*56}\n"
            f"  State: STIFF\n"
            f"  Joints: {self._get_joint_str()}\n"
            f"\n"
            f"  NOTE: Sunrise app MUST be in joint impedance\n"
            f"  mode with zero stiffness for compliant to work!\n"
            f"{'='*56}"
        )

        while True:
            key = get_keypress(0.1)
            if key is None:
                continue

            if key == " ":
                if not self.compliant:
                    self._enter_compliant()
                else:
                    self._enter_stiff()

            elif key in ("q", "Q"):
                if self.compliant:
                    self.get_logger().info("  Stiffening before exit...")
                    self._enter_stiff()
                self.get_logger().info("  Exiting.")
                rclpy.shutdown()
                return


def main(args=None):
    rclpy.init(args=args)
    node = HandGuideToggle()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()