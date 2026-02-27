"""
Launch file for the scan & merge pipeline.

Launches:
  1. robot_state_publisher (loads URDF, broadcasts TF)
  2. RealSense camera node
  3. scan_and_merge_node

Assumes:
  - lbr_ros2 stack is running separately (FRI driver, controllers)
  - MoveIt2 move_group is running separately

Usage:
  ros2 launch scan_and_merge scan_and_merge.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── Arguments ──
        DeclareLaunchArgument(
            "robot_name", default_value="lbr",
            description="Robot name prefix used in URDF",
        ),
        DeclareLaunchArgument(
            "camera_serial_no", default_value="",
            description="RealSense camera serial number (empty = any)",
        ),
        DeclareLaunchArgument(
            "pointcloud_enable", default_value="true",
            description="Enable point cloud output from RealSense",
        ),

        # ── RealSense Camera ──
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="camera",
            namespace="camera",
            parameters=[{
                "serial_no": LaunchConfiguration("camera_serial_no"),
                "pointcloud.enable": LaunchConfiguration("pointcloud_enable"),
                "align_depth.enable": True,
                "depth_module.profile": "640x480x30",
                "rgb_camera.profile": "640x480x30",
            }],
            output="screen",
        ),

        # ── Scan & Merge Node (delayed to let camera start) ──
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package="scan_and_merge",
                    executable="scan_and_merge_node",
                    name="scan_and_merge_node",
                    output="screen",
                    parameters=[{
                        "robot_name": LaunchConfiguration("robot_name"),
                    }],
                ),
            ],
        ),
    ])
