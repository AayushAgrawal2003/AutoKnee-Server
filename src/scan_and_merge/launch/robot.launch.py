"""
Launch: KUKA Hardware + MoveIt

Brings up:
  1. lbr_bringup hardware (robot driver, controllers, robot_state_publisher)
  2. MoveIt move_group
  3. EE pose publisher (/ee_pose topic)

Usage:
  ros2 launch scan_and_merge robot.launch.py
  ros2 launch scan_and_merge robot.launch.py model:=med14
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare



def generate_launch_description():
    model = LaunchConfiguration("model")
    robot_name = LaunchConfiguration("robot_name")

    return LaunchDescription([

        # ── Arguments ──
        DeclareLaunchArgument("model", default_value="med7",
                              description="Robot model [med7, med14, iiwa7, iiwa14]"),
        DeclareLaunchArgument("robot_name", default_value="lbr",
                              description="Robot name prefix"),

        # ── 1. Hardware Bringup ──
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("lbr_bringup"), "launch", "hardware.launch.py"
                ])
            ),
            launch_arguments={
                "model": model,
                "robot_name": robot_name,
            }.items(),
        ),

        # ── Tool Tip Frame (10mm X, 10mm Z offset from flange) ──


        # ── 2. MoveIt (delayed to let hardware connect first) ──
        TimerAction(
            period=5.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        PathJoinSubstitution([
                            FindPackageShare("lbr_bringup"), "launch", "move_group.launch.py"
                        ])
                    ),
                    launch_arguments={
                        "model": model,
                        "mode": "hardware",
                        "rviz": "false",
                    }.items(),
                ),
            ],
        ),

        # ── 3. EE Pose Publisher (delayed to let TF tree populate) ──
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='scan_and_merge',
                    executable='ee_publisher',
                    output='screen',
                ),
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
        #     arguments=['--x', '0.0732', '--z', '0.107', '--y', '-0.00179',
        #             '--frame-id', 'lbr_link_7', '--child-frame-id', 'tool_tip'], 
        arguments=['--x', '0.0732', '--z', '0.107', '--y', '-0.00179',
           '--roll', '0', '--pitch', '1.5708', '--yaw', '0x',
           '--frame-id', 'lbr_link_7', '--child-frame-id', 'tool_tip']
    )])