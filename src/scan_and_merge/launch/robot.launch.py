"""
Launch: KUKA Hardware + MoveIt + IR Tracking

Brings up:
  1. lbr_bringup hardware (robot driver, controllers, robot_state_publisher)
  2. MoveIt move_group
  3. EE pose publisher (/ee_marker_pos topic)
  4. IR tracking node (Polaris Vega bone tracker → KUKA frame)
  5. Tool tip static TF
  6. RViz

Usage:
  ros2 launch scan_and_merge robot.launch.py
  ros2 launch scan_and_merge robot.launch.py model:=med14
  ros2 launch scan_and_merge robot.launch.py run_ir_tracking:=false
  ros2 launch scan_and_merge robot.launch.py vega_ip:=169.254.9.239
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    model = LaunchConfiguration("model")
    robot_name = LaunchConfiguration("robot_name")
    run_ir_tracking = LaunchConfiguration("run_ir_tracking")
    vega_ip = LaunchConfiguration("vega_ip")
    ir_hz = LaunchConfiguration("ir_hz")

    return LaunchDescription([

        # ── Arguments ──
        DeclareLaunchArgument("model", default_value="med7",
                              description="Robot model [med7, med14, iiwa7, iiwa14]"),
        DeclareLaunchArgument("robot_name", default_value="lbr",
                              description="Robot name prefix"),
        DeclareLaunchArgument("run_ir_tracking", default_value="true",
                              description="Launch IR tracking node (Polaris Vega)"),
        DeclareLaunchArgument("vega_ip", default_value="169.254.9.239",
                              description="Known Polaris Vega IP (speeds up discovery)"),
        DeclareLaunchArgument("ir_hz", default_value="50.0",
                              description="IR tracker poll/publish rate in Hz"),

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

        # ── 2. Tool Tip Frame ──
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['--x', '0.0732', '--z', '0.107', '--y', '-0.00179',
                       '--roll', '0', '--pitch', '1.5708', '--yaw', '0',
                       '--frame-id', 'lbr_link_7', '--child-frame-id', 'tool_tip'],
        ),

        # ── 3. MoveIt (delayed to let hardware connect first) ──
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

        # ── 4. EE Pose Publisher (delayed to let TF tree populate) ──
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

        # ── 5. IR Tracking Node (delayed to let /ee_marker_pos start) ──
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    condition=IfCondition(run_ir_tracking),
                    package='ir_tracking',
                    executable='ir_tracking_node',
                    name='ir_tracking_node',
                    output='screen',
                    parameters=[{
                        'hz': ir_hz,
                        'vega_ip': vega_ip,
                    }],
                ),
            ],
        ),

        # ── 6. RViz ──
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", PathJoinSubstitution([
                FindPackageShare("scan_and_merge"), "rviz", "main.rviz"
            ])],
            output="screen",
        ),
    ])
