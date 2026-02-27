"""
Launch: KUKA Hardware + MoveIt

Brings up:
  1. lbr_bringup hardware (robot driver, controllers, robot_state_publisher)
  2. MoveIt move_group

Usage:
  ros2 launch scan_and_merge robot.launch.py
  ros2 launch scan_and_merge robot.launch.py model:=med14
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
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
    ])