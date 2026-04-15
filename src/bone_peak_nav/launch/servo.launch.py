"""
Launches bone_peak_nav_node (MoveIt plan+execute — no servo).

Run AFTER `ros2 launch scan_and_merge robot.launch.py` which brings up
hardware + joint_trajectory_controller + move_group, and
`ros2 launch scan_and_merge scan.launch.py` which publishes the
tracked bone clouds on /tracked/femur and /tracked/tibia.

Usage:
  ros2 launch bone_peak_nav servo.launch.py
  ros2 launch bone_peak_nav servo.launch.py velocity_scaling:=0.2
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("standoff_distance", default_value="0.02"),
        DeclareLaunchArgument("velocity_scaling", default_value="0.6"),
        DeclareLaunchArgument("planning_time", default_value="50.0"),
        DeclareLaunchArgument("num_planning_attempts", default_value="5"),
        DeclareLaunchArgument("tracking_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("movement_threshold_m", default_value="0.0003"),
        DeclareLaunchArgument(
            "goto_pose_input_topic",
            default_value="/surgical_plan/probe_pose",
            description="Topic goto_pose_node subscribes to for target poses",
        ),

        Node(
            package="bone_peak_nav",
            executable="bone_peak_nav_node",
            output="screen",
            parameters=[{
                "standoff_distance": LaunchConfiguration("standoff_distance"),
                "velocity_scaling": LaunchConfiguration("velocity_scaling"),
                "planning_time": LaunchConfiguration("planning_time"),
                "num_planning_attempts": LaunchConfiguration("num_planning_attempts"),
                "tracking_rate_hz": LaunchConfiguration("tracking_rate_hz"),
                "movement_threshold_m": LaunchConfiguration("movement_threshold_m"),
            }],
        ),

        # Foot pedal — publishes std_msgs/Bool on /pedal_press.
        # Crashes occasionally on UTF-8 decode of serial garbage at open;
        # respawn so launch keeps restarting until it comes up clean.
        Node(
            package="scan_and_merge",
            executable="foot_pedal_node",
            output="screen",
            respawn=True,
            respawn_delay=1.0,
        ),

        # Go-to-target-pose node — listens on goto_pose_input_topic for a
        # PoseStamped, then executes on 'go' command via /goto_pose/command.
        Node(
            package="scan_and_merge",
            executable="goto_pose_node",
            output="screen",
            parameters=[{
                "input_topic": LaunchConfiguration("goto_pose_input_topic"),
                "velocity_scaling": LaunchConfiguration("velocity_scaling"),
                "planning_time": LaunchConfiguration("planning_time"),
                "num_planning_attempts": LaunchConfiguration("num_planning_attempts"),
                "tracking_rate_hz": LaunchConfiguration("tracking_rate_hz"),
                "movement_threshold_m": LaunchConfiguration("movement_threshold_m"),
            }],
        ),
    ])
