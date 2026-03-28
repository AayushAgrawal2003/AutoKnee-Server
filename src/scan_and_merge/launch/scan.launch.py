"""
Launch: Camera + Static TF + RViz + Scan Node + Detect Node + Waypoint Viz + Bone Cloud Mover

Brings up:
  1. RealSense camera (pointcloud + aligned depth)
  2. Static TF: lbr_link_7 -> camera_link
  3. RViz with scan_and_merge config
  4. scan_and_merge_node (delayed, runs in xterm for interactive input)
  5. detect_and_merge_node (optional, YOLO detection + filtered merging)
  6. waypoint_visualizer (optional, shows waypoint trajectory in RViz)
  7. bone_cloud_mover (rigid-body tracking of aligned clouds with IR trackers)
  8. ros2 bag record (optional, records all key topics for offline experiments)

Topic naming:
  /bone_scan/{tibia,femur}     — filtered+denoised camera scan clouds
  /bone_model/{tibia,femur}    — ICP-aligned reference PLY (model moved to match scan)
  /model_frame/{tibia,femur}   — reference PLY in its own local frame (for live tracking)
  /tracked/{tibia,femur}       — live marker-tracked bone model (from bone_cloud_mover)
  /calibration/ref_to_tracker_*— calibration 4x4 matrices
  /kuka_frame/bone_pose_*      — live IR tracker poses

Usage:
  # Original scan workflow (unchanged)
  ros2 launch scan_and_merge scan.launch.py

  # Just visualize waypoints in RViz (no robot motion)
  ros2 launch scan_and_merge scan.launch.py \
    scan_node:=false visualize_waypoints:=true \
    load_waypoints:=~/scan_output/waypoints.npy

  # Run YOLO detect node instead of scan node
  ros2 launch scan_and_merge scan.launch.py \
    run_detect:=true scan_node:=false \
    weights:=~/weights/best.pt \
    load_waypoints:=~/scan_output/waypoints.npy

  # Record a rosbag for offline experiments
  ros2 launch scan_and_merge scan.launch.py \
    run_detect:=true scan_node:=false \
    record_bag:=true bag_dir:=~/autoknee_bags \
    weights:=~/weights/best.pt \
    load_waypoints:=~/scan_output/waypoints.npy

  # Enable perpendicular camera adjustment (reorients toward bone centroid using wrist joints)
  ros2 launch scan_and_merge scan.launch.py \
    run_detect:=true scan_node:=false \
    perpendicular_adjust:=true \
    weights:=~/weights/best.pt \
    load_waypoints:=~/scan_output/waypoints.npy

NOTE: The scan node needs interactive terminal input (ENTER to record / start).
      It launches in an xterm window. Install if you don't have it:
        sudo apt install xterm
      Or set use_xterm:=false and run the node manually in a separate terminal.

      The detect node does NOT need interactive input — it replays waypoints
      automatically once launched.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    ExecuteProcess,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg_share = FindPackageShare("scan_and_merge")
    detect_config = PathJoinSubstitution([pkg_share, "config", "detect_node.yaml"])
    mover_config = PathJoinSubstitution([pkg_share, "config", "bone_mover.yaml"])

    # ── Launch Configurations ──
    cam_x = LaunchConfiguration("cam_x")
    cam_y = LaunchConfiguration("cam_y")
    cam_z = LaunchConfiguration("cam_z")
    cam_roll = LaunchConfiguration("cam_roll")
    cam_pitch = LaunchConfiguration("cam_pitch")
    cam_yaw = LaunchConfiguration("cam_yaw")
    load_waypoints = LaunchConfiguration("load_waypoints")
    velocity_scaling = LaunchConfiguration("velocity_scaling")
    scan = LaunchConfiguration("scan")
    use_xterm = LaunchConfiguration("use_xterm")
    launch_rviz = LaunchConfiguration("rviz")

    # Detect node configs
    run_detect = LaunchConfiguration("run_detect")
    scan_node_enabled = LaunchConfiguration("scan_node")
    weights = LaunchConfiguration("weights")
    target_classes = LaunchConfiguration("target_classes")
    confidence = LaunchConfiguration("confidence")
    use_seg_mask = LaunchConfiguration("use_seg_mask")

    # Waypoint visualizer
    visualize_waypoints = LaunchConfiguration("visualize_waypoints")

    # Bone cloud mover
    run_bone_mover = LaunchConfiguration("run_bone_mover")

    # Multi-orientation calibration
    multi_orientation = LaunchConfiguration("multi_orientation")
    n_orientations = LaunchConfiguration("n_orientations")

    # Bone cloud mover calibration paths
    ref_to_tracker_tibia = LaunchConfiguration("ref_to_tracker_tibia")
    ref_to_tracker_femur = LaunchConfiguration("ref_to_tracker_femur")

    # Perpendicular view adjustment
    perpendicular_adjust = LaunchConfiguration("perpendicular_adjust")

    # Rosbag recording
    record_bag = LaunchConfiguration("record_bag")
    bag_dir    = LaunchConfiguration("bag_dir")

    return LaunchDescription([

        # ── Arguments: Camera mount transform ──
        DeclareLaunchArgument("cam_x", default_value="0.03775",
                              description="Camera X offset from lbr_link_7 (meters)"),
        DeclareLaunchArgument("cam_y", default_value="-0.00900",
                              description="Camera Y offset from lbr_link_7 (meters)"),
        DeclareLaunchArgument("cam_z", default_value="0.18123",
                              description="Camera Z offset from lbr_link_7 (meters)"),
        DeclareLaunchArgument("cam_roll", default_value="3.14",
                              description="Camera roll from lbr_link_7 (radians)"),
        DeclareLaunchArgument("cam_pitch", default_value="-1.57079632679",
                              description="Camera pitch from lbr_link_7 (radians)"),
        DeclareLaunchArgument("cam_yaw", default_value="0",
                              description="Camera yaw from lbr_link_7 (radians)"),

        # ── Arguments: Scan node ──
        DeclareLaunchArgument("load_waypoints", default_value="~/scan_output/waypoints.npy",
                              description="Path to waypoints.npy"),
        DeclareLaunchArgument("velocity_scaling", default_value="0.1",
                              description="Robot velocity scaling 0.0-1.0"),
        DeclareLaunchArgument("scan", default_value="true",
                              description="Capture point clouds and images (false = replay only)"),
        DeclareLaunchArgument("use_xterm", default_value="true",
                              description="Launch scan node in xterm for interactive input"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="Launch RViz"),

        # ── Arguments: Detect node ──
        DeclareLaunchArgument("run_detect", default_value="true",
                              description="Launch detect_and_merge_node"),
        DeclareLaunchArgument("scan_node", default_value="false",
                              description="Launch scan_and_merge_node (disable if only detecting)"),
        DeclareLaunchArgument("weights", default_value="~/scan_output/best.pt",
                              description="Path to YOLO .pt weights"),
        DeclareLaunchArgument("target_classes", default_value="",
                              description="YOLO class IDs to keep (comma-sep or JSON list, empty = all)"),
        DeclareLaunchArgument("confidence", default_value="0.8",
                              description="YOLO confidence threshold"),
        DeclareLaunchArgument("use_seg_mask", default_value="true",
                              description="Use segmentation masks instead of bounding boxes"),

        # ── Arguments: Waypoint visualizer ──
        DeclareLaunchArgument("visualize_waypoints", default_value="false",
                              description="Launch waypoint visualizer (shows trajectory in RViz)"),

        # ── Arguments: Bone cloud mover ──
        DeclareLaunchArgument("run_bone_mover", default_value="true",
                              description="Launch bone cloud mover (tracks aligned clouds with IR trackers)"),
        DeclareLaunchArgument("ref_to_tracker_tibia", default_value="",
                              description="Path to T_ref_to_tracker .npy for tibia (multi-orient calibration)"),
        DeclareLaunchArgument("ref_to_tracker_femur", default_value="",
                              description="Path to T_ref_to_tracker .npy for femur (multi-orient calibration)"),

        # ── Arguments: Multi-orientation calibration ──
        DeclareLaunchArgument("multi_orientation", default_value="true",
                              description="Enable multi-orientation ICP calibration mode"),
        DeclareLaunchArgument("n_orientations", default_value="2",
                              description="Number of bone orientations to collect"),

        # ── Arguments: Perpendicular view adjustment ──
        DeclareLaunchArgument("perpendicular_adjust", default_value="false",
                              description="Reorient camera perpendicular to bone centroid at each waypoint "
                                          "(uses top 2-3 wrist joints only)"),

        # ── Arguments: Rosbag recording ──
        DeclareLaunchArgument("record_bag", default_value="false",
                              description="Record a rosbag of all key topics for offline experiments"),
        DeclareLaunchArgument("bag_dir", default_value="~/autoknee_bags",
                              description="Base output directory for rosbag files "
                                          "(a timestamped subfolder is created automatically)"),

        # ── 1. RealSense Camera ──
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="camera",
            namespace="camera_arm",
            parameters=[{
                "pointcloud.enable": True,
                "pointcloud.stream_filter": 2,    # RS2_STREAM_COLOR
                "pointcloud.stream_index_filter": 0,
                "align_depth.enable": True,
                "enable_color": True,
                "depth_module.profile": "640x480x30",
                "rgb_camera.profile": "640x480x30",
            }],
            output="screen",
        ),

        # ── 2. Static TF: lbr_link_7 -> camera_link ──
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="camera_ee_tf",
            arguments=[
                "--x", cam_x,
                "--y", cam_y,
                "--z", cam_z,
                "--roll", cam_roll,
                "--pitch", cam_pitch,
                "--yaw", cam_yaw,
                "--frame-id", "lbr_link_7",
                "--child-frame-id", "camera_link",
            ],
            output="screen",
        ),

        # ── 3. RViz ──

        # ── 4. Scan Node (original, in xterm) ──
        TimerAction(
            period=4.0,
            actions=[
                # Option A: xterm (interactive)
                ExecuteProcess(
                    condition=IfCondition(PythonExpression([
                        "'", scan_node_enabled, "' == 'true' and '",
                        use_xterm, "' == 'true'",
                    ])),
                    cmd=[
                        "xterm", "-e",
                        "bash", "-c",
                        PythonExpression([
                            "'source ~/lbr-stack/install/setup.bash && "
                            "ros2 run scan_and_merge scan_and_merge_node --ros-args"
                            " -p load_waypoints:=",
                            "\"'", " + '", load_waypoints, "' + '", "\"'",
                            " + '", " -p velocity_scaling:=", "' + '", velocity_scaling, "'",
                            " + '", " -p scan:=", "' + '", scan, "'",
                            " + '", "; exec bash'",
                        ]),
                    ],
                    output="screen",
                ),
                # Option B: inline (no interactive input)
                Node(
                    condition=IfCondition(PythonExpression([
                        "'", scan_node_enabled, "' == 'true' and '",
                        use_xterm, "' == 'false'",
                    ])),
                    package="scan_and_merge",
                    executable="scan_and_merge_node",
                    name="scan_and_merge_node",
                    output="screen",
                    parameters=[{
                        "load_waypoints": load_waypoints,
                        "velocity_scaling": velocity_scaling,
                        "scan": scan,
                    }],
                ),
            ],
        ),

        # ── 5. Detect & Merge Node ──
        TimerAction(
            period=5.0,  # wait for camera + TF to be ready
            actions=[
                Node(
                    condition=IfCondition(run_detect),
                    package="scan_and_merge",
                    executable="detect_and_merge_node",
                    name="detect_and_merge_node",
                    output="screen",
                    parameters=[
                        detect_config,
                        {   # CLI overrides
                            "load_waypoints": load_waypoints,
                            "velocity_scaling": velocity_scaling,
                            "multi_orientation": multi_orientation,
                            "n_orientations": n_orientations,
                            "perpendicular_adjust": perpendicular_adjust,
                        },
                    ],
                ),
            ],
        ),

        # ── 6. Waypoint Visualizer ──
        TimerAction(
            period=2.0,  # start quickly, just needs TF
            actions=[
                Node(
                    condition=IfCondition(visualize_waypoints),
                    package="scan_and_merge",
                    executable="waypoint_visualizer",
                    name="waypoint_visualizer",
                    output="screen",
                    parameters=[{
                        "waypoints_file": load_waypoints,
                        "publish_rate": 1.0,
                        "show_camera_frame": True,
                        "use_tf_fk": True,
                    }],
                ),
            ],
        ),

        # ── 7. Bone Cloud Mover (tracks aligned clouds with IR trackers) ──
        Node(
            condition=IfCondition(run_bone_mover),
            package="scan_and_merge",
            executable="bone_cloud_mover",
            name="bone_cloud_mover",
            output="screen",
            parameters=[mover_config],
        ),

        # ── 8. Rosbag Recording (optional, record_bag:=true to enable) ──
        # Records a timestamped bag under bag_dir/ containing all key pipeline topics:
        #   /bone_scan/*        filtered+denoised camera scan clouds
        #   /bone_model/*       ICP-aligned reference model clouds
        #   /model_frame/*      model-frame reference for live tracking
        #   /tracked/*          live marker-tracked bone output
        #   /calibration/*      calibration matrices
        #   /kuka_frame/*       IR tracker poses
        #   /multi_orient/*     calibration status
        #   /lbr/joint_states   robot joint states
        #   camera RGB + depth  for full replay capability
        TimerAction(
            period=3.0,  # give camera + nodes time to start
            actions=[
                ExecuteProcess(
                    condition=IfCondition(record_bag),
                    cmd=[
                        "bash", "-c", [
                            "mkdir -p ", bag_dir,
                            " && ros2 bag record"
                            " -o ", bag_dir, "/bag_$(date +%Y%m%d_%H%M%S)",
                            " /bone_scan/tibia"
                            " /bone_scan/femur"
                            " /bone_model/tibia"
                            " /bone_model/femur"
                            " /model_frame/tibia"
                            " /model_frame/femur"
                            " /tracked/tibia"
                            " /tracked/femur"
                            " /calibration/ref_to_tracker_tibia"
                            " /calibration/ref_to_tracker_femur"
                            " /kuka_frame/bone_pose_tibia"
                            " /kuka_frame/bone_pose_femur"
                            " /multi_orient/status"
                            " /multi_orient/command"
                            " /lbr/joint_states"
                            " /camera_arm/camera/color/image_rect_raw"
                            " /camera_arm/camera/aligned_depth_to_color/image_raw"
                            " /camera_arm/camera/aligned_depth_to_color/camera_info",
                        ],
                    ],
                    output="screen",
                    shell=False,
                ),
            ],
        ),
    ])
