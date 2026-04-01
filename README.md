# AUTOKnee

ROS 2 (Humble) workspace for autonomous bone scanning and registration using a **KUKA LBR Med 7** manipulator with a wrist-mounted **Intel RealSense** depth camera and an **NDI Polaris Vega** optical tracker. The system moves the robot through taught waypoints, detects bones with a YOLO segmentation model, builds per-bone point clouds, denoises them, registers them against reference models using ICP, and optionally solves a multi-orientation calibration to track bones in real time.

## Pipeline Overview

```
Teach waypoints → Replay & capture RGB+Depth at each pose
        ↓
  YOLO inference → per-instance segmentation masks
        ↓
  Mask depth → back-project to 3D (camera frame)
        ↓
  Left/right bone separation (by image x-position)
        ↓
  Transform to base frame (lbr_link_0) via TF
        ↓
  (Optional) Perpendicular view adjustment → re-capture
        ↓
  Cross-view consistency → SOR → radius filter → voxel downsample → smooth
        ↓
  ICP registration (exhaustive rotation + FPFH coarse → multi-scale point-to-plane)
        ↓
  (Optional) Multi-orientation calibration → T_ref_to_tracker solve
        ↓
  Live bone tracking via IR trackers + calibration
```

## Packages

```
src/
├── scan_and_merge/        # Main package: scanning, detection, merging, registration, calibration
├── ir_tracking/           # NDI Polaris Vega IR optical tracking with EKF smoothing
├── audio_capture/         # Microphone capture + Vosk speech recognition for voice commands
├── lbr_fri_ros2_stack/    # KUKA LBR FRI driver & MoveIt2 configuration
├── fri/                   # KUKA FRI SDK (C++)
└── lbr_fri_idl/           # KUKA FRI IDL definitions
```

---

## scan_and_merge

Core package containing the detection/registration pipeline, denoising, ICP, multi-orientation calibration, and live bone tracking.

### Structure

```
scan_and_merge/
├── scan_and_merge/
│   ├── detect_and_merge_node.py     # Main pipeline: YOLO detection + merge + registration + calibration
│   ├── scan_and_merge_node.py       # Waypoint teach/replay + raw capture
│   ├── cloud_denoise.py             # Denoising & smoothing utilities
│   ├── icp_registration.py          # ICP alignment (coarse + fine)
│   ├── multi_orientation_solver.py  # Least-squares T_ref_to_tracker solver
│   ├── mo_utils.py                  # SE(3) math helpers + MultiOrientHelper
│   ├── bone_cloud_mover.py          # Live bone tracking node
│   ├── waypoint_visualizer.py       # RViz waypoint visualization
│   ├── cloud_publisher.py           # Debug: publish any PLY/NPY to RViz
│   ├── marker_pub.py                # EE pose publisher (TF → PoseStamped)
│   └── replay_waypoints.py          # Print/visualize saved waypoints (standalone)
├── launch/
│   ├── scan.launch.py               # Main launch file
│   └── robot.launch.py
├── config/
│   ├── detect_node.yaml             # detect_and_merge_node parameters
│   └── bone_mover.yaml              # bone_cloud_mover parameters
├── resource/
│   ├── tibia_new.ply                # Reference tibia model
│   └── femur_new.ply                # Reference femur model
├── rviz/                            # RViz configs
├── foxglove/                        # Foxglove configs
├── setup.py
└── package.xml
```

### Nodes

#### `detect_and_merge_node`

Main pipeline node. Replays saved waypoints, runs YOLO detection at each pose, produces denoised per-bone point clouds, registers them to reference models, and optionally runs multi-orientation calibration.

**Two operating modes:**

1. **Single-Orientation** (`multi_orientation=false`) — scan → detect → merge → denoise → ICP register → publish.
2. **Multi-Orientation Calibration** (`multi_orientation=true`) — collects N orientation measurements. At each orientation: scan → detect → merge → denoise → ICP → user accepts/rejects. After collection, runs a least-squares solver to find `T_ref_to_tracker` (the fixed offset between the reference model and its IR tracker).

**Subscribed Topics:**

| Topic | Type | Description |
|---|---|---|
| `/lbr/joint_states` | `JointState` | Current joint positions |
| `/camera_arm/camera/color/image_rect_raw` | `Image` | RGB image |
| `/camera_arm/camera/aligned_depth_to_color/image_raw` | `Image` | Aligned depth (uint16, mm) |
| `/camera_arm/camera/aligned_depth_to_color/camera_info` | `CameraInfo` | Camera intrinsics |
| `/kuka_frame/bone_pose_tibia` | `PoseStamped` | IR tracker pose for tibia (multi-orientation mode) |
| `/kuka_frame/bone_pose_femur` | `PoseStamped` | IR tracker pose for femur (multi-orientation mode) |
| `/multi_orient/command` | `String` | User commands: `"both"`, `"femur"`, `"tibia"`, `"neither"`, `"ready"`, `"solve"` |

**Published Topics:**

| Topic | Type | Description |
|---|---|---|
| `/bone_scan/tibia` | `PointCloud2` | Denoised tibia scan in base frame |
| `/bone_scan/femur` | `PointCloud2` | Denoised femur scan in base frame |
| `/bone_model/tibia` | `PointCloud2` | Reference tibia model aligned to scan (ICP) |
| `/bone_model/femur` | `PointCloud2` | Reference femur model aligned to scan (ICP) |
| `/model_frame/tibia` | `PointCloud2` | Reference model in its local frame (post-calibration) |
| `/model_frame/femur` | `PointCloud2` | Reference model in its local frame (post-calibration) |
| `/calibration/ref_to_tracker_tibia` | `Float64MultiArray` | 4x4 calibration matrix (flattened) |
| `/calibration/ref_to_tracker_femur` | `Float64MultiArray` | 4x4 calibration matrix (flattened) |
| `/multi_orient/status` | `String` | Status messages for multi-orientation mode |

**Parameters (`config/detect_node.yaml`):**

| Parameter | Default | Description |
|---|---|---|
| `load_waypoints` | `~/scan_output/new_waypoints.npy` | Path to waypoints file |
| `weights` | `~/scan_output/best.pt` | Path to YOLO `.pt` weights |
| `target_classes` | `""` | Comma-sep class IDs or JSON list (empty = all) |
| `confidence` | `0.8` | YOLO detection threshold |
| `velocity_scaling` | `0.1` | Robot motion speed factor |
| `max_depth_m` | `1.0` | Maximum depth to keep (meters) |
| `settle_time` | `1.5` | Wait time after reaching waypoint (seconds) |
| `use_seg_mask` | `true` | Use segmentation masks vs bounding boxes |
| `denoise` | `true` | Enable denoising pipeline |
| `cross_cloud_dist` | `0.005` | Cross-view support distance (m) |
| `cross_cloud_min_views` | `2` | Min viewpoints for a point to survive |
| `sor_k` | `20` | SOR neighbor count |
| `sor_std` | `2.0` | SOR std ratio |
| `radius_filter` | `0.01` | Radius outlier removal search radius (m) |
| `radius_min_neighbors` | `5` | Radius outlier removal min neighbors |
| `denoise_voxel_size` | `0.001` | Voxel downsample size (m) |
| `smooth_k` | `10` | Laplacian smoothing neighbor count |
| `smooth_iterations` | `1` | Laplacian smoothing passes |
| `surface_smooth` | `false` | Enable Poisson reconstruction + Taubin smoothing |
| `poisson_depth` | `5` | Poisson reconstruction octree depth |
| `poisson_density_quantile` | `0.1` | Poisson low-density vertex removal quantile |
| `taubin_iterations` | `5` | Taubin smoothing iterations |
| `taubin_mu` | `-0.53` | Taubin smoothing mu parameter |
| `register` | `true` | Enable ICP registration |
| `tibia_reference` | `resource/tibia_new.ply` | Path to tibia reference PLY |
| `femur_reference` | `resource/femur_new.ply` | Path to femur reference PLY |
| `icp_coarse_method` | `"hybrid"` | Coarse alignment: `"hybrid"`, `"fpfh"`, or `"centroid"` |
| `icp_voxel_size` | `0.002` | ICP voxel size (m) |
| `multi_orientation` | `true` | Enable multi-orientation calibration mode |
| `n_orientations` | `4` | Number of orientations to collect |
| `tracker_avg_samples` | `50` | IR tracker pose averaging samples |
| `perpendicular_adjust` | `true` | Enable camera perpendicular view adjustment |
| `perp_max_angle_deg` | `25.0` | Max rotation angle for perpendicular adjustment |
| `perp_free_joints` | `3` | Number of wrist joints free for perp adjustment |

#### `scan_and_merge_node`

Teaches and replays joint-space waypoints on the KUKA arm. Three modes:

- **TEACH** — manually guide the robot, press ENTER to record waypoints.
- **SCAN** — replay waypoints, capture RGB images + depth point clouds at each pose.
- **MERGE** — transform all captured clouds to base frame and merge.

Output: `~/scan_output/` with subdirs for images and clouds.

#### `bone_cloud_mover`

Live bone tracking node. Two modes:

- **Pre-calibration (anchor+delta):** latches the ICP-aligned model at capture time, applies relative tracker motion to update the model position.
- **Post-calibration (direct):** receives `T_ref_to_tracker` calibration, applies it to live tracker poses, publishes tracked bone clouds.

**Published Topics:** `/tracked/tibia`, `/tracked/femur` — live-tracked bone models in base frame.

#### `waypoint_visualizer`

Loads `waypoints.npy` and visualizes in RViz as numbered spheres, trajectory lines, tool direction arrows, and arm ghost overlays. Publishes `/waypoint_markers` (MarkerArray) and `/waypoint_poses` (PoseArray).

#### `cloud_publisher`

Debug utility. Loads any `.ply` or `.npy` file and publishes as `PointCloud2` at 1 Hz.

```bash
ros2 run scan_and_merge cloud_publisher --ros-args \
  -p file:=~/detect_output/bone_left_clean_20260328.ply \
  -p frame:=lbr_link_0 \
  -p topic:=/debug/cloud
```

#### `marker_pub`

Publishes end-effector pose from TF as `PoseStamped` on `/ee_marker_pos` at 50 Hz. Used by `ir_tracking` for calibration.

---

## ir_tracking

NDI Polaris Vega IR optical tracking package. Discovers the Vega camera on the network, tracks bone and end-effector tools, transforms poses into the KUKA base frame, and smooths them with an Extended Kalman Filter.

### Nodes

#### `ir_tracking_node`

- Discovers and connects to the Polaris Vega optical camera (multi-strategy: direct IP, ARP, mDNS, subnet scan).
- Tracks 3 ROM tools: femur, tibia, end-effector (ISTAR).
- One-shot calibration: uses `/ee_marker_pos` (KUKA FK + CAD offset) to solve `T_kuka_cam = T_kuka_rom_initial @ inv(T_cam_rom_initial)`.
- EKF smoothing per tracker (constant-velocity model, handles dropouts).

**Published Topics:**

| Topic | Type | Description |
|---|---|---|
| `/kuka_frame/pose_ee` | `PoseStamped` | End-effector tracker in KUKA base frame |
| `/kuka_frame/bone_pose_femur` | `PoseStamped` | Femur tracker in KUKA base frame |
| `/kuka_frame/bone_pose_tibia` | `PoseStamped` | Tibia tracker in KUKA base frame |
| `/kuka_frame/drift` | `Float64` | Translational drift between predicted and actual EE (m) |

### Supporting Modules

- **`pose_ekf.py`** — 12-state constant-velocity EKF for 6-DOF pose smoothing. Handles jitter, dropout bridging (predict-only when tracker lost), configurable noise parameters.
- **`vega_discover.py`** — multi-strategy network discovery for Polaris Vega (direct IP → ARP → mDNS → subnet scan). Includes NDI handshake verification.

---

## audio_capture

Real-time audio capture and offline speech recognition for voice commands during surgery.

### Nodes

#### `mic_publisher`

Captures audio from a specified microphone device (default: `"CMTECK"`) and publishes raw PCM as `Int16MultiArray` on `/audio`.

#### `speech_recognizer`

Subscribes to `/audio`, runs Vosk (Kaldi-based) offline speech recognition, publishes recognized text as `String` on `/speech`. Handles stereo→mono conversion and resampling (44100 Hz → 16 kHz).

---

## Usage

### 1. Teach Waypoints

```bash
ros2 launch scan_and_merge scan.launch.py scan_node:=true run_detect:=false
# Follow interactive prompts to teach waypoints
# Saves to ~/scan_output/waypoints.npy
```

### 2. Single-Orientation Scan + Registration

```bash
ros2 launch scan_and_merge scan.launch.py \
  run_detect:=true \
  multi_orientation:=false \
  weights:=~/scan_output/best.pt \
  load_waypoints:=~/scan_output/new_waypoints.npy \
  use_seg_mask:=true \
  confidence:=0.8
```

### 3. Multi-Orientation Calibration

```bash
ros2 launch scan_and_merge scan.launch.py \
  run_detect:=true \
  multi_orientation:=true \
  n_orientations:=4 \
  perpendicular_adjust:=true \
  run_bone_mover:=true
```

At each orientation:
1. The robot scans and registers bones.
2. Review in RViz, then publish a command on `/multi_orient/command`:
   - `"both"` — accept both bones
   - `"femur"` / `"tibia"` — accept one
   - `"neither"` — reject
3. Reposition bones, then send `"ready"` to start the next orientation (or `"solve"` to run the solver early).

After all orientations, the solver computes `T_ref_to_tracker` per bone with iterative outlier rejection.

### 4. Launch Arguments

| Argument | Default | Description |
|---|---|---|
| `run_detect` | `true` | Launch detect_and_merge_node |
| `scan_node` | `false` | Launch scan_and_merge_node (teach mode) |
| `weights` | `~/scan_output/best.pt` | YOLO weights path |
| `load_waypoints` | `~/scan_output/new_waypoints.npy` | Waypoints file |
| `confidence` | `0.8` | YOLO confidence threshold |
| `use_seg_mask` | `true` | Use segmentation masks |
| `multi_orientation` | `true` | Enable multi-orientation mode |
| `n_orientations` | `4` | Orientations to collect |
| `perpendicular_adjust` | `true` | Camera reorientation toward bone |
| `run_bone_mover` | `true` | Launch bone_cloud_mover for live tracking |
| `record_bag` | `false` | Record ROS bag |
| `rviz` | `true` | Launch RViz |

## Output Files

All outputs saved to `~/detect_output/`:

```
detect_output/
├── detections/                            # Annotated YOLO images per waypoint
│   └── det_wp_0_bone_left_*.png
├── clouds/                                # Per-instance camera-frame clouds
│   └── cloud_wp_0_bone_left_*.npy
├── bone_left_raw_*.ply                    # Raw merged (base frame)
├── bone_right_raw_*.ply
├── bone_left_clean_*.ply                  # Denoised
├── bone_right_clean_*.ply
├── detected_merged_clean_*.ply            # Both bones combined (denoised)
├── tibia_icp_T_ref2base_*.npy             # 4x4 ICP transform
├── femur_icp_T_ref2base_*.npy
├── tibia_ref_in_base_*.ply                # Reference model aligned to scan
├── femur_ref_in_base_*.ply
├── T_ref_to_tracker_tibia_*.npy           # Multi-orient calibration result
├── T_ref_to_tracker_femur_*.npy
├── multi_orient_tibia_*.npy               # Full orientation data
├── multi_orient_femur_*.npy
├── multi_orient_report_*.json             # Calibration summary + metrics
└── manifest.json                          # Run metadata
```

## Denoising Pipeline

Applied per-bone across all waypoint views:

1. **Cross-cloud consistency** — removes points seen in fewer than `cross_cloud_min_views` viewpoints (eliminates phantom geometry from depth noise / edge artifacts).
2. **Statistical Outlier Removal (SOR)** — removes points whose k-NN mean distance exceeds the global mean + σ threshold.
3. **Radius outlier removal** — removes points with too few neighbors within a radius.
4. **Voxel downsample** — uniform grid averaging.
5. **Laplacian smoothing** — moves each point toward its k-NN centroid (conservative, 50% weight).
6. *(Optional)* **Surface smoothing** — Poisson reconstruction + Taubin smoothing for smoother surfaces.

## ICP Registration

Aligns reference PLY mesh models onto the scanned bone clouds:

1. **Load reference** — if mesh (has triangles), sample 50k surface points; if point cloud, use directly.
2. **Auto-scale detection** — detects mm vs m mismatch and corrects.
3. **Coarse alignment** — hybrid method: exhaustive rotation search (24 axis-aligned + 8 PCA sign combos) scored with FPFH features.
4. **Fine alignment** — multi-scale point-to-plane ICP (8 mm → 4 mm → 2 mm) with Cauchy robust loss and divergence guard.
5. **Output** — reference model transformed into `lbr_link_0` frame, overlaid on scan.

## Multi-Orientation Calibration

Solves for the fixed transform `T_ref_to_tracker` between each bone's reference model frame and its IR tracker:

1. Collect N orientations (default 4). At each: average tracker poses, scan, ICP register.
2. Per-orientation estimate: `T_est_i = inv(T_tracker_i) @ T_icp_i`.
3. Rotation solved via weighted Wahba SVD; translation via weighted least squares.
4. Iterative outlier rejection (> 3× median residuals) until stable.
5. Validation: per-orientation residuals reported as translation (mm) and rotation (degrees).

## Dependencies

```bash
# ROS 2
sudo apt install ros-humble-moveit ros-humble-realsense2-camera

# Python
pip3 install ultralytics open3d scipy opencv-python-headless vosk sounddevice --break-system-packages
```

## Setup

```bash
cd ~/AUTOKnee-server

colcon build --packages-select scan_and_merge ir_tracking audio_capture
source install/setup.bash
```

## Hardware

- **Robot:** KUKA LBR Med 7 (7-DOF medical manipulator)
- **Camera:** Intel RealSense D4xx, wrist-mounted on `lbr_link_7`
- **Tracker:** NDI Polaris Vega optical tracker with ROM tools for femur, tibia, and end-effector
- **Microphone:** USB microphone (default: CMTECK) for voice commands
- **Workspace:** Two bone phantoms (tibia + femur) placed in front of the robot

## Troubleshooting

**CUDA/PyTorch mismatch** (`undefined symbol: __nvJitLinkComplete`):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

**ROS2 parameter type error** (`InvalidParameterTypeException` for `target_classes`):
Already handled with `ParameterDescriptor(type=PARAMETER_STRING)`.

**RealSense segfault** (exit code -11):
Relaunch — this is a known intermittent driver issue.

**ICP bad alignment:**
Try centroid coarse alignment: `-p icp_coarse_method:=centroid`. Check scale ratio in logs — if reference PLY is in mm and scan is in meters, auto-scaling should correct it but verify in logs.

**Polaris Vega not found:**
Run the discovery module standalone to diagnose: `python3 -m ir_tracking.vega_discover`. Ensure the Vega is on the same subnet or provide a direct IP.
