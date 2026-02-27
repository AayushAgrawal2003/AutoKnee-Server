# NEEpolean — YOLO-Guided Bone Scanning & Registration

ROS 2 (Humble) package for autonomous bone scanning using a **KUKA LBR Med 7** manipulator with a wrist-mounted **Intel RealSense** depth camera. The system moves the robot through taught waypoints, detects bones with a YOLO segmentation model, builds per-bone point clouds, denoises them, and registers them against reference CAD/mesh models using ICP.

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
  Cross-view consistency filter → SOR → radius filter → voxel downsample → smooth
        ↓
  ICP registration (FPFH coarse + multi-scale point-to-plane)
        ↓
  Publish aligned reference models as PointCloud2 in RViz
```

## Package Structure

```
scan_and_merge/
├── scan_and_merge/
│   ├── scan_and_merge_node.py      # Waypoint teach/replay + raw capture
│   ├── detect_and_merge_node.py    # YOLO detection + filtered cloud merge + registration
│   ├── cloud_denoise.py            # Denoising & smoothing utilities
│   ├── icp_registration.py         # ICP alignment utilities
│   ├── cloud_publisher.py          # Debug: publish any PLY/NPY to RViz
│   └── replay_waypoints.py         # Visualize saved waypoints
├── launch/
│   └── scan.launch.py
├── resource/
│   ├── tibia_shell.ply             # Reference tibia model
│   └── femur_shell.ply             # Reference femur model
├── config/
│   └── main.rviz
├── setup.py
├── package.xml
└── .gitignore
```

## Nodes

### `scan_and_merge_node`

Teaches and replays joint-space waypoints on the KUKA arm. At each waypoint, captures raw RGB, depth, and point cloud data.

**Mode:** `teach` (interactive) or `replay` (from saved waypoints).

### `detect_and_merge_node`

Main pipeline node. Replays saved waypoints, runs YOLO detection at each pose, and produces denoised per-bone point clouds registered to reference models.

**Subscribed Topics:**
| Topic | Type | Description |
|---|---|---|
| `/lbr/joint_states` | `JointState` | Current joint positions |
| `/camera_arm/camera/color/image_rect_raw` | `Image` | RGB image |
| `/camera_arm/camera/aligned_depth_to_color/image_raw` | `Image` | Aligned depth (uint16, mm) |
| `/camera_arm/camera/aligned_depth_to_color/camera_info` | `CameraInfo` | Camera intrinsics |

**Published Topics:**
| Topic | Type | Description |
|---|---|---|
| `/registered/tibia` | `PointCloud2` | Denoised tibia scan in base frame |
| `/registered/femur` | `PointCloud2` | Denoised femur scan in base frame |
| `/reference/tibia` | `PointCloud2` | Reference tibia model aligned to scan |
| `/reference/femur` | `PointCloud2` | Reference femur model aligned to scan |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `load_waypoints` | `""` | Path to `waypoints.npy` |
| `weights` | `""` | Path to YOLO `.pt` weights |
| `target_classes` | `""` | Comma-sep class IDs or JSON list (empty = all) |
| `confidence` | `0.5` | YOLO detection threshold |
| `use_seg_mask` | `False` | Use segmentation masks (True) or bounding boxes (False) |
| `velocity_scaling` | `0.1` | Robot motion speed factor |
| `max_depth_m` | `1.0` | Maximum depth to keep (meters) |
| `settle_time` | `1.5` | Wait time after reaching waypoint (seconds) |
| `denoise` | `True` | Enable denoising pipeline |
| `cross_cloud_dist` | `0.005` | Cross-view support distance (m) |
| `cross_cloud_min_views` | `2` | Min viewpoints for a point to survive |
| `sor_k` | `20` | Statistical outlier removal: neighbor count |
| `sor_std` | `2.0` | Statistical outlier removal: std ratio |
| `radius_filter` | `0.01` | Radius outlier removal: search radius (m) |
| `radius_min_neighbors` | `5` | Radius outlier removal: min neighbor count |
| `denoise_voxel_size` | `0.001` | Voxel downsample size (m) |
| `smooth_k` | `10` | Laplacian smoothing: neighbor count |
| `smooth_iterations` | `1` | Laplacian smoothing: passes |
| `register` | `True` | Enable ICP registration |
| `tibia_reference` | auto (package resource) | Path to tibia reference PLY |
| `femur_reference` | auto (package resource) | Path to femur reference PLY |
| `icp_coarse_method` | `"fpfh"` | Coarse alignment: `"fpfh"` or `"centroid"` |
| `icp_voxel_size` | `0.002` | ICP voxel size (m) |

### `cloud_publisher`

Debug utility. Loads any `.ply` or `.npy` point cloud and publishes it as `PointCloud2` at 1 Hz.

```bash
ros2 run scan_and_merge cloud_publisher --ros-args \
  -p file:=~/detect_output/bone_left_raw_20260227.ply \
  -p frame:=lbr_link_0 \
  -p topic:=/debug/cloud
```

## Usage

### 1. Teach Waypoints

```bash
ros2 launch scan_and_merge scan.launch.py scan_node:=true
# Follow interactive prompts to teach 5 waypoints
# Saves to ~/scan_output/waypoints.npy
```

### 2. Run Detection + Registration Pipeline

```bash
ros2 launch scan_and_merge scan.launch.py \
  run_detect:=true \
  scan_node:=false \
  weights:=~/weights/best.pt \
  load_waypoints:=~/scan_output/waypoints.npy \
  use_seg_mask:=true \
  confidence:=0.8
```

### 3. Class Filtering

```bash
# Only detect classes 0 and 1
ros2 launch scan_and_merge scan.launch.py \
  run_detect:=true scan_node:=false \
  weights:=~/weights/best.pt \
  load_waypoints:=~/scan_output/waypoints.npy \
  target_classes:="0,1"
```

## Output Files

All outputs are saved to `~/detect_output/`:

```
detect_output/
├── detections/              # Annotated YOLO images per waypoint
│   ├── det_wp_0_*.png
│   └── ...
├── clouds/                  # Per-instance camera-frame clouds
│   ├── cloud_wp_0_bone_left_*.npy
│   └── ...
├── bone_left_raw_*.ply      # Raw merged tibia (base frame)
├── bone_right_raw_*.ply     # Raw merged femur (base frame)
├── bone_left_clean_*.ply    # Denoised tibia
├── bone_right_clean_*.ply   # Denoised femur
├── tibia_ref_in_base_*.ply  # Reference model aligned to scan
├── femur_ref_in_base_*.ply  # Reference model aligned to scan
├── tibia_icp_T_ref2base_*.npy  # 4x4 ICP transform
├── femur_icp_T_ref2base_*.npy
├── detected_merged_clean_*.ply # Both bones combined
└── manifest.json            # Run metadata
```

## Denoising Pipeline

Applied per-bone across all waypoint views:

1. **Cross-cloud consistency** — removes points only seen in one viewpoint (phantom geometry from depth noise or edge artifacts)
2. **Statistical Outlier Removal (SOR)** — removes points whose k-NN mean distance exceeds the global mean + σ threshold
3. **Radius outlier removal** — removes points with too few neighbors within a radius
4. **Voxel downsample** — uniform grid averaging
5. **Laplacian smoothing** — moves each point toward its k-NN centroid (conservative, 50% weight)

## ICP Registration

Aligns reference PLY mesh models onto the scanned bone clouds:

1. **Load reference** — if mesh (has triangles), sample 50k surface points; if point cloud, use directly
2. **Coarse alignment** — FPFH feature extraction + RANSAC matching (or centroid fallback)
3. **Fine alignment** — multi-scale point-to-plane ICP at 3 resolution levels (8mm → 4mm → 2mm)
4. **Output** — reference model transformed into `lbr_link_0` frame, overlaid on scan

## Dependencies

```bash
# ROS 2
sudo apt install ros-humble-moveit ros-humble-realsense2-camera

# Python
pip3 install ultralytics open3d scipy opencv-python-headless --break-system-packages
```

## Setup

```bash
cd ~/lbr-stack

# Add entry points in scan_and_merge/setup.py:
#   "detect_and_merge_node = scan_and_merge.detect_and_merge_node:main",
#   "cloud_publisher = scan_and_merge.cloud_publisher:main",

# Place reference PLYs in scan_and_merge/resource/
# Add to setup.py data_files if not already:
#   (os.path.join('share', package_name, 'resource'), glob('resource/*.ply')),

colcon build --packages-select scan_and_merge
source install/setup.bash
```

## Hardware

- **Robot:** KUKA LBR Med 7 (7-DOF medical manipulator)
- **Camera:** Intel RealSense D4xx, wrist-mounted on lbr_link_7
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
Try centroid coarse alignment instead of FPFH: `-p icp_coarse_method:=centroid`. Check scale ratio in logs — if reference PLY is in mm and scan is in meters, ICP will fail.
