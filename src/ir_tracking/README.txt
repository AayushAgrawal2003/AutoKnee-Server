IR Tracking Package — AutoKNEE (Team B)
========================================

Unified Polaris Vega IR tracking + camera-to-KUKA base frame transform.


QUICK START
-----------
    python3 scripts/ir_tracking_node.py
    python3 scripts/ir_tracking_node.py --hz 50 --ip 169.254.9.239


WHAT IT DOES
------------
Single ROS 2 node that handles the entire pipeline:

  1. Discovers the Polaris Vega camera on the network
  2. Connects and tracks 3 ROM tools with EKF smoothing:
       - Femur   (BBT-110017Rev1-FemurTracker-SPH, with +10.897mm X correction)
       - Tibia   (BBT-TrackerA-Gray_Polaris)
       - EE      (ISTAR-APPLE01)
  3. Subscribes to /istar_global (PoseStamped) — the ISTAR ROM origin
     in KUKA base frame, published upstream by the KUKA node
     (= KUKA EE FK + constant CAD offset applied externally)
  4. One-shot calibration: computes T_kuka_cam from first valid pair of
     Polaris EE reading vs /istar_global
  5. Continuously publishes all 3 tracker poses in KUKA base frame
  6. Publishes registration drift (live Polaris-transformed vs /istar_global)


FILE STRUCTURE
--------------
    ir_tracking/
    +-- README.txt               <- this file
    +-- roms/
    |   +-- BBT-110017Rev1-FemurTracker-SPH.rom
    |   +-- BBT-TrackerA-Gray_Polaris.rom
    |   +-- ISTAR-APPLE01.rom
    +-- scripts/
        +-- ir_tracking_node.py  <- MAIN ENTRY POINT (run this)
        +-- vega_discover.py     <- Polaris Vega network discovery
        +-- pose_ekf.py          <- 12-state EKF for 6-DOF pose smoothing
        +-- vega_multi_track.py  <- standalone multi-tracker CLI (debug)
        +-- vega_viz_3d.py       <- PyQt6 3D visualization UI (debug)


TOPICS
------
Subscribes:
    /istar_global              PoseStamped   ISTAR ROM origin in KUKA base frame

Publishes:
    /kuka_frame/pose_ee           PoseStamped   EE in KUKA base frame
    /kuka_frame/bone_pose_femur   PoseStamped   Femur in KUKA base frame
    /kuka_frame/bone_pose_tibia   PoseStamped   Tibia in KUKA base frame
    /kuka_frame/drift             Float64       Translational drift (meters)

All PoseStamped outputs: frame_id='kuka_base', positions in meters,
orientations as quaternions (x, y, z, w).

Drift is also logged to console every 2 seconds in mm.


CALIBRATION
-----------
Math (one-shot, frozen at startup):

    T_kuka_cam = T_kuka_rom_initial @ inv(T_cam_rom_initial)

    where:
      T_kuka_rom_initial = first /istar_global reading
      T_cam_rom_initial  = first Polaris EE tracker reading (in camera frame)

After calibration:

    T_kuka_tracker = T_kuka_cam @ T_cam_tracker    (for each tracker)

Drift (every frame):

    predicted = T_kuka_cam @ polaris_ee_live
    actual    = latest /istar_global
    drift     = ||predicted.position - actual.position||


/istar_global TOPIC
-------------------
This topic must be published by an upstream node (e.g., the KUKA controller
bridge). It represents the ISTAR ROM tool origin position in the KUKA robot
base frame. This is computed as:

    KUKA EE FK pose + constant CAD offset from top disk center to ROM origin

The CAD offset (in the mount STL coordinate frame) is:
    (-73.02, -1.79, -70.48) mm = (-0.07302, -0.00179, -0.07048) m

This offset must be applied in the KUKA EE's local frame before publishing.
The offset was derived by comparing:
  - Top disk center (bolt-hole fit): (-0.19, -0.43, 149.69) mm
  - ISTAR ROM origin (marker surface center): (-73.21, -2.22, 79.21) mm
NOTE: axis mapping from STL to KUKA EE frame must be verified on the robot.


DEPENDENCIES
------------
    pip install numpy sksurgerynditracker ndicapy
    # ROS 2 (Humble or later): rclpy, geometry_msgs, std_msgs

    For debug tools only:
    pip install pyqt6 pyqtgraph    (vega_viz_3d.py)


CLI ARGUMENTS
-------------
    --hz FLOAT    Poll/publish rate in Hz (default: 50)
    --ip IP       Known Vega IP to skip discovery (e.g. 169.254.9.239)
