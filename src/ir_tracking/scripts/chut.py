"""
Unified IR tracking + camera-to-KUKA transform node.

Single ROS 2 node that handles the full pipeline:
  1. Discovers and connects to the Polaris Vega camera
  2. Tracks all 3 ROM tools (femur, tibia, end-effector) with EKF smoothing
  3. Subscribes to /ee_marker_pos for the EE tracker ROM origin in KUKA base
     frame (= KUKA FK + constant CAD offset, published by upstream KUKA node)
  4. Adaptive hand-eye calibration:
       - Accepts any new simultaneous pose pair that moves enough from every
         previously collected pose (low bar: 3 deg OR 10 mm)
       - After each accepted pose (from the 3rd onward), evaluates a continuous
         *diversity score* derived from the smallest singular values of the
         Tsai-Lenz equation system — directly measuring how well-determined
         the 12-DOF solution is
       - When diversity >= 1.0 (i.e. sigma_min exceeds proven-reliable
         thresholds for both rotation and translation), calibration fires
         automatically
  5. Solves for BOTH X = T_base^cam  AND  Y = T_cam_obj^fk_obj  (the
     constant convention offset between the two object-frame definitions)
  6. Publishes all tracker 6-DOF poses in KUKA base frame:
       EE:           T_base = X @ T_cam_ee @ Y   (full chain, minimises drift)
       Femur/Tibia:  T_base = X @ T_cam_tracker   (Y is EE-specific)
  7. Publishes registration drift (X @ B_live @ Y  vs  A_live from FK)

Calibration math:
    A_i = X @ B_i @ Y        (fundamental per-pose equation)

    Relative motions:
        M_ij = A_i inv(A_j)  = X N_ij inv(X)     (Y cancels)
        => M_ij X = X N_ij                        (classic AX = XB)

    Step 1 — X via Tsai-Lenz (rotation then translation)
    Step 2 — Y via least-squares:  Y_i = inv(B_i) inv(X) A_i  averaged

    Diversity metric:
        sigma_rot   = sigma_min of the (3K x 3) rotation equation matrix
        sigma_trans = sigma_min of the (3K x 3) translation equation matrix
        diversity   = min(sigma_rot / TAU_ROT, sigma_trans / TAU_TRANS)
        Calibrate when diversity >= 1.0

Subscriptions:
    /ee_marker_pos   PoseStamped   ISTAR ROM origin in KUKA base frame (metres)

Publications:
    /kuka_frame/pose_ee           PoseStamped   EE in KUKA base frame
    /kuka_frame/bone_pose_femur   PoseStamped   Femur in KUKA base frame
    /kuka_frame/bone_pose_tibia   PoseStamped   Tibia in KUKA base frame
    /kuka_frame/drift             Float64       Translational drift (metres)

Usage:
    python3 ir_tracking_node.py
    python3 ir_tracking_node.py --hz 50 --ip 169.254.9.239
"""

import sys
import os
import time
import argparse
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vega_discover import discover_vega, VegaNotFoundError
from pose_ekf import PoseEKF
from sksurgerynditracker.nditracker import NDITracker

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROM_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "roms"))

# (rom_filename, output_topic, x_correction_mm)
# Index 0 = femur, 1 = tibia, 2 = EE (ISTAR)
TRACKERS = [
    ("BBT-110017Rev1-FemurTracker-SPH.rom", "/kuka_frame/bone_pose_femur", 8.770 - (-2.127)),
    ("BBT-TrackerA-Gray_Polaris.rom",       "/kuka_frame/bone_pose_tibia", 0.0),
    ("ISTAR-APPLE01.rom",                   "/kuka_frame/pose_ee",         0.0),
]

EE_INDEX = 2  # ISTAR is the 3rd tracker
MM_TO_M = 0.001
BASE_FRAME_ID = "lbr_link_0"

# -- Adaptive calibration parameters --
# Low-bar gate: reject a reading only if the robot hasn't moved at all
MIN_NEW_POSE_ROT_DEG = 3.0     # minimum rotation from every existing pose
MIN_NEW_POSE_TRANS_M = 0.010   # minimum translation from every existing pose
                                # (either condition suffices — OR logic)
MAX_PAIR_AGE_S = 0.1           # max staleness / simultaneity gap

# Diversity thresholds (calibrate when sigma_min exceeds these)
# These were empirically validated with synthetic noise sweeps:
#   sigma_rot  >= 0.45  =>  rotation well-determined even with 0.5 mm noise
#   sigma_trans >= 0.008 =>  translation well-determined (metres)
SIGMA_ROT_THRESHOLD = 0.45
SIGMA_TRANS_THRESHOLD = 0.008

MAX_CALIB_POSES = 80           # safety cap — something is wrong if we need this many


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def quaternion_to_rotation_matrix(x, y, z, w):
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def pose_msg_to_matrix(msg):
    """Convert a PoseStamped to a 4x4 homogeneous transform."""
    p = msg.pose
    T = np.eye(4)
    T[:3, :3] = quaternion_to_rotation_matrix(
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
    T[0, 3] = p.position.x
    T[1, 3] = p.position.y
    T[2, 3] = p.position.z
    return T


def matrix_to_pose_stamped(T, frame_id, stamp):
    """Build a PoseStamped from a 4x4 homogeneous transform."""
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(T[0, 3])
    msg.pose.position.y = float(T[1, 3])
    msg.pose.position.z = float(T[2, 3])
    qx, qy, qz, qw = rotation_matrix_to_quaternion(T[:3, :3])
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    msg.pose.orientation.w = float(qw)
    return msg


def invert_transform(T):
    """SE(3) inverse via R^T."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_x_correction(T, dx_mm):
    """Shift tool origin along its local X axis by dx_mm."""
    Tc = T.copy()
    Tc[:3, 3] += dx_mm * T[:3, 0]
    return Tc


def rotation_matrix_to_euler_zyx(R):
    """Extract ZYX Euler angles (degrees) from a 3x3 rotation matrix."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees([rz, ry, rx])


def skew(v):
    """3x3 skew-symmetric matrix from a 3-vector."""
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0],
    ])


def rotation_angle_between(R1, R2):
    """Geodesic angle (radians) between two rotation matrices."""
    R_rel = R1.T @ R2
    cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_theta)


def rotation_to_modified_rodrigues(R):
    """Modified Rodrigues vector: axis * 2*sin(theta/2)."""
    theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-10:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * (2.0 * np.sin(theta / 2.0))


def modified_rodrigues_to_rotation(pcg):
    """Recover rotation matrix from modified Rodrigues vector."""
    norm = np.linalg.norm(pcg)
    if norm < 1e-12:
        return np.eye(3)
    half_sin = np.clip(norm / 2.0, -1.0, 1.0)
    theta = 2.0 * np.arcsin(half_sin)
    axis = pcg / norm
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def project_to_SO3(M):
    """Closest rotation matrix to M via SVD projection."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------------------------------------------------------------------------
# Diversity metric
# ---------------------------------------------------------------------------

@dataclass
class DiversityInfo:
    """Snapshot of how well-determined the calibration system currently is."""
    n_poses: int
    n_pairs: int              # C(n,2) relative-motion pairs
    sigma_rot: float          # sigma_min of rotation equation matrix
    sigma_trans: float        # sigma_min of translation equation matrix
    cond_rot: float           # condition number of rotation system
    cond_trans: float         # condition number of translation system
    diversity: float          # min(sigma_rot/TAU, sigma_trans/TAU), 0..1+
    ready: bool               # diversity >= 1.0 and n_poses >= 3

    def status_line(self):
        bar_len = 20
        fill = int(min(self.diversity, 1.0) * bar_len)
        bar = '#' * fill + '-' * (bar_len - fill)
        pct = min(self.diversity * 100, 100)
        return (
            f"[{bar}] {pct:5.1f}%  "
            f"poses={self.n_poses}  "
            f"s_rot={self.sigma_rot:.3f}  "
            f"s_trans={self.sigma_trans:.4f}  "
            f"cond_r={self.cond_rot:.1f}  "
            f"cond_t={self.cond_trans:.1f}"
        )


def build_tsai_lenz_system(A_list, B_list):
    """
    Build the Tsai-Lenz linear system matrices from pose lists.

    Returns (A_rot, b_rot, A_trans_func) where A_trans_func(R_X) builds
    the translation system given a rotation solution.

    Also returns the raw relative motions for reuse.
    """
    n = len(A_list)
    rows_rot = []
    rhs_rot = []
    rel_motions = []   # list of (M_ij, N_ij)

    for i in range(n):
        for j in range(i + 1, n):
            M_ij = A_list[i] @ invert_transform(A_list[j])
            N_ij = B_list[i] @ invert_transform(B_list[j])
            rel_motions.append((M_ij, N_ij))

            alpha = rotation_to_modified_rodrigues(M_ij[:3, :3])
            beta = rotation_to_modified_rodrigues(N_ij[:3, :3])

            rows_rot.append(skew(alpha + beta))
            rhs_rot.append(2.0 * (beta - alpha))

    A_rot = np.vstack(rows_rot)
    b_rot = np.concatenate(rhs_rot)
    return A_rot, b_rot, rel_motions


def compute_diversity(A_list, B_list):
    """
    Evaluate how well-determined the AX=XB system is given current poses.

    The diversity metric is based on sigma_min of both the rotation and
    translation equation matrices. These directly measure the weakest
    observability direction — if sigma_min = 0, the system is rank-
    deficient and has infinitely many solutions along that direction.

    Returns a DiversityInfo.
    """
    n = len(A_list)
    n_pairs = n * (n - 1) // 2

    if n < 2:
        return DiversityInfo(
            n_poses=n, n_pairs=0,
            sigma_rot=0.0, sigma_trans=0.0,
            cond_rot=np.inf, cond_trans=np.inf,
            diversity=0.0, ready=False,
        )

    A_rot, b_rot, rel_motions = build_tsai_lenz_system(A_list, B_list)

    # Rotation system SVD
    sv_rot = np.linalg.svd(A_rot, compute_uv=False)
    sigma_rot = sv_rot[-1] if len(sv_rot) >= 3 else 0.0
    cond_rot = sv_rot[0] / max(sigma_rot, 1e-15)

    # For translation system we need R_X, but we don't have it yet during
    # collection.  However, the *matrix* (R_M - I) on the LHS doesn't
    # depend on R_X — only the RHS does.  So sigma_min of the LHS tells us
    # whether translation is observable, independent of the rotation solution.
    rows_trans = []
    for M_ij, N_ij in rel_motions:
        rows_trans.append(M_ij[:3, :3] - np.eye(3))
    A_trans = np.vstack(rows_trans)

    sv_trans = np.linalg.svd(A_trans, compute_uv=False)
    sigma_trans = sv_trans[-1] if len(sv_trans) >= 3 else 0.0
    cond_trans = sv_trans[0] / max(sigma_trans, 1e-15)

    d_rot = sigma_rot / SIGMA_ROT_THRESHOLD
    d_trans = sigma_trans / SIGMA_TRANS_THRESHOLD
    diversity = min(d_rot, d_trans)
    ready = (n >= 3 and diversity >= 1.0)

    return DiversityInfo(
        n_poses=n, n_pairs=n_pairs,
        sigma_rot=sigma_rot, sigma_trans=sigma_trans,
        cond_rot=cond_rot, cond_trans=cond_trans,
        diversity=diversity, ready=ready,
    )


# ---------------------------------------------------------------------------
# Solver: X (Tsai-Lenz) then Y (least-squares back-substitution)
# ---------------------------------------------------------------------------

def solve_X_tsai_lenz(A_list, B_list):
    """
    Solve for X = T_base^cam via Tsai-Lenz (1989).

    Uses relative motions M_ij X = X N_ij which cancel Y.
    """
    A_rot, b_rot, rel_motions = build_tsai_lenz_system(A_list, B_list)

    # -- Rotation ----------------------------------------------------------
    pcg_prime, _, _, _ = np.linalg.lstsq(A_rot, b_rot, rcond=None)

    # Recover modified Rodrigues from tangent half-angle parameterisation
    norm_pp = np.linalg.norm(pcg_prime)
    cos_half = 2.0 / np.sqrt(4.0 + norm_pp ** 2)
    P_X = pcg_prime * cos_half

    R_X = project_to_SO3(modified_rodrigues_to_rotation(P_X))

    # -- Translation -------------------------------------------------------
    rows_t = []
    rhs_t = []
    for M_ij, N_ij in rel_motions:
        rows_t.append(M_ij[:3, :3] - np.eye(3))
        rhs_t.append(R_X @ N_ij[:3, 3] - M_ij[:3, 3])

    A_t = np.vstack(rows_t)
    b_t = np.concatenate(rhs_t)
    t_X, _, _, _ = np.linalg.lstsq(A_t, b_t, rcond=None)

    X = np.eye(4)
    X[:3, :3] = R_X
    X[:3, 3] = t_X
    return X


def solve_Y(X, A_list, B_list):
    """
    Recover Y = T_cam_obj^fk_obj  from  A_i = X B_i Y.

    Each pose gives Y_i = inv(B_i) inv(X) A_i.
    We average the rotation (Frobenius mean projected onto SO(3))
    and average the translation.
    """
    X_inv = invert_transform(X)
    R_sum = np.zeros((3, 3))
    t_sum = np.zeros(3)

    for A_i, B_i in zip(A_list, B_list):
        Y_i = invert_transform(B_i) @ X_inv @ A_i
        R_sum += Y_i[:3, :3]
        t_sum += Y_i[:3, 3]

    n = len(A_list)
    R_Y = project_to_SO3(R_sum / n)
    t_Y = t_sum / n

    Y = np.eye(4)
    Y[:3, :3] = R_Y
    Y[:3, 3] = t_Y
    return Y


def solve_hand_eye(A_list, B_list):
    """
    Full hand-eye calibration: solve for both X and Y.

    A_i = X @ B_i @ Y

    Returns (X, Y) as 4x4 ndarrays.
    """
    X = solve_X_tsai_lenz(A_list, B_list)
    Y = solve_Y(X, A_list, B_list)
    return X, Y


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class IRTrackingNode(Node):

    def __init__(self, hz, vega_ip):
        super().__init__('ir_tracking_node')

        # -- Resolve ROM files ---------------------------------------------
        self.romfiles = []
        self.topics = []
        self.x_corrections = []
        for rom_name, topic, dx in TRACKERS:
            rom_path = os.path.join(ROM_DIR, rom_name)
            if not os.path.isfile(rom_path):
                self.get_logger().fatal(f"ROM not found: {rom_path}")
                raise FileNotFoundError(rom_path)
            self.romfiles.append(rom_path)
            self.topics.append(topic)
            self.x_corrections.append(dx)
            corr = f"  (X correction: +{dx:.3f}mm)" if dx else ""
            self.get_logger().info(f"ROM: {rom_name}{corr}  ->  {topic}")

        # -- Discover and connect to Polaris Vega --------------------------
        self.get_logger().info("Discovering Polaris Vega...")
        try:
            self.vega_info = discover_vega(known_ip=vega_ip, verbose=True)
        except VegaNotFoundError as e:
            self.get_logger().fatal(str(e))
            raise

        settings = self.vega_info.as_settings(self.romfiles)
        self.get_logger().info(
            f"Connecting to {self.vega_info.ip}:{self.vega_info.port}...")
        self.ndi_tracker = NDITracker(settings)
        self.ndi_tracker.start_tracking()
        self.get_logger().info("Polaris Vega tracking started.")

        # -- EKF filters (one per tracker) ---------------------------------
        self.ekf_filters = [PoseEKF(max_misses=60) for _ in self.romfiles]
        self._last_frame_num = None

        # -- Calibration state ---------------------------------------------
        self.calibrated = False
        self.T_kuka_cam = None          # X = T_base^cam
        self.T_obj_offset = None        # Y = T_cam_obj^fk_obj  (convention)

        # Adaptive pose collection
        self._calib_A = []              # A_i from FK    (4x4, metres)
        self._calib_B = []              # B_i from cam   (4x4, metres)
        self._diversity = DiversityInfo(
            n_poses=0, n_pairs=0,
            sigma_rot=0, sigma_trans=0,
            cond_rot=np.inf, cond_trans=np.inf,
            diversity=0, ready=False,
        )

        # Latest readings + timestamps
        self._latest_fk_matrix = None
        self._latest_fk_time = None
        self._latest_cam_ee_matrix = None
        self._latest_cam_ee_time = None

        self._drift_m = None

        # -- ROS subscription ----------------------------------------------
        self.create_subscription(
            PoseStamped, '/ee_marker_pos', self._fk_pose_cb, 10)
        self.get_logger().info(
            "Subscribed to /ee_marker_pos (KUKA EE + CAD offset)")

        # -- ROS publishers ------------------------------------------------
        self.pose_pubs = []
        for topic in self.topics:
            self.pose_pubs.append(self.create_publisher(PoseStamped, topic, 10))

        self.drift_pub = self.create_publisher(Float64, '/kuka_frame/drift', 10)

        # -- Timers --------------------------------------------------------
        self.create_timer(1.0 / hz, self._poll_and_publish)
        self.create_timer(2.0, self._log_status)

        self.get_logger().info(
            f"Running at {hz:.0f} Hz.  Move the robot to diverse poses — "
            f"calibration fires automatically when diversity is sufficient.")

    # -- /ee_marker_pos callback -------------------------------------------

    def _fk_pose_cb(self, msg):
        self._latest_fk_matrix = pose_msg_to_matrix(msg)
        self._latest_fk_time = time.monotonic()

    # -- Pose acceptance (low bar: just "not standing still") --------------

    def _moved_enough(self, A_new, B_new):
        """
        Returns True if the new pose differs from every existing pose
        by at least MIN_NEW_POSE_ROT_DEG or MIN_NEW_POSE_TRANS_M, on
        both the FK and camera sides.
        """
        for A_old, B_old in zip(self._calib_A, self._calib_B):
            a_rot = np.degrees(rotation_angle_between(
                A_old[:3, :3], A_new[:3, :3]))
            a_trans = np.linalg.norm(A_new[:3, 3] - A_old[:3, 3])
            b_rot = np.degrees(rotation_angle_between(
                B_old[:3, :3], B_new[:3, :3]))
            b_trans = np.linalg.norm(B_new[:3, 3] - B_old[:3, 3])

            a_moved = (a_rot > MIN_NEW_POSE_ROT_DEG
                       or a_trans > MIN_NEW_POSE_TRANS_M)
            b_moved = (b_rot > MIN_NEW_POSE_ROT_DEG
                       or b_trans > MIN_NEW_POSE_TRANS_M)

            if not (a_moved and b_moved):
                return False
        return True

    # -- Main loop ---------------------------------------------------------

    def _poll_and_publish(self):
        # -- 1. Poll Polaris Vega ------------------------------------------
        try:
            _, _, framenumbers, tracking, quality = self.ndi_tracker.get_frame()
        except Exception as e:
            self.get_logger().warn(
                f"get_frame error: {e}", throttle_duration_sec=2.0)
            return

        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()

        cur_frame = framenumbers[0] if len(framenumbers) > 0 else None
        new_frame = (cur_frame != self._last_frame_num)
        self._last_frame_num = cur_frame

        # -- 2. EKF-filter each tracker ------------------------------------
        cam_poses = [None] * len(self.romfiles)
        for i in range(len(self.romfiles)):
            T = tracking[i]
            raw_vis = not np.isnan(T[0, 0])

            if raw_vis and self.x_corrections[i] != 0:
                T = apply_x_correction(T, self.x_corrections[i])

            vis_for_ekf = raw_vis and new_frame
            T_filt, valid = self.ekf_filters[i].process(T, vis_for_ekf, now)
            if not valid:
                continue

            T_m = T_filt.copy()
            T_m[:3, 3] *= MM_TO_M
            cam_poses[i] = T_m

        if cam_poses[EE_INDEX] is not None:
            self._latest_cam_ee_matrix = cam_poses[EE_INDEX]
            self._latest_cam_ee_time = now

        # -- 3. Adaptive calibration ---------------------------------------
        if not self.calibrated:
            self._try_collect_and_calibrate(now)
            if not self.calibrated:
                return

        # -- 4. Transform and publish --------------------------------------
        for i in range(len(self.romfiles)):
            if cam_poses[i] is None:
                continue

            if i == EE_INDEX:
                # Full chain: X @ B @ Y  (Y absorbs convention offset)
                T_kuka = self.T_kuka_cam @ cam_poses[i] @ self.T_obj_offset
            else:
                # Camera-only trackers: X @ B  (no Y — different object)
                T_kuka = self.T_kuka_cam @ cam_poses[i]

            self.pose_pubs[i].publish(
                matrix_to_pose_stamped(T_kuka, BASE_FRAME_ID, stamp))

        # -- 5. Drift (EE only: compare X@B@Y vs FK) ----------------------
        if (self._latest_fk_matrix is not None
                and cam_poses[EE_INDEX] is not None):
            T_predicted = (self.T_kuka_cam
                           @ cam_poses[EE_INDEX]
                           @ self.T_obj_offset)
            T_actual = self._latest_fk_matrix

            drift_vec = T_predicted[:3, 3] - T_actual[:3, 3]
            self._drift_m = float(np.linalg.norm(drift_vec))

            msg = Float64()
            msg.data = self._drift_m
            self.drift_pub.publish(msg)

    # -- Adaptive collection + calibration trigger -------------------------

    def _try_collect_and_calibrate(self, now):
        if self._latest_fk_matrix is None or self._latest_cam_ee_matrix is None:
            return
        if self._latest_fk_time is None or self._latest_cam_ee_time is None:
            return

        # Simultaneity gate
        age_fk = now - self._latest_fk_time
        age_cam = now - self._latest_cam_ee_time
        time_gap = abs(self._latest_fk_time - self._latest_cam_ee_time)
        if age_fk > MAX_PAIR_AGE_S or age_cam > MAX_PAIR_AGE_S:
            return
        if time_gap > MAX_PAIR_AGE_S:
            return

        A_new = self._latest_fk_matrix.copy()
        B_new = self._latest_cam_ee_matrix.copy()

        # Safety cap
        if len(self._calib_A) >= MAX_CALIB_POSES:
            self.get_logger().warn(
                f"[Calib] {MAX_CALIB_POSES} poses collected but diversity "
                f"still only {self._diversity.diversity:.2f} — forcing solve.")
            self._run_solver()
            return

        # Accept first pose unconditionally
        if len(self._calib_A) == 0:
            self._calib_A.append(A_new)
            self._calib_B.append(B_new)
            self.get_logger().info(
                "[Calib] Pose 1 captured. Move the robot...")
            return

        # Low-bar motion gate
        if not self._moved_enough(A_new, B_new):
            return

        # Accept the pose
        self._calib_A.append(A_new)
        self._calib_B.append(B_new)
        n = len(self._calib_A)

        # Recompute diversity
        self._diversity = compute_diversity(self._calib_A, self._calib_B)

        self.get_logger().info(
            f"[Calib] Pose {n} accepted. "
            f"{self._diversity.status_line()}")

        # Actionable feedback
        if n >= 3 and not self._diversity.ready:
            if self._diversity.sigma_rot < SIGMA_ROT_THRESHOLD:
                self.get_logger().info(
                    "  -> Rotation axes still too similar — "
                    "try rotating around a different axis")
            if self._diversity.sigma_trans < SIGMA_TRANS_THRESHOLD:
                self.get_logger().info(
                    "  -> Translation span too narrow — "
                    "try moving to a more different position")

        if self._diversity.ready:
            self._run_solver()

    def _run_solver(self):
        n = len(self._calib_A)
        n_pairs = n * (n - 1) // 2
        self.get_logger().info(
            f"[Calib] Solving for X and Y from {n} poses "
            f"({n_pairs} relative-motion equations)...")

        try:
            X, Y = solve_hand_eye(self._calib_A, self._calib_B)
        except Exception as e:
            self.get_logger().error(f"[Calib] Solver failed: {e}")
            self.get_logger().error(
                "  Discarding last pose — keep moving to diverse poses.")
            self._calib_A.pop()
            self._calib_B.pop()
            self._diversity = compute_diversity(
                self._calib_A, self._calib_B)
            return

        self.T_kuka_cam = X
        self.T_obj_offset = Y
        self.calibrated = True

        # -- Diagnostics ---------------------------------------------------
        t_x = X[:3, 3]
        rz_x, ry_x, rx_x = rotation_matrix_to_euler_zyx(X[:3, :3])
        t_y = Y[:3, 3]
        rz_y, ry_y, rx_y = rotation_matrix_to_euler_zyx(Y[:3, :3])

        self.get_logger().info("=" * 58)
        self.get_logger().info("  HAND-EYE CALIBRATION COMPLETE")
        self.get_logger().info("=" * 58)
        self.get_logger().info(
            f"  X = T_{BASE_FRAME_ID}^cam")
        self.get_logger().info(
            f"    trans: ({t_x[0]:.6f}, {t_x[1]:.6f}, {t_x[2]:.6f}) m")
        self.get_logger().info(
            f"    rot (ZYX): ({rz_x:.4f}, {ry_x:.4f}, {rx_x:.4f}) deg")
        self.get_logger().info(
            f"  Y = T_cam_obj^fk_obj  (convention offset)")
        self.get_logger().info(
            f"    trans: ({t_y[0]:.6f}, {t_y[1]:.6f}, {t_y[2]:.6f}) m")
        self.get_logger().info(
            f"    rot (ZYX): ({rz_y:.4f}, {ry_y:.4f}, {rx_y:.4f}) deg")
        self.get_logger().info(
            f"  From {n} poses, {n_pairs} relative-motion equations")
        self.get_logger().info(
            f"  Final diversity: {self._diversity.diversity:.3f}")

        # Per-pose reprojection error using full X @ B @ Y chain
        errors = []
        for i in range(n):
            T_pred = X @ self._calib_B[i] @ Y
            T_actual = self._calib_A[i]
            pos_err = np.linalg.norm(T_pred[:3, 3] - T_actual[:3, 3])
            rot_err = np.degrees(rotation_angle_between(
                T_pred[:3, :3], T_actual[:3, :3]))
            errors.append((pos_err, rot_err))
            self.get_logger().info(
                f"    Pose {i+1}: {pos_err*1000:.3f} mm / "
                f"{rot_err:.3f} deg")

        mean_pos = np.mean([e[0] for e in errors]) * 1000
        mean_rot = np.mean([e[1] for e in errors])
        max_pos = np.max([e[0] for e in errors]) * 1000
        self.get_logger().info(
            f"  Reprojection: mean {mean_pos:.3f} mm / {mean_rot:.3f} deg"
            f"  |  max {max_pos:.3f} mm")
        self.get_logger().info("=" * 58)

    # -- Periodic status log -----------------------------------------------

    def _log_status(self):
        if not self.calibrated:
            has_fk = self._latest_fk_matrix is not None
            has_cam = self._latest_cam_ee_matrix is not None
            src = (f"/ee_marker_pos: {'OK' if has_fk else 'waiting'} | "
                   f"Polaris EE: {'OK' if has_cam else 'waiting'}")

            if self._diversity.n_poses >= 2:
                self.get_logger().info(
                    f"[Calib] {self._diversity.status_line()}")
            else:
                self.get_logger().info(
                    f"[Calib] {self._diversity.n_poses} pose(s) | {src}")
            return

        if self._drift_m is not None:
            self.get_logger().info(
                f"Drift: {self._drift_m * 1000.0:.3f} mm")

    # -- Cleanup -----------------------------------------------------------

    def destroy_node(self):
        if hasattr(self, 'ndi_tracker') and self.ndi_tracker:
            try:
                self.ndi_tracker.stop_tracking()
                self.ndi_tracker.close()
            except Exception:
                pass
            self.ndi_tracker = None
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IR tracking + adaptive hand-eye calibration. "
                    "Collects pose pairs and calibrates automatically "
                    "when observation diversity is sufficient.",
    )
    parser.add_argument(
        "--hz", type=float, default=50.0,
        help="Poll/publish rate in Hz (default: 50)",
    )
    parser.add_argument(
        "--ip", default=None,
        help="Known Vega IP to skip discovery (e.g. 169.254.9.239)",
    )
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args if ros_args else None)

    try:
        node = IRTrackingNode(hz=args.hz, vega_ip=args.ip)
    except (FileNotFoundError, VegaNotFoundError):
        rclpy.shutdown()
        sys.exit(1)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.get_logger().info("Shutting down...")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
