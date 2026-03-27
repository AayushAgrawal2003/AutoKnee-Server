#!/usr/bin/env python3
"""
Multi-Orientation ICP Refinement Solver

Given N bone orientations, each with:
  - T_icp_i:     4x4 transform mapping reference model -> base frame (from ICP)
  - T_tracker_i: 4x4 bone tracker pose in base frame (from IR tracking)
  - fitness_i:   ICP fitness score
  - rmse_i:      ICP inlier RMSE

Solves for T_ref_to_tracker, the fixed rigid offset between the reference
model frame and the tracker body frame, using weighted least squares on SE(3).

Pure numpy — no ROS or scipy dependencies.
"""

import numpy as np


def _invert_transform(T):
    """SE(3) inverse via R^T, t' = -R^T @ t."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def solve_ref_to_tracker(orientations):
    """
    Solve for T_ref_to_tracker given multiple orientation measurements.

    Args:
        orientations: list of dicts, each with keys:
            T_tracker: 4x4 numpy array, tracker pose in base frame
            T_icp:     4x4 numpy array, ICP transform (ref model -> base frame)
            fitness:   float, ICP fitness score (0 to 1)
            rmse:      float, ICP inlier RMSE (meters)

    Returns:
        dict with:
            T_ref_to_tracker: 4x4 numpy array
            R_optimal:        3x3 rotation matrix
            t_optimal:        3-vector translation
            weights:          normalized weight array
    """
    n = len(orientations)
    if n == 0:
        raise ValueError("Need at least 1 orientation")

    # Per-orientation estimates: T_est_i = inv(T_tracker_i) @ T_icp_i
    R_ests = []
    t_ests = []
    raw_weights = []

    for ori in orientations:
        T_tracker = ori["T_tracker"]
        T_icp = ori["T_icp"]
        fitness = ori["fitness"]
        rmse = ori["rmse"]

        T_est = _invert_transform(T_tracker) @ T_icp
        R_ests.append(T_est[:3, :3])
        t_ests.append(T_est[:3, 3])

        # Weight: fitness / rmse (higher fitness, lower rmse = better)
        w = fitness / max(rmse, 1e-9)
        raw_weights.append(w)

    weights = np.array(raw_weights)
    weights = weights / weights.sum()

    # ── Rotation: Wahba's problem via SVD ──
    M = np.zeros((3, 3))
    for i in range(n):
        M += weights[i] * R_ests[i]

    U, S, Vt = np.linalg.svd(M)
    # Ensure proper rotation (det = +1, not reflection)
    d = np.linalg.det(U @ Vt)
    R_optimal = U @ np.diag([1.0, 1.0, d]) @ Vt

    # ── Translation: weighted least squares ──
    # From: t_icp_i = R_tracker_i @ t_x + t_tracker_i
    # So:   t_x = R_tracker_i^T @ (t_icp_i - t_tracker_i)
    t_optimal = np.zeros(3)
    for i, ori in enumerate(orientations):
        R_tracker = ori["T_tracker"][:3, :3]
        t_tracker = ori["T_tracker"][:3, 3]
        t_icp = ori["T_icp"][:3, 3]
        t_optimal += weights[i] * (R_tracker.T @ (t_icp - t_tracker))

    # Assemble result
    T_ref_to_tracker = np.eye(4)
    T_ref_to_tracker[:3, :3] = R_optimal
    T_ref_to_tracker[:3, 3] = t_optimal

    return {
        "T_ref_to_tracker": T_ref_to_tracker,
        "R_optimal": R_optimal,
        "t_optimal": t_optimal,
        "weights": weights,
    }


def validate_result(T_ref_to_tracker, orientations):
    """
    Compute per-orientation residuals to assess calibration quality.

    Args:
        T_ref_to_tracker: 4x4 numpy array (the solved calibration)
        orientations:     list of dicts (same as solve_ref_to_tracker input)

    Returns:
        dict with:
            residuals:       list of dicts with trans_mm and rot_deg
            outlier_indices: list of int indices flagged as outliers
            mean_trans_mm:   float
            mean_rot_deg:    float
    """
    residuals = []

    for ori in orientations:
        T_tracker = ori["T_tracker"]
        T_icp = ori["T_icp"]

        # Predicted ICP result from calibration
        T_predicted = T_tracker @ T_ref_to_tracker
        # Error transform
        T_error = _invert_transform(T_predicted) @ T_icp

        # Translation residual (mm)
        trans_err = np.linalg.norm(T_error[:3, 3]) * 1000.0

        # Rotation residual (degrees)
        trace_val = np.trace(T_error[:3, :3])
        cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        rot_err = np.degrees(np.arccos(cos_angle))

        residuals.append({
            "trans_mm": trans_err,
            "rot_deg": rot_err,
        })

    trans_vals = np.array([r["trans_mm"] for r in residuals])
    rot_vals = np.array([r["rot_deg"] for r in residuals])

    # Outlier detection: > 3x median
    outlier_indices = []
    if len(residuals) >= 3:
        med_trans = np.median(trans_vals)
        med_rot = np.median(rot_vals)
        for i in range(len(residuals)):
            if (trans_vals[i] > 3.0 * max(med_trans, 0.1) or
                    rot_vals[i] > 3.0 * max(med_rot, 0.1)):
                outlier_indices.append(i)

    return {
        "residuals": residuals,
        "outlier_indices": outlier_indices,
        "mean_trans_mm": float(trans_vals.mean()),
        "mean_rot_deg": float(rot_vals.mean()),
    }


def solve_with_outlier_rejection(orientations, max_rounds=2):
    """
    Solve T_ref_to_tracker with iterative outlier rejection.

    Args:
        orientations: list of dicts (same as solve_ref_to_tracker input)
        max_rounds:   max outlier rejection iterations

    Returns:
        dict with:
            T_ref_to_tracker: 4x4 numpy array (final calibration)
            R_optimal:        3x3 rotation
            t_optimal:        3-vector translation
            weights:          normalized weights
            validation:       validation dict from validate_result
            rejected_indices: list of orientation indices rejected as outliers
            n_orientations_used: how many orientations contributed to final result
    """
    if len(orientations) < 1:
        raise ValueError("Need at least 1 orientation")

    active_indices = list(range(len(orientations)))
    all_rejected = []

    for round_idx in range(max_rounds):
        active_oris = [orientations[i] for i in active_indices]

        if len(active_oris) < 1:
            raise ValueError("All orientations rejected — cannot solve")

        result = solve_ref_to_tracker(active_oris)
        validation = validate_result(result["T_ref_to_tracker"], active_oris)

        if not validation["outlier_indices"] or len(active_oris) <= 2:
            break

        # Map outlier indices back to original indices
        new_rejected = [active_indices[i] for i in validation["outlier_indices"]]
        all_rejected.extend(new_rejected)
        active_indices = [i for i in active_indices if i not in new_rejected]

        if len(active_indices) < 1:
            raise ValueError("All orientations rejected — cannot solve")

    # Final solve with clean set
    active_oris = [orientations[i] for i in active_indices]
    result = solve_ref_to_tracker(active_oris)
    validation = validate_result(result["T_ref_to_tracker"], active_oris)

    return {
        "T_ref_to_tracker": result["T_ref_to_tracker"],
        "R_optimal": result["R_optimal"],
        "t_optimal": result["t_optimal"],
        "weights": result["weights"],
        "validation": validation,
        "rejected_indices": all_rejected,
        "n_orientations_used": len(active_indices),
    }
