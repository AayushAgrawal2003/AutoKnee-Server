#!/usr/bin/env python3
"""
Point Cloud Denoising & Smoothing Utilities

Methods:
  1. Statistical Outlier Removal (SOR):
     For each point, compute mean distance to k nearest neighbors.
     Remove points where mean_dist > global_mean + std_ratio * global_std.
     → Removes isolated noisy points far from local structure.

  2. Radius Outlier Removal:
     For each point, count neighbors within a radius.
     Remove points with fewer than min_neighbors.
     → Removes sparse scattered points.

  3. Cross-Cloud Consistency Filter:
     After merging multiple views, points that were only seen from
     one viewpoint and have no nearby support from other viewpoints
     are likely noise. For each point from cloud_i, check if any
     point from cloud_j (j != i) is within a distance threshold.
     Remove unsupported points.
     → Removes phantom points that appear in only one view.

  4. Voxel Smoothing:
     Downsample to voxels, then optionally smooth normals.
     → Produces cleaner surface.

Usage:
  from cloud_denoise import denoise_pipeline

  # Full pipeline on a merged cloud
  clean_pts, clean_cols = denoise_pipeline(points, colors)

  # Or use individual functions
  from cloud_denoise import statistical_outlier_removal, radius_outlier_removal
"""

import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    from scipy.spatial import KDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ──────────────────────────────────────────────────────────────────────
# 1. Statistical Outlier Removal
# ──────────────────────────────────────────────────────────────────────
def statistical_outlier_removal(points, colors=None, k=20, std_ratio=2.0):
    """
    Remove points whose mean distance to k nearest neighbors
    exceeds (global_mean + std_ratio * global_std).

    Args:
        points:    (N, 3) array
        colors:    (N, 3) array or None
        k:         number of nearest neighbors to consider
        std_ratio: multiplier on std deviation for threshold

    Returns:
        filtered_points, filtered_colors, inlier_mask
    """
    if HAS_OPEN3D and len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        cl, inlier_idx = pcd.remove_statistical_outlier(
            nb_neighbors=k, std_ratio=std_ratio
        )

        mask = np.zeros(len(points), dtype=bool)
        mask[inlier_idx] = True

        return points[mask], (colors[mask] if colors is not None else None), mask

    elif HAS_SCIPY and len(points) > 0:
        # Fallback: scipy KDTree
        tree = KDTree(points)
        dists, _ = tree.query(points, k=k + 1)  # +1 because self is included
        mean_dists = dists[:, 1:].mean(axis=1)   # exclude self

        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        threshold = global_mean + std_ratio * global_std

        mask = mean_dists < threshold
        return points[mask], (colors[mask] if colors is not None else None), mask

    else:
        print("[WARN] Neither open3d nor scipy available for SOR")
        mask = np.ones(len(points), dtype=bool)
        return points, colors, mask


# ──────────────────────────────────────────────────────────────────────
# 2. Radius Outlier Removal
# ──────────────────────────────────────────────────────────────────────
def radius_outlier_removal(points, colors=None, radius=0.01, min_neighbors=5):
    """
    Remove points with fewer than min_neighbors within radius.

    Args:
        points:        (N, 3) array
        colors:        (N, 3) array or None
        radius:        search radius in meters
        min_neighbors: minimum neighbor count to keep a point

    Returns:
        filtered_points, filtered_colors, inlier_mask
    """
    if HAS_OPEN3D and len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        cl, inlier_idx = pcd.remove_radius_outlier(
            nb_points=min_neighbors, radius=radius
        )

        mask = np.zeros(len(points), dtype=bool)
        mask[inlier_idx] = True

        return points[mask], (colors[mask] if colors is not None else None), mask

    elif HAS_SCIPY and len(points) > 0:
        tree = KDTree(points)
        counts = tree.query_ball_point(points, r=radius, return_length=True)
        # counts includes self, so compare against min_neighbors + 1
        mask = counts >= (min_neighbors + 1)
        return points[mask], (colors[mask] if colors is not None else None), mask

    else:
        print("[WARN] Neither open3d nor scipy available for radius filter")
        mask = np.ones(len(points), dtype=bool)
        return points, colors, mask


# ──────────────────────────────────────────────────────────────────────
# 3. Cross-Cloud Consistency Filter
# ──────────────────────────────────────────────────────────────────────
def cross_cloud_consistency_filter(cloud_list, distance_threshold=0.005, min_views=2):
    """
    Remove points that don't have support from at least (min_views - 1) OTHER
    viewpoint clouds within distance_threshold.

    This catches phantom geometry that only appears in one scan — if a point
    from cloud_i has no neighbor within distance_threshold in any cloud_j,
    it's likely noise.

    Args:
        cloud_list: list of dicts, each with:
                      "points_base": (N, 3) already transformed to base frame
                      "colors":      (N, 3) or None
        distance_threshold: max distance (m) to count as "supported"
        min_views: minimum number of views a point must be seen in
                   (1 = no filtering, 2 = must have support from at least 1 other cloud)

    Returns:
        filtered_cloud_list: same structure, with unsupported points removed
    """
    if not HAS_SCIPY:
        print("[WARN] scipy not available for cross-cloud consistency filter")
        return cloud_list

    n_clouds = len(cloud_list)
    if n_clouds < 2 or min_views < 2:
        return cloud_list

    # Build KDTree for each cloud
    trees = []
    for c in cloud_list:
        trees.append(KDTree(c["points_base"]))

    filtered = []
    for i in range(n_clouds):
        pts_i = cloud_list[i]["points_base"]
        cols_i = cloud_list[i].get("colors")

        # Count how many OTHER clouds have a point within threshold
        support_count = np.zeros(len(pts_i), dtype=int)

        for j in range(n_clouds):
            if i == j:
                continue
            # For each point in cloud_i, find distance to nearest point in cloud_j
            dists, _ = trees[j].query(pts_i)
            support_count += (dists < distance_threshold).astype(int)

        # Keep points supported by at least (min_views - 1) other clouds
        mask = support_count >= (min_views - 1)

        filtered.append({
            "points_base": pts_i[mask],
            "colors": cols_i[mask] if cols_i is not None else None,
            "label": cloud_list[i].get("label", f"cloud_{i}"),
        })

    return filtered


# ──────────────────────────────────────────────────────────────────────
# 4. Voxel Downsample + Surface Smoothing
# ──────────────────────────────────────────────────────────────────────
def voxel_downsample(points, colors=None, voxel_size=0.002):
    """
    Voxel grid downsampling — averages points within each voxel.

    Args:
        points:     (N, 3) array
        colors:     (N, 3) array or None
        voxel_size: voxel edge length in meters

    Returns:
        downsampled_points, downsampled_colors
    """
    if HAS_OPEN3D and len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_down = pcd.voxel_down_sample(voxel_size)

        pts_out = np.asarray(pcd_down.points)
        cols_out = np.asarray(pcd_down.colors) if colors is not None else None
        return pts_out, cols_out
    else:
        # Naive fallback: grid-based averaging
        grid_idx = np.floor(points / voxel_size).astype(int)
        _, unique_idx, inverse = np.unique(
            grid_idx, axis=0, return_index=True, return_inverse=True
        )
        pts_out = np.zeros((len(unique_idx), 3))
        cols_out = np.zeros((len(unique_idx), 3)) if colors is not None else None
        counts = np.zeros(len(unique_idx))

        for i in range(len(points)):
            v = inverse[i]
            pts_out[v] += points[i]
            if colors is not None:
                cols_out[v] += colors[i]
            counts[v] += 1

        pts_out /= counts[:, None]
        if cols_out is not None:
            cols_out /= counts[:, None]
        return pts_out, cols_out


def smooth_cloud(points, colors=None, k=10, iterations=1):
    """
    Laplacian smoothing: move each point toward the centroid
    of its k nearest neighbors.

    Gentle smoothing that reduces noise while preserving shape.

    Args:
        points:     (N, 3) array
        colors:     (N, 3) array or None (colors are NOT smoothed)
        k:          neighbor count
        iterations: number of smoothing passes

    Returns:
        smoothed_points, colors (unchanged)
    """
    if not HAS_SCIPY or len(points) == 0:
        return points, colors

    pts = points.copy()
    for _ in range(iterations):
        tree = KDTree(pts)
        _, idx = tree.query(pts, k=k + 1)  # +1 for self
        neighbor_idx = idx[:, 1:]           # exclude self
        centroids = pts[neighbor_idx].mean(axis=1)
        # Move halfway toward centroid (conservative smoothing)
        pts = 0.5 * pts + 0.5 * centroids

    return pts, colors


# ──────────────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────────────
def denoise_pipeline(points, colors=None,
                     sor_k=20, sor_std=2.0,
                     radius=0.01, min_neighbors=5,
                     voxel_size=0.001,
                     smooth_k=10, smooth_iters=1,
                     verbose=True):
    """
    Full denoising pipeline for a single merged cloud:
      1. Statistical outlier removal
      2. Radius outlier removal
      3. Voxel downsample
      4. Laplacian smoothing

    For cross-cloud consistency, call cross_cloud_consistency_filter()
    BEFORE merging, then pass the merged result here.

    Args & Returns: points (N,3), colors (N,3) or None
    """
    n_start = len(points)
    if verbose:
        print(f"  Denoise pipeline: {n_start} input points")

    # Step 1: Statistical outlier removal
    points, colors, mask1 = statistical_outlier_removal(
        points, colors, k=sor_k, std_ratio=sor_std
    )
    if verbose:
        print(f"    SOR (k={sor_k}, std={sor_std}): "
              f"{n_start} → {len(points)} ({n_start - len(points)} removed)")

    # Step 2: Radius outlier removal
    n_before = len(points)
    points, colors, mask2 = radius_outlier_removal(
        points, colors, radius=radius, min_neighbors=min_neighbors
    )
    if verbose:
        print(f"    Radius (r={radius}, min={min_neighbors}): "
              f"{n_before} → {len(points)} ({n_before - len(points)} removed)")

    # Step 3: Voxel downsample
    n_before = len(points)
    points, colors = voxel_downsample(points, colors, voxel_size=voxel_size)
    if verbose:
        print(f"    Voxel ({voxel_size*1000:.1f}mm): "
              f"{n_before} → {len(points)}")

    # Step 4: Laplacian smoothing
    if smooth_iters > 0:
        points, colors = smooth_cloud(
            points, colors, k=smooth_k, iterations=smooth_iters
        )
        if verbose:
            print(f"    Smooth (k={smooth_k}, iters={smooth_iters}): done")

    if verbose:
        print(f"  Denoise complete: {n_start} → {len(points)} points")

    return points, colors


def denoise_per_bone_pipeline(waypoint_clouds,
                              cross_dist=0.005, min_views=2,
                              sor_k=20, sor_std=2.0,
                              radius=0.01, min_neighbors=5,
                              voxel_size=0.001,
                              smooth_k=10, smooth_iters=1,
                              verbose=True):
    """
    Full pipeline for the bone scanning workflow:
      1. Transform per-waypoint clouds to base frame
      2. Cross-cloud consistency filter (per bone)
      3. Merge consistent clouds
      4. SOR + Radius + Voxel + Smooth

    Args:
        waypoint_clouds: list of dicts from detect_and_merge_node, each with:
            bone_id, points_cam, colors, rotation, translation, label
        cross_dist:   distance threshold for cross-cloud support (meters)
        min_views:    min viewpoints a point needs support from

    Returns:
        dict: {
            "bone_left":  {"points": (N,3), "colors": (N,3)},
            "bone_right": {"points": (N,3), "colors": (N,3)},
            "combined":   {"points": (N,3), "colors": (N,3)},
        }
    """
    from collections import defaultdict

    # ── Group by bone_id ──
    by_bone = defaultdict(list)
    for entry in waypoint_clouds:
        by_bone[entry["bone_id"]].append(entry)

    results = {}

    for bone_id in ["bone_left", "bone_right"]:
        entries = by_bone.get(bone_id, [])
        if not entries:
            if verbose:
                print(f"  {bone_id}: no clouds, skipping")
            continue

        if verbose:
            print(f"\n  ── {bone_id}: {len(entries)} waypoint clouds ──")

        # Step 1: Transform each cloud to base frame
        cloud_list = []
        for entry in entries:
            pts = entry["points_cam"]
            cols = entry["colors"]
            rot = entry["rotation"]
            trans = entry["translation"]
            transformed = (rot @ pts.T).T + trans
            cloud_list.append({
                "points_base": transformed,
                "colors": cols,
                "label": entry["label"],
            })
            if verbose:
                print(f"    {entry['label']}: {len(pts)} pts")

        # Step 2: Cross-cloud consistency
        if len(cloud_list) >= 2 and min_views >= 2:
            n_before = sum(len(c["points_base"]) for c in cloud_list)
            cloud_list = cross_cloud_consistency_filter(
                cloud_list,
                distance_threshold=cross_dist,
                min_views=min_views,
            )
            n_after = sum(len(c["points_base"]) for c in cloud_list)
            if verbose:
                print(f"    Cross-cloud (dist={cross_dist}, views={min_views}): "
                      f"{n_before} → {n_after} ({n_before - n_after} removed)")

        # Step 3: Merge consistent clouds
        all_pts = [c["points_base"] for c in cloud_list if len(c["points_base"]) > 0]
        all_cols = [c["colors"] for c in cloud_list
                    if c["colors"] is not None and len(c["colors"]) > 0]

        if not all_pts:
            if verbose:
                print(f"    {bone_id}: no points after consistency filter!")
            continue

        merged_pts = np.vstack(all_pts)
        merged_cols = np.vstack(all_cols) if all_cols else None

        # Step 4: Denoise pipeline
        clean_pts, clean_cols = denoise_pipeline(
            merged_pts, merged_cols,
            sor_k=sor_k, sor_std=sor_std,
            radius=radius, min_neighbors=min_neighbors,
            voxel_size=voxel_size,
            smooth_k=smooth_k, smooth_iters=smooth_iters,
            verbose=verbose,
        )

        results[bone_id] = {"points": clean_pts, "colors": clean_cols}

    # ── Combined ──
    combined_pts = []
    combined_cols = []
    for bone_id in ["bone_left", "bone_right"]:
        if bone_id in results:
            combined_pts.append(results[bone_id]["points"])
            if results[bone_id]["colors"] is not None:
                combined_cols.append(results[bone_id]["colors"])

    if combined_pts:
        results["combined"] = {
            "points": np.vstack(combined_pts),
            "colors": np.vstack(combined_cols) if combined_cols else None,
        }

    return results