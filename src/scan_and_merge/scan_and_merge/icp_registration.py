#!/usr/bin/env python3
"""
ICP Registration Utilities

Aligns a reference PLY model onto a scanned point cloud.

The scan is FIXED (target). The reference model is MOVED (source).
Open3D ICP convention: source is transformed to align with target.

  register_bone(scan_pts_base, scan_cols, ref_ply_path)
    → returns aligned reference points in base frame
"""

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required for ICP registration")


def load_reference_mesh(ply_path, n_sample_points=50000, voxel_size=0.001):
    """Load reference PLY (mesh or cloud), sample + downsample + normals."""
    mesh = o3d.io.read_triangle_mesh(ply_path)

    if mesh.has_triangles() and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=n_sample_points)
        print(f"    Ref mesh: {len(mesh.triangles)} tris → {len(pcd.points)} sampled pts")
    else:
        pcd = o3d.io.read_point_cloud(ply_path)
        print(f"    Ref cloud: {len(pcd.points)} pts")

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30)
    )
    print(f"    Ref after downsample: {len(pcd_down.points)} pts")
    return pcd_down


def arrays_to_pcd(points, colors=None, voxel_size=0.001):
    """Numpy arrays → open3d PointCloud, downsampled with normals."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=30)
    )
    return pcd_down


def _print_cloud_stats(name, pcd):
    pts = np.asarray(pcd.points)
    print(f"    {name}: {len(pts)} pts, "
          f"x=[{pts[:,0].min():.4f},{pts[:,0].max():.4f}] "
          f"y=[{pts[:,1].min():.4f},{pts[:,1].max():.4f}] "
          f"z=[{pts[:,2].min():.4f},{pts[:,2].max():.4f}]")
    extent = pts.max(axis=0) - pts.min(axis=0)
    print(f"    {name} extent: [{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}]m")


def register_bone(scan_points, scan_colors, reference_ply_path,
                  coarse_method="fpfh", voxel_size=0.002, init_transform=None):
    """
    Register reference PLY onto scan cloud.

    The SCAN is fixed (target). The REFERENCE is moved (source).

    Args:
        scan_points:        (N, 3) bone scan in lbr_link_0 frame
        scan_colors:        (N, 3) or None
        reference_ply_path: path to reference .ply model
        coarse_method:      "fpfh" or "centroid"
        voxel_size:         base voxel size
        init_transform:     optional 4x4 numpy array from a previous good run.
                            If provided, skips coarse alignment entirely and
                            uses this as the ICP starting pose.

    Returns:
        dict:
            transform:          4x4 (ref model → base frame)
            aligned_ref_points: (M, 3) reference in base frame
            aligned_ref_colors: (M, 3) or None
            scan_points:        (K, 3) scan in base frame (downsampled)
            scan_colors:        (K, 3) or None
            fitness, rmse
    """
    print(f"\n  ── ICP Registration ──")

    # Load + prepare
    ref_pcd = load_reference_mesh(reference_ply_path, voxel_size=voxel_size)
    scan_pcd = arrays_to_pcd(scan_points, scan_colors, voxel_size=voxel_size)

    _print_cloud_stats("Scan (TARGET, fixed)", scan_pcd)
    _print_cloud_stats("Ref  (SOURCE, moved)", ref_pcd)

    # Check scale compatibility
    scan_extent = np.asarray(scan_pcd.points).ptp(axis=0)
    ref_extent = np.asarray(ref_pcd.points).ptp(axis=0)
    scale_ratio = np.linalg.norm(ref_extent) / max(np.linalg.norm(scan_extent), 1e-6)
    print(f"    Scale ratio (ref/scan): {scale_ratio:.3f}")
    if scale_ratio > 5.0 or scale_ratio < 0.2:
        print(f"    WARNING: scale mismatch! ref may be in mm vs scan in m. "
              f"Ratio={scale_ratio:.1f}")

    # ── Coarse alignment ──
    if init_transform is not None:
        print("    Coarse: using provided init transform (skipping FPFH/centroid)")
        coarse_T = init_transform.copy()
    elif coarse_method == "fpfh":
        print("    Coarse: FPFH RANSAC...")
        coarse_T = _fpfh_registration(ref_pcd, scan_pcd, voxel_size * 2)

        # Check if FPFH actually worked — evaluate fitness at coarse result
        eval_result = o3d.pipelines.registration.evaluate_registration(
            ref_pcd, scan_pcd, voxel_size * 20, coarse_T
        )
        if eval_result.fitness < 0.1:
            print(f"    FPFH failed (fitness={eval_result.fitness:.4f}), "
                  f"falling back to centroid alignment...")
            coarse_T = _centroid_alignment(ref_pcd, scan_pcd)
    else:
        print("    Coarse: centroid alignment...")
        coarse_T = _centroid_alignment(ref_pcd, scan_pcd)

    # Debug: where does coarse put the reference?
    ref_coarse = o3d.geometry.PointCloud(ref_pcd)
    ref_coarse.transform(coarse_T)
    _print_cloud_stats("Ref after coarse", ref_coarse)

    # ── Fine: multi-scale ICP ──
    # Use wider correspondence distances to handle rough coarse alignment
    print("    Fine: multi-scale point-to-plane ICP...")
    scales = [voxel_size * 5, voxel_size * 2, voxel_size]
    dists  = [voxel_size * 50, voxel_size * 20, voxel_size * 10]

    current_T = coarse_T.copy()
    for i, (vs, md) in enumerate(zip(scales, dists)):
        src_d = ref_pcd.voxel_down_sample(vs)
        tgt_d = scan_pcd.voxel_down_sample(vs)
        src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs*3, max_nn=30))
        tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs*3, max_nn=30))

        result = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d,  # source=ref, target=scan
            max_correspondence_distance=md,
            init=current_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=100, relative_fitness=1e-7, relative_rmse=1e-7,
            ),
        )
        current_T = result.transformation
        print(f"      Scale {i+1} (voxel={vs*1000:.1f}mm, dist={md*1000:.1f}mm): "
              f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")

    final_T = current_T
    fitness = result.fitness
    rmse = result.inlier_rmse
    print(f"    Final: fitness={fitness:.4f}, RMSE={rmse:.6f}")

    # If ICP failed badly, retry with centroid coarse alignment
    if fitness < 0.3 and coarse_method == "fpfh" and init_transform is None:
        print(f"    ICP fitness too low ({fitness:.4f}), retrying with centroid...")
        coarse_T = _centroid_alignment(ref_pcd, scan_pcd)

        current_T = coarse_T.copy()
        for i, (vs, md) in enumerate(zip(scales, dists)):
            src_d = ref_pcd.voxel_down_sample(vs)
            tgt_d = scan_pcd.voxel_down_sample(vs)
            src_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs*3, max_nn=30))
            tgt_d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs*3, max_nn=30))

            result = o3d.pipelines.registration.registration_icp(
                src_d, tgt_d,
                max_correspondence_distance=md,
                init=current_T,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=200, relative_fitness=1e-7, relative_rmse=1e-7,
                ),
            )
            current_T = result.transformation
            print(f"      Retry scale {i+1}: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")

        if result.fitness > fitness:
            final_T = current_T
            fitness = result.fitness
            rmse = result.inlier_rmse
            print(f"    Retry improved: fitness={fitness:.4f}, RMSE={rmse:.6f}")
        else:
            print(f"    Retry did not improve, keeping original")

    # Apply transform to reference → now in base frame
    aligned_ref = o3d.geometry.PointCloud(ref_pcd)
    aligned_ref.transform(final_T)
    _print_cloud_stats("Ref ALIGNED (in base frame)", aligned_ref)

    return {
        "transform": final_T,
        "ref_points": np.asarray(ref_pcd.points),
        "ref_colors": np.asarray(ref_pcd.colors) if ref_pcd.has_colors() else None,
        "aligned_ref_points": np.asarray(aligned_ref.points),
        "aligned_ref_colors": np.asarray(aligned_ref.colors) if aligned_ref.has_colors() else None,
        "scan_points": np.asarray(scan_pcd.points),
        "scan_colors": np.asarray(scan_pcd.colors) if scan_pcd.has_colors() else None,
        "fitness": fitness,
        "rmse": rmse,
    }


def _centroid_alignment(source, target):
    """Translate source centroid to target centroid."""
    src_c = np.asarray(source.points).mean(axis=0)
    tgt_c = np.asarray(target.points).mean(axis=0)
    T = np.eye(4)
    T[:3, 3] = tgt_c - src_c
    return T


def _fpfh_registration(source, target, voxel_size):
    """FPFH + RANSAC coarse registration. source→target."""
    r_feat = voxel_size * 5

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=r_feat, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=r_feat, max_nn=100))

    dist_thresh = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    print(f"      FPFH RANSAC: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
    return result.transformation