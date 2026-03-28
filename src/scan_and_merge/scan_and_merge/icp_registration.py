#!/usr/bin/env python3
"""
ICP Registration Utilities (improved)

Aligns a reference PLY model onto a scanned point cloud.

The scan is FIXED (target). The reference model is MOVED (source).
Open3D ICP convention: source is transformed to align with target.

  register_bone(scan_pts_base, scan_cols, ref_ply_path)
    → returns aligned reference points in base frame

Changes vs original:
  1. Default coarse_method changed to "hybrid" — races PCA-flip-search,
     centroid, and FPFH, picks best automatically.
  2. Multi-scale ICP now uses Cauchy robust loss to handle partial overlap
     (scan only covers one side of the bone, so ~30% of reference points
     have no valid correspondence — Cauchy down-weights those outliers).
  3. PCA coarse alignment with 4 axis-flip search resolves the eigenvector
     sign ambiguity that caused "aligning to random surfaces".
  4. 6-stage coarse-to-fine refinement (was 3 stages).
  5. .ptp() replaced with .max() - .min() for NumPy 2.x compat.

Backward compatible:  coarse_method="fpfh" and "centroid" still work.
"""

import numpy as np
import copy as _copy

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required for ICP registration")


# ─────────────────────────────────────────────────────────────
# I/O helpers (unchanged)
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Public API  (same signature — drop-in replacement)
# ─────────────────────────────────────────────────────────────

def register_bone(scan_points, scan_colors, reference_ply_path,
                  coarse_method="hybrid", voxel_size=0.002, init_transform=None):
    """
    Register reference PLY onto scan cloud.

    The SCAN is fixed (target). The REFERENCE is moved (source).

    Args:
        scan_points:        (N, 3) bone scan in lbr_link_0 frame
        scan_colors:        (N, 3) or None
        reference_ply_path: path to reference .ply model
        coarse_method:      "hybrid" (recommended), "fpfh", or "centroid"
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
    scan_pts = np.asarray(scan_pcd.points)
    ref_pts = np.asarray(ref_pcd.points)
    scan_extent = scan_pts.max(axis=0) - scan_pts.min(axis=0)
    ref_extent  = ref_pts.max(axis=0) - ref_pts.min(axis=0)
    scale_ratio = np.linalg.norm(ref_extent) / max(np.linalg.norm(scan_extent), 1e-6)
    print(f"    Scale ratio (ref/scan): {scale_ratio:.3f}")
    if scale_ratio > 5.0 or scale_ratio < 0.2:
        print(f"    WARNING: scale mismatch! ref may be in mm vs scan in m. "
              f"Ratio={scale_ratio:.1f}")

    # ── Coarse alignment ──
    if init_transform is not None:
        print("    Coarse: using provided init transform (skipping coarse)")
        coarse_T = init_transform.copy()

    elif coarse_method == "hybrid":
        print("    Coarse: hybrid (PCA-flip-search + centroid + FPFH) ...")
        coarse_T = _hybrid_coarse(ref_pcd, scan_pcd, voxel_size)

    elif coarse_method == "fpfh":
        print("    Coarse: FPFH RANSAC...")
        coarse_T = _fpfh_registration(ref_pcd, scan_pcd, voxel_size * 2)

        # Check if FPFH actually worked
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

    # ── Fine: multi-scale ICP with Cauchy robust loss ──
    print("    Fine: multi-scale point-to-plane ICP (Cauchy loss)...")
    final_T = _multiscale_icp_cauchy(ref_pcd, scan_pcd, coarse_T, voxel_size)

    # Evaluate
    eval_r = o3d.pipelines.registration.evaluate_registration(
        ref_pcd, scan_pcd, voxel_size * 10, final_T
    )
    fitness = eval_r.fitness
    rmse = eval_r.inlier_rmse
    print(f"    Final: fitness={fitness:.4f}, RMSE={rmse:.6f}")

    # If ICP failed badly, retry with centroid (only if we didn't already try it)
    if fitness < 0.3 and coarse_method != "hybrid" and init_transform is None:
        print(f"    ICP fitness too low ({fitness:.4f}), retrying with centroid...")
        coarse_T = _centroid_alignment(ref_pcd, scan_pcd)
        retry_T = _multiscale_icp_cauchy(ref_pcd, scan_pcd, coarse_T, voxel_size)
        retry_eval = o3d.pipelines.registration.evaluate_registration(
            ref_pcd, scan_pcd, voxel_size * 10, retry_T
        )
        if retry_eval.fitness > fitness:
            final_T = retry_T
            fitness = retry_eval.fitness
            rmse = retry_eval.inlier_rmse
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


# ─────────────────────────────────────────────────────────────
# Coarse alignment strategies
# ─────────────────────────────────────────────────────────────

def _centroid_alignment(source, target):
    """Translate source centroid to target centroid."""
    src_c = np.asarray(source.points).mean(axis=0)
    tgt_c = np.asarray(target.points).mean(axis=0)
    T = np.eye(4)
    T[:3, 3] = tgt_c - src_c
    return T


def _pca_alignment(source, target, flip_x=1, flip_y=1):
    """PCA rotation+translation with configurable axis flips."""
    def _pca(pcd):
        pts = np.asarray(pcd.points)
        c = pts.mean(axis=0)
        vals, vecs = np.linalg.eigh(np.cov((pts - c).T))
        return c, vecs[:, np.argsort(vals)[::-1]]

    sc, sa = _pca(source)
    tc, ta = _pca(target)

    sa[:, 0] *= flip_x
    sa[:, 1] *= flip_y
    sa[:, 2] = np.cross(sa[:, 0], sa[:, 1])

    R = ta @ np.linalg.inv(sa)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tc - R @ sc
    return T


def _quick_evaluate(source, target, coarse_T):
    """Quick 3-stage Cauchy ICP + Chamfer eval to score a coarse alignment."""
    s_tmp = o3d.geometry.PointCloud(source)
    s_tmp.transform(coarse_T)
    s_tmp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    cT = np.eye(4)
    for md in [0.03, 0.01, 0.005]:
        loss = o3d.pipelines.registration.CauchyLoss(k=md * 0.3)
        r = o3d.pipelines.registration.registration_icp(
            s_tmp, target, md, cT,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=150))
        cT = r.transformation

    s_eval = o3d.geometry.PointCloud(s_tmp)
    s_eval.transform(cT)
    d_fwd = np.asarray(s_eval.compute_point_cloud_distance(target))
    d_rev = np.asarray(target.compute_point_cloud_distance(s_eval))
    return np.mean(d_fwd) + np.mean(d_rev)


def _hybrid_coarse(source, target, voxel_size):
    """
    Race multiple coarse strategies, each followed by a quick Cauchy ICP,
    and return the one with the lowest Chamfer distance.

    Strategies:
      A) PCA with 4 axis-flip combos  (handles orientation ambiguity)
      B) Centroid translation          (fast fallback)
      C) FPFH RANSAC                   (feature-based)
    """
    candidates = []

    # A: PCA flips (4 combos)
    for fx in [1, -1]:
        for fy in [1, -1]:
            T = _pca_alignment(source, target, fx, fy)
            chamfer = _quick_evaluate(source, target, T)
            candidates.append((chamfer, T, f"PCA({fx},{fy})"))

    # B: Centroid
    T_c = _centroid_alignment(source, target)
    chamfer = _quick_evaluate(source, target, T_c)
    candidates.append((chamfer, T_c, "Centroid"))

    # C: FPFH
    try:
        T_f = _fpfh_registration(source, target, voxel_size * 2)
        chamfer = _quick_evaluate(source, target, T_f)
        candidates.append((chamfer, T_f, "FPFH"))
    except Exception as e:
        print(f"      FPFH failed: {e}")

    candidates.sort(key=lambda x: x[0])
    best_chamfer, best_T, best_label = candidates[0]

    print(f"    Hybrid coarse candidates:")
    for ch, _, lab in candidates[:5]:
        marker = " ◀ winner" if lab == best_label else ""
        print(f"      {lab}: chamfer={ch:.6f}{marker}")

    return best_T


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


# ─────────────────────────────────────────────────────────────
# Fine alignment: multi-scale ICP with Cauchy robust loss
# ─────────────────────────────────────────────────────────────

def _multiscale_icp_cauchy(source, target, init_T, voxel_size):
    """
    6-stage coarse-to-fine ICP with Cauchy loss.

    The Cauchy kernel down-weights outlier correspondences, which is
    critical when the reference model extends beyond the partial scan.
    """
    max_dists = [
        voxel_size * 25,   # 50 mm — capture gross misalignment
        voxel_size * 15,   # 30 mm
        voxel_size * 10,   # 20 mm
        voxel_size * 5,    # 10 mm
        voxel_size * 2.5,  #  5 mm
        voxel_size * 1,    #  2 mm — fine polish
    ]

    current_T = init_T.copy()

    for i, md in enumerate(max_dists):
        vs = max(md * 0.3, voxel_size)
        src_d = source.voxel_down_sample(vs)
        tgt_d = target.voxel_down_sample(vs)
        src_d.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 3, max_nn=30))
        tgt_d.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 3, max_nn=30))

        loss = o3d.pipelines.registration.CauchyLoss(k=md * 0.3)

        result = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d,
            max_correspondence_distance=md,
            init=current_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=200,
                relative_fitness=1e-7,
                relative_rmse=1e-7,
            ),
        )
        current_T = result.transformation
        print(f"      Scale {i+1} (dist={md*1000:.1f}mm): "
              f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")

    return current_T