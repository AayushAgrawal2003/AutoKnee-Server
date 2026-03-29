#!/usr/bin/env python3
"""
ICP Registration Utilities (v3 — robust coarse alignment)

Aligns a reference PLY model onto a scanned point cloud.

The scan is FIXED (target). The reference model is MOVED (source).
Open3D ICP convention: source is transformed to align with target.

  register_bone(scan_pts_base, scan_cols, ref_ply_path)
    → returns aligned reference points in base frame

Key improvements over v2:
  1. Auto-detects mm-vs-m scale mismatch and converts reference to meters.
  2. Exhaustive rotation search: 24 axis-aligned rotations + 8 PCA sign
     combos + FPFH, all scored with trimmed scan→ref distance (correct
     metric for partial overlap where scan sees only one side of the bone).
  3. Top candidates refined with quick ICP before selecting winner.
  4. Multi-scale ICP has divergence guard — reverts if fitness drops.
  5. Backward compatible: same register_bone() signature.
"""

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required for ICP registration")


# ─────────────────────────────────────────────────────────────
# I/O helpers
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
        dict with transform, aligned_ref_points, aligned_ref_colors,
        scan_points, scan_colors, fitness, rmse
    """
    print(f"\n  ── ICP Registration (v3) ──")

    # Load + prepare
    ref_pcd = load_reference_mesh(reference_ply_path, voxel_size=voxel_size)
    scan_pcd = arrays_to_pcd(scan_points, scan_colors, voxel_size=voxel_size)

    _print_cloud_stats("Scan (TARGET, fixed)", scan_pcd)
    _print_cloud_stats("Ref  (SOURCE, moved)", ref_pcd)

    # ── Auto-scale: detect mm vs m mismatch and fix ──
    ref_pcd = _auto_scale_reference(ref_pcd, scan_pcd)

    # ── Coarse alignment ──
    if init_transform is not None:
        print("    Coarse: using provided init transform (skipping coarse)")
        coarse_T = init_transform.copy()

    elif coarse_method == "hybrid":
        print("    Coarse: exhaustive rotation search ...")
        coarse_T = _exhaustive_coarse(ref_pcd, scan_pcd, voxel_size)

    elif coarse_method == "fpfh":
        print("    Coarse: FPFH RANSAC...")
        coarse_T = _fpfh_registration(ref_pcd, scan_pcd, voxel_size * 2)
        eval_result = o3d.pipelines.registration.evaluate_registration(
            ref_pcd, scan_pcd, voxel_size * 20, coarse_T
        )
        if eval_result.fitness < 0.1:
            print(f"    FPFH failed (fitness={eval_result.fitness:.4f}), "
                  f"falling back to exhaustive search...")
            coarse_T = _exhaustive_coarse(ref_pcd, scan_pcd, voxel_size)

    else:
        print("    Coarse: centroid alignment...")
        coarse_T = _centroid_alignment(ref_pcd, scan_pcd)

    # Debug
    ref_coarse = o3d.geometry.PointCloud(ref_pcd)
    ref_coarse.transform(coarse_T)
    _print_cloud_stats("Ref after coarse", ref_coarse)

    # ── Fine: multi-scale ICP with divergence guard ──
    print("    Fine: multi-scale point-to-plane ICP (Cauchy loss)...")
    final_T = _multiscale_icp_guarded(ref_pcd, scan_pcd, coarse_T, voxel_size)

    # Evaluate
    eval_r = o3d.pipelines.registration.evaluate_registration(
        ref_pcd, scan_pcd, voxel_size * 10, final_T
    )
    fitness = eval_r.fitness
    rmse = eval_r.inlier_rmse
    print(f"    Final: fitness={fitness:.4f}, RMSE={rmse:.6f}")

    # Apply transform to reference
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
# Auto-scale: detect and fix mm ↔ m mismatch
# ─────────────────────────────────────────────────────────────

def _auto_scale_reference(ref_pcd, scan_pcd):
    """If reference is in mm and scan in m (or vice versa), rescale reference."""
    ref_pts = np.asarray(ref_pcd.points)
    scan_pts = np.asarray(scan_pcd.points)
    ref_diag = np.linalg.norm(ref_pts.max(axis=0) - ref_pts.min(axis=0))
    scan_diag = np.linalg.norm(scan_pts.max(axis=0) - scan_pts.min(axis=0))

    ratio = ref_diag / max(scan_diag, 1e-9)
    print(f"    Scale ratio (ref/scan): {ratio:.3f}")

    if ratio > 100:
        # Reference is in mm, scan is in m — scale ref down
        scale = 1.0 / 1000.0
        print(f"    AUTO-SCALE: ref appears to be in mm, converting to meters (÷1000)")
    elif ratio < 0.01:
        # Reference is in m, scan is in mm
        scale = 1000.0
        print(f"    AUTO-SCALE: ref appears to be in m, scan in mm (×1000)")
    elif ratio > 5:
        scale = 1.0 / ratio
        print(f"    AUTO-SCALE: significant scale mismatch, rescaling ref by {scale:.4f}")
    elif ratio < 0.2:
        scale = 1.0 / ratio
        print(f"    AUTO-SCALE: significant scale mismatch, rescaling ref by {scale:.4f}")
    else:
        return ref_pcd

    # Scale the reference
    pts_scaled = ref_pts * scale
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(pts_scaled)
    if ref_pcd.has_colors():
        new_pcd.colors = ref_pcd.colors
    if ref_pcd.has_normals():
        new_pcd.normals = ref_pcd.normals

    _print_cloud_stats("Ref (after scale)", new_pcd)
    return new_pcd


# ─────────────────────────────────────────────────────────────
# Coarse alignment: exhaustive rotation search
# ─────────────────────────────────────────────────────────────

def _centroid_alignment(source, target):
    """Translate source centroid to target centroid."""
    src_c = np.asarray(source.points).mean(axis=0)
    tgt_c = np.asarray(target.points).mean(axis=0)
    T = np.eye(4)
    T[:3, 3] = tgt_c - src_c
    return T


def _make_transform(R, src_centroid, tgt_centroid):
    """Build 4x4 that rotates source about its centroid, then translates to target centroid."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tgt_centroid - R @ src_centroid
    return T


def _generate_octahedral_rotations():
    """Generate all 24 rotation matrices of the octahedral group.
    These are all axis-aligned rotations (90° increments around x/y/z)."""
    rots = []
    # All signed permutations of identity axes
    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    for i in range(3):
        for si in [1, -1]:
            for j in range(3):
                if j == i:
                    continue
                for sj in [1, -1]:
                    # third axis determined by cross product (right-hand rule)
                    a = axes[i] * si
                    b = axes[j] * sj
                    c = np.cross(a, b)
                    R = np.column_stack([a, b, c])
                    if abs(np.linalg.det(R) - 1.0) < 0.01:
                        rots.append(R)
    return rots


def _generate_pca_rotations(source, target):
    """Generate rotation candidates from PCA with all 8 sign combos."""
    def _pca(pcd):
        pts = np.asarray(pcd.points)
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        return c, vecs[:, idx]

    sc, sa = _pca(source)
    tc, ta = _pca(target)

    rotations = []
    for fx in [1, -1]:
        for fy in [1, -1]:
            for fz in [1, -1]:
                sa_flip = sa.copy()
                sa_flip[:, 0] *= fx
                sa_flip[:, 1] *= fy
                sa_flip[:, 2] *= fz
                # Ensure right-handedness
                if np.linalg.det(sa_flip) < 0:
                    sa_flip[:, 2] *= -1
                R = ta @ np.linalg.inv(sa_flip)
                # Only keep proper rotations
                if abs(np.linalg.det(R) - 1.0) < 0.01:
                    rotations.append(R)
    return rotations


def _generate_so3_samples(n_divisions=3):
    """Generate roughly uniform rotation samples on SO(3) using
    Euler angle grid. n_divisions=3 → ~72 rotations."""
    rots = []
    step = np.pi / (2 * n_divisions)  # 30° for n_divisions=3
    for alpha in np.arange(0, 2 * np.pi, step):
        for beta in np.arange(0, np.pi + 1e-9, step):
            for gamma in np.arange(0, 2 * np.pi, step):
                # ZYZ Euler angles
                ca, sa_ = np.cos(alpha), np.sin(alpha)
                cb, sb = np.cos(beta), np.sin(beta)
                cg, sg = np.cos(gamma), np.sin(gamma)
                Rz1 = np.array([[ca, -sa_, 0], [sa_, ca, 0], [0, 0, 1]])
                Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
                Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
                R = Rz1 @ Ry @ Rz2
                rots.append(R)
    return rots


def _score_alignment_trimmed(source_pts, target_tree, trim_pct=0.75):
    """Score alignment using trimmed scan→source distance.

    For partial overlap: every SCAN point should have a nearby REF point.
    We measure target→source distance (how well the scan is covered by ref),
    and take the trimmed mean to ignore the worst outliers.

    Lower is better.
    """
    # target_tree is built from scan. source_pts is transformed ref.
    # We want: for each scan point, distance to nearest ref point.
    # So build tree from source_pts (ref), query with target (scan).
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    # distances from scan to nearest ref
    dists = np.asarray(target_tree.compute_point_cloud_distance(source_pcd))

    # But we actually want ref→scan coverage too for the visible part.
    # Better: use scan→ref (each scan point should have a close ref point)
    dists_scan_to_ref = np.asarray(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            np.asarray(target_tree.points)
        )).compute_point_cloud_distance(source_pcd)
    )

    # Trimmed mean of scan→ref distances
    n = len(dists_scan_to_ref)
    k = max(1, int(n * trim_pct))
    sorted_d = np.sort(dists_scan_to_ref)
    return np.mean(sorted_d[:k])


def _score_transform_fast(ref_pts, scan_pcd, T, trim_pct=0.75):
    """Apply transform to ref_pts, score against scan using trimmed scan→ref."""
    R = T[:3, :3]
    t = T[:3, 3]
    ref_transformed = (R @ ref_pts.T).T + t

    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_transformed)

    # For each scan point, find distance to nearest transformed ref point
    scan_pts = np.asarray(scan_pcd.points)
    dists = np.asarray(scan_pcd.compute_point_cloud_distance(ref_pcd))

    n = len(dists)
    k = max(1, int(n * trim_pct))
    return np.mean(np.sort(dists)[:k])


def _quick_icp_refine(source, target, init_T, max_dists=[0.03, 0.01, 0.005]):
    """Quick 3-stage Cauchy ICP to refine a coarse alignment. Returns (T, score)."""
    s_tmp = o3d.geometry.PointCloud(source)
    s_tmp.transform(init_T)
    s_tmp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    current_T = np.eye(4)
    best_fitness = 0.0
    for md in max_dists:
        loss = o3d.pipelines.registration.CauchyLoss(k=md * 0.3)
        r = o3d.pipelines.registration.registration_icp(
            s_tmp, target, md, current_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        current_T = r.transformation
        best_fitness = max(best_fitness, r.fitness)

    # Compose: final_T = current_T @ init_T
    composed_T = current_T @ init_T

    # Score: trimmed scan→ref distance
    ref_transformed = np.asarray(o3d.geometry.PointCloud(source).transform(composed_T).points)
    ref_pcd_t = o3d.geometry.PointCloud()
    ref_pcd_t.points = o3d.utility.Vector3dVector(ref_transformed)
    dists = np.asarray(target.compute_point_cloud_distance(ref_pcd_t))
    n = len(dists)
    k = max(1, int(n * 0.75))
    score = np.mean(np.sort(dists)[:k])

    return composed_T, score


def _exhaustive_coarse(source, target, voxel_size):
    """
    Exhaustive rotation search for coarse alignment.

    Generates rotation candidates from:
      A) All 24 axis-aligned rotations (octahedral group)
      B) PCA with 8 sign combos (4 unique proper rotations)
      C) FPFH RANSAC
    Scores each with trimmed scan→ref distance, refines top candidates
    with quick ICP, returns the best.
    """
    src_pts = np.asarray(source.points)
    src_c = src_pts.mean(axis=0)
    tgt_c = np.asarray(target.points).mean(axis=0)

    # ── Generate all candidate transforms ──
    candidates = []  # (score, T, label)

    # A: 24 axis-aligned rotations
    oct_rots = _generate_octahedral_rotations()
    for i, R in enumerate(oct_rots):
        T = _make_transform(R, src_c, tgt_c)
        score = _score_transform_fast(src_pts, target, T)
        candidates.append((score, T, f"Oct-{i}"))

    # B: PCA rotations (up to 8 sign combos → ~4 proper rotations)
    pca_rots = _generate_pca_rotations(source, target)
    for i, R in enumerate(pca_rots):
        T = _make_transform(R, src_c, tgt_c)
        score = _score_transform_fast(src_pts, target, T)
        candidates.append((score, T, f"PCA-{i}"))

    # C: Centroid-only (identity rotation)
    T_c = _centroid_alignment(source, target)
    score = _score_transform_fast(src_pts, target, T_c)
    candidates.append((score, T_c, "Centroid"))

    # D: FPFH RANSAC
    try:
        T_f = _fpfh_registration(source, target, voxel_size * 2)
        score = _score_transform_fast(src_pts, target, T_f)
        candidates.append((score, T_f, "FPFH"))
    except Exception as e:
        print(f"      FPFH failed: {e}")

    # E: Fast Global Registration
    try:
        T_fgr = _fgr_registration(source, target, voxel_size * 2)
        score = _score_transform_fast(src_pts, target, T_fgr)
        candidates.append((score, T_fgr, "FGR"))
    except Exception as e:
        print(f"      FGR failed: {e}")

    # Sort by score and take top candidates for ICP refinement
    candidates.sort(key=lambda x: x[0])

    print(f"    Pre-ICP ranking (top 8 of {len(candidates)}):")
    for sc, _, lab in candidates[:8]:
        print(f"      {lab}: trimmed_dist={sc:.6f}")

    # ── Refine top candidates with quick ICP ──
    n_refine = min(8, len(candidates))
    refined = []
    for sc, T, lab in candidates[:n_refine]:
        T_ref, icp_score = _quick_icp_refine(source, target, T)
        refined.append((icp_score, T_ref, lab))

    refined.sort(key=lambda x: x[0])

    print(f"    Post-ICP ranking (top 5):")
    for sc, _, lab in refined[:5]:
        marker = " ◀ winner" if lab == refined[0][2] else ""
        print(f"      {lab}: trimmed_dist={sc:.6f}{marker}")

    return refined[0][1]


# ─────────────────────────────────────────────────────────────
# Feature-based coarse registration
# ─────────────────────────────────────────────────────────────

def _fpfh_registration(source, target, voxel_size):
    """FPFH + RANSAC coarse registration. source→target."""
    r_feat = voxel_size * 5

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=r_feat, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=r_feat, max_nn=100))

    dist_thresh = voxel_size * 3.0
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, src_fpfh, tgt_fpfh,
        mutual_filter=False,  # don't require mutual — partial overlap breaks this
        max_correspondence_distance=dist_thresh,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999),
    )
    print(f"      FPFH RANSAC: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
    return result.transformation


def _fgr_registration(source, target, voxel_size):
    """Fast Global Registration — often better than RANSAC for partial overlap."""
    r_feat = voxel_size * 5

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=r_feat, max_nn=100))
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=r_feat, max_nn=100))

    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source, target, src_fpfh, tgt_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel_size * 3.0,
        ),
    )
    print(f"      FGR: fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f}")
    return result.transformation


# ─────────────────────────────────────────────────────────────
# Fine alignment: multi-scale ICP with divergence guard
# ─────────────────────────────────────────────────────────────

def _multiscale_icp_guarded(source, target, init_T, voxel_size):
    """
    6-stage coarse-to-fine ICP with Cauchy loss and transform-based
    divergence detection.

    Instead of comparing fitness across stages (which naturally drops at
    tighter thresholds with partial overlap), we detect divergence by
    checking if the transform jumps too far between stages.
    """
    max_dists = [
        voxel_size * 25,   # 50 mm
        voxel_size * 15,   # 30 mm
        voxel_size * 10,   # 20 mm
        voxel_size * 5,    # 10 mm
        voxel_size * 2.5,  #  5 mm
        voxel_size * 1,    #  2 mm
    ]

    # Fixed evaluation threshold for consistent scoring
    eval_dist = voxel_size * 10

    current_T = init_T.copy()
    best_T = init_T.copy()
    best_eval = o3d.pipelines.registration.evaluate_registration(
        source, target, eval_dist, init_T)
    best_score = best_eval.fitness

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

        # Evaluate at fixed threshold for consistent scoring
        new_eval = o3d.pipelines.registration.evaluate_registration(
            source, target, eval_dist, result.transformation)

        print(f"      Scale {i+1} (dist={md*1000:.1f}mm): "
              f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.6f} "
              f"[eval@{eval_dist*1000:.0f}mm: {new_eval.fitness:.4f}]")

        # Check for divergence: large rotation jump from current best
        R_diff = result.transformation[:3, :3] @ np.linalg.inv(current_T[:3, :3])
        angle_jump = np.degrees(np.arccos(
            np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))

        if angle_jump > 45 and best_score > 0.3:
            print(f"      ⚠ Large rotation jump ({angle_jump:.1f}°), reverting")
            current_T = best_T.copy()
            continue

        # Accept if evaluation score improved or didn't drop badly
        if new_eval.fitness >= best_score * 0.85:
            current_T = result.transformation
            if new_eval.fitness >= best_score:
                best_score = new_eval.fitness
                best_T = result.transformation.copy()
        else:
            print(f"      ⚠ Eval fitness dropped ({new_eval.fitness:.4f} < "
                  f"{best_score*0.85:.4f}), reverting")
            current_T = best_T.copy()

    return best_T


# ─────────────────────────────────────────────────────────────
# Legacy aliases for backward compatibility
# ─────────────────────────────────────────────────────────────

def _hybrid_coarse(source, target, voxel_size):
    """Legacy wrapper — calls exhaustive search."""
    return _exhaustive_coarse(source, target, voxel_size)


def _pca_alignment(source, target, flip_x=1, flip_y=1):
    """Legacy PCA alignment with configurable axis flips."""
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


def _multiscale_icp_cauchy(source, target, init_T, voxel_size):
    """Legacy alias for guarded ICP."""
    return _multiscale_icp_guarded(source, target, init_T, voxel_size)
