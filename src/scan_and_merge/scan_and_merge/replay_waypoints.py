#!/usr/bin/env python3
"""
Utility: Load previously saved waypoints and visualize or replay them.

Usage:
  python3 replay_waypoints.py ~/scan_output/waypoints.npy
  python3 replay_waypoints.py ~/scan_output/waypoints.npy --visualize
"""

import argparse
import numpy as np
import os


def print_waypoints(waypoints):
    """Pretty print saved waypoints."""
    print(f"\nLoaded {len(waypoints)} waypoints:\n")
    print(f"  {'#':<4} {'A1':>8} {'A2':>8} {'A3':>8} {'A4':>8} {'A5':>8} {'A6':>8} {'A7':>8}")
    print(f"  {'':─<4} {'':─>8} {'':─>8} {'':─>8} {'':─>8} {'':─>8} {'':─>8} {'':─>8}")

    for i, wp in enumerate(waypoints):
        deg = np.degrees(wp)
        print(f"  {i+1:<4} {deg[0]:>7.1f}° {deg[1]:>7.1f}° {deg[2]:>7.1f}° "
              f"{deg[3]:>7.1f}° {deg[4]:>7.1f}° {deg[5]:>7.1f}° {deg[6]:>7.1f}°")

    print()


def visualize_waypoints(waypoints):
    """Visualize waypoints using FK and matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib required for visualization. pip install matplotlib")
        return

    # Simple FK using the KUKA Med 7 URDF joint origins
    # DH-like parameters extracted from URDF
    joint_origins = [
        (0.0, 0.0, 0.1475),      # A1 from base
        (0.0, -0.0105, 0.1925),   # A2 from link1
        (0.0, 0.0105, 0.2075),    # A3 from link2
        (0.0, 0.0105, 0.1925),    # A4 from link3
        (0.0, -0.0105, 0.2075),   # A5 from link4
        (0.0, -0.0707, 0.1925),   # A6 from link5
        (0.0, 0.0707, 0.091),     # A7 from link6
    ]
    joint_axes = [
        (0, 0, 1),   # A1: Z
        (0, 1, 0),   # A2: Y
        (0, 0, 1),   # A3: Z
        (0, -1, 0),  # A4: -Y
        (0, 0, 1),   # A5: Z
        (0, 1, 0),   # A6: Y
        (0, 0, 1),   # A7: Z
    ]
    ee_offset = np.array([0.0, 0.0, 0.189])

    def rot_axis(axis, angle):
        """Rotation matrix around an axis by angle (radians)."""
        ax = np.array(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        c, s = np.cos(angle), np.sin(angle)
        x, y, z = ax
        return np.array([
            [c + x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
            [y*x*(1-c)+z*s,   c + y*y*(1-c), y*z*(1-c)-x*s],
            [z*x*(1-c)-y*s,   z*y*(1-c)+x*s, c + z*z*(1-c)],
        ])

    def fk(q):
        """Compute link positions for visualization."""
        positions = [np.array([0, 0, 0])]
        T = np.eye(4)

        for i in range(7):
            # Translation to joint
            t = np.eye(4)
            t[:3, 3] = joint_origins[i]
            T = T @ t

            # Rotation
            r = np.eye(4)
            r[:3, :3] = rot_axis(joint_axes[i], q[i])
            T = T @ r

            positions.append(T[:3, 3].copy())

        # EE
        t_ee = np.eye(4)
        t_ee[:3, 3] = ee_offset
        T = T @ t_ee
        positions.append(T[:3, 3].copy())

        return np.array(positions)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.viridis(np.linspace(0, 1, len(waypoints)))

    for i, wp in enumerate(waypoints):
        pts = fk(wp)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                "-o", color=colors[i], label=f"WP {i+1}",
                markersize=4, linewidth=2)
        # Mark EE
        ax.scatter(*pts[-1], s=100, color=colors[i], marker="^", edgecolors="black")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("KUKA Med 7 - Recorded Waypoints (FK)")
    ax.legend()

    # Equal aspect ratio
    all_pts = np.vstack([fk(wp) for wp in waypoints])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="View/replay saved waypoints")
    parser.add_argument("waypoints_file", help="Path to waypoints.npy")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Plot 3D visualization of FK")
    args = parser.parse_args()

    if not os.path.exists(args.waypoints_file):
        print(f"File not found: {args.waypoints_file}")
        return

    waypoints = np.load(args.waypoints_file)
    print_waypoints(waypoints)

    if args.visualize:
        visualize_waypoints(waypoints)


if __name__ == "__main__":
    main()
