"""
Multi-tracker detection and position publishing.

Connects to the Polaris Vega via vega_discover, loads multiple ROM tool
definitions, and continuously publishes 6-DOF poses for every visible tracker.

The default mode loads the femur (SPH) and gray (Polaris) ROMs. Because the
femur tracker uses Polaris-type flat disc reflectors but only has an SPH ROM
supported by the NDI system, the reported femur position is corrected for the
systematic X-offset difference (SPH=8.770mm vs Polaris=-2.127mm = 10.897mm
along the tool's local X axis).

Usage:
    python3 vega_multi_track.py
    python3 vega_multi_track.py --path /path/a.rom /path/b.rom
    python3 vega_multi_track.py --measure
"""

import sys
import os
import time
import signal
import atexit
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vega_discover import discover_vega, VegaNotFoundError

from sksurgerynditracker.nditracker import NDITracker

# ---------------------------------------------------------------------------
# ROM paths and X-offset correction
# ---------------------------------------------------------------------------
ROM_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "roms"))

FEMUR_SPH_ROM = os.path.join(ROM_DIR, "BBT-110017Rev1-FemurTracker-SPH.rom")
GRAY_POLARIS_ROM = os.path.join(ROM_DIR, "BBT-TrackerA-Gray_Polaris.rom")

# X-offset correction for femur tracker (SPH ROM + Polaris flat disc markers).
# SPH ROM encodes marker X = 8.770mm; actual flat disc X = -2.127mm.
# The NDI-computed tool origin is shifted by -(8.770 - (-2.127)) = -10.897mm
# along the tool's local X axis. We correct by adding +10.897mm along tool X.
FEMUR_SPH_X_CORRECTION_MM = 8.770 - (-2.127)  # = 10.897

FEMUR_ROM_BASENAME = os.path.basename(FEMUR_SPH_ROM)


def _rom_label(path):
    return os.path.splitext(os.path.basename(path))[0]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

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


def apply_x_correction(T, dx_mm):
    """Shift the tool origin along its local X axis by dx_mm."""
    T_corrected = T.copy()
    T_corrected[:3, 3] += dx_mm * T[:3, 0]
    return T_corrected


# ---------------------------------------------------------------------------
# Tracker wrapper with cleanup
# ---------------------------------------------------------------------------

tracker = None

def cleanup():
    global tracker
    if tracker is not None:
        try:
            tracker.stop_tracking()
            tracker.close()
        except Exception:
            pass
        tracker = None

atexit.register(cleanup)
signal.signal(signal.SIGTERM, lambda *_: (cleanup(), sys.exit(0)))


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def _print_frame(labels, tracking, quality, frame_num, timestamp):
    """Print a human-readable frame to stdout."""
    lines = [f"{DIM}--- frame {frame_num}  t={timestamp:.3f}s ---{RESET}"]

    for i, label in enumerate(labels):
        T = tracking[i]
        q = quality[i]
        visible = not np.isnan(T[0, 0])

        if visible:
            x, y, z = T[0, 3], T[1, 3], T[2, 3]
            rz, ry, rx = rotation_matrix_to_euler_zyx(T[:3, :3])
            status = f"{GREEN}TRACKING{RESET}"
            pos = f"pos=({x:8.2f}, {y:8.2f}, {z:8.2f}) mm"
            rot = f"euler=({rz:7.2f}, {ry:7.2f}, {rx:7.2f})deg"
            qual = f"q={float(q):.4f}"
            lines.append(f"  {CYAN}{label:<40}{RESET} {status}  {pos}  {rot}  {qual}")
        else:
            lines.append(f"  {CYAN}{label:<40}{RESET} {RED}NOT VISIBLE{RESET}")

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global tracker

    parser = argparse.ArgumentParser(
        description="Multi-tracker detection & position publishing via Polaris Vega.",
    )
    parser.add_argument(
        "--path", nargs="+", metavar="ROM",
        help="ROM file paths to track",
    )
    parser.add_argument(
        "--measure", action="store_true",
        help="Measure tracking rate (Hz); prints result on Ctrl+C",
    )
    parser.add_argument(
        "--corrected", action="store_true",
        help="Apply +10.897mm X-offset correction to femur tracker (SPH ROM with Polaris flat disc markers)",
    )
    parser.add_argument(
        "--ekf", action="store_true",
        help="Apply EKF smoothing and dropout bridging to all tracked poses",
    )
    args = parser.parse_args()

    # Resolve ROM files and per-tool X corrections
    if args.path:
        romfiles = [os.path.abspath(p) for p in args.path]
    else:
        romfiles = [FEMUR_SPH_ROM, GRAY_POLARIS_ROM]

    # Build per-tool X corrections (only when --corrected is set)
    x_corrections = [0.0] * len(romfiles)
    if args.corrected:
        for i, rf in enumerate(romfiles):
            if os.path.basename(rf) == FEMUR_ROM_BASENAME:
                x_corrections[i] = FEMUR_SPH_X_CORRECTION_MM

    # Validate ROM files exist
    for rf in romfiles:
        if not os.path.isfile(rf):
            print(f"ROM file not found: {rf}")
            sys.exit(1)

    labels = [_rom_label(rf) for rf in romfiles]

    if not args.measure:
        print(f"Loading {len(romfiles)} tracker(s):")
        for label, path, dx in zip(labels, romfiles, x_corrections):
            corr = f"  (X correction: +{dx:.3f}mm)" if dx != 0 else ""
            print(f"  {label}{corr}")
            print(f"    {path}")

    # Discover Vega
    try:
        info = discover_vega(verbose=not args.measure)
    except VegaNotFoundError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    # Connect
    settings = info.as_settings(romfiles)
    print(f"Connecting to {info.ip}:{info.port}...")
    tracker = NDITracker(settings)
    tracker.start_tracking()

    # EKF instances (one per tracker)
    ekf_filters = None
    if args.ekf:
        from pose_ekf import PoseEKF
        ekf_filters = [PoseEKF(max_misses=60) for _ in romfiles]

    if args.measure:
        print("Measuring tracking rate... press Ctrl+C to stop.")
    else:
        ekf_str = " + EKF" if args.ekf else ""
        print(f"Tracking started — {len(romfiles)} tool(s){ekf_str}\n")

    t0 = time.monotonic()
    frame_count = 0

    try:
        while True:
            port_handles, timestamps, framenumbers, tracking, quality = tracker.get_frame()
            frame_count += 1

            if not args.measure:
                now = time.monotonic()
                # Apply X-offset corrections
                for i, dx in enumerate(x_corrections):
                    if dx != 0 and not np.isnan(tracking[i][0, 0]):
                        tracking[i] = apply_x_correction(tracking[i], dx)

                # Apply EKF if enabled
                if ekf_filters:
                    for i in range(len(romfiles)):
                        vis = not np.isnan(tracking[i][0, 0])
                        T_filt, valid = ekf_filters[i].process(
                            tracking[i], vis, now)
                        if valid:
                            tracking[i] = T_filt
                        # If EKF predicts during dropout, mark as visible
                        # by leaving the filtered transform in place

                elapsed = now - t0
                _print_frame(labels, tracking, quality, frame_count, elapsed)

    except KeyboardInterrupt:
        elapsed = time.monotonic() - t0

        if args.measure:
            hz = frame_count / elapsed if elapsed > 0 else 0
            print(f"\n{BOLD}Measurement results:{RESET}")
            print(f"  Frames:   {frame_count}")
            print(f"  Duration: {elapsed:.3f}s")
            print(f"  Rate:     {hz:.1f} Hz")
        else:
            print(f"\n{DIM}Interrupted after {frame_count} frames.{RESET}")

    print("Stopping tracker...")
    tracker.stop_tracking()
    tracker.close()
    tracker = None
    print("Done.")


if __name__ == "__main__":
    main()
