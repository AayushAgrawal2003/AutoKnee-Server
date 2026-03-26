"""
Constant-velocity Extended Kalman Filter for 6-DOF pose smoothing.

State (12): [x, y, z, vx, vy, vz, rz, ry, rx, wz, wy, wx]
Measurement (6): [x, y, z, rz, ry, rx]

Handles:
- Jitter smoothing via Kalman update
- Dropout bridging via predict-only when tracking is lost
- Euler angle wrapping in the innovation step
"""

import numpy as np


class PoseEKF:
    """Per-tracker Kalman filter for 6-DOF pose."""

    def __init__(self, max_misses=60, q_pos=0.5, q_vel=50.0,
                 q_ang=0.001, q_angvel=0.5, r_pos=1.0, r_ang=0.01):
        """
        Args:
            max_misses: frames of prediction before giving up
            q_pos/q_vel: process noise for position/velocity (mm^2)
            q_ang/q_angvel: process noise for angle/angular vel (rad^2)
            r_pos: measurement noise position (mm^2)
            r_ang: measurement noise angle (rad^2)
        """
        self.n = 12
        self.m = 6
        self.max_misses = max_misses

        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 1000.0
        self.initialized = False
        self.miss_count = 0
        self._last_time = None

        # Measurement matrix: pick [x,y,z,rz,ry,rx] from state
        self.H = np.zeros((self.m, self.n))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 6] = 1.0
        self.H[4, 7] = 1.0
        self.H[5, 8] = 1.0

        self.R = np.diag([r_pos, r_pos, r_pos, r_ang, r_ang, r_ang])

        self._q_pos = q_pos
        self._q_vel = q_vel
        self._q_ang = q_ang
        self._q_angvel = q_angvel

    def _F(self, dt):
        F = np.eye(self.n)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        F[6, 9] = dt
        F[7, 10] = dt
        F[8, 11] = dt
        return F

    def _Q(self, dt):
        # Piecewise constant acceleration noise model
        dt2 = dt * dt
        dt3 = dt2 * dt
        q = np.zeros((self.n, self.n))
        for i, qi in [(0, self._q_pos), (1, self._q_pos), (2, self._q_pos)]:
            vi = i + 3
            q[i, i] = dt3 / 3 * qi
            q[i, vi] = dt2 / 2 * qi
            q[vi, i] = dt2 / 2 * qi
            q[vi, vi] = dt * qi
        for i, qi in [(6, self._q_ang), (7, self._q_ang), (8, self._q_ang)]:
            vi = i + 3
            q[i, i] = dt3 / 3 * qi
            q[i, vi] = dt2 / 2 * qi
            q[vi, i] = dt2 / 2 * qi
            q[vi, vi] = dt * qi
        # Add base velocity/angular velocity noise
        for i in range(3, 6):
            q[i, i] += self._q_vel * dt
        for i in range(9, 12):
            q[i, i] += self._q_angvel * dt
        return q

    def _predict(self, dt):
        F = self._F(dt)
        Q = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def _update(self, z):
        innovation = z - self.H @ self.x
        # Wrap angle innovations to [-pi, pi]
        for i in range(3, 6):
            innovation[i] = (innovation[i] + np.pi) % (2.0 * np.pi) - np.pi
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form

    def process(self, T_4x4, visible, t_now):
        """
        Process one frame.

        Args:
            T_4x4: 4x4 homogeneous transform (may contain NaN if not visible)
            visible: bool, whether tracker was detected this frame
            t_now: monotonic timestamp (seconds)

        Returns:
            (T_filtered, is_valid): filtered 4x4 transform, and whether
            the output should be displayed
        """
        if self._last_time is None:
            self._last_time = t_now

        dt = t_now - self._last_time
        dt = max(dt, 1e-4)  # avoid zero dt
        self._last_time = t_now

        if not visible:
            if not self.initialized:
                return T_4x4, False
            self.miss_count += 1
            if self.miss_count > self.max_misses:
                return T_4x4, False
            self._predict(dt)
            return self._state_to_T(), True

        z = self._T_to_z(T_4x4)

        if not self.initialized:
            self.x[:3] = z[:3]
            self.x[6:9] = z[3:6]
            self.P = np.eye(self.n) * 100.0
            # Zero velocity initially
            self.P[3, 3] = self.P[4, 4] = self.P[5, 5] = 1000.0
            self.P[9, 9] = self.P[10, 10] = self.P[11, 11] = 1000.0
            self.initialized = True
            self.miss_count = 0
            return T_4x4, True

        self._predict(dt)
        self._update(z)
        self.miss_count = 0
        return self._state_to_T(), True

    def reset(self):
        """Reset filter state."""
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 1000.0
        self.initialized = False
        self.miss_count = 0
        self._last_time = None

    @staticmethod
    def _T_to_z(T):
        """Extract [x, y, z, rz, ry, rx] from a 4x4 transform."""
        pos = T[:3, 3]
        R = T[:3, :3]
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0
        return np.array([pos[0], pos[1], pos[2], rz, ry, rx])

    @staticmethod
    def _state_to_T():
        """Would need self — defined below."""
        pass

    def _state_to_T(self):
        """Reconstruct 4x4 transform from EKF state."""
        T = np.eye(4)
        T[:3, 3] = self.x[:3]
        rz, ry, rx = self.x[6], self.x[7], self.x[8]
        cz, sz = np.cos(rz), np.sin(rz)
        cy, sy = np.cos(ry), np.sin(ry)
        cx, sx = np.cos(rx), np.sin(rx)
        T[0, 0] = cz * cy
        T[0, 1] = cz * sy * sx - sz * cx
        T[0, 2] = cz * sy * cx + sz * sx
        T[1, 0] = sz * cy
        T[1, 1] = sz * sy * sx + cz * cx
        T[1, 2] = sz * sy * cx - cz * sx
        T[2, 0] = -sy
        T[2, 1] = cy * sx
        T[2, 2] = cy * cx
        return T
