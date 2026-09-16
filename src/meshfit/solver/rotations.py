"""Rotation helpers shared by every fit.

`yaw_of` and `tilt_of` are the two numbers that describe an upright-constrained
pose: how far round it is turned, and how far it leans. Tilt is the one that
cannot be changed by a yaw rotation -- `Rz` preserves the z-component of any
vector, so it leaves pitch and roll algebraically untouched."""

from __future__ import annotations

import numpy as np

from ..frames import Y_UP, Frame
from ..pose import Pose


def rz(theta: float) -> np.ndarray:
    """Rotation about world +Z (the only free rotational axis when upright)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(R: np.ndarray, frame: Frame = Y_UP) -> float:
    """Yaw [rad] of a rotation assumed to be `Rz(yaw) @ frame.R_up`.

    A tilted `R` is projected rather than rejected: the returned yaw is the
    best upright approximation. Pair it with `tilt_of` when you need to know
    whether that projection discarded anything.
    """
    M = R @ frame.R_up.T
    return float(np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1]))


def tilt_of(R: np.ndarray, frame: Frame = Y_UP) -> float:
    """Angle [deg] between the object's up axis and world +Z.

    Zero for any upright pose. Large values from `fit_similarity` are the
    signal that either the object really is fallen, or `canonical_up` is wrong.
    """
    up_world = R @ frame.up
    return float(np.degrees(np.arccos(np.clip(up_world[2], -1.0, 1.0))))

def _rms(pose: Pose, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pose.apply(x) - y) ** 2, axis=1))))


def project_upright(pose: Pose, frame: Frame = Y_UP) -> Pose:
    """Drop a pose's tilt, keeping its yaw, scale and translation."""
    return Pose(R=rz(yaw_of(pose.R, frame)) @ frame.R_up,
                scale=pose.scale.copy(), t=pose.t.copy(), rms=pose.rms)
