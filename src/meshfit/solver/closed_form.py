"""Closed-form pose fits on paired points.

Direct solves -- no iteration, no learning rate, no local minima. Each takes
matched point pairs and returns the `Pose` that best maps one set onto the
other under some constraint.

The constraint is the point. `fit_upright` cannot return a tilted object
because tilt is not in its parameter space. Outliers can bias the answer, but
they cannot tip the object over. `fit_similarity` removes that constraint and
exists as a competing hypothesis -- when it fits decisively better, the object
probably really is tilted.

Being closed form is also what makes robustness affordable: RANSAC needs a
thousand hypotheses, and a thousand microsecond solves is nothing while a
thousand nonlinear ones is not."""

from __future__ import annotations

import numpy as np

from ..frames import Y_UP, Frame
from ..pose import Pose
from .rotations import _rms, rz


def fit_upright(x_canonical: np.ndarray, y_world: np.ndarray,
                frame: Frame = Y_UP) -> Pose:
    """4-DoF fit: yaw + isotropic scale + translation. Upright by construction.

    Closed form, because under the upright constraint the problem separates:
    yaw comes from the 2D cross-covariance in the support plane (and is
    scale-independent), scale is then a 1D least squares, and translation
    follows from the centroids.
    """
    x = x_canonical @ frame.R_up.T                 # canonical, now Z-up
    xm, ym = x.mean(axis=0), y_world.mean(axis=0)
    xc, yc = x - xm, y_world - ym

    theta = float(np.arctan2(
        np.sum(xc[:, 0] * yc[:, 1] - xc[:, 1] * yc[:, 0]),
        np.sum(xc[:, 0] * yc[:, 0] + xc[:, 1] * yc[:, 1]),
    ))
    R_yaw = rz(theta)
    s = float(np.sum(yc * (xc @ R_yaw.T)) / max(np.sum(xc**2), 1e-12))
    s = max(s, 1e-6)

    pose = Pose.similarity(R=R_yaw @ frame.R_up, s=s, t=ym - s * (R_yaw @ xm))
    pose.rms = _rms(pose, x_canonical, y_world)
    return pose


def fit_similarity(x_canonical: np.ndarray, y_world: np.ndarray) -> Pose:
    """7-DoF Umeyama fit: unconstrained rotation + isotropic scale + translation.

    The competing hypothesis to `fit_upright`. Nothing stops it returning a
    tilted pose, which is exactly what makes it informative.
    """
    xm, ym = x_canonical.mean(axis=0), y_world.mean(axis=0)
    xc, yc = x_canonical - xm, y_world - ym

    U, D, Vt = np.linalg.svd(yc.T @ xc / len(xc))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:    # forbid reflections
        S[2, 2] = -1.0
    R = U @ S @ Vt

    s = float(np.trace(np.diag(D) @ S) / max(np.mean(np.sum(xc**2, axis=1)), 1e-12))
    s = max(s, 1e-6)

    pose = Pose.similarity(R=R, s=s, t=ym - s * (R @ xm))
    pose.rms = _rms(pose, x_canonical, y_world)
    return pose


def fit_anisotropic(x_canonical: np.ndarray, y_world: np.ndarray,
                    prior: np.ndarray, frame: Frame = Y_UP,
                    reg: float = 0.1, init_theta: float = 0.0,
                    iters: int = 8) -> Pose:
    """Upright fit with PER-AXIS scale: `y ~= Rz(theta) @ (S * R_up @ x) + t`.

    General anisotropic Procrustes has no closed form, but the yaw-only
    constraint makes it separable: the vertical axis is a 1D ridge solve, and
    the horizontal pair alternates between scales and yaw, converging in a few
    passes.

    Each axis is ridge-regularised toward `prior[axis]` with strength
    `reg * mean-per-axis-energy`. Axes the correspondences barely constrain --
    the unobserved depth of a single-view object -- stay at the prior instead
    of collapsing; well-observed axes follow the data.
    """
    if not frame.is_axis_aligned:
        raise ValueError("fit_anisotropic needs an axis-aligned canonical_up "
                         "(per-axis scales are reordered by abs(R_up))")

    x = x_canonical @ frame.R_up.T                  # canonical, Z-up: (u, v, up)
    xm, ym = x.mean(axis=0), y_world.mean(axis=0)
    xc, yc = x - xm, y_world - ym

    prior_zup = np.asarray(prior, dtype=float) @ np.abs(frame.R_up.T)
    lam = reg * float(np.sum(xc**2)) / 3.0

    # vertical axis decouples entirely: plain 1D ridge least squares
    s_up = ((np.sum(xc[:, 2] * yc[:, 2]) + lam * prior_zup[2])
            / (np.sum(xc[:, 2] ** 2) + lam))

    # horizontal pair: alternate (scales | yaw)
    theta = float(init_theta)
    s_u, s_v = prior_zup[0], prior_zup[1]
    for _ in range(iters):
        c, s_ = np.cos(theta), np.sin(theta)
        rotated = np.stack([c * yc[:, 0] + s_ * yc[:, 1],
                            -s_ * yc[:, 0] + c * yc[:, 1]], axis=1)
        s_u = ((np.sum(xc[:, 0] * rotated[:, 0]) + lam * prior_zup[0])
               / (np.sum(xc[:, 0] ** 2) + lam))
        s_v = ((np.sum(xc[:, 1] * rotated[:, 1]) + lam * prior_zup[1])
               / (np.sum(xc[:, 1] ** 2) + lam))
        scaled = np.stack([s_u * xc[:, 0], s_v * xc[:, 1]], axis=1)
        theta_next = float(np.arctan2(
            np.sum(scaled[:, 0] * yc[:, 1] - scaled[:, 1] * yc[:, 0]),
            np.sum(scaled[:, 0] * yc[:, 0] + scaled[:, 1] * yc[:, 1]),
        ))
        if abs(theta_next - theta) < 1e-9:
            theta = theta_next
            break
        theta = theta_next

    scale_zup = np.maximum([s_u, s_v, s_up], 1e-6)
    scale_canonical = scale_zup @ np.abs(frame.R_up)      # back to canonical order

    pose = Pose(R=rz(theta) @ frame.R_up,
                scale=scale_canonical,
                t=ym - rz(theta) @ (scale_zup * xm))
    pose.rms = _rms(pose, x_canonical, y_world)
    return pose
