"""The joint nonlinear solve.

Where the closed-form fits answer 'what pose maps these points onto those', this
answers 'what pose best explains these PIXELS' -- which is a different and
better-posed question, because a matcher measures pixels and a depth map only
guesses at metres."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..correspond import Correspondence
from ..frames import Y_UP, Frame
from ..pose import Pose
from .rotations import _rms, rz, tilt_of, yaw_of


def _pack(pose: Pose, frame: Frame, free_rotation: bool) -> np.ndarray:
    """Pose -> optimiser vector.

    Rotation is one yaw angle (7-parameter form) or a rotation vector
    (9-parameter form). A rotation vector has no gimbal lock and no constraint
    to maintain, which is what makes it well behaved under a least-squares step.

    Scale is carried as log-scale either way: positive without a constraint,
    and with all three axes on a common footing so a step means the same thing
    for each.
    """
    if free_rotation:
        return np.concatenate([Rotation.from_matrix(pose.R).as_rotvec(),
                               np.log(pose.scale), pose.t])
    return np.concatenate([[yaw_of(pose.R, frame)], np.log(pose.scale), pose.t])


def _unpack(v: np.ndarray, frame: Frame, free_rotation: bool) -> Pose:
    if free_rotation:
        return Pose(R=Rotation.from_rotvec(v[0:3]).as_matrix(),
                    scale=np.exp(v[3:6]), t=v[6:9])
    return Pose(R=rz(v[0]) @ frame.R_up, scale=np.exp(v[1:4]), t=v[4:7])


def fit_joint(
    corrs: list[Correspondence],
    init: Pose,
    frame: Frame = Y_UP,
    *,
    sigma_px: float = 1.0,
    sigma_depth: float = 0.05,
    huber_px: float = 3.0,
    max_iters: int = 100,
    free_rotation: bool = False,
) -> Pose:
    """Rotation + per-axis scale + translation, solved TOGETHER in pixels.

    Two things set this apart from the closed-form fits:

    It minimises pixels, not metres. A matcher measures a 2D keypoint to about
    a pixel; lifting that through a depth map to get a 3D target folds the
    depth estimator's error into the measurement, and a 3D residual then
    weights a ~1px lateral observation and a ~10cm depth guess equally. Here
    the 2D term stays 2D, and the 3D pairs enter only as a weak prior through
    `sigma_depth`.

    That prior is not optional for single-view input. Reprojection alone is
    gauge-degenerate under one camera: slide the object along the viewing ray
    while scaling it proportionally and every pixel is unchanged. The depth
    term is what makes scale observable at all, which is why it is a prior
    rather than a residual to be minimised away.

    It also solves yaw and the three scales jointly, which no closed-form fit
    here can: anisotropy cannot be iterated (the set {R diag(S)} is not closed
    under composition), so the alternative is running yaw first and scales
    after and hoping they agree. An optimiser updates a parameter vector and
    never composes transforms, so the obstruction does not apply.

    `free_rotation` selects the parameterisation, and should match whatever the
    refinement stage concluded:

        False   7 params: yaw, three scales, translation. Pitch and roll are
                pinned to zero, so an upright object stays upright.
        True    9 params: rotation vector, three scales, translation. Keeps and
                refines a lean.

    Getting this wrong is not a small error. Polishing a tilted pose with the
    7-parameter form cannot represent the answer, so it stands the object up
    and the fit collapses -- on a 62-degree object we measured reprojection
    going from 12px to 53px.

    `sigma_px` and `sigma_depth` are the relative trust in each term -- their
    ratio is what does the work, not their absolute values.

    Note that a large RATIO between the per-axis scales is not by itself a sign
    of trouble: generated meshes have non-uniform canonical extents, so a 1.7x
    scale on a squat axis can be what makes the placed object isotropic. Judge
    anisotropy on `canonical_extent * scale`, never on the scales alone.
    """

    usable = [c for c in corrs if len(c)]
    if not usable:
        raise ValueError("no correspondences to fit")
    for c in usable:
        c.validate()

    if not frame.is_axis_aligned:
        raise ValueError("fit_joint needs an axis-aligned canonical_up")

    tilt = tilt_of(init.R, frame)
    if tilt > 1.0 and not free_rotation:
        warnings.warn(
            f"fit_joint is in its 7-parameter form, which pins pitch and roll "
            f"to zero, but the initial pose leans {tilt:.1f} deg. That tilt "
            f"cannot survive. Pass free_rotation=True to keep it.",
            stacklevel=2)

    def residuals(v: np.ndarray) -> np.ndarray:
        pose = _unpack(v, frame, free_rotation)
        out = []
        for c in usable:
            world = pose.apply(c.canonical)
            uv, in_front = c.project(world)
            # a point behind the camera gets a large but finite penalty, so the
            # optimiser is pushed back into the valid region instead of hitting
            # a non-finite residual and giving up
            err = np.where(in_front[:, None], uv - c.pixel, 10.0 * huber_px)
            out.append((err * c.weight[:, None] / sigma_px).ravel())
            if c.world is not None:
                out.append(((world - c.world) * c.weight[:, None] / sigma_depth).ravel())
        return np.concatenate(out)

    v0 = _pack(Pose(R=init.R, scale=np.maximum(init.scale, 1e-6), t=init.t),
               frame, free_rotation)
    result = least_squares(
        residuals, v0,
        loss="huber", f_scale=huber_px,      # matcher outliers survive RANSAC
        max_nfev=max_iters * (len(v0) + 1),
        x_scale="jac",
    )

    pose = _unpack(result.x, frame, free_rotation)
    x_all, y_all = [], []
    for c in usable:
        if c.world is not None:
            x_all.append(c.canonical)
            y_all.append(c.world)
    pose.rms = (_rms(pose, np.concatenate(x_all), np.concatenate(y_all))
                if x_all else float(np.sqrt(np.mean(result.fun**2))))
    return pose


def reprojection_rms(pose: Pose, corrs: list[Correspondence]) -> float:
    """Mean reprojection error [px] -- the metric `fit_joint` actually reduces,
    and the one to report next to a 3D residual rather than instead of it."""
    errs = []
    for c in corrs:
        if not len(c):
            continue
        uv, in_front = c.project(pose.apply(c.canonical))
        if in_front.any():
            errs.append(np.linalg.norm(uv[in_front] - c.pixel[in_front], axis=1))
    if not errs:
        return float("inf")
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
