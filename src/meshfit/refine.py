"""The render-match-lift refinement loop.

One iteration: place the mesh at the current pose, render it into every view,
match each render against the masked observation, lift both sides of every
match to 3D, and solve ONE delta over all views' pairs at once. Repeat.

Each iteration solves the pose OUTRIGHT rather than composing a correction
onto the previous one. Correspondences give canonical points and where they
were observed, which is everything a fit needs, and re-solving avoids
accumulating whatever the last estimate got wrong.

Two hypotheses compete every iteration:

    upright   Rz(yaw) @ R_up -- pitch and roll pinned to zero. The object
              stands on the support surface no matter what the matcher says.
    free      any rotation. Can recover a genuinely fallen or leaning object,
              and can equally invent a lean out of correspondence noise.

`tilt_mode="auto"` fits both and keeps the free one only if it leans by more
than `min_tilt_deg` AND renders to a better silhouette.

The angle threshold is the important half. The free fit has two extra degrees
of freedom, so its residual is ALWAYS lower -- comparing fits alone picks it
every time. But a small tilt and a large one mean different things: a few
degrees is what correspondence noise produces, while an object leaning 60
degrees is a claim no amount of noise makes by accident. Requiring a large
angle keeps the cases where tilt is real and rejects the ones where it is
drift.

There is deliberately no mode that inherits the initialisation's pitch and roll
untouched. That was the old behaviour and it guaranteed nothing -- an object
stayed at whatever lean the generator happened to guess, neither verified nor
correctable.

The renderer and matcher are arguments, not imports. They are the only parts
that need a GPU, a compiled rasteriser or a downloaded checkpoint, so injecting
them keeps this loop -- where the actual logic lives -- testable with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .correspond import Correspondence, lift_by_raycast, lift_from_pointmap
from .frames import Y_UP, Frame
from .metrics import mean_mask_iou
from .pose import Pose
from .solver import fit_similarity, fit_upright, ransac, tilt_of, yaw_of
from .views import Render, View


class Renderer(Protocol):
    """Draw a world-placed mesh into a view."""

    def __call__(self, mesh, intrinsic: np.ndarray, cam2world: np.ndarray,
                 image_hw: tuple[int, int]) -> Render: ...


class Matcher(Protocol):
    """Find correspondences between a render and an observation.

    Returns (kpts_rendered (N,2), kpts_observed (N,2), confidence (N,)).
    """

    def __call__(self, rendered_rgb: np.ndarray,
                 observed_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@dataclass
class RefineResult:
    pose: Pose
    correspondences: list[Correspondence] = field(default_factory=list)
    iterations: int = 0
    matches: int = 0
    inliers: int = 0
    converged: bool = False
    #: Views skipped for too little coverage or too few matches, per iteration.
    skipped: list[int] = field(default_factory=list)
    #: True when a free-rotation delta beat the upright one decisively enough
    #: to be believed -- i.e. the object looks genuinely tilted.
    used_free_rotation: bool = False
    #: Tilt [deg] of the final pose. Nonzero only via free rotation, since a
    #: yaw delta provably cannot change it.
    tilt_deg: float = 0.0

    @property
    def found_nothing(self) -> bool:
        """True when the loop never produced a usable delta.

        Worth checking explicitly: the honest outcome here is "I could not
        improve this", which looks identical to "this was already perfect" if
        you only inspect the pose.
        """
        return self.iterations == 0


def refine(
    mesh,
    pose: Pose,
    views: list[View],
    render: Renderer,
    match: Matcher,
    frame: Frame = Y_UP,
    *,
    iters: int = 3,
    min_matches: int = 10,
    min_coverage: int = 100,
    inlier_thresh: float = 0.05,
    ransac_iters: int = 1000,
    seed: int = 0,
    tilt_mode: str = "auto",
    min_tilt_deg: float = 20.0,
    debug=None,
) -> RefineResult:
    """Refine `pose` by matching renders of `mesh` against `views`.

    `mesh` is in its CANONICAL frame; the loop places it at the current pose
    each iteration rather than mutating vertices, so repeated passes do not
    accumulate drift.

    Views that render almost nothing (`min_coverage` pixels) or yield too few
    matches are skipped rather than allowed to contribute noise -- but if every
    view is skipped the result reports `found_nothing` instead of quietly
    returning the input pose as though it had been confirmed.

    `tilt_mode` is "auto", "upright" or "free". "upright" forces pitch and roll
    to zero, which is what you want for anything resting on a surface; "free"
    always re-estimates them; "auto" takes the free fit only when it leans more
    than `min_tilt_deg` and renders to a better silhouette than the upright one.
    """
    result = RefineResult(pose=pose.copy())

    for iteration in range(iters):
        placed = _placed_copy(mesh, result.pose)
        src_all, dst_all, pixels_all, views_all = [], [], [], []
        skipped = 0

        for view in views:
            drawn = render(placed, view.intrinsic, view.cam2world, view.image_hw)
            if debug is not None:
                debug.render(f"refine{iteration}", views.index(view), drawn)
            if drawn.coverage < min_coverage:
                skipped += 1
                continue

            k_render, k_observed, _conf = match(drawn.rgb, view.masked_rgb())
            on_object = _inside(k_render, drawn.mask)
            k_render, k_observed = k_render[on_object], k_observed[on_object]
            if debug is not None:
                debug.matches(f"refine{iteration}", views.index(view), drawn.rgb,
                              view.masked_rgb(), k_render, k_observed)
            if len(k_render) < min_matches:
                skipped += 1
                continue

            src, ok_src = lift_by_raycast(k_render, placed, view.intrinsic,
                                          view.cam2world)
            if view.pointmap is None:
                raise ValueError("refine needs per-view pointmaps to lift "
                                 "observed keypoints to 3D")
            dst, ok_dst = lift_from_pointmap(k_observed[ok_src], view.pointmap,
                                             view.cam2world, view.image_hw)
            src = src[ok_dst]
            if len(src) < min_matches:
                skipped += 1
                continue

            src_all.append(src)
            dst_all.append(dst)
            pixels_all.append(k_observed[ok_src][ok_dst])
            views_all.append(view)

        result.skipped.append(skipped)
        if not src_all:
            break

        # World points on the rendered surface, mapped back to the canonical
        # frame. Exact regardless of whether the current pose is any good --
        # the mesh was literally placed there, so inverting that placement
        # recovers the mesh's own coordinates.
        src = result.pose.inverse_apply(np.concatenate(src_all))
        dst = np.concatenate(dst_all)

        previous = result.pose
        pose, inliers, used_free = _choose(
            src, dst, frame, tilt_mode, min_tilt_deg, inlier_thresh,
            ransac_iters, seed + iteration,
            score=lambda p: _silhouette_score(mesh, p, views, render))

        result.used_free_rotation = used_free
        result.pose = pose
        result.iterations = iteration + 1
        result.matches = len(src)
        result.inliers = int(inliers.sum())
        result.correspondences = _pack(src, dst, pixels_all, views_all, src_all,
                                       inliers)
        result.tilt_deg = tilt_of(pose.R, frame)

        if _converged(previous, pose, frame):
            result.converged = True
            break

    return result


def _choose(src: np.ndarray, dst: np.ndarray, frame: Frame, tilt_mode: str,
            min_tilt_deg: float, inlier_thresh: float, iters: int, seed: int,
            score):
    """Fit the pose outright, choosing between the upright and free hypotheses."""
    if tilt_mode not in ("auto", "upright", "free"):
        raise ValueError(f"tilt_mode must be auto|upright|free, got {tilt_mode!r}")

    def upright_fit(x, y):
        return fit_upright(x, y, frame)

    if tilt_mode == "upright":
        pose, inliers = ransac(src, dst, upright_fit, min_samples=2,
                               inlier_thresh=inlier_thresh, iters=iters, seed=seed)
        return pose, inliers, False

    free, free_inliers = ransac(src, dst, fit_similarity, min_samples=3,
                                inlier_thresh=inlier_thresh, iters=iters, seed=seed)
    if tilt_mode == "free":
        return free, free_inliers, True

    upright, upright_inliers = ransac(src, dst, upright_fit, min_samples=2,
                                      inlier_thresh=inlier_thresh, iters=iters,
                                      seed=seed)

    # A lean smaller than the threshold is what noise produces; do not pay for
    # the plausibility guarantee to buy it.
    if tilt_of(free.R, frame) < min_tilt_deg:
        return upright, upright_inliers, False

    # Large enough to be a real claim -- now make it earn its place on the
    # metric that matters, rather than on a residual it is guaranteed to win.
    if score(free) > score(upright):
        return free, free_inliers, True
    return upright, upright_inliers, False


def _silhouette_score(mesh, pose: Pose, views, render) -> float:
    """Mean rendered silhouette overlap across views."""
    return 0.0 if render is None else mean_mask_iou(views, mesh, pose, render)


def _inside(kpts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Which keypoints land on the rendered object.

    A match anchored to background pixels of the render corresponds to no
    surface at all, so it cannot be lifted and would only feed RANSAC noise.
    Cheaper and far more meaningful than thresholding a matcher's confidence.
    """
    if not len(kpts):
        return np.zeros(0, bool)
    H, W = mask.shape
    u = np.clip(np.round(kpts[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(kpts[:, 1]).astype(int), 0, H - 1)
    return mask[v, u]


def _placed_copy(mesh, pose: Pose):
    placed = mesh.copy()
    placed.vertices = pose.apply(np.asarray(mesh.vertices, dtype=float))
    return placed


def _converged(before: Pose, after: Pose, frame: Frame, *,
               scale_tol: float = 1e-3, trans_tol: float = 1e-3,
               yaw_tol_deg: float = 0.1) -> bool:
    """Did this iteration move the pose enough to be worth another?"""

    scale_change = float(np.max(np.abs(after.scale / np.maximum(before.scale, 1e-9) - 1.0)))
    yaw_change = abs(yaw_of(after.R, frame) - yaw_of(before.R, frame))
    return (scale_change < scale_tol
            and float(np.linalg.norm(after.t - before.t)) < trans_tol
            and yaw_change < np.radians(yaw_tol_deg))


def _pack(src_canonical: np.ndarray, dst: np.ndarray, pixels_all, views_all,
          src_all, inliers: np.ndarray) -> list[Correspondence]:
    """Split the pooled inliers back out per view.

    These are TRUE correspondences -- a matcher said so -- unlike the
    nearest-neighbour guesses ICP works with, so later stages reuse them rather
    than re-associating. They are already in canonical coordinates, which is
    what lets `fit_joint` re-solve from scratch.
    """
    out, offset = [], 0
    for chunk, pixels, view in zip(src_all, pixels_all, views_all, strict=True):
        span = slice(offset, offset + len(chunk))
        keep = inliers[span]
        offset += len(chunk)
        if not keep.any():
            continue
        out.append(Correspondence(canonical=src_canonical[span][keep],
                                  pixel=pixels[keep], view=view,
                                  world=dst[span][keep]))
    return out
