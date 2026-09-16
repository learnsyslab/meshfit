"""Getting a starting pose.

Refinement is local. It renders the mesh at the current pose and matches that
render against the photograph, so if the mesh starts far enough away that it
renders off-frame or at wildly wrong size, there is nothing to match and the
loop reports failure. Worse, the constrained delta is yaw-only by design, so
refinement can never fix an object that starts lying on its side. Orientation
and rough scale have to be right BEFORE the first render.

Three ways to get there, in descending order of how much you should trust them:

`from_generator`  A pose-aware backend (SAM3D, recgen) already solved this.
                  Use it. Nothing here beats the model that made the mesh.
`from_search`     Render a ring of yaw hypotheses and keep the best. For
                  pose-blind generators (TRELLIS and most image-to-3D).
`from_bbox`       Centroid and extent, no search. Cheap, and correct only up
                  to yaw -- fine as a seed for the search, rarely on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .frames import Y_UP, Frame
from .metrics import mask_iou
from .pose import Pose
from .solver import project_upright, rz
from .views import Observation, View


@dataclass
class Symmetry:
    """How much the yaw sweep actually preferred its winner.

    Not a probability -- a shape descriptor for the score curve. A chair with a
    distinctive back gives one sharp peak. A square table gives four equal ones.
    A vase gives a flat line.

    The distinction matters downstream, not visually: picking the wrong peak on
    a symmetric object costs nothing in a render (which is why the scores tie)
    but can put a cabinet's drawers against the wall, or a mug's handle where no
    gripper can reach.
    """

    margin: float                  # (best - best rival) / best, 0 when tied
    k_fold: int                    # peaks within noise of the winner
    scores: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    yaws: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def is_ambiguous(self) -> bool:
        """True when some other yaw explains the image about as well."""
        return self.k_fold > 1 or self.margin < 0.05


@dataclass
class Initialization:
    pose: Pose
    source: str                    # generator | search | bbox | icp
    symmetry: Symmetry | None = None

    @property
    def trustworthy(self) -> bool:
        return self.symmetry is None or not self.symmetry.is_ambiguous


# ---------------------------------------------------------------------------
# from the generator
# ---------------------------------------------------------------------------


def from_generator(T_world_canonical: np.ndarray,
                   scale: float | np.ndarray,
                   frame: Frame = Y_UP,
                   *, force_upright: bool = False) -> Initialization:
    """Adopt a pose-aware generator's own estimate.

    `T_world_canonical` is the rigid 4x4 and `scale` the metric scale, matching
    what SAM3D writes to `result.json`. The pose must already be in the same
    world frame as the point cloud; a generator reporting in camera frame needs
    composing with `cam2world` first.

    The pose is adopted as given. `force_upright` projects out any tilt, but is
    off by default: plenty of objects really are tilted -- a controller dropped
    on a desk, a fallen bottle -- and an initialiser has no evidence with which
    to tell those apart from a generator being a few degrees sloppy. That is a
    decision for the constrained-vs-free hypothesis test downstream, which has
    residuals to compare.

    The cost of leaving tilt in is real, though: refinement is yaw-only, so
    whatever tilt is here survives to the end unless a later stage projects it
    out deliberately.
    """
    T = np.asarray(T_world_canonical, dtype=float).reshape(4, 4)
    pose = Pose(R=T[:3, :3], scale=scale, t=T[:3, 3])
    if force_upright:
        pose = project_upright(pose, frame)
    return Initialization(pose=pose, source="generator")


# ---------------------------------------------------------------------------
# from the point cloud alone
# ---------------------------------------------------------------------------


def from_bbox(mesh, observation: Observation, frame: Frame = Y_UP,
              yaw: float = 0.0, percentile: float = 1.0) -> Initialization:
    """Match the mesh's bounding box to the observed points'.

    Scale comes from the VERTICAL extent, not the mean of all three. For an
    object standing on a floor the full height is usually visible, whereas the
    horizontal extents of a single-view cloud cover only the front surface and
    would bias the scale small. Translation likewise aligns the base and the
    horizontal centre rather than the centroids, since a partial cloud's
    centroid sits toward the camera.

    Yaw is not estimated -- there is nothing in a bounding box to estimate it
    from. Seed `from_search` with this.

    Extents are percentile-trimmed, not min/max: masked clouds carry depth
    bleed from the silhouette edge, and a 1% tail is enough to inflate an
    extent several-fold (see `Observation.robust_bounds`).
    """
    verts_up = np.asarray(mesh.vertices, dtype=float) @ frame.R_up.T
    mesh_extent = np.percentile(verts_up, 100.0 - percentile, axis=0) \
        - np.percentile(verts_up, percentile, axis=0)
    lo, hi = observation.robust_bounds(percentile)

    scale = float((hi[2] - lo[2]) / max(mesh_extent[2], 1e-9))
    R = rz(yaw) @ frame.R_up

    # place the base on the observed base, and centre it horizontally
    scaled = (scale * verts_up) @ rz(yaw).T
    target = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    def _mid(axis: int) -> float:
        lo_a = np.percentile(scaled[:, axis], percentile)
        hi_a = np.percentile(scaled[:, axis], 100.0 - percentile)
        return float((lo_a + hi_a) / 2)

    current = np.array([_mid(0), _mid(1),
                        float(np.percentile(scaled[:, 2], percentile))])

    return Initialization(
        pose=Pose.similarity(R=R, s=scale, t=target - current),
        source="bbox",
    )


# ---------------------------------------------------------------------------
# yaw search
# ---------------------------------------------------------------------------


def from_search(
    mesh,
    observation: Observation,
    render=None,
    match=None,
    frame: Frame = Y_UP,
    *,
    yaw_steps: int = 36,
    top_k: int = 3,
    view: View | None = None,
    debug=None,
) -> Initialization:
    """Score a ring of yaw hypotheses and keep the best.

    A full SO(3) template search is three degrees of freedom. Here the object
    is upright by assumption and the world is gravity-aligned, so two of them
    are already pinned and only yaw is free -- a ring of `yaw_steps`, not a
    sphere of hundreds. Translation and scale come from `from_bbox` and are not
    searched either.

    Two stages, because matchers are expensive. Every hypothesis is scored by
    silhouette overlap, which needs no appearance and no lighting; only the
    `top_k` survivors are matched. With 36 hypotheses that is 36 cheap renders
    and 3 matcher calls rather than 36 of each.

    Pass `match=None` to stop after the silhouette stage -- useful when no GPU
    is available, at the cost of relying on outline alone.
    """
    if view is None:
        view = observation.views[0]

    yaws = np.linspace(0.0, 2.0 * np.pi, yaw_steps, endpoint=False)
    candidates = [from_bbox(mesh, observation, frame, yaw=y).pose for y in yaws]
    scores = np.array([mask_iou(view, mesh, p, render) for p in candidates])

    symmetry = _symmetry_of(scores, yaws)
    order = np.argsort(scores)[::-1]

    if debug is not None:
        # Lazy on purpose: debug thumbnails are the ONLY reason this module
        # would depend on the drawing layer, and paying that at import time
        # to serve an optional feature is the wrong trade.
        from .viz import overlay
        thumbs = [_thumb(overlay(view, mesh, candidates[int(i)], render))
                  for i in range(0, len(candidates), max(1, len(candidates) // 12))]
        debug.sweep("init", scores, yaws, thumbs)

    if match is None:
        return Initialization(pose=candidates[int(order[0])], source="search",
                              symmetry=symmetry)

    best_pose, best_inliers = candidates[int(order[0])], -1
    for idx in order[:top_k]:
        inliers = _count_matches(mesh, candidates[int(idx)], view, render, match)
        if inliers > best_inliers:
            best_pose, best_inliers = candidates[int(idx)], inliers

    return Initialization(pose=best_pose, source="search", symmetry=symmetry)


def _thumb(img: np.ndarray, width: int = 160) -> np.ndarray:

    im = Image.fromarray(np.asarray(img).astype(np.uint8))
    return np.asarray(im.resize((width, max(1, round(width * im.height / im.width)))))


def _count_matches(mesh, pose: Pose, view: View, render, match) -> int:
    """How many correspondences this hypothesis supports. Ties on the outline
    are broken by texture, which is what silhouettes cannot see."""
    if render is None:
        return 0
    placed = mesh.copy()
    placed.vertices = pose.apply(np.asarray(mesh.vertices, dtype=float))
    drawn = render(placed, view.intrinsic, view.cam2world, view.image_hw)
    if not drawn.coverage:
        return 0
    k_render, _k_obs, _conf = match(drawn.rgb, view.masked_rgb())
    return len(k_render)


def _symmetry_of(scores: np.ndarray, yaws: np.ndarray,
                 tolerance: float = 0.02) -> Symmetry:
    """Describe the sweep's score curve.

    Rivals are counted as PEAKS, not as high samples: adjacent bins on the same
    hill are one answer seen twice, while a separate hill is a genuinely
    different answer. Counting samples would call every smooth peak ambiguous.
    """
    best = float(scores.max())
    if best <= 0:
        return Symmetry(margin=0.0, k_fold=len(scores), scores=scores, yaws=yaws)

    peaks = [i for i in range(len(scores))
             if scores[i] >= scores[i - 1]
             and scores[i] >= scores[(i + 1) % len(scores)]]
    near_best = [i for i in peaks if scores[i] >= best - tolerance]

    rivals = [scores[i] for i in peaks if scores[i] < best - tolerance]
    margin = (best - max(rivals)) / best if rivals else (0.0 if len(near_best) > 1 else 1.0)

    return Symmetry(margin=float(margin), k_fold=max(len(near_best), 1),
                    scores=scores, yaws=yaws)
