"""The input contract.

meshfit takes an object's observations and the scene's unified point cloud,
and returns the pose that places a generated mesh into that cloud's frame.

Preconditions, all of them load-bearing and none of them checkable from the
data alone -- meshfit validates shapes and dtypes, not semantics:

- the cloud is GRAVITY-ALIGNED, +Z up. Everything "upright" means is defined
  against that. A tilted cloud yields confident, wrong, upright-looking poses.
- everything is METRIC, in METRES. Thresholds throughout (inlier radii,
  penetration depths) are absolute distances; a cloud in centimetres makes
  them silently 100x too tight.
- `cam2world` maps camera -> that same world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class View:
    """One camera's observation of the object.

    Attributes:
        rgb: `(H, W, 3)` `uint8`, 0-255. The photograph, unmasked.
        mask: `(H, W)` `bool`. True on the object. Defines what meshfit is
            fitting to -- a mask covering part of the object fits the mesh to
            that part.
        intrinsic: `(3, 3)` `float64`. Pinhole `K` in pixels, for `rgb`'s
            resolution: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.
        cam2world: `(4, 4)` `float64`. Camera -> world. OpenCV camera
            convention: **+X right, +Y down, +Z forward**. The world is Z-up,
            gravity-aligned and metric. The rotation block must be orthonormal;
            a scale smuggled into it is rejected by `validate`.
        pointmap: `(h, w, 3)` `float64` or None. Per-pixel points in the
            **camera** frame, metres, with `z <= 0` marking invalid. May be at
            a different resolution than `rgb` -- consumers rescale. Without it
            the observation has no metric information and refinement cannot
            lift matches to 3D.
    """

    rgb: np.ndarray
    mask: np.ndarray
    intrinsic: np.ndarray
    cam2world: np.ndarray
    pointmap: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.rgb = np.asarray(self.rgb)
        self.mask = np.asarray(self.mask).astype(bool)
        self.intrinsic = np.asarray(self.intrinsic, dtype=float).reshape(3, 3)
        self.cam2world = np.asarray(self.cam2world, dtype=float).reshape(4, 4)
        if self.pointmap is not None:
            self.pointmap = np.asarray(self.pointmap, dtype=float)

    @property
    def image_hw(self) -> tuple[int, int]:
        return int(self.rgb.shape[0]), int(self.rgb.shape[1])

    def validate(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"rgb must be (H,W,3), got {self.rgb.shape}")
        if self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {self.rgb.dtype}")
        if self.mask.shape != self.rgb.shape[:2]:
            raise ValueError(f"mask {self.mask.shape} does not match rgb {self.rgb.shape[:2]}")
        if not self.mask.any():
            raise ValueError("mask is empty -- the object is not visible in this view")
        # a c2w whose rotation block is not orthonormal is the classic symptom
        # of a scale factor smuggled into the extrinsic
        Rc = self.cam2world[:3, :3]
        if not np.allclose(Rc @ Rc.T, np.eye(3), atol=1e-4):
            raise ValueError("cam2world rotation block is not orthonormal "
                             "(scale baked into the extrinsic?)")
        if self.pointmap is not None and (
                self.pointmap.ndim != 3 or self.pointmap.shape[2] != 3):
            raise ValueError(f"pointmap must be (h,w,3), got {self.pointmap.shape}")

    def masked_rgb(self) -> np.ndarray:
        """RGB with everything but the object zeroed -- what the matcher sees."""
        out = self.rgb.copy()
        out[~self.mask] = 0
        return out


@dataclass
class Render:
    """What a renderer produces: pixels and a silhouette, nothing more.

    Deliberately no depth. Requiring one would make every renderer responsible
    for matching our depth convention and precision, when the 3D actually
    needed is recoverable exactly by intersecting the mesh that was just drawn
    (`correspond.lift_by_raycast`). Renderers draw; geometry stays geometry.

    It lives here rather than in the refinement loop so a renderer can be
    written against the contract without importing its consumer.

    Attributes:
        rgb: `(H, W, 3)` `uint8`. The mesh drawn from the view's camera.
        mask: `(H, W)` `bool`. True where the mesh covers -- the rendered
            silhouette, which is also how `metrics.mask_iou` scores a pose.
    """

    rgb: np.ndarray
    mask: np.ndarray

    @property
    def coverage(self) -> int:
        return int(self.mask.sum())


@dataclass
class Observation:
    """Everything meshfit knows about one object: its views, plus the subset
    of the unified cloud that belongs to it.

    Measured DEPTH is what makes the problem metric. From one photograph a
    small nearby object and a large distant one are indistinguishable --
    doubling an object's size and doubling its distance reprojects to the same
    pixels exactly -- so a single view leaves size and distance tied together
    as one unknown. A depth measurement frees both.

    That depth lives in each view's `pointmap`, which is where every
    correspondence gets its 3D target. `points` is a separate, optional thing:
    a pooled cloud used to seed a search when no pose is supplied, and to draw
    the object in the 3D viewer. Nothing in the solve path reads it.


    Attributes:
        views: One or more `View`. All in the same world frame.
        points: `(N, 3)` `float64`, metres, **world** frame, or None. The
            OBJECT's points only -- lifted from masked pixels, not the whole
            scene. OPTIONAL: needed only to seed a search when no pose is
            given (`init.from_bbox`). Refinement and the joint solve do not
            read it; their 3D comes from each view's `pointmap`.
        colors: `(N, 3)` `uint8` or None. The pixel colour each point came
            from. Used only for visualisation.
    """

    views: list[View]
    points: np.ndarray | None = None
    colors: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.points is not None:
            self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        if self.colors is not None:
            self.colors = np.asarray(self.colors, dtype=np.uint8).reshape(-1, 3)
            if len(self.colors) != len(self.points):
                raise ValueError(f"{len(self.colors)} colours for {len(self.points)} points")

    def validate(self) -> None:
        if not self.views:
            raise ValueError("need at least one view")
        for i, v in enumerate(self.views):
            try:
                v.validate()
            except ValueError as e:
                raise ValueError(f"view[{i}]: {e}") from e
        if self.points is not None and len(self.points) < 3:
            raise ValueError(f"need >=3 object points, got {len(self.points)}")

    def require_points(self, why: str) -> np.ndarray:
        """`points`, or a message naming what wanted them.

        Only the search initialisers need a pooled cloud, so a caller who
        supplies a generator pose never has to build one. Saying which stage
        asked beats a bare AttributeError.
        """
        if self.points is None:
            raise ValueError(
                f"{why} needs Observation.points, which is None. Build it with "
                f"`io.object_points_from_views(views)`, or supply `init` so no "
                f"search is required.")
        return self.points

    @property
    def centroid(self) -> np.ndarray:
        return self.require_points("centroid").mean(axis=0)

    @property
    def extent(self) -> np.ndarray:
        """Full world-frame bbox extent [m]. Sensitive to outliers -- prefer
        `robust_extent` for anything that feeds a pose."""
        pts = self.require_points("extent")
        return pts.max(axis=0) - pts.min(axis=0)

    def robust_bounds(self, percentile: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Per-axis (low, high) with the outer `percentile` trimmed off each end.

        Masked point clouds bleed. Pixels on the object's silhouette edge pick
        up depth from whatever is behind it, so a handful of points land metres
        away and min/max stops describing the object at all -- on real data we
        have seen a 15cm object report a 70cm extent from a 1% tail.
        """
        pts = self.require_points("robust_bounds")
        return (np.percentile(pts, percentile, axis=0),
                np.percentile(pts, 100.0 - percentile, axis=0))

    def robust_extent(self, percentile: float = 1.0) -> np.ndarray:
        lo, hi = self.robust_bounds(percentile)
        return hi - lo
