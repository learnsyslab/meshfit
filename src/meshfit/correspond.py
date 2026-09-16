"""Correspondences -- what the solvers actually consume.

A correspondence set ties canonical mesh points to where they were observed.
Every solver in `meshfit.solver` takes some projection of this:

    closed-form fits   want the 3D pairs (canonical, world)
    fit_joint          wants the 2D pixels, and treats the 3D pairs as a prior

Keeping both on one object is deliberate. The 2D keypoint is what the matcher
actually measured, to about a pixel; the 3D point is that same keypoint pushed
through a depth map, so it carries the depth estimator's error too. Solvers
that can tell the difference should be able to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .views import View


@dataclass
class Correspondence:
    """Matched points between a mesh and one view of it.

    Attributes:
        canonical: `(N, 3)` `float64`. Points in the MESH's canonical frame,
            not the world -- which is what lets a solver re-estimate the pose
            from scratch rather than only correct it.
        pixel: `(N, 2)` `float64`, `(u, v)` in `view`'s pixels. What the
            matcher actually measured, to roughly a pixel.
        view: The `View` these were matched against.
        world: `(N, 3)` `float64` or None. `pixel` lifted to the world frame
            through the pointmap. Carries the depth estimator's error as well
            as the matcher's, so solvers treat it as a prior rather than a
            measurement.
        weight: `(N,)` `float64`. Per-match confidence; defaults to ones.
    """

    canonical: np.ndarray
    pixel: np.ndarray
    view: View
    world: np.ndarray | None = None
    weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.canonical = np.asarray(self.canonical, dtype=float).reshape(-1, 3)
        self.pixel = np.asarray(self.pixel, dtype=float).reshape(-1, 2)
        if self.world is not None:
            self.world = np.asarray(self.world, dtype=float).reshape(-1, 3)
        if self.weight is None:
            self.weight = np.ones(len(self.canonical))
        else:
            self.weight = np.asarray(self.weight, dtype=float).reshape(-1)

    def __len__(self) -> int:
        return len(self.canonical)

    def validate(self) -> None:
        n = len(self.canonical)
        for name, arr in (("pixel", self.pixel), ("world", self.world),
                          ("weight", self.weight)):
            if arr is not None and len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} entries, expected {n}")

    @property
    def world2cam(self) -> np.ndarray:
        return np.linalg.inv(self.view.cam2world)

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> pixels, plus a validity mask.

        Points at or behind the camera plane have no projection; they are
        reported rather than silently producing a mirrored pixel.
        """
        w2c = self.world2cam
        cam = points_world @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        in_front = z > 1e-6
        safe_z = np.where(in_front, z, 1.0)
        K = self.view.intrinsic
        uv = np.stack([
            K[0, 0] * cam[:, 0] / safe_z + K[0, 2],
            K[1, 1] * cam[:, 1] / safe_z + K[1, 2],
        ], axis=1)
        return uv, in_front


def stack_world_pairs(corrs: list[Correspondence]) -> tuple[np.ndarray, np.ndarray]:
    """Pool every correspondence's 3D pairs across views, for the closed-form
    fits -- which have no notion of which camera a pair came from."""
    usable = [c for c in corrs if c.world is not None and len(c)]
    if not usable:
        raise ValueError("no correspondences carry world points")
    return (np.concatenate([c.canonical for c in usable]),
            np.concatenate([c.world for c in usable]))


# ---------------------------------------------------------------------------
# lifting: 2D keypoints -> 3D world points
# ---------------------------------------------------------------------------


def lift_by_raycast(kpts: np.ndarray, mesh_world, intrinsic: np.ndarray,
                    cam2world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rendered keypoints -> world, by intersecting the mesh directly.

    The alternative is reading a depth buffer, which would make every renderer
    responsible for agreeing with us about depth conventions and precision.
    Casting rays at the mesh instead keeps the renderer's job to producing
    pixels, gives exact surface points rather than quantised ones, and works
    identically whatever drew the image.

    `mesh_world` must already be placed at the pose that was rendered. Rays are
    cast from the camera centre through each pixel; the nearest hit is taken,
    which is the visible surface. Keypoints that miss the mesh -- antialiased
    edges, a matcher firing on background -- are reported invalid.
    """
    K = intrinsic
    n = len(kpts)
    dirs_cam = np.stack([
        (kpts[:, 0] - K[0, 2]) / K[0, 0],
        (kpts[:, 1] - K[1, 2]) / K[1, 1],
        np.ones(n),
    ], axis=-1)
    dirs = dirs_cam @ cam2world[:3, :3].T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.tile(cam2world[:3, 3], (n, 1))

    hits, ray_index, _tri = mesh_world.ray.intersects_location(
        origins, dirs, multiple_hits=False)

    points = np.zeros((n, 3))
    valid = np.zeros(n, bool)
    if len(ray_index):
        points[ray_index] = hits
        valid[ray_index] = True
    return points[valid], valid


def lift_from_pointmap(kpts: np.ndarray, pointmap: np.ndarray,
                       cam2world: np.ndarray,
                       image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Observed keypoints -> world, via a per-pixel camera-frame pointmap.

    The pointmap may be at a different resolution than the image the matcher
    ran on, so keypoints are rescaled first. Sampling is bilinear because these
    points come from a smooth depth field and the sub-pixel position is real
    information -- unlike the rendered side.
    """
    H_img, W_img = image_hw
    H_pm, W_pm = pointmap.shape[:2]
    u = kpts[:, 0] * (W_pm / W_img)
    v = kpts[:, 1] * (H_pm / H_img)
    cam = _bilinear(pointmap, u, v)
    valid = cam[:, 2] > 0.0
    world = cam @ cam2world[:3, :3].T + cam2world[:3, 3]
    return world[valid], valid


def _bilinear(field: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    H, W = field.shape[:2]
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
    du, dv = (u - u0)[:, None], (v - v0)[:, None]
    return ((1 - dv) * ((1 - du) * field[v0, u0] + du * field[v0, u1])
            + dv * ((1 - du) * field[v1, u0] + du * field[v1, u1]))
