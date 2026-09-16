"""How well does a placed mesh agree with what was observed?

These are scorers, not pictures. They live below `viz` because the solve path
needs them -- refinement compares hypotheses by silhouette overlap and the
pipeline gates every stage on it -- and a solver that has to import a drawing
module to score a pose has its layering backwards.

Everything here is blind to depth. An object at twice the distance and twice
the size covers exactly the same pixels, so a perfect score means "lands in the
right place on screen", never "is in the right place". Read these beside a 3D
residual, never instead of one.
"""

from __future__ import annotations

import numpy as np

from .pose import Pose
from .views import View


def silhouette(view: View, mesh, pose: Pose, render=None) -> np.ndarray:
    """Boolean mask of where `mesh` at `pose` lands in `view`.

    With a renderer this is the rasterised silhouette. Without one it splats
    projected vertices -- coarse, but it needs no GPU and still answers whether
    the object is in the right PLACE, which is the failure worth catching
    first.
    """
    H, W = view.image_hw
    if render is not None:
        return render(_placed(mesh, pose), view.intrinsic, view.cam2world,
                      view.image_hw).mask

    world = pose.apply(np.asarray(mesh.vertices, dtype=float))
    w2c = np.linalg.inv(view.cam2world)
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    cam = cam[cam[:, 2] > 1e-6]

    K = view.intrinsic
    u = np.round(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    mask = np.zeros((H, W), bool)
    mask[v[inside], u[inside]] = True
    return mask


def mask_iou(view: View, mesh, pose: Pose, render=None) -> float:
    """Silhouette agreement in [0, 1] between the placed mesh and the mask."""
    predicted = silhouette(view, mesh, pose, render)
    union = (predicted | view.mask).sum()
    return float((predicted & view.mask).sum() / union) if union else 0.0


def mean_mask_iou(views, mesh, pose: Pose, render=None) -> float:
    """`mask_iou` averaged over views -- what the stage gate scores on."""
    scores = [mask_iou(v, mesh, pose, render) for v in views]
    return float(np.mean(scores)) if scores else 0.0


def _placed(mesh, pose: Pose):
    placed = mesh.copy()
    placed.vertices = pose.apply(np.asarray(mesh.vertices, dtype=float))
    return placed
