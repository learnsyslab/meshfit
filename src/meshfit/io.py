"""Reading and writing a meshfit problem.

Upstream pipelines each have their own on-disk layout, and none of them is
meshfit's. Rather than teach meshfit to read all of them, converters normalise
into the one format here -- an `Observation` plus the mesh it belongs to, and
optionally whatever pose the generator already had an opinion about.

    <case>/
        mesh.glb            the mesh, in its generator's canonical frame
        views.npz           rgb, mask, intrinsic, cam2world, pointmap (V-first)
        points.npy          (N,3) the object's points, world frame, metres
        init.json           optional: a pose-aware generator's estimate
        meta.json           canonical_up, provenance, any reference pose

Keeping `points.npy` alongside the pointmaps is deliberate redundancy: the
points ARE derivable from mask + pointmap + cam2world, but which pixels counted
and how they were subsampled is a decision, and a case file should record
decisions rather than make readers repeat them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .frames import DEFAULT_UP
from .pose import Pose
from .views import Observation, View

VIEW_KEYS = ("rgb", "mask", "intrinsic", "cam2world", "pointmap")


@dataclass
class Case:
    """A complete meshfit problem, plus whatever is known about the answer."""

    mesh: object
    observation: Observation
    init: Pose | None = None          # a pose-aware generator's estimate
    reference: Pose | None = None     # a known-good pose, for comparison
    meta: dict = None                 # canonical_up, provenance, labels

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}

    @property
    def canonical_up(self) -> str:
        return self.meta.get("canonical_up", DEFAULT_UP)


def object_points_from_views(views: list[View], max_points: int = 20000,
                             seed: int = 0, masked: bool = True
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Lift pixels with valid depth into the world frame, with their colours.

    `masked=True` gives the object; `masked=False` gives the whole frame, which
    is what makes a 3D view legible -- a floating cloud of object points tells
    you nothing about whether the object sits ON the desk.

    Subsampled because a full-resolution mask can be millions of points and
    nothing downstream benefits: ICP and bbox statistics converge long before
    that, and the cost is linear.
    """
    points, colors = [], []
    for view in views:
        if view.pointmap is None:
            continue
        H, W = view.pointmap.shape[:2]
        keep = view.pointmap[..., 2] > 0
        if masked:
            mask = view.mask if view.mask.shape == (H, W) else _resize_mask(view.mask, (H, W))
            keep = keep & mask
        if not keep.any():
            continue
        cam = view.pointmap[keep]
        points.append(cam @ view.cam2world[:3, :3].T + view.cam2world[:3, 3])
        rgb = view.rgb if view.rgb.shape[:2] == (H, W) else _resize_rgb(view.rgb, (H, W))
        colors.append(rgb[keep])

    if not points:
        raise ValueError("no pixels with valid depth -- check masks and pointmaps")
    points = np.concatenate(points)
    colors = np.concatenate(colors).astype(np.uint8)
    if len(points) > max_points:
        idx = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]
    return points, colors


def scene_points_from_views(views: list[View], max_points: int = 120000,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The whole frame, for context around the object."""
    return object_points_from_views(views, max_points, seed, masked=False)


def _resize_rgb(rgb: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    H, W = hw
    ys = (np.arange(H) * rgb.shape[0] / H).astype(int).clip(0, rgb.shape[0] - 1)
    xs = (np.arange(W) * rgb.shape[1] / W).astype(int).clip(0, rgb.shape[1] - 1)
    return rgb[np.ix_(ys, xs)]


def _resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour, so a boolean stays boolean and no edge pixel is
    invented by interpolation."""
    H, W = hw
    ys = (np.arange(H) * mask.shape[0] / H).astype(int).clip(0, mask.shape[0] - 1)
    xs = (np.arange(W) * mask.shape[1] / W).astype(int).clip(0, mask.shape[1] - 1)
    return mask[np.ix_(ys, xs)]


# ---------------------------------------------------------------------------


def save_case(path, mesh, observation: Observation, *, init: Pose | None = None,
              reference: Pose | None = None, meta: dict | None = None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # GLB keeps UVs and the PBR material; OBJ/PLY would not.
    mesh.export(path / "mesh.glb")
    np.savez_compressed(
        path / "views.npz",
        **{key: np.stack([_view_field(v, key) for v in observation.views])
           for key in VIEW_KEYS},
    )
    if observation.points is not None:
        np.save(path / "points.npy", observation.points.astype(np.float32))
    if observation.colors is not None:
        np.save(path / "colors.npy", observation.colors)

    if init is not None:
        _write_json(path / "init.json", init.to_dict())
    payload = dict(meta or {})
    payload.setdefault("canonical_up", DEFAULT_UP)
    payload["n_views"] = len(observation.views)
    payload["n_points"] = 0 if observation.points is None else len(observation.points)
    if reference is not None:
        payload["reference_pose"] = reference.to_dict()
    _write_json(path / "meta.json", payload)
    return path


def load_case(path) -> Case:

    path = Path(path)
    mesh = trimesh.load(path / "mesh.glb", force="mesh", process=False)
    arrays = np.load(path / "views.npz")
    views = [
        View(rgb=arrays["rgb"][i], mask=arrays["mask"][i],
             intrinsic=arrays["intrinsic"][i], cam2world=arrays["cam2world"][i],
             pointmap=arrays["pointmap"][i])
        for i in range(len(arrays["rgb"]))
    ]
    points_path, colors_path = path / "points.npy", path / "colors.npy"
    observation = Observation(
        views=views,
        points=np.load(points_path) if points_path.exists() else None,
        colors=np.load(colors_path) if colors_path.exists() else None)

    meta = _read_json(path / "meta.json") if (path / "meta.json").exists() else {}
    init = _pose_from_dict(_read_json(path / "init.json")) \
        if (path / "init.json").exists() else None
    reference = _pose_from_dict(meta["reference_pose"]) if "reference_pose" in meta else None
    return Case(mesh=mesh, observation=observation, init=init,
                reference=reference, meta=meta)


def _view_field(view: View, key: str) -> np.ndarray:
    if key == "pointmap":
        if view.pointmap is None:
            raise ValueError("every view needs a pointmap to be saved")
        return view.pointmap.astype(np.float32)
    return {"rgb": view.rgb, "mask": view.mask,
            "intrinsic": view.intrinsic.astype(np.float32),
            "cam2world": view.cam2world.astype(np.float32)}[key]


def _pose_from_dict(d: dict) -> Pose:
    T = np.asarray(d["T_world_canonical"], dtype=float)
    scale = np.asarray(d["scale_aniso"], dtype=float) if "scale_aniso" in d \
        else float(d["scale_metric"])
    return Pose(R=T[:3, :3], scale=scale, t=T[:3, 3])


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
