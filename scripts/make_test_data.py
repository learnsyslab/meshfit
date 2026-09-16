"""Build meshfit test cases from upstream pipeline outputs.

Each source stores its data differently and none of them matches meshfit's
contract, so every converter's job is the same: produce views with RGB, an
object mask, calibrated cameras and metric camera-frame points, plus the
object's points in a gravity-aligned world frame.

    digital-sister   everything present already; also carries a posealign pose
                     we can compare against
    recgen           pose-aware: ships its own pose estimate, so this is the
                     `from_generator` path
    trellis          a mesh and an image, nothing else. Depth and mask have to
                     be produced first -- the `from_search` path

Usage:
    python scripts/make_test_data.py digital-sister <scene_dir> <obj_id> [...]
    python scripts/make_test_data.py recgen <recgen_out_dir> <image>
    python scripts/make_test_data.py list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshfit.io import object_points_from_views, save_case
from meshfit.pose import Pose
from meshfit.views import Observation, View

OUT_ROOT = Path(__file__).resolve().parents[1] / "test_data"


# ---------------------------------------------------------------------------
# digital-sister
# ---------------------------------------------------------------------------


def from_digital_sister(scene_dir: Path, obj_id: str, out: Path) -> Path:
    """An ObjectRecord plus its generated mesh.

    The record's `cam2world_ground_aligned_list` is already the gravity-aligned
    metric frame meshfit requires, so no transform is needed -- only a rename
    from digital-sister's per-view list names to meshfit's per-view objects.
    """
    import trimesh

    record = np.load(scene_dir / "perception" / "objects" / obj_id / "views.npz")
    views = [
        View(rgb=record["rgb_image_list"][i],
             mask=record["object_mask_list"][i],
             intrinsic=record["intrinsic_list"][i],
             cam2world=record["cam2world_ground_aligned_list"][i],
             pointmap=record["pointmap_list"][i])
        for i in range(len(record["rgb_image_list"]))
    ]
    for v in views:
        v.validate()

    points, colors = object_points_from_views(views)
    observation = Observation(views=views, points=points, colors=colors)
    mesh = trimesh.load(scene_dir / "meshes" / obj_id / "raw.glb",
                        force="mesh", process=False)

    entry = _scene_entry(scene_dir, obj_id)
    reference = _pose_from_contract(entry["pose"]) if "pose" in entry else None

    # The generation backend declares its own canonical up axis and, when it is
    # pose-aware, its pose estimate. Both are exactly what meshfit wants: read
    # them rather than assuming the TRELLIS convention.
    mesh_meta = json.loads((scene_dir / "meshes" / obj_id / "meta.json").read_text())
    init = None
    if mesh_meta.get("pose_aware") and "T_world_canonical_init" in mesh_meta:
        T = np.asarray(mesh_meta["T_world_canonical_init"], dtype=float)
        init = Pose(R=T[:3, :3], scale=float(mesh_meta["scale_metric_init"]), t=T[:3, 3])

    return save_case(
        out, mesh, observation, init=init, reference=reference,
        meta={"canonical_up": mesh_meta.get("up_axis", "+Y"),
              "backend": mesh_meta.get("backend", "unknown"),
              "pose_aware": bool(mesh_meta.get("pose_aware", False)),
              "source": f"digital-sister:{scene_dir.name}/{obj_id}",
              "label": entry.get("label", ""),
              "note": "reference_pose is posealign's own output, not ground truth"},
    )


def _scene_entry(scene_dir: Path, obj_id: str) -> dict:
    scene = json.loads((scene_dir / "scene.json").read_text())
    for obj in scene["objects"]:
        if obj["id"] == obj_id:
            return obj
    raise KeyError(f"{obj_id} not in {scene_dir/'scene.json'}")


def _pose_from_contract(pose: dict) -> Pose:
    T = np.asarray(pose["T_world_canonical"], dtype=float)
    scale = np.asarray(pose["scale_aniso"], float) if "scale_aniso" in pose \
        else float(pose["scale_metric"])
    return Pose(R=T[:3, :3], scale=scale, t=T[:3, 3])


# ---------------------------------------------------------------------------
# recgen
# ---------------------------------------------------------------------------


def from_recgen(recgen_dir: Path, record_pkl: Path, out: Path) -> Path:
    """recgen's mesh and pose, plus the LiveWorldGen record it was run on.

    recgen reports its pose in a NORMALISED CAMERA frame, not the world, so it
    has to be composed through two more transforms before meshfit can use it:

        canonical --(pose)--> ncam --(cam2ncam^-1)--> camera --(cam2world)--> world

    Since `cam2ncam` is a uniform scale plus a translation, the whole chain
    collapses back into a single similarity, which is what meshfit wants.

    Uses `textured_mesh.glb`, recgen's texture-baked export: low-poly geometry
    with a real 1024px PBR texture, which carries about three times the
    appearance contrast of `mesh.obj`'s per-vertex colours. That matters
    directly -- a feature matcher works on what it can see.

    The GLB is written in glTF's +Y-up convention while the pose was estimated
    against the +Z-up `mesh.obj` frame, so the pose is composed with that
    rotation rather than the mesh being rewritten. Recording the convention is
    what `canonical_up` is for; rewriting vertices would only hide it.
    """
    import trimesh

    md = json.loads((recgen_dir / "metadata.json").read_text())
    record = _load_lwg_record(record_pkl)
    views = _views_from_lwg(record)
    points, colors = object_points_from_views(views)
    observation = Observation(views=views, points=points, colors=colors)

    # Full resolution: embree raycasts ~4M rays/s even on a million faces, so
    # there is nothing to buy by decimating -- and decimation costs the texture
    # the matcher depends on.
    mesh = trimesh.load(recgen_dir / "textured_mesh.glb", force="mesh", process=False)

    # +Y-up glb -> the +Z-up frame the pose was estimated in
    R_Y2Z = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    init = _recgen_pose_to_world(md, views[0].cam2world, canonical_rotation=R_Y2Z)
    return save_case(
        out, mesh, observation, init=init,
        meta={"canonical_up": "+Y",      # glTF convention, as exported
              "backend": "recgen",
              "pose_aware": True,
              "source": f"recgen:{recgen_dir.name}",
              "label": recgen_dir.name,
              "note": "init composed canonical->ncam->camera->world"},
    )


def _recgen_pose_to_world(md: dict, cam2world: np.ndarray,
                          canonical_rotation: np.ndarray | None = None) -> Pose:
    """Collapse recgen's pose and `cam2ncam` into one world-frame similarity.

    With ncam->cam being `sigma * I` plus offset `u`, and cam->world being
    (R_c, t_c):

        p_world = (R_c R)(sigma s) p_can + R_c (sigma t + u) + t_c
    """
    R = np.asarray(md["pose"]["rotation_matrix"], dtype=float)
    t = np.asarray(md["pose"]["translation"], dtype=float)
    s = float(md["pose"]["scale"])

    n2c = np.linalg.inv(np.asarray(md["cam2ncam"], dtype=float))
    sigma = float(n2c[0, 0])
    if not np.allclose(n2c[:3, :3], sigma * np.eye(3), atol=1e-6):
        raise ValueError("cam2ncam is not a uniform scale; cannot collapse to a similarity")
    u = n2c[:3, 3]

    R_c, t_c = cam2world[:3, :3], cam2world[:3, 3]
    R_world = R_c @ R
    if canonical_rotation is not None:
        # the mesh is stored in a different canonical frame than the pose was
        # estimated in; fold that rotation in rather than rewriting vertices
        R_world = R_world @ canonical_rotation
    return Pose(R=R_world, scale=sigma * s, t=R_c @ (sigma * t + u) + t_c)


def _load_lwg_record(path: Path) -> dict:
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


def _views_from_lwg(record: dict) -> list[View]:
    """LiveWorldGen uses the same per-view list names as digital-sister."""
    views = [
        View(rgb=np.asarray(record["rgb_image_list"][i]),
             mask=np.asarray(record["object_mask_list"][i]),
             intrinsic=np.asarray(record["intrinsic_list"][i]),
             cam2world=np.asarray(record["cam2world_ground_aligned_list"][i]),
             pointmap=np.asarray(record["pointmap_list"][i]))
        for i in range(len(record["rgb_image_list"]))
    ]
    for v in views:
        v.validate()
    return views


# ---------------------------------------------------------------------------
# TRELLIS.2 (pose-blind)
# ---------------------------------------------------------------------------


def from_trellis(glb: Path, view_npz: Path, mask_npy: Path, out: Path,
                 label: str = "") -> Path:
    """A TRELLIS.2 mesh plus a reconstruction of the photo it came from.

    TRELLIS.2 is pose-blind: it returns a canonical mesh and nothing else. So
    unlike the other converters there is no `init.json` here, and meshfit has
    to find the pose itself -- this is the case that exercises `from_search`.

    The view side is produced separately, because it needs models that live in
    other environments: DA3 for metric depth and a camera, then text-prompted
    segmentation for the mask. Both write plain arrays, which is all this needs.
    """
    import trimesh

    arrays = np.load(view_npz)
    mask = np.load(mask_npy)
    view = View(rgb=arrays["rgb"], mask=mask, intrinsic=arrays["intrinsic"],
                cam2world=arrays["cam2world"], pointmap=arrays["pointmap"])
    view.validate()

    points, colors = object_points_from_views([view])
    observation = Observation(views=[view], points=points, colors=colors)

    mesh = trimesh.load(glb, force="mesh", process=False)

    return save_case(
        out, mesh, observation,
        meta={"canonical_up": "+Y",          # glTF convention; TRELLIS.2 follows it
              "backend": "trellis2",
              "pose_aware": False,
              "source": f"trellis2:{glb.stem}",
              "label": label or glb.stem,
              "note": "no generator pose -- meshfit must search for the yaw"},
    )


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)

    ds = sub.add_parser("digital-sister")
    ds.add_argument("scene_dir", type=Path)
    ds.add_argument("obj_ids", nargs="+")

    rg = sub.add_parser("recgen")
    rg.add_argument("recgen_dir", type=Path)
    rg.add_argument("record_pkl", type=Path)
    rg.add_argument("--name", default=None)

    tr = sub.add_parser("trellis")
    tr.add_argument("glb", type=Path)
    tr.add_argument("view_npz", type=Path)
    tr.add_argument("mask_npy", type=Path)
    tr.add_argument("--name", default=None)
    tr.add_argument("--label", default="")

    sub.add_parser("list")

    args = ap.parse_args(argv)

    if args.source == "list":
        for case in sorted(OUT_ROOT.glob("*/meta.json")):
            meta = json.loads(case.read_text())
            print(f"  {case.parent.name:28s} views={meta.get('n_views')} "
                  f"points={meta.get('n_points')} up={meta.get('canonical_up')} "
                  f"src={meta.get('source','')}")
        return 0

    if args.source == "trellis":
        name = args.name or f"trellis_{args.glb.stem}"
        path = from_trellis(args.glb, args.view_npz, args.mask_npy,
                            OUT_ROOT / name, args.label)
        meta = json.loads((path / "meta.json").read_text())
        print(f"wrote {path.relative_to(OUT_ROOT.parent)}  "
              f"({meta['n_views']} view(s), {meta['n_points']} pts, pose_aware=False)")
        return 0

    if args.source == "recgen":
        name = args.name or f"recgen_{args.recgen_dir.name}"
        path = from_recgen(args.recgen_dir, args.record_pkl, OUT_ROOT / name)
        meta = json.loads((path / "meta.json").read_text())
        print(f"wrote {path.relative_to(OUT_ROOT.parent)}  "
              f"({meta['n_views']} view(s), {meta['n_points']} pts)")
        return 0

    for obj_id in args.obj_ids:
        name = f"{args.scene_dir.name}_{obj_id}"
        path = from_digital_sister(args.scene_dir, obj_id, OUT_ROOT / name)
        meta = json.loads((path / "meta.json").read_text())
        print(f"wrote {path.relative_to(OUT_ROOT.parent)}  "
              f"({meta['n_views']} view(s), {meta['n_points']} pts, "
              f"label={meta.get('label','')!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
