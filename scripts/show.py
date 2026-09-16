"""Look at an alignment.

    python scripts/show.py overlay <case> [<case> ...]   # write a PNG panel
    python scripts/show.py viser   <case>                # interactive 3D
    python scripts/show.py list                          # available cases

Two views because they fail differently. The overlay catches wrong PLACEMENT --
and convention bugs, which produce a respectable residual and a visibly
mirrored picture. viser catches wrong DEPTH, which an overlay cannot show at
all: an object twice as far away and twice as large covers the same pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshfit.io import load_case
from meshfit.pose import Pose
from meshfit.viz import mask_iou, overlay

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "test_data"


def _case_dir(name: str) -> Path:
    """Accept a bare case name, a path relative to the repo, or an absolute one.

    Tab-completion produces `test_data/<case>`, so prepending the cases root
    unconditionally would double it.
    """
    for candidate in (CASES / name, Path(name), ROOT / name):
        if (candidate / "meta.json").exists():
            return candidate
    known = sorted(d.parent.name for d in CASES.glob("*/meta.json"))
    raise SystemExit(f"no case at {name!r}. known cases:\n  " + "\n  ".join(known))


def _poses(case, case_dir: Path):
    """Every pose this case has, in pipeline order.

    Including meshfit's own, from `pose.json`. Leaving it out meant the viewer
    showed the generator's guess and someone else's reference but never this
    tool's answer -- and a pose-blind case with neither had nothing to show at
    all.
    """
    out = []
    if case.init is not None:
        out.append((f"init ({case.meta.get('backend', 'generator')})", case.init))
    fitted = _fitted_pose(case_dir)
    if fitted is not None:
        out.append(("meshfit", fitted))
    if case.reference is not None:
        out.append(("reference", case.reference))
    return out


def _fitted_pose(case_dir: Path):
    """meshfit's result, if `meshfit fit` has been run on this case."""
    path = case_dir / "pose.json"
    if not path.exists():
        return None
    pose = json.loads(path.read_text())["pose"]
    T = np.asarray(pose["T_world_canonical"], dtype=float)
    scale = np.asarray(pose["scale_aniso"], dtype=float) if "scale_aniso" in pose \
        else float(pose["scale_metric"])
    return Pose(R=T[:3, :3], scale=scale, t=T[:3, 3])


def cmd_overlay(names: list[str], out_path: Path, width: int = 384,
                use_renderer: bool = True) -> None:
    from PIL import Image

    render = None
    if use_renderer:
        from meshfit.render import BlenderRenderer
        render = BlenderRenderer()

    rows = []
    for name in names:
        case_dir = _case_dir(name)
        case = load_case(case_dir)
        view = case.observation.views[0]
        H, W = view.image_hw
        size = (width, int(width * H / W))

        panels = [np.asarray(Image.fromarray(view.rgb).resize(size))]
        for label, pose in _poses(case, case_dir):
            iou = mask_iou(view, case.mesh, pose, render)
            print(f"  {case.meta.get('label', name):22s} {label:22s} IoU {iou:.3f}")
            img = overlay(view, case.mesh, pose, render)
            panels.append(np.asarray(Image.fromarray(img).resize(size)))
        rows.append(np.concatenate(panels, axis=1))

    if render is not None:
        render.close()

    width_px = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 0), (0, width_px - r.shape[1]), (0, 0))) for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(padded, axis=0)).save(out_path)
    print(f"\nwrote {out_path}")
    first = _case_dir(names[0])
    labels = [label for label, _ in _poses(load_case(first), first)]
    print("columns: raw RGB | " + " | ".join(labels))
    print("green = mesh, red = observed mask, yellow = agreement")


def cmd_viser(name: str, port: int) -> None:
    from meshfit.viz import show

    case_dir = _case_dir(name)
    case = load_case(case_dir)
    poses = _poses(case, case_dir)
    if not poses:
        raise SystemExit(
            f"{name} has no pose to show. Run `meshfit fit test_data/{name}` "
            f"first -- a pose-blind case has no generator pose either.")
    # last pose is the most refined; show the first alongside it for comparison
    before = poses[0][1] if len(poses) > 1 else None
    show(case.observation, case.mesh, poses[-1][1], before=before, port=port)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("overlay")
    o.add_argument("cases", nargs="+")
    o.add_argument("-o", "--out", type=Path, default=ROOT / "out" / "overlay.png")
    o.add_argument("--width", type=int, default=384)
    o.add_argument("--no-render", action="store_true",
                   help="skip Blender; splat vertices instead (coarse, no GPU)")

    v = sub.add_parser("viser")
    v.add_argument("case")
    v.add_argument("--port", type=int, default=8080)

    sub.add_parser("list")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for meta in sorted(CASES.glob("*/meta.json")):
            import json
            m = json.loads(meta.read_text())
            print(f"  {meta.parent.name:32s} {m.get('label',''):22s} "
                  f"up={m.get('canonical_up')} views={m.get('n_views')}")
        return 0
    if args.cmd == "overlay":
        cmd_overlay(args.cases, args.out, args.width, not args.no_render)
        return 0
    cmd_viser(args.case, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
