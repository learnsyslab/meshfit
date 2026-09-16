"""Command line: run a fit on a saved case and write the pose out as JSON.

    meshfit fit <case> [-o pose.json] [--overlay out.png]

`<case>` is a directory in meshfit's own format (see `meshfit.io`); converters
for upstream pipelines live in `scripts/make_test_data.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .api import fit
from .frames import Frame
from .io import load_case
from .solver import tilt_of, yaw_of
from .viz import DebugSink
from .viz import overlay as make_overlay


def _add_fit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("case", type=Path, help="case directory")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="write the pose JSON here (default: <case>/pose.json)")
    p.add_argument("--overlay", type=Path, default=None,
                   help="also write a before/after overlay image")
    p.add_argument("--no-init", action="store_true",
                   help="ignore the generator's pose and search instead")
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--no-polish", action="store_true")
    p.add_argument("--force-upright", action="store_true",
                   help="project tilt out of the final pose")
    p.add_argument("--iters", type=int, default=3, help="refinement iterations")
    p.add_argument("--yaw-steps", type=int, default=36)
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug", nargs="?", const="", default=None, metavar="DIR",
                   help="save every intermediate render, match and overlay "
                        "(default: out/debug/<case>)")
    p.add_argument("--tilt", choices=("auto", "upright", "free"), default="auto",
                   help="auto: fit both and take the tilted one only if it wins "
                        "decisively; upright: pin pitch/roll to zero; "
                        "free: always re-estimate them")
    p.add_argument("--min-tilt", type=float, default=20.0, metavar="DEG",
                   help="in auto mode, the smallest lean worth believing "
                        "(default 20); below this the upright fit is kept")
    p.add_argument("--no-gate", action="store_true",
                   help="keep every stage even if it worsens silhouette overlap")
    p.add_argument("--cpu-only", action="store_true",
                   help="no renderer or matcher; initialisation only")


def cmd_fit(args) -> int:

    case = load_case(args.case)
    render = matcher = None
    if not args.cpu_only:
        from .matcher import RoMaMatcher
        from .render import BlenderRenderer
        render = BlenderRenderer()
        matcher = RoMaMatcher(device=args.device)

    debug = None
    if args.debug is not None:
        root = Path(args.debug) if args.debug else Path("out") / "debug" / args.case.name
        debug = DebugSink(root)

    started = time.perf_counter()
    result = fit(
        case.mesh, case.observation,
        canonical_up=case.canonical_up,
        init=None if args.no_init else case.init,
        render=render, match=matcher,
        do_refine=not args.no_refine, do_polish=not args.no_polish,
        force_upright=args.force_upright, gate_on_iou=not args.no_gate,
        tilt_mode=args.tilt, min_tilt_deg=args.min_tilt,
        debug=debug,
        refine_iters=args.iters, yaw_steps=args.yaw_steps,
    )
    elapsed = time.perf_counter() - started

    frame = Frame.from_up(case.canonical_up)
    print(f"\n{args.case.name}  ({elapsed:.1f}s)")
    print(f"  init      : {result.init_source}")
    for stage in result.stages:
        print(f"  {stage['stage']:10s}: "
              + ", ".join(f"{k}={v}" for k, v in stage.items() if k != "stage"))
    print(f"  pose      : yaw {np.degrees(yaw_of(result.pose.R, frame)):+.2f} deg  "
          f"tilt {tilt_of(result.pose.R, frame):.2f} deg  "
          f"scale {np.round(result.pose.scale, 4)}  t {np.round(result.pose.t, 3)}")
    if not np.isnan(result.reprojection_rms_px):
        print(f"  reproj    : {result.reprojection_rms_px:.2f} px "
              f"({result.inliers}/{result.matches} inliers)")
    for message in result.warnings:
        print(f"  note      : {message}")
    if result.symmetry is None:
        print("  WARNING   : yaw ambiguity was NOT assessed (the pose came from "
              "the generator, so no orientation sweep ran). Do not treat this "
              "pose as orientation-reliable without checking.")
    elif result.symmetry.is_ambiguous:
        print(f"  WARNING   : yaw is ambiguous "
              f"(k_fold={result.symmetry.k_fold}, margin={result.symmetry.margin:.3f}) "
              f"-- the image cannot distinguish this orientation from others")

    out = args.out or (args.case / "pose.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"  wrote     : {out}")

    if args.overlay is not None:

        view = case.observation.views[0]
        panels = [view.rgb]
        if case.init is not None:
            panels.append(make_overlay(view, case.mesh, case.init, render))
        panels.append(make_overlay(view, case.mesh, result.pose, render))
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.concatenate(panels, axis=1)).save(args.overlay)
        print(f"  overlay   : {args.overlay}  (raw | init | fitted)")

    if debug is not None:
        print(f"  debug     : {debug.root}  ({len(list(debug.root.iterdir()))} images)")
    if render is not None:
        render.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="meshfit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_fit_args(sub.add_parser("fit", help="fit one case"))
    args = ap.parse_args(argv)
    return cmd_fit(args)


if __name__ == "__main__":
    sys.exit(main())
