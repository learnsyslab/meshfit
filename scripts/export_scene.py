"""Freeze an aligned case into a single GLB the docs can load in a browser.

    python scripts/export_scene.py complex_tabletop_4_obj_001 ...

The viser viewer needs a Python process and the test data, neither of which a
reader of the documentation has. This writes the same picture -- the observed
frame as a coloured point cloud, the fitted mesh sitting in it, the generator's
initial pose beside it -- as one static file any glTF viewer can open.

The point cloud is subsampled and heavy meshes are decimated, because these are
shipped in the repository and fetched over the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meshfit.io import load_case, scene_points_from_views
from meshfit.pose import Pose

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "test_data"
OUT = ROOT / "docs" / "scenes"

# glTF is +Y-up, meshfit's world is +Z-up.
ZUP_TO_YUP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


def fitted_pose(case_dir: Path) -> Pose:
    """meshfit's own answer, written by `meshfit fit`."""
    payload = json.loads((case_dir / "pose.json").read_text())["pose"]
    T = np.asarray(payload["T_world_canonical"], dtype=float)
    scale = np.asarray(payload.get("scale_aniso") or [payload["scale_metric"]] * 3)
    return Pose(R=T[:3, :3], scale=scale, t=T[:3, 3])


def to_linear(colors: np.ndarray) -> np.ndarray:
    """sRGB bytes -> linear bytes.

    glTF vertex colours are linear, and a viewer converts them back on the way
    to the screen. Handing it the sRGB values straight off the camera washes
    the whole cloud out.
    """
    x = colors[:, :3].astype(np.float64) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return np.rint(lin * 255).astype(np.uint8)


def fix_material(mesh):
    """Make the material render as the diffuse surface it is.

    trimesh writes no `metallicFactor`, and glTF's default for it is 1.0 -- a
    mirror. The texture then reads as a dark smear in any PBR viewer.
    """
    material = getattr(mesh.visual, "material", None)
    if material is not None and hasattr(material, "metallicFactor"):
        material.metallicFactor = 0.0
        material.roughnessFactor = 0.7
    return mesh


def lighten(mesh, max_faces: int, max_texture: int):
    """Cut a mesh down to something a browser can fetch.

    A generated mesh can carry a million faces and a 4K texture, which is 36 MB
    of GLB and more detail than a viewer this size can show. Decimation drops
    the UVs, so the texture is baked to vertex colours first.
    """
    if len(mesh.faces) > max_faces:
        colors = np.asarray(mesh.visual.to_color().vertex_colors)
        small = mesh.copy().simplify_quadric_decimation(face_count=max_faces)
        # Decimation returns bare geometry, so the colours are carried over by
        # nearest original vertex. They vary smoothly over a surface, and the
        # new vertices are a subset of the old ones up to the collapses.
        _, nearest = cKDTree(mesh.vertices).query(small.vertices)
        small.visual = trimesh.visual.ColorVisuals(
            small, vertex_colors=to_linear(colors[nearest]))
        return small
    image = getattr(getattr(mesh.visual, "material", None), "baseColorTexture", None)
    if image is not None and max(image.size) > max_texture:
        mesh = mesh.copy()
        scale = max_texture / max(image.size)
        mesh.visual.material.baseColorTexture = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    return fix_material(mesh)


def posed_mesh(mesh, pose: Pose):
    """The mesh in the world frame."""
    out = mesh.copy()
    out.vertices = pose.apply(np.asarray(mesh.vertices, dtype=float))
    _ = out.vertex_normals             # exported as NORMAL; without it the
                                       # viewer falls back to flat shading
    return out


def ghost_mesh(mesh, pose: Pose, max_faces: int):
    """The generator's initial pose, drawn later as a translucent shell.

    Its own geometry rather than a second node on the fitted mesh: the two
    poses differ by an anisotropic similarity, and a node transform carrying
    one is not something a viewer can decompose back out.
    """
    shell = mesh.copy()
    shell.visual = trimesh.visual.ColorVisuals(shell)
    if len(shell.faces) > max_faces:
        shell = shell.simplify_quadric_decimation(face_count=max_faces)
    return posed_mesh(shell, pose)


def crop_to_object(points, colors, centre, extent, margin: float):
    """Drop points far from the object.

    A full frame is mostly wall and floor, which costs bytes and pushes the
    object into a corner of the viewer's default framing.
    """
    radius = float(np.max(extent)) * margin
    keep = np.all(np.abs(points - centre) <= radius, axis=1)
    return points[keep], colors[keep]


def export(name: str, *, max_points: int, max_faces: int, max_texture: int,
           margin: float) -> Path:
    case_dir = CASES / name
    case = load_case(case_dir)
    pose = fitted_pose(case_dir)
    mesh = posed_mesh(lighten(case.mesh, max_faces, max_texture), pose)

    points, colors = scene_points_from_views(case.observation.views,
                                             max_points=max_points * 4)
    points, colors = crop_to_object(points, colors, mesh.bounds.mean(axis=0),
                                    mesh.extents, margin)
    if len(points) > max_points:
        keep = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[keep], colors[keep]

    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="mesh", geom_name="mesh")
    scene.add_geometry(trimesh.PointCloud(points, colors=to_linear(colors)),
                       node_name="points", geom_name="points")
    if case.init is not None:
        scene.add_geometry(ghost_mesh(case.mesh, case.init, max_faces // 4),
                           node_name="init", geom_name="init")
    scene.apply_transform(ZUP_TO_YUP)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.glb"
    out.write_bytes(scene.export(file_type="glb"))
    print(f"{name:32s} {len(points):>7,} points  "
          f"{'init' if case.init is not None else '    '}  "
          f"{out.stat().st_size / 1e6:5.2f} MB")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case names under test_data/")
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--max-faces", type=int, default=60000)
    parser.add_argument("--max-texture", type=int, default=512)
    parser.add_argument("--margin", type=float, default=2.5,
                        help="half-width of the kept region, in object extents")
    args = parser.parse_args()

    names = args.cases or sorted(d.parent.name for d in CASES.glob("*/pose.json"))
    for name in names:
        export(name, max_points=args.max_points, max_faces=args.max_faces,
               max_texture=args.max_texture, margin=args.margin)


if __name__ == "__main__":
    main()
