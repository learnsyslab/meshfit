<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/learnsyslab/meshfit/raw/main/docs/img/logo_dark.gif">
  <img alt="meshfit" src="https://github.com/learnsyslab/meshfit/raw/main/docs/img/logo.gif">
</picture>
<!-- -------------------------------------------------------------------------------- -->
<div align="center">

  **Place a generated mesh where the object actually is.**

  [![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
  [![Tests](https://github.com/learnsyslab/meshfit/actions/workflows/testing.yml/badge.svg)](https://github.com/learnsyslab/meshfit/actions/workflows/testing.yml)
  [![Ruff](https://github.com/learnsyslab/meshfit/actions/workflows/ruff.yml/badge.svg)](https://github.com/learnsyslab/meshfit/actions/workflows/ruff.yml)
  [![Docs](https://github.com/learnsyslab/meshfit/actions/workflows/docs.yml/badge.svg)](https://learnsyslab.github.io/meshfit)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

meshfit recovers the metric pose of a mesh in a real scene (position, per-axis scale and orientation) from RGB, object masks and a gravity-aligned point cloud.

**It works with any mesh.** Generated, scanned, or a CAD part; `glb`, `obj`, `ply`, `stl`, `step`, or anything else [trimesh](https://trimesh.org) reads. meshfit uses only vertices, faces and, for appearance, the material. The one thing it must be told is **which axis is up in the mesh's own frame**, since that is what "upright" is measured against and no file format reliably records it.

Generated meshes are the motivating case, since they arrive wrong in *proportion* as well as pose, which is why scale is per-axis rather than a single number. 

A pose-aware backend ([SAM 3D](https://github.com/facebookresearch/sam-3d-objects), [RecGen](https://github.com/TRI-ML/recgen)) passes its estimate as `init` and meshfit refines it. Without one ([TRELLIS.2](https://github.com/microsoft/trellis.2), [TRELLIS](https://github.com/microsoft/TRELLIS), or a CAD part) meshfit finds the orientation by search. Same code path either way.

```python
import meshfit

result = meshfit.fit(
    mesh,                       # trimesh, in the generator's canonical frame
    observation,                # views (RGB, mask, K, cam2world, pointmap) + object points
    canonical_up="+Y",          # glTF convention
    init=generator_pose,        # optional: SAM 3D / recgen already had an opinion
)

result.pose.matrix()            # T_world_canonical (rigid)
result.pose.scale               # per-canonical-axis metric scale
result.confidence.ambiguous     # is this object's yaw determinable at all?
```

## Documentation

[learnsyslab.github.io/meshfit](https://learnsyslab.github.io/meshfit): installation, user guide, examples, and API reference.

## Features

- **Generator-agnostic.** Nothing in the package branches on which model made the mesh. TRELLIS.2, SAM 3D and recgen differ only in whether they pass `init`
- **Metric.** Scale and distance come from measured depth, not from appearance, which fixes only a viewing ray
- **Upright when it should be.** The constrained fit pins pitch and roll to zero; a free-rotation fit is adopted only when the lean is large *and* renders better
- **Anisotropic scale.** Generated meshes get proportions wrong, so scale is per-canonical-axis, solved jointly with pose
- **Every stage must prove itself.** A stage is kept only if it renders a better silhouette than its input
- **Reports ambiguity.** A symmetric object's yaw is flagged rather than committed to silently

## Installation

```bash
pip install meshfit
```

Developer install ([pixi](https://pixi.sh/) recommended):

```bash
git clone https://github.com/learnsyslab/meshfit.git
cd meshfit
pixi run install-dev
pixi run test
```

<!-- ## Results

Silhouette IoU against the observed mask, single view, on meshes from three different generators.

| object | generator | init | meshfit |
|---|---|---|---|
| game controller | SAM 3D | 0.201 | **0.951** |
| toy figurine | SAM 3D | 0.441 | **0.912** |
| toilet paper roll | SAM 3D | 0.331 | **0.855** |
| drawer unit | recgen | 0.863 | **0.927** |
| windmill souvenir | TRELLIS.2 | *(no pose)* | **0.870** |

TRELLIS.2 is pose-blind, so the windmill had no generator pose and meshfit found the orientation by search. -->

## Citation

```bibtex
@software{meshfit2026,
  title  = {meshfit: Metric Pose Recovery for Generated Meshes},
  author = {Li, Jim Yun-Jin},
  year   = {2026},
  url    = {https://github.com/learnsyslab/meshfit},
}
```
