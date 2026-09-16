---
hide:
  - navigation
---

![meshfit](img/logo.gif#only-light)
![meshfit](img/logo_dark.gif#only-dark)

<!-- # meshfit -->

**Place a generated mesh where the object actually is.**

Putting a mesh into a real scene with the right position, metric size and
orientation is a problem every real2sim pipeline ends up solving again. meshfit
does that and nothing else.

```python
import meshfit

result = meshfit.fit(mesh, observation, canonical_up="+Y", init=generator_pose)
result.pose.matrix()          # T_world_canonical
result.pose.scale             # per-canonical-axis metric scale
```

## Why meshfit

**Any mesh, from anywhere.** Generated, scanned, or a CAD part; `glb`, `obj`,
`ply`, `stl`, `step`, or anything else [trimesh](https://trimesh.org) reads.
meshfit touches only vertices, faces and the material. The one thing it must be
told is **which axis is up in the mesh's own frame**, since that is what
"upright" is measured against and no file format reliably records it.

Generated meshes are the motivating case, since they arrive wrong in
*proportion* as well as pose, a chair too wide for its height. That is why scale
is per-axis rather than one number.

A pose-aware backend passes its estimate as `init`. Without one, meshfit finds
the orientation by search. Same code path.

**Measured depth is what makes the answer metric.** From one photograph you
cannot tell a small object nearby from a large one further away: double an
object's size, move it twice as far, and it covers the same pixels. A single
image fixes the object's direction but leaves size and distance tied together.
Each view's `pointmap` measures that distance, and every correspondence is
lifted through it.


<!-- ## Results

Silhouette IoU against the observed mask, single view, meshes from three
different generators.

| object | generator | init | meshfit |
|---|---|---|---|
| game controller | SAM 3D | 0.201 | **0.951** |
| toy figurine | SAM 3D | 0.441 | **0.912** |
| toilet paper roll | SAM 3D | 0.331 | **0.855** |
| drawer unit | recgen | 0.863 | **0.927** |
| windmill souvenir | TRELLIS.2 | *no pose* | **0.870** | -->

## Install

```bash
pip install git+https://github.com/learnsyslab/meshfit.git
```

## Next

<div class="grid cards" markdown>

- **[Installation](get-started/installation.md)** — pip, or a pinned dev environment
- **[Quick Start](get-started/quick-start.md)** — one object, start to finish
- **[User Guide](user-guide/index.md)** — the input contract and what each stage does
- **[API Reference](api/index.md)** — every public type and function

</div>
