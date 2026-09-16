# Initialisation

Refinement is local. It renders the mesh at the current pose and matches that
render against the photograph, so a mesh that starts far enough away renders
off-frame or at the wrong size and there is nothing to match. Orientation and
rough scale have to be right *before* the first render.

## From the generator

A pose-aware backend already solved this, and nothing here beats the model that
made the mesh.

```python
from meshfit.init import from_generator
init = from_generator(T_world_canonical, scale_metric, frame)
```

The pose is adopted **as given**. `force_upright=True` projects out any tilt
but is off by default. Plenty of objects really are tilted, like a controller
dropped on a desk or a fallen bottle, and an initialiser has no evidence to tell
those apart from a generator being a few degrees sloppy. That decision belongs
to [refinement](tilt.md), which has residuals to compare.

The pose must be in the same world frame as the point cloud. A generator
reporting in camera frame needs composing with `cam2world` first.

## By search

For a pose-blind generator, meshfit finds its own.

A full SO(3) template search is three degrees of freedom. The object is upright
by assumption and the world is gravity-aligned, which pins two of them, leaving
only **yaw** free. That is a ring of hypotheses rather than a sphere of
hundreds.

```python
from meshfit.init import from_search
init = from_search(mesh, observation, render, match, frame, yaw_steps=36)
```

Two stages, because matchers are expensive:

1. every hypothesis is scored by **silhouette overlap**, which needs no
   appearance, no lighting and no samples
2. only the top few are matched

With 36 hypotheses that is 36 cheap renders and 3 matcher calls, rather than 36
of each.

## By bounding box

!!! info "This is the only stage that reads `Observation.points`"
    Refinement and polish take their 3D from each view's `pointmap`, so a case
    with a generator pose needs no pooled cloud at all.

```python
from meshfit.init import from_bbox
init = from_bbox(mesh, observation, frame)
```

Scale comes from the **vertical** extent, not the mean of all three. For an
object standing on a surface the full height is usually visible, whereas the
horizontal extents of a single-view cloud cover only the front and would bias
the scale small.

Extents are percentile-trimmed rather than min/max, see the bleed warning in
[Input contract](inputs.md).

Yaw is not estimated, since a bounding box carries no yaw information. This is
a seed for `from_search` rather than an answer on its own, and it is weakest on
flat objects where height is the smallest dimension.
