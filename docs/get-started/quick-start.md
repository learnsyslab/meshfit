# Quick Start

## From the command line

meshfit reads a **case** directory holding the mesh, views and object points
(see [Input contract](../user-guide/inputs.md)).

```bash
meshfit fit test_data/my_object --overlay out/fit.png
```

```
my_object  (11.2s)
  init      : source=generator, mask_iou=0.2012
  refine    : iterations=3, matches=3181, free_rotation=True, tilt_deg=61.479,
              accepted=True, mask_iou_before=0.2012, mask_iou_after=0.9394
  polish    : params=9, reproj_px_before=12.437, reproj_px_after=9.023,
              accepted=True
  pose      : yaw -80.70 deg  tilt 55.08 deg  scale [0.1815 0.2082 0.1878]
  wrote     : test_data/my_object/pose.json
```

Every stage reports whether it was **accepted**. A stage that does not improve
the rendered silhouette is discarded.

## From Python

```python
import meshfit
from meshfit.io import load_case
from meshfit.render import BlenderRenderer
from meshfit.matcher import RoMaMatcher

case = load_case("test_data/my_object")

result = meshfit.fit(
    case.mesh,
    case.observation,
    canonical_up=case.canonical_up,
    init=case.init,                  # None for a pose-blind generator
    render=BlenderRenderer(),
    match=RoMaMatcher(),
)

print(result.pose.matrix())          # T_world_canonical (rigid)
print(result.pose.scale)             # per-canonical-axis metric scale
print(result.to_dict())              # the published JSON schema
```

## Building a case from your own data

```python
from meshfit.views import View, Observation
from meshfit.io import save_case

views = [View(rgb=rgb, mask=mask, intrinsic=K, cam2world=c2w, pointmap=pointmap)]
observation = Observation(views=views)

save_case("test_data/my_object", mesh, observation, init=generator_pose,
          meta={"canonical_up": "+Y"})
```

If you have **no** pose to pass, meshfit has to search for the orientation, and
that needs a pooled cloud to size the object against:

```python
from meshfit.io import object_points_from_views

points, colors = object_points_from_views(views)
observation = Observation(views=views, points=points, colors=colors)
```

The world must be **Z-up, gravity-aligned and in metres**. meshfit validates
shapes and dtypes but cannot check those, and getting them wrong produces
confident, wrong answers rather than errors.

## Looking at the result

```bash
meshfit fit test_data/my_object --overlay out/fit.png --debug
python scripts/show.py viser test_data/my_object
```

The overlay catches wrong placement, viser catches wrong depth. See
[Visualisation](../user-guide/visualisation.md).
