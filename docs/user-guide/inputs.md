# Input contract

## `View` — one camera's observation

| field | shape | dtype | frame / units | notes |
|---|---|---|---|---|
| `rgb` | `(H, W, 3)` | `uint8` | 0–255 | the photograph, unmasked |
| `mask` | `(H, W)` | `bool` | — | True on the object |
| `intrinsic` | `(3, 3)` | `float64` | pixels | `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`, at `rgb`'s resolution |
| `cam2world` | `(4, 4)` | `float64` | metres | camera → world; rotation block must be orthonormal |
| `pointmap` | `(h, w, 3)` | `float64` | metres, **camera** frame | `z <= 0` marks invalid; may differ in resolution from `rgb` |

## `Observation` — the views, and optionally the object's points

| field | shape | dtype | frame / units | notes |
|---|---|---|---|---|
| `views` | — | `list[View]` | — | all in the same world frame |
| `points` | `(N, 3)` | `float64` | metres, **world** | optional; seeds the yaw search when no pose is given. No solver reads it |
| `colors` | `(N, 3)` | `uint8` | 0–255 | the pixel each point came from; visualisation only |

Plus the mesh in its generator's canonical frame, and `canonical_up` (default
`"+Y"`) saying which axis is up in *that* frame.

!!! note "Camera convention"
    `cam2world` uses the **OpenCV** convention: **+X right, +Y down, +Z
    forward**. Getting this wrong produces a plausible-looking pose with a
    respectable residual and a visibly mirrored overlay, which is what
    [`--overlay`](visualisation.md) is for.

## Why depth is required

One photograph cannot tell you how big something is. Double an object's size
and move it twice as far from the camera, and it covers exactly the same pixels.
A single view pins down direction, shape and orientation, but leaves size and
distance locked together as one unknown, and a depth measurement frees both.

That depth is the per-view `pointmap`. Every correspondence the matcher finds is
lifted through it to get a 3D target, and those targets are the depth prior in
[Polish](polish.md).

`Observation.points` is not that. It is a pooled cloud of the masked pixels, no
solver reads it, and it is optional:

```python
Observation(views=views)          # no points at all
```

It exists to seed a bounding box when no pose is supplied
([`from_bbox`](initialisation.md)), and to draw the object in the 3D viewer. Ask
for `extent`, `centroid` or `robust_bounds` without it and meshfit tells you
which stage wanted it and how to build one.

## Two preconditions meshfit cannot check

!!! warning "The world is Z-up and gravity-aligned"
    Everything "upright" means is defined against world +Z. A tilted world
    yields confident, wrong, upright-looking poses. The residual will not reveal
    it, because a wrong gravity direction is a different self-consistent
    parameterisation, not an outlier.

!!! warning "Everything is metric, in metres"
    Thresholds throughout are absolute distances: RANSAC inlier radii, depth
    priors, convergence tolerances. A cloud in centimetres makes them silently
    100x too tight.

`validate()` checks shapes, dtypes, that the mask is non-empty and that
`cam2world`'s rotation block is orthonormal (a non-orthonormal one is the classic
symptom of scale smuggled into an extrinsic). Semantics it cannot check.

## canonical_up

glTF specifies +Y as up, so `"+Y"` is the right default for GLB input and
TRELLIS-style generators follow it. But the format's convention and the object's
canonicalisation are different claims: a generator can emit a chair lying on its
side inside a perfectly valid GLB, OBJ and PLY carry no convention at all, and
USD stages from Omniverse are authored Z-up.

Getting it wrong does not raise and does not inflate the residual. The solver
returns a confident fit with the object on its side. State it explicitly when
your generator is not glTF-conventional.

## Building an Observation

```python
import numpy as np
from meshfit.views import View, Observation
from meshfit.io import object_points_from_views

view = View(
    rgb=rgb,               # (H, W, 3) uint8
    mask=mask,             # (H, W)    bool
    intrinsic=K,           # (3, 3)    float64, pixels
    cam2world=c2w,         # (4, 4)    float64, OpenCV convention, metres
    pointmap=pointmap,     # (h, w, 3) float64, camera frame, metres
)
observation = Observation(views=[view])
observation.validate()
```

That is enough for `meshfit.fit(..., init=generator_pose)`. Add a cloud only if
you have no pose and need the yaw search, or want the object drawn in viser:

```python
points, colors = object_points_from_views([view])   # (N,3) float64, (N,3) uint8
observation = Observation(views=[view], points=points, colors=colors)
```

`object_points_from_views` subsamples: a full-resolution mask can be millions of
points and nothing downstream benefits.

!!! tip "Masked clouds bleed"
    Pixels on a silhouette edge pick up depth from whatever is behind the
    object, so a handful of points land metres away. Use
    `Observation.robust_bounds()` rather than min/max for anything that feeds a
    pose. On real data we have seen a 15 cm object report a 70 cm extent from a
    1% tail.
