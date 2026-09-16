# Refinement

One iteration: place the mesh, render it into every view, match each render
against the masked observation, lift both sides of every match to 3D, solve the
pose. Repeat.

```python
from meshfit.refine import refine
result = refine(mesh, pose, views, render, match, frame, iters=3)
```

Each iteration solves the pose outright rather than correcting the previous
estimate, so errors do not accumulate across iterations.

Two hypotheses compete every iteration, one upright and one with rotation free.
See [Tilt](tilt.md).

## Failure

If every view renders too little of the mesh, or produces too few matches, the
result reports `found_nothing` and leaves the pose alone. Check it before
trusting the pose: "I could not improve this" and "this was already correct"
look the same if you only read the numbers.

```python
out = refine(mesh, pose, views, render, match)
if out.found_nothing:
    ...
```
