# User Guide

meshfit runs three stages and keeps whichever ones earn their place.

```
initialise ──▶ refine ──▶ polish
```

| stage | what it does | needs |
|---|---|---|
| [initialise](initialisation.md) | a pose to start from: the generator's, or one found by yaw search | a renderer and `Observation.points`, if searching |
| [refine](refinement.md) | render, match, lift, re-solve the pose outright | renderer + matcher |
| [polish](polish.md) | one least-squares pass minimising reprojection error in pixels | correspondences from refine |

Each stage is optional and each is replaceable. A pose-aware backend makes the
first redundant, no GPU makes the second impossible, and the third only helps
once correspondences exist. A stage that is skipped says so in
`provenance.stages`.

## The one rule

**A stage is kept only if it renders a better silhouette than its input.**

Refinement is not monotonic. On a textureless or repetitive object, say a white
drawer unit with four identical fronts, a matcher can be confidently and
consistently wrong, and the resulting pose ends up worse than the one it started
from. So every stage record carries `accepted`, `mask_iou_before` and
`mask_iou_after`, and a rejected stage is a normal outcome rather than a
failure.

## Where to go next

- [Input contract](inputs.md) — what meshfit needs, and the two preconditions it
  cannot check
- [Tilt and plausibility](tilt.md) — when the upright constraint is given up
- [Symmetry and confidence](symmetry.md) — when the yaw is not determinable
- [Output schema](output.md) — the JSON, field by field
