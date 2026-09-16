# Tilt and plausibility

Every iteration fits two hypotheses and picks one.

| | rotation | pitch / roll | guarantees |
|---|---|---|---|
| **upright** | yaw only | pinned to 0 | the object stands, always |
| **free** | any | re-estimated | none, but a real lean is reachable |

```bash
meshfit fit <case> --tilt auto      # default
meshfit fit <case> --tilt upright   # plausibility over accuracy
meshfit fit <case> --tilt free      # accuracy over plausibility
```

Most objects on a floor or table really are upright, and the upright fit makes a
whole class of failure impossible: a bad matcher cannot lay a chair on its side
if "on its side" is not in the vocabulary. But some objects genuinely lean, and
the upright fit can never reach them.

You cannot just take whichever fits better. The free fit has two extra degrees
of freedom, so its residual is always at least as low, and choosing on residual
picks it every time. `auto` requires two things instead:

1. the lean exceeds `--min-tilt` (default **20°**)
2. it renders a better silhouette than the upright fit

## Both outcomes, on real objects

| object | upright | free | verdict |
|---|---|---|---|
| game controller | 0.919, flat | **0.951**, leaning 55° | free is right, it really does lean |
| windmill souvenir | **0.870**, flat | 0.853, leaning 13.5° | free is wrong, it sits flat |

Same mechanism, opposite outcomes. The windmill's 13.5° fell below the threshold
and was rejected, and it reached the same score upright, so the lean bought
nothing.
