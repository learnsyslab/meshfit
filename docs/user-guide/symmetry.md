# Symmetry and confidence

The yaw search scores every hypothesis, giving a curve over 0–360°. Its shape
says how much to trust the winner.

```
chair (distinctive back)     square table (4-fold)      vase (continuous)
 │      ╱╲                    │  ╱╲   ╱╲   ╱╲   ╱╲      │
 │     ╱  ╲                   │ ╱  ╲ ╱  ╲ ╱  ╲ ╱  ╲     │──────────────────
 │────╱    ╲────────          │╯    ╰╯    ╰╯    ╰╯  ╰   │
 └───────────────────         └──────────────────────   └──────────────────
  one clear answer             four equally good          every answer ties
```

```python
result.symmetry.margin        # (best - best rival) / best
result.symmetry.k_fold        # peaks within noise of the winner
result.symmetry.is_ambiguous  # margin small, or k_fold > 1
```

## Why it is reported instead of hidden

On a symmetric object the wrong peak costs nothing visually. That is why the
scores tie. It can still break what comes next:

- A box-shaped cabinet looks the same at 0° and 180°. Pick wrong and the render
  is fine, but the drawers open into the wall.
- A mug's body gives a flat yaw curve, yet the handle's position is the whole
  point for a gripper.

Without the flag, a coin flip and a confident answer look identical.
