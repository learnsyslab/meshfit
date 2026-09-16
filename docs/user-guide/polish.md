# Polish

One Levenberg–Marquardt pass that minimises **reprojection error in pixels**.

## Why pixels and not metres

A matcher locates a keypoint to about a pixel. Pushing that through a depth map
to get a 3D target mixes in the depth estimator's error, and a 3D residual then
treats a 1 px measurement and a 10 cm depth guess as equally reliable.

$$
\min_{\theta,S,t}\ \sum_i \rho\!\left(\frac{\lVert \pi(K,T(x_i)) - u_i\rVert^2}{\sigma_{px}^2}\right)
\;+\; \sum_i \frac{\lVert T(x_i) - y_i\rVert^2}{\sigma_{depth}^2}
$$

Reprojection carries a low $\sigma$ and constrains five of seven degrees of
freedom tightly. The depth prior carries a high $\sigma$ and constrains the one
direction nothing else can see.

On synthetic data with 10 cm depth noise and 0.5 px matcher noise this beats a
3D-3D fit by roughly 10x in translation and 20x in scale.

!!! note "The depth prior is not optional under one camera"
    With a single view, reprojection alone cannot fix scale: slide the object
    along the viewing ray, scale it proportionally, and every pixel is
    unchanged. The depth term is what makes scale observable.

## Seven parameters, or nine

The parameterisation follows what refinement concluded, and appears in the
output as `params`:

| refine chose | params | rotation | scale | translation |
|---|---|---|---|---|
| upright | **7** | yaw | per-axis | 3 |
| free | **9** | full | per-axis | 3 |

The 7-parameter form cannot represent a tilted pose. Given one it returns an
upright object, which is why it is only used when refinement settled on upright.
