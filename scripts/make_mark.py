"""Animate the meshfit mark: a mesh triangle fitting itself to measured points.

    pixi run python scripts/make_mark.py

A triangle starts rotated, oversized and offset -- a generated mesh's pose
before anything corrects it -- and converges onto three measured points, with
correspondence lines shrinking as it lands. Three points because that is
exactly the minimal set `fit_similarity` needs to solve a similarity.

Written as a GIF rather than an animated SVG because GitHub strips animation
from SVGs in READMEs. Frames are drawn at 4x and downsampled, since PIL's
draw primitives are not anti-aliased.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SS = 4                       # supersample factor


class Theme:
    def __init__(self, bg, point, mesh, fill, link, ghost):
        self.bg, self.point, self.mesh = bg, point, mesh
        self.fill, self.link, self.ghost = fill, link, ghost


LIGHT = Theme(bg=(255, 255, 255), point=(30, 34, 40), mesh=(52, 58, 66),
              fill=(52, 58, 66, 30), link=(120, 128, 138), ghost=(120, 128, 138))
DARK = Theme(bg=(24, 27, 31), point=(238, 242, 246), mesh=(210, 216, 222),
             fill=(210, 216, 222, 30), link=(128, 136, 146), ghost=(128, 136, 146))


def targets(size: int) -> np.ndarray:
    """The measured points: a deliberately scalene triangle, so the fit has a
    unique answer rather than a symmetric family of them."""
    c = size / 2
    r = size * 0.30
    angles = (-90.0, 28.0, 152.0)
    return np.array([[c + r * math.cos(math.radians(a)),
                      c + r * math.sin(math.radians(a))] for a in angles])


def ease_out(t: float, power: float = 3.0) -> float:
    """Fast then settling -- a fit converges, it does not arrive at constant
    speed."""
    return 1.0 - (1.0 - t) ** power


def pose_at(t: float, pts: np.ndarray, size: int,
            turn_deg: float, grow: float, slide) -> np.ndarray:
    """The triangle part-way through being fitted.

    A similarity: rotation, uniform scale, translation -- the same three
    degrees of freedom the solver recovers, interpolated to zero.
    """
    e = ease_out(t)
    centre = pts.mean(axis=0)
    angle = math.radians(turn_deg) * (1.0 - e)
    scale = 1.0 + (grow - 1.0) * (1.0 - e)
    offset = np.array(slide) * size * (1.0 - e)

    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    return (pts - centre) @ R.T * scale + centre + offset


def draw_favicon(size: int, theme: Theme) -> Image.Image:
    """The settled mark, drawn heavy enough to survive a browser tab.

    The animation's hairline triangle and small dots disappear below about
    32px. A favicon is read as a silhouette, so this fills the triangle and
    enlarges the points rather than scaling the same drawing down.
    """
    s = size * SS
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    pts = targets(size) * SS

    d.polygon([tuple(p) for p in pts], fill=(*theme.mesh, 255))
    r = 9.0 * SS
    for p in pts:
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(*theme.point, 255))
        d.ellipse([p[0] - r * 0.42, p[1] - r * 0.42, p[0] + r * 0.42, p[1] + r * 0.42],
                  fill=(*theme.bg, 255))          # a hole, so a point reads as a point
    return img.resize((size, size), Image.LANCZOS)


def draw_frame(t: float, size: int, theme: Theme, show_links: bool) -> Image.Image:
    s = size * SS
    img = Image.new("RGB", (s, s), theme.bg)
    d = ImageDraw.Draw(img, "RGBA")

    pts = targets(size) * SS
    tri = pose_at(t, targets(size), size, turn_deg=-146.0, grow=1.52,
                  slide=(0.06, -0.05)) * SS

    # Where it started, FADING as the fit converges. Held at constant strength
    # it reads as a second triangle rather than as history.
    ghost_alpha = int(90 * max(0.0, 1.0 - ease_out(t) * 1.15))
    if ghost_alpha > 4:
        start = pose_at(0.0, targets(size), size, -146.0, 1.52, (0.06, -0.05)) * SS
        d.line([tuple(p) for p in start] + [tuple(start[0])],
               fill=(*theme.ghost, ghost_alpha), width=int(1.8 * SS), joint="curve")

    # Correspondence lines: what pulls each vertex onto its point. Strongest
    # when the error is largest, gone once there is nothing left to correct.
    if show_links:
        for a, b in zip(tri, pts, strict=True):
            gap = float(np.linalg.norm(a - b))
            if gap <= 2.0 * SS:
                continue
            alpha = int(235 * min(1.0, gap / (size * 0.28 * SS)))
            d.line([tuple(a), tuple(b)], fill=(*theme.link, alpha),
                   width=max(1, int(1.6 * SS)))

    d.polygon([tuple(p) for p in tri], fill=theme.fill)
    d.line([tuple(p) for p in tri] + [tuple(tri[0])],
           fill=theme.mesh, width=int(3.2 * SS), joint="curve")

    r = 5.2 * SS
    for p in pts:                                        # the measurements
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=theme.point)

    return img.resize((size, size), Image.LANCZOS)


FONT = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def _load_font(px: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(FONT, px)
    except OSError:
        return ImageFont.load_default()


def wordmark(text: str, cap_px: int, colour, tracking: float = 0.06):
    """`text` in bold, on transparency, with a little letter-spacing.

    Drawn glyph by glyph because PIL has no tracking control, and a technical
    wordmark wants the letters slightly apart -- set solid, MESHFIT reads as
    one long word.
    """
    from PIL import ImageFont  # noqa: F401

    font = _load_font(cap_px)
    gap = int(cap_px * tracking)
    widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in text]
    total = sum(widths) + gap * (len(text) - 1)

    ascent, descent = font.getmetrics()
    img = Image.new("RGBA", (total + cap_px // 4, ascent + descent), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = 0
    for ch, w in zip(text, widths, strict=True):
        d.text((x - font.getbbox(ch)[0], 0), ch, font=font, fill=colour)
        x += w + gap
    return img.crop(img.getbbox())


def lockup(out: Path, text: str, theme: Theme, cap_px: int, mark_px: int,
           frames: int, hold: int, ms: int, links: bool, animate: bool) -> None:
    """Mark on the left, wordmark on the right, on transparency.

    Transparent so one file works on a light README, a dark README and both
    docs themes without a background fighting the page.
    """
    word = wordmark(text, cap_px, theme.mesh)
    gap = int(mark_px * 0.30)
    height = max(mark_px, word.height)
    width = mark_px + gap + word.width
    pad = int(height * 0.12)

    def compose(mark_rgba: Image.Image) -> Image.Image:
        canvas = Image.new("RGBA", (width + 2 * pad, height + 2 * pad), (0, 0, 0, 0))
        canvas.alpha_composite(mark_rgba, (pad, pad + (height - mark_px) // 2))
        canvas.alpha_composite(word, (pad + mark_px + gap, pad + (height - word.height) // 2))
        return canvas

    if not animate:
        compose(frame_rgba(1.0, mark_px, theme, links)).save(out)
        print(f"  {out}  ({width + 2 * pad}x{height + 2 * pad})")
        return

    seq = [compose(frame_rgba(i / (frames - 1), mark_px, theme, links))
           for i in range(frames)]
    seq += [seq[-1]] * hold
    flat = []
    for f in seq:                      # GIF has no alpha blending, only a key
        bg = Image.new("RGB", f.size, theme.bg)
        bg.paste(f, (0, 0), f)
        flat.append(bg)
    flat[0].save(out, save_all=True, append_images=flat[1:], duration=ms, loop=0,
                 optimize=True, disposal=2)
    print(f"  {out}  ({flat[0].size[0]}x{flat[0].size[1]}, {len(flat)} frames)")


def frame_rgba(t: float, size: int, theme: Theme, links: bool) -> Image.Image:
    """One mark frame with a transparent background, for compositing."""
    opaque = draw_frame(t, size, theme, links)
    rgba = opaque.convert("RGBA")
    bg = np.array(theme.bg, dtype=np.int16)
    arr = np.array(rgba, dtype=np.int16)
    # distance from the theme background becomes coverage
    dist = np.abs(arr[..., :3] - bg).max(axis=2)
    arr[..., 3] = np.clip(dist * 6, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def build(out: Path, size: int, frames: int, hold: int, ms: int, theme: Theme,
          links: bool) -> None:
    seq = [draw_frame(i / (frames - 1), size, theme, links) for i in range(frames)]
    seq += [seq[-1]] * hold                              # rest on the answer
    out.parent.mkdir(parents=True, exist_ok=True)
    seq[0].save(out, save_all=True, append_images=seq[1:], duration=ms, loop=0,
                optimize=True, disposal=2)
    print(f"  {out}  ({size}px, {len(seq)} frames)")
    still = out.with_suffix(".png")
    seq[-1].save(still)
    print(f"  {still}  (final frame)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "docs" / "img")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--frames", type=int, default=44)
    ap.add_argument("--hold", type=int, default=18, help="frames to rest on the fit")
    ap.add_argument("--ms", type=int, default=40, help="per-frame duration")
    ap.add_argument("--no-links", action="store_true")
    ap.add_argument("--marks", action="store_true",
                    help="also emit the standalone mark, without the wordmark")
    ap.add_argument("--text", default="MESHFIT")
    ap.add_argument("--cap", type=int, default=150, help="wordmark cap height, px")
    ap.add_argument("--mark", type=int, default=190, help="mark size in the lockup, px")
    args = ap.parse_args(argv)

    links = not args.no_links
    if args.marks:          # standalone mark, without the wordmark beside it
        build(args.out / "mark.gif", args.size, args.frames, args.hold, args.ms, LIGHT, links)
        build(args.out / "mark_dark.gif", args.size, args.frames, args.hold, args.ms, DARK, links)

    # favicon: the settled mark, square, no wordmark -- at 32px a word is a smudge
    draw_favicon(256, LIGHT).save(args.out / "favicon.png")
    draw_favicon(256, DARK).save(args.out / "favicon_light.png")
    print(f"  {args.out / 'favicon.png'}  (256px, solid; _light variant too)")

    for name, theme, animate in (("logo.png", LIGHT, False),
                                 ("logo_white.png", DARK, False),
                                 ("logo.gif", LIGHT, True),
                                 ("logo_dark.gif", DARK, True)):
        lockup(args.out / name, args.text, theme, args.cap, args.mark,
               args.frames, args.hold, args.ms, links, animate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
