#!/usr/bin/env python3
"""Draw the symphony-dt mark: three score staff lines that converge at an anchor and continue as a
forecast trajectory inside a widening uncertainty band. Okabe-Ito colours, safe for colour blindness.

Writes SVG (light, dark), PNG at 512 and 1024, favicons and the GitHub social preview to docs/assets/logo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cairosvg

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "logo"

THEMES = {
    "light": {"line": "#0072B2", "band": "#56B4E9", "band_opacity": "0.35", "dot": "#E69F00", "bg": None},
    "dark": {"line": "#56B4E9", "band": "#56B4E9", "band_opacity": "0.28", "dot": "#E69F00", "bg": None},
}


def mark(theme: dict, *, size: int = 512, background: str | None = None) -> str:
    t = theme
    bg = f'<rect width="512" height="512" rx="96" fill="{background}"/>' if background else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="{size}" height="{size}" role="img" aria-label="symphony-dt">
  <title>symphony-dt</title>
  {bg}
  <g transform="translate(0 40)">
  <path d="M256 262 C 320 230, 380 170, 462 92 L 462 238 C 392 248, 322 262, 256 262 Z" fill="{t["band"]}" fill-opacity="{t["band_opacity"]}"/>
  <g fill="none" stroke="{t["line"]}" stroke-width="18" stroke-linecap="round" stroke-linejoin="round">
    <path d="M50 190 C 150 190, 190 262, 256 262"/>
    <path d="M50 262 L 256 262"/>
    <path d="M50 334 C 150 334, 190 262, 256 262"/>
    <path d="M256 262 C 322 250, 392 214, 462 166"/>
  </g>
  <circle cx="256" cy="262" r="22" fill="{t["dot"]}"/>
  </g>
</svg>
"""


def social() -> str:
    t = THEMES["dark"]
    inner = mark(t, size=420).split("\n", 1)[1].rsplit("</svg>", 1)[0]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640">
  <rect width="1280" height="640" fill="#0B1F33"/>
  <g transform="translate(70 80) scale(0.82)">{inner}</g>
  <text x="560" y="300" font-family="Helvetica, Arial, sans-serif" font-size="92" font-weight="700" fill="#FFFFFF">symphony-dt</text>
  <text x="564" y="368" font-family="Helvetica, Arial, sans-serif" font-size="34" fill="#BFE3F7">Bayesian digital twins for</text>
  <text x="564" y="414" font-family="Helvetica, Arial, sans-serif" font-size="34" fill="#BFE3F7">longitudinal clinical cohorts</text>
  <text x="564" y="500" font-family="Helvetica, Arial, sans-serif" font-size="26" fill="#E69F00">Developed by Ahmed Hemedan</text>
</svg>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (OUT / f"symphony-dt-{name}.svg").write_text(mark(theme))
    light = mark(THEMES["light"])
    for size in (512, 1024):
        cairosvg.svg2png(
            bytestring=light.encode(),
            write_to=str(OUT / f"symphony-dt-{size}.png"),
            output_width=size,
            output_height=size,
        )
    dark = mark(THEMES["dark"])
    cairosvg.svg2png(
        bytestring=dark.encode(),
        write_to=str(OUT / "symphony-dt-dark-512.png"),
        output_width=512,
        output_height=512,
    )
    fav = mark(THEMES["light"], background="#FFFFFF")
    (OUT / "favicon.svg").write_text(fav)
    for size in (16, 32, 48, 180):
        cairosvg.svg2png(
            bytestring=fav.encode(),
            write_to=str(OUT / f"favicon-{size}.png"),
            output_width=size,
            output_height=size,
        )
    subprocess.run(
        [
            "magick",
            str(OUT / "favicon-16.png"),
            str(OUT / "favicon-32.png"),
            str(OUT / "favicon-48.png"),
            str(OUT / "favicon.ico"),
        ],
        check=True,
    )
    soc = social()
    (OUT / "social-preview.svg").write_text(soc)
    cairosvg.svg2png(
        bytestring=soc.encode(),
        write_to=str(OUT / "social-preview.png"),
        output_width=1280,
        output_height=640,
    )


if __name__ == "__main__":
    main()
