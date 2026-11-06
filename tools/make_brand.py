#!/usr/bin/env python3
"""Generate the symphony-dt brand: the twin tulip mark and its variants.

The mark is Ahmed Hemedan's SYMPHONY tulip. The red tulip is the international symbol of
Parkinson's disease. Its left half is the patient, solid. Its right half is the digital twin, drawn
as posterior samples: dots that grow denser toward the seam, where the twin is most certain.

Outputs (docs/assets/brand/): mark SVG light and dark, app icon, wordmark light and dark, PNG at
512 and 1024, favicons (a simplified mark that stays legible at 16 px), the GitHub social preview,
and the ASCII tulip used by `symphony-dt about`.

Run with Homebrew cairo on the library path on macOS:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python tools/make_brand.py
"""

from __future__ import annotations

import math
import random
import subprocess
from pathlib import Path

import cairosvg

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "brand"

RED, RED_DARK = "#B4233A", "#8E1A2D"
NAVY, INK, SLATE, MUTED = "#1E2A4A", "#1B2230", "#9AA3B2", "#5B6474"
NAVY_ON_DARK, INK_ON_DARK, MUTED_ON_DARK = "#A9BBE3", "#EEF1F6", "#A7B0BF"
PAPER, NIGHT = "#F7F8FB", "#121A2E"

PETAL = "M64 94C49 94 35 84 34 63C33.4 51 34.5 40 37 28C44 34 50 41 53.5 48C56.5 38 60 30 64 20Z"
VEIN = "M53.5 48C55 62 57 78 62 92"
LEAF = "M62 118C52 116 42 108 38 96C48 98 57 104 62 112Z"
MIRROR = "translate(1.4 0) translate(128 0) scale(-1 1)"
SHIFT = "translate(-1.4 0)"


def _dots(seed: int = 7, spacing: float = 4.1, simple: bool = False) -> str:
    """Posterior-sample dots: a jittered lattice, larger near the seam and the petal's heart."""
    rng = random.Random(seed)
    out = []
    step = 7.5 if simple else spacing
    y = 16.0
    row = 0
    while y < 124:
        x = 64.5 + (step / 2 if row % 2 else 0.0)
        while x < 100:
            jx = 0.0 if simple else rng.uniform(-0.6, 0.6)
            jy = 0.0 if simple else rng.uniform(-0.6, 0.6)
            cx, cy = x + jx, y + jy
            seam = math.exp(-(((cx - 66) / 16) ** 2))
            heart = math.exp(-(((cx - 76) / 13) ** 2 + ((cy - 60) / 26) ** 2))
            r = (2.3 if simple else 0.62) + (0.0 if simple else 0.95 * max(seam * 0.7, heart))
            out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/>')
            x += step
        y += step * 0.87
        row += 1
    return "".join(out)


def tulip(*, navy: str, outline: str, simple: bool = False, uid: str = "t") -> str:
    """The tulip group in a 128 x 128 frame, optically centred."""
    stroke = 2.0 if simple else 1.2
    return (
        f'<g transform="translate(0 -6.5)">'
        f'<defs><clipPath id="{uid}"><path d="{PETAL}" transform="{MIRROR}"/>'
        f'<path d="{LEAF}" transform="{MIRROR}"/></clipPath></defs>'
        f'<path d="{PETAL}" fill="{RED}" transform="{SHIFT}"/>'
        + (
            ""
            if simple
            else f'<path d="{VEIN}" fill="none" stroke="{RED_DARK}" stroke-width="1.6" stroke-linecap="round" '
            f'transform="{SHIFT}"/>'
        )
        + f'<path d="M64 94V121" stroke="{navy}" stroke-width="{5 if simple else 4}" stroke-linecap="round" '
        f'transform="{SHIFT}"/>'
        f'<path d="{LEAF}" fill="{navy}" transform="{SHIFT}"/>'
        f'<g clip-path="url(#{uid})" fill="{navy}">{_dots(simple=simple)}</g>'
        f'<path d="{PETAL}" fill="none" stroke="{outline}" stroke-width="{stroke}" transform="{MIRROR}"/>'
        f'<path d="{LEAF}" fill="none" stroke="{outline}" stroke-width="{stroke}" transform="{MIRROR}"/>'
        f"</g>"
    )


def svg(body: str, w: int, h: int, vb: str, title: str = "symphony-dt") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{vb}" role="img" '
        f'aria-label="{title}"><title>{title}</title>{body}</svg>\n'
    )


def mark(theme: str = "light", *, simple: bool = False) -> str:
    dark = theme == "dark"
    body = tulip(
        navy=NAVY_ON_DARK if dark else NAVY,
        outline="#6E7A93" if dark else SLATE,
        simple=simple,
        uid=f"twin-{theme}{'-s' if simple else ''}",
    )
    return svg(
        body,
        512,
        512,
        "0 0 128 128",
        "symphony-dt: a tulip, the Parkinson's disease symbol, whose right half is a digital twin "
        "drawn as posterior samples",
    )


def app_icon(theme: str = "light") -> str:
    dark = theme == "dark"
    bg = NIGHT if dark else PAPER
    inner = tulip(
        navy=NAVY_ON_DARK if dark else NAVY, outline="#6E7A93" if dark else SLATE, uid=f"icon-{theme}"
    )
    body = (
        f'<rect width="128" height="128" rx="28" fill="{bg}"/>'
        f'<g transform="translate(14.08 14.08) scale(0.78)">{inner}</g>'
    )
    return svg(body, 512, 512, "0 0 128 128")


def wordmark(theme: str = "light") -> str:
    dark = theme == "dark"
    ink, muted = (INK_ON_DARK, MUTED_ON_DARK) if dark else (INK, MUTED)
    inner = tulip(
        navy=NAVY_ON_DARK if dark else NAVY, outline="#6E7A93" if dark else SLATE, uid=f"wm-{theme}"
    )
    body = (
        f'<g transform="translate(0 8) scale(1.0)">{inner}</g>'
        f'<text x="150" y="78" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="58" '
        f'font-weight="700" letter-spacing="-1" fill="{ink}">symphony<tspan fill="{RED}">-dt</tspan></text>'
        f'<text x="152" y="108" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="17.5" '
        f'fill="{muted}">Bayesian digital twins of Parkinson\'s disease progression</text>'
    )
    return svg(body, 1300, 290, "0 0 650 145")


def social() -> str:
    inner = tulip(navy=NAVY, outline=SLATE, uid="social")
    chips = [("Stan model of record", 0), ("calibrated forecasts", 1), ("reproducible", 2)]
    chip_svg = ""
    x = 560
    for label, _ in chips:
        w = 36 + 9.6 * len(label)
        chip_svg += (
            f'<rect x="{x}" y="430" width="{w:.0f}" height="44" rx="22" fill="#FFFFFF" '
            f'stroke="#D6DBE4"/><text x="{x + w / 2:.0f}" y="459" text-anchor="middle" '
            f'font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="20" fill="{INK}">'
            f"{label}</text>"
        )
        x += w + 14
    body = (
        f'<rect width="1280" height="640" fill="{PAPER}"/>'
        f'<rect x="0" y="0" width="1280" height="8" fill="{RED}"/>'
        f'<g transform="translate(70 90) scale(3.6)">{inner}</g>'
        f'<text x="560" y="250" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="104" '
        f'font-weight="700" letter-spacing="-2" fill="{INK}">symphony<tspan fill="{RED}">-dt</tspan></text>'
        f'<text x="564" y="318" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="34" '
        f'fill="{MUTED}">Bayesian digital twins of</text>'
        f'<text x="564" y="362" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="34" '
        f'fill="{MUTED}">Parkinson\'s disease progression</text>'
        f"{chip_svg}"
        f'<text x="564" y="560" font-family="Helvetica Neue, Helvetica, Arial, sans-serif" font-size="24" '
        f'font-weight="700" fill="{NAVY}">Developed by Ahmed Hemedan</text>'
    )
    return svg(body, 1280, 640, "0 0 1280 640")


ASCII_TULIP = r"""
            /\    /\    /\
           /##\  /#|:\  /::\
          |####\/##|::\/::::|
          |########|::::::::|
           \#######|:::::::/
            '.#####|:::::.'
              '-.##|::.-'
         .--.      |      .:.
        (####'-.   |   .-'::)
         '--.__'-. | .-'__:'
                '-.|.-'
                   |
          s y m p h o n y - d t
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    def write(name: str, text: str, *pngs: int) -> None:
        (OUT / f"{name}.svg").write_text(text)
        for size in pngs:
            cairosvg.svg2png(
                bytestring=text.encode(),
                write_to=str(OUT / f"{name}-{size}.png"),
                output_width=size,
                output_height=size,
            )

    write("symphony-dt-mark", mark("light"), 512, 1024)
    write("symphony-dt-mark-dark", mark("dark"), 512)
    write("symphony-dt-icon", app_icon("light"), 512, 1024)
    write("symphony-dt-icon-dark", app_icon("dark"), 512)
    for theme in ("light", "dark"):
        text = wordmark(theme)
        (OUT / f"symphony-dt-wordmark-{theme}.svg").write_text(text)
        cairosvg.svg2png(
            bytestring=text.encode(),
            write_to=str(OUT / f"symphony-dt-wordmark-{theme}.png"),
            output_width=1950,
            output_height=435,
        )
    fav = mark("light", simple=True)
    (OUT / "favicon.svg").write_text(fav)
    for size in (16, 32, 48, 180):
        cairosvg.svg2png(
            bytestring=fav.encode(),
            write_to=str(OUT / f"favicon-{size}.png"),
            output_width=size,
            output_height=size,
        )
    subprocess.run(
        ["magick", *(str(OUT / f"favicon-{s}.png") for s in (16, 32, 48)), str(OUT / "favicon.ico")],
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
    (OUT / "ascii-tulip.txt").write_text(ASCII_TULIP.lstrip("\n"))


if __name__ == "__main__":
    main()
