#!/usr/bin/env python3
"""Generate the two explanatory figures of symphony-dt as SVG and PNG.

1. twin-concept: the digital twin of Parkinson's disease in one picture, from irregular clinic visits
   to a calibrated, gated forecast.
2. method-overview: the v2 pipeline, from data and adapters through the frozen pd_v2 model, Stan
   inference, the convergence gates and evaluation, with the governance band underneath.

Every curve is drawn from simple synthetic functions. No data of any kind enters these figures.
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python tools/make_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import cairosvg
from make_brand import NAVY, SLATE, tulip

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
MATH = "STIXGeneral, STIX Two Text, Cambria Math, Times New Roman, serif"
INK, MUTED, LINE, PANEL, CARD = "#1B2230", "#5B6474", "#D6DBE4", "#F3F5F9", "#FFFFFF"
M_COL, C_COL, N_COL, RED = "#2E3A8C", "#B85A0D", "#0F7B6C", "#B4233A"


def t(
    x: float,
    y: float,
    s: str,
    size: float = 14,
    *,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
    family: str = FONT,
    italic: bool = False,
) -> str:
    style = ' font-style="italic"' if italic else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}"{style}>{s}</text>'
    )


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = CARD,
    stroke: str = LINE,
    rx: float = 10,
    dash: bool = False,
) -> str:
    d = ' stroke-dasharray="5 4"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.2"{d}/>'
    )


def badge(x: float, y: float, label: str, colour: str = INK, r: float = 11) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{colour}"/>' + t(
        x, y + 4.5, label, 12.5, weight=700, fill="#FFFFFF", anchor="middle"
    )


def arrow(x1: float, y1: float, x2: float, y2: float, colour: str = "#6B7385", width: float = 2.2) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" stroke-width="{width}" '
        f'marker-end="url(#arrow)"/>'
    )


def polyline(points: list[tuple[float, float]], colour: str, width: float = 2.4, dash: str = "") -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{d}/>'
    )


def band(upper: list[tuple[float, float]], lower: list[tuple[float, float]], colour: str, opacity: float) -> str:
    pts = upper + lower[::-1]
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + "Z"
    return f'<path d="{d}" fill="{colour}" fill-opacity="{opacity}"/>'


DEFS = (
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#6B7385"/></marker></defs>'
)


def coupling(cx: float, cy: float, scale: float = 1.0) -> str:
    """The sparse coupling graph: M drives C and N, N drives C, every domain drives itself."""
    m, c, n = (cx - 40 * scale, cy - 22 * scale), (cx + 40 * scale, cy - 22 * scale), (cx, cy + 26 * scale)
    r = 14 * scale
    parts = []
    for (x1, y1), (x2, y2) in ((m, c), (m, n), (n, c)):
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy)
        parts.append(arrow(x1 + dx / d * r, y1 + dy / d * r, x2 - dx / d * (r + 3), y2 - dy / d * (r + 3)))
    for (x, y), up in ((m, True), (c, True), (n, False)):
        sy = -1 if up else 1
        parts.append(
            f'<path d="M{x - 6:.1f} {y + sy * (r - 2):.1f} C{x - 16:.1f} {y + sy * (r + 16):.1f} '
            f'{x + 16:.1f} {y + sy * (r + 16):.1f} {x + 7:.1f} {y + sy * (r - 1):.1f}" fill="none" '
            f'stroke="#6B7385" stroke-width="1.8" marker-end="url(#arrow)"/>'
        )
    for (x, y), lab, col in ((m, "M", M_COL), (c, "C", C_COL), (n, "N", N_COL)):
        parts.append(badge(x, y, lab, col, r))
    return "".join(parts)


def sigmoid_path(
    x0: float, x1: float, y0: float, y1: float, mid: float, k: float, n: int = 60
) -> list[tuple[float, float]]:
    pts = []
    for i in range(n + 1):
        u = i / n
        v = 1 / (1 + math.exp(-k * (u - mid)))
        pts.append((x0 + u * (x1 - x0), y0 + v * (y1 - y0)))
    return pts


# --------------------------------------------------------------------------------------------- 1
def twin_concept() -> str:
    W, H = 1500, 720
    p = [f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', DEFS]
    p.append(t(40, 58, "The digital twin of Parkinson's disease", 30, weight=700))
    p.append(
        t(
            40,
            88,
            "From irregular clinic visits to a calibrated forecast that must pass fixed gates",
            16.5,
            fill=MUTED,
        )
    )
    for i, (lab, name, col) in enumerate((("M", "motor", M_COL), ("C", "cognitive", C_COL), ("N", "autonomic", N_COL))):
        x = 1100 + i * 132
        p.append(badge(x, 52, lab, col, 12))
        p.append(t(x + 20, 57, name, 16))

    cols = [
        (40, "1", "THE PATIENT", "a record in the clinic"),
        (530, "2", "THE TWIN", "a Bayesian state-space model"),
        (1020, "3", "THE FORECAST", "uncertainty you can check"),
    ]
    for x, num, title, sub in cols:
        p.append(box(x, 118, 440, 452, fill=PANEL, stroke="#E3E7EE", rx=16))
        p.append(badge(x + 30, 152, num, INK, 13))
        p.append(t(x + 52, 158, title, 18, weight=700))
        p.append(t(x + 30, 188, sub, 15, fill=MUTED))
    for x in (486, 976):
        p.append(arrow(x, 344, x + 38, 344, width=3))

    # 1. the patient: tulip and irregular visits for three scores
    p.append(f'<g transform="translate(58 205) scale(0.92)">{tulip(navy=NAVY, outline=SLATE, uid="cpt")}</g>')
    p.append(t(190, 236, "visits at irregular times", 14.5, fill=MUTED))
    times = [0, 0.55, 1.4, 1.9, 2.0]
    for row, (lab, col, ys) in enumerate(
        (
            ("MDS-UPDRS III", M_COL, [0.30, 0.36, 0.40, 0.47, 0.45]),
            ("MoCA (reversed)", C_COL, [0.15, 0.14, 0.20, 0.22, 0.24]),
            ("SCOPA-AUT", N_COL, [0.20, 0.22, 0.27, 0.26, 0.33]),
        )
    ):
        y0 = 268 + row * 62
        p.append(t(190, y0, lab, 13.5, weight=700, fill=col))
        base = y0 + 36
        p.append(f'<line x1="190" y1="{base}" x2="452" y2="{base}" stroke="{LINE}" stroke-width="1.2"/>')
        pts = [(190 + tt * 125, base - v * 55) for tt, v in zip(times, ys, strict=True)]
        p.append(polyline(pts, col, 1.6))
        for x, y in pts:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="#FFFFFF" stroke="{col}" stroke-width="2.2"/>')
    p.append(f'<line x1="440" y1="250" x2="440" y2="470" stroke="{INK}" stroke-width="1.4" stroke-dasharray="4 4"/>')
    p.append(t(440, 488, "today", 13, fill=MUTED, anchor="middle"))
    p.append(t(64, 520, "three scores, sometimes missing; dose (LEDD),", 14.5, fill=MUTED))
    p.append(t(64, 542, "age, sex and optional pathway markers", 14.5, fill=MUTED))

    # 2. the twin: latent severities and coupling
    p.append(t(560, 228, "hidden severity in three domains only rises", 14.5, fill=MUTED))
    for _lab, col, mid, k, y0, y1 in (
        ("M", M_COL, 0.45, 6.0, 368, 250),
        ("C", C_COL, 0.75, 7.5, 380, 300),
        ("N", N_COL, 0.6, 6.5, 374, 278),
    ):
        p.append(polyline(sigmoid_path(570, 820, y0, y1, mid, k), col, 3.0))
    p.append(f'<line x1="570" y1="386" x2="830" y2="386" stroke="{LINE}" stroke-width="1.2"/>')
    p.append(t(570, 404, "time", 12.5, fill=MUTED))
    p.append(coupling(880, 300, 0.85))
    p.append(t(880, 362, "sparse coupling", 13, fill=MUTED, anchor="middle"))
    eq = [("s", True), ("(t + h) = s(t) + h · softplus(r", False), ("i", True), (" + C x) · ξ", False)]
    x = 560
    p.append(
        f'<text x="{x}" y="448" font-family="{MATH}" font-size="19" fill="{INK}">'
        '<tspan font-style="italic">s</tspan>(<tspan font-style="italic">t</tspan> + <tspan '
        'font-style="italic">h</tspan>) = <tspan font-style="italic">s</tspan>(<tspan font-style="italic">t'
        '</tspan>) + <tspan font-style="italic">h</tspan> · softplus(<tspan font-style="italic">r</tspan>'
        '<tspan dy="5" font-size="13" font-style="italic">i</tspan><tspan dy="-5"> </tspan>+ <tspan '
        'font-style="italic">C x</tspan>) ⊙ ξ</text>'
    )
    del eq, x
    p.append(t(560, 488, "learned from the whole cohort, then updated", 14.5, fill=MUTED))
    p.append(t(560, 510, "on this person's visits; each score is a noisy,", 14.5, fill=MUTED))
    p.append(t(560, 532, "bounded reading of one domain", 14.5, fill=MUTED))

    # 3. the forecast: fan chart, held-out visits and an event probability
    x0, x1, yb = 1050, 1430, 420
    hist = [(x0 + i * 30, yb - 40 - 6 * i + (3 if i % 2 else -2)) for i in range(5)]
    ax = hist[-1][0]
    p.append(f'<line x1="{x0}" y1="{yb + 10}" x2="{x1}" y2="{yb + 10}" stroke="{LINE}" stroke-width="1.2"/>')
    fan_x = [ax + i * 15 for i in range(19)]
    med = [(x, hist[-1][1] - 0.22 * (x - ax)) for x in fan_x]
    for width, op in ((1.0, 0.16), (0.45, 0.32)):
        up = [(x, y - width * 0.40 * (x - ax) ** 0.88) for x, y in med]
        lo = [(x, y + width * 0.40 * (x - ax) ** 0.88) for x, y in med]
        p.append(band(up, lo, M_COL, op))
    p.append(polyline(med, M_COL, 2.6))
    p.append(polyline(hist, INK, 2.0))
    for xx, yy in hist:
        p.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{INK}"/>')
    for xx, dy in ((ax + 90, -6), (ax + 180, 12), (ax + 255, -14)):
        yy = hist[-1][1] - 0.22 * (xx - ax) + dy
        p.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.4" fill="#FFFFFF" stroke="{RED}" stroke-width="2.2"/>')
    p.append(
        f'<line x1="{ax}" y1="230" x2="{ax}" y2="{yb + 10}" stroke="{INK}" stroke-width="1.4" stroke-dasharray="4 4"/>'
    )
    p.append(t(ax - 6, 246, "anchor", 13, fill=MUTED, anchor="end"))
    p.append(t(1050, 228, "50% and 95% intervals from posterior draws", 14.5, fill=MUTED))
    p.append(f'<circle cx="1058" cy="462" r="4.4" fill="#FFFFFF" stroke="{RED}" stroke-width="2.2"/>')
    p.append(t(1070, 467, "held-out visits score the forecast", 14.5, fill=MUTED))
    p.append(t(1050, 500, "coverage and CRPS against LOCF and a", 14.5, fill=MUTED))
    p.append(t(1050, 522, "mixed model; event risks such as", 14.5, fill=MUTED))
    p.append(
        f'<text x="1050" y="547" font-family="{MATH}" font-size="18" fill="{INK}">P(ΔUPDRS-III ≥ 5 within '
        f"2 years)</text>"
    )

    # the band underneath
    p.append(box(40, 598, 1420, 92, fill="#FFFFFF", stroke="#E3E7EE", rx=16))
    items = [
        ("Frozen model of record", "pd_v2, sha256 checked at import"),
        ("Fixed convergence gates", "R-hat ≤ 1.01, ESS ≥ 400, 4 chains"),
        ("Pre-registered scoring", "frozen specification, logged deviations"),
        ("No data in the package", "synthetic tests, privacy gate"),
    ]
    for i, (head, sub) in enumerate(items):
        x = 70 + i * 352
        p.append(f'<circle cx="{x}" cy="644" r="6" fill="{RED if i == 0 else NAVY}"/>')
        p.append(t(x + 16, 638, head, 16, weight=700))
        p.append(t(x + 16, 662, sub, 14, fill=MUTED))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        f"aria-label=\"The digital twin of Parkinson's disease\"><title>The digital twin of Parkinson's "
        f"disease</title>{''.join(p)}</svg>\n"
    )


# --------------------------------------------------------------------------------------------- 2
def code(x: float, y: float, w: float, s: str) -> str:
    return f'<rect x="{x}" y="{y - 17}" width="{w}" height="25" rx="5" fill="#EEF1F6"/>' + t(
        x + 9, y + 1, s, 13.5, family="SF Mono, Menlo, Consolas, monospace"
    )


def math_line(x: float, y: float, html: str, size: float = 17) -> str:
    return f'<text x="{x}" y="{y}" font-family="{MATH}" font-size="{size}" fill="{INK}">{html}</text>'


def i(s: str) -> str:
    return f'<tspan font-style="italic">{s}</tspan>'


def sub(s: str) -> str:
    return f'<tspan dy="5" font-size="12">{s}</tspan><tspan dy="-5"> </tspan>'


def method_overview() -> str:
    W, H = 2000, 1300
    p = [f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>', DEFS]
    p.append(t(40, 62, "symphony-dt method overview", 34, weight=700))
    p.append(
        t(
            40,
            96,
            "From any longitudinal cohort to forecasts that pass fixed gates, scored and governed",
            18,
            fill=MUTED,
        )
    )
    for k, (lab, name, col) in enumerate((("M", "motor", M_COL), ("C", "cognitive", C_COL), ("N", "autonomic", N_COL))):
        x = 1480 + k * 170
        p.append(badge(x, 64, lab, col, 14))
        p.append(t(x + 24, 70, name, 19))

    cols = [
        (40, 300, "1", "DATA", "one canonical dataset"),
        (370, 470, "2", "MODEL", "pd_v2, the frozen model of record"),
        (870, 360, "3", "INFERENCE", "Stan NUTS via cmdstanpy"),
        (1260, 330, "4", "GATES", "converged, or it is not"),
        (1620, 340, "5", "EVALUATE", "calibration you can check"),
    ]
    top, bottom = 130, 1010
    for x, w, num, title, subtitle in cols:
        p.append(box(x, top, w, bottom - top, fill=PANEL, stroke="#E3E7EE", rx=18))
        p.append(badge(x + 32, top + 44, num, INK, 14))
        p.append(t(x + 56, top + 51, title, 21, weight=700))
        p.append(t(x + 22, top + 84, subtitle, 16, fill=MUTED))
    for x in (344, 844, 1234, 1594):
        p.append(arrow(x, 560, x + 22, 560, width=3))

    # 1. DATA
    x = 58
    p.append(box(x, 240, 264, 152))
    p.append(t(x + 16, 268, "TwinDataset", 17, weight=700))
    for k, s in enumerate(
        (
            "participants, observations,",
            "visits and markers;",
            "time in whole days;",
            "coded validation errors",
        )
    ):
        p.append(t(x + 16, 294 + k * 23, s, 15, fill=MUTED))
    p.append(box(x, 410, 264, 268))
    p.append(t(x + 16, 438, "Clinic visits", 17, weight=700))
    p.append(t(x + 16, 462, "irregular times", 15, fill=MUTED))
    dots = [0, 22, 38, 92, 116, 132, 200]
    p.append(f'<line x1="{x + 22}" y1="490" x2="{x + 22 + 200}" y2="490" stroke="{INK}" stroke-width="1.6"/>')
    for d in dots:
        p.append(f'<circle cx="{x + 22 + d}" cy="490" r="6" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>')
    for k, (name, rng, col) in enumerate(
        (
            ("MDS-UPDRS III", "0 to 132", M_COL),
            ("MoCA", "0 to 30, reversed", C_COL),
            ("SCOPA-AUT", "0 to 69", N_COL),
        )
    ):
        y = 528 + k * 50
        p.append(f'<rect x="{x + 16}" y="{y - 16}" width="5" height="42" rx="2" fill="{col}"/>')
        p.append(t(x + 32, y, name, 16, weight=700))
        p.append(t(x + 32, y + 22, rng, 15, fill=MUTED))
    p.append(box(x, 696, 264, 166))
    p.append(t(x + 16, 724, "Adapters, by YAML mapping", 17, weight=700))
    for k, s in enumerate(
        (
            "PPMI download layout",
            "OMOP CDM, HL7 FHIR R4",
            "CDISC SDTM, wide, long",
            "validity envelope (ENV codes)",
            "harmonisation, cited only",
        )
    ):
        p.append(t(x + 16, 750 + k * 22, s, 15, fill=MUTED))
    p.append(box(x, 880, 264, 108, dash=True))
    p.append(t(x + 16, 908, "Synthetic cohorts", 17, weight=700))
    p.append(code(x + 16, 944, 232, "simulate_cohort(n)"))
    p.append(t(x + 16, 976, "no participant data ships", 14.5, fill=MUTED))

    # 2. MODEL
    x = 388
    p.append(box(x, 240, 434, 120))
    p.append(t(x + 16, 268, "Progression drive", 17, weight=700))
    p.append(
        math_line(
            x + 16,
            304,
            f"{i('r')}{sub(i('i'))} = μ{sub(i('r'))} + Γ {i('d')}{sub(i('i'))} + "
            f"{i('A z')}{sub(i('i'))} + σ{sub(i('r'))} ⊙ {i('b')}{sub(i('i'))}",
        )
    )
    p.append(math_line(x + 16, 338, f"{i('b')}{sub(i('i'))}, {i('s')}{sub(i('i'))}(0) ~ N(0, {i('I')}{sub('3')})"))
    p.append(arrow(x + 217, 362, x + 217, 380))
    p.append(box(x, 384, 434, 300))
    p.append(t(x + 16, 412, "Latent dynamics between visits", 17, weight=700))
    p.append(
        f'<text x="{x + 16}" y="446" font-family="{MATH}" font-size="17" fill="{INK}">{i("s")} = ('
        f'<tspan fill="{M_COL}" font-weight="700" font-style="italic">M</tspan>, '
        f'<tspan fill="{C_COL}" font-weight="700" font-style="italic">C</tspan>, '
        f'<tspan fill="{N_COL}" font-weight="700" font-style="italic">N</tspan>), '
        f'{i("x")} = logit<tspan dy="-7" font-size="12">−1</tspan><tspan dy="7"> </tspan>({i("s")})</text>'
    )
    p.append(math_line(x + 16, 478, f"{i('s')}({i('t')} + {i('h')}) = {i('s')}({i('t')})"))
    p.append(math_line(x + 46, 506, f"+ {i('h')} · softplus({i('r')}{sub(i('i'))} + {i('C x')}({i('t')})) ⊙ ξ"))
    p.append(math_line(x + 16, 538, f"ξ = exp(−½{i('h')}σ{sub('ξ')}² + √{i('h')} σ{sub('ξ')} ⊙ ε)"))
    p.append(box(x + 14, 556, 190, 116, fill="#F8F9FC"))
    p.append(coupling(x + 109, 610, 0.9))
    p.append(t(x + 222, 584, "coupling C", 16, weight=700))
    for k, s in enumerate(("six free entries;", "steps of at most", "0.5 years, counted", "in integer days")):
        p.append(t(x + 222, 608 + k * 20, s, 14.5, fill=MUTED))
    p.append(arrow(x + 217, 686, x + 217, 704))
    p.append(box(x, 708, 434, 154))
    p.append(t(x + 16, 736, "Bounded observation", 17, weight=700))
    p.append(
        math_line(
            x + 16,
            770,
            f"μ = {i('L')} + ({i('U')} − {i('L')}) logit<tspan dy='-7' "
            f"font-size='12'>−1</tspan><tspan dy='7'> </tspan>({i('a')} + {i('b s')}), MoCA reflected",
        )
    )
    p.append(math_line(x + 16, 802, f"{i('y')} ~ N(μ, σ²) truncated, or censored at [{i('L')}, {i('U')}]"))
    p.append(t(x + 16, 838, "LEDD enters the UPDRS-III mean only", 14.5, fill=MUTED))
    p.append(box(x, 880, 434, 108, fill="#FFFFFF", stroke=RED))
    p.append(t(x + 16, 908, "Frozen and checked", 17, weight=700, fill=RED))
    p.append(t(x + 16, 934, "model.stan byte-identical to the study;", 14.5, fill=MUTED))
    p.append(t(x + 16, 956, "sha256 checked at import, in tests, in CI", 14.5, fill=MUTED))
    p.append(t(x + 16, 978, "primary arm: A01 off, A02 on, A03 MoCA", 14.5, fill=MUTED))

    # 3. INFERENCE
    x = 888
    for k, (y, h, letter, head, lines, cmd) in enumerate(
        (
            (
                240,
                210,
                "a",
                "Population posterior",
                [
                    "at least four chains in parallel;",
                    "chain-level resume with",
                    "the same seed and chain id;",
                    "a run manifest per fit",
                ],
                "fit()",
            ),
            (
                472,
                190,
                "b",
                "Held-out design",
                ["mask each test participant", "after a 2-year anchor;", "fit on everything else"],
                "split_at_anchor()",
            ),
            (
                684,
                302,
                "c",
                "Forecast",
                [
                    "thin to 400 draws; standalone",
                    "generated quantities give",
                    "y_rep_c: the observation model",
                    "on the posterior path, free",
                    "after the anchor",
                    "fast route: EXPERIMENTAL",
                ],
                "forecast_exact()",
            ),
        )
    ):
        p.append(box(x, y, 324, h))
        p.append(badge(x + 26, y + 28, letter, MUTED, 12))
        p.append(t(x + 46, y + 34, head, 17, weight=700))
        for j, s in enumerate(lines):
            p.append(t(x + 16, y + 64 + j * 23, s, 15, fill=MUTED))
        p.append(code(x + 16, y + h - 26, 200, cmd))
        if k < 2:
            p.append(arrow(x + 162, y + h + 2, x + 162, y + h + 18))

    # 4. GATES
    x = 1278
    p.append(box(x, 240, 294, 352))
    p.append(t(x + 16, 268, "Converged only if", 17, weight=700))
    for k, (lines) in enumerate(
        (
            ["max rank-normalised", "split R-hat ≤ 1.01"],
            ["min bulk ESS ≥ 400"],
            ["at least four", "retained chains"],
        )
    ):
        y = 306 + [0, 70, 120][k]
        p.append(badge(x + 28, y - 5, str(k + 1), INK, 12))
        for j, s in enumerate(lines):
            p.append(t(x + 50, y + j * 23, s, 15.5))
    p.append(f'<line x1="{x + 16}" y1="490" x2="{x + 278}" y2="490" stroke="{LINE}"/>')
    for j, s in enumerate(("a chain is excluded when its", "mean lp__ is more than 5 sd", "below the best chain")):
        p.append(t(x + 16, 516 + j * 23, s, 14.5, fill=MUTED))
    p.append(box(x, 612, 294, 186))
    p.append(t(x + 16, 640, "Reported beside", 17, weight=700))
    for j, s in enumerate(
        (
            "divergences, tree depth,",
            "E-BFMI and tail ESS;",
            "R-hat and ESS from CmdStan's",
            "own stansummary",
        )
    ):
        p.append(t(x + 16, 666 + j * 23, s, 15, fill=MUTED))
    p.append(box(x, 818, 294, 170))
    p.append(t(x + 16, 846, "Changing a threshold", 17, weight=700))
    for j, s in enumerate(
        ("needs a GateOverride with a", "reason and an author; it is", "logged and warned, never", "silent")
    ):
        p.append(t(x + 16, 872 + j * 23, s, 15, fill=MUTED))

    # 5. EVALUATE
    x = 1638
    p.append(box(x, 240, 304, 230))
    p.append(t(x + 16, 268, "Twin against baselines", 17, weight=700))
    p.append(t(x + 16, 292, "LOCF and a mixed model by EM", 15, fill=MUTED))
    p.append(t(x + 16, 330, "paired CRPS difference,", 15, fill=MUTED))
    p.append(t(x + 16, 352, "twin minus each baseline", 15, fill=MUTED))
    p.append(f'<line x1="{x + 30}" y1="404" x2="{x + 274}" y2="404" stroke="{INK}" stroke-width="1.6"/>')
    p.append(f'<line x1="{x + 152}" y1="390" x2="{x + 152}" y2="418" stroke="{INK}" stroke-width="2"/>')
    p.append(t(x + 152, 386, "0", 14, anchor="middle"))
    p.append(t(x + 30, 440, "favours the twin", 13.5, fill=MUTED))
    p.append(t(x + 274, 440, "favours baseline", 13.5, fill=MUTED, anchor="end"))
    p.append(box(x, 488, 304, 200))
    p.append(t(x + 16, 516, "Coverage and events", 17, weight=700))
    for j, s in enumerate(
        (
            "50% and 95% intervals;",
            "+5 UPDRS-III, −2 MoCA:",
            "Brier, skill and AUC;",
            "participant bootstrap;",
            "cells under 10 suppressed",
        )
    ):
        p.append(t(x + 16, 542 + j * 23, s, 15, fill=MUTED))
    p.append(box(x, 706, 304, 282))
    p.append(t(x + 16, 734, "Calibration of the sampler", 17, weight=700))
    for j, s in enumerate(
        (
            "SBC ranks with Bonferroni",
            "or BH, and the reported",
            "uncorrected family risk;",
            "PIT and boundary checks",
        )
    ):
        p.append(t(x + 16, 760 + j * 23, s, 15, fill=MUTED))
    hb = [44, 52, 40, 48, 46, 50, 42, 47]
    for k, hh in enumerate(hb):
        p.append(f'<rect x="{x + 30 + k * 31}" y="{950 - hh}" width="24" height="{hh}" fill="#B9C0CC"/>')
    p.append(f'<line x1="{x + 24}" y1="904" x2="{x + 280}" y2="904" stroke="{INK}" stroke-dasharray="5 4"/>')

    # governance band
    p.append(box(40, 1040, 1920, 220, fill=PANEL, stroke="#E3E7EE", rx=18))
    p.append(t(76, 1086, "GOVERNANCE AND REPRODUCIBILITY", 21, weight=700))
    p.append(t(76, 1114, "every run leaves a record; the data never leave", 16, fill=MUTED))
    steps = [
        ("symphony-dt simulate, fit,", "gates, forecast, evaluate"),
        ("Run manifest", "versions, seeds, digests"),
        ("Frozen specification", "hash before scoring"),
        ("Provenance", "W3C PROV-JSON"),
        ("Cards and checklist", "model, data, TRIPOD+AI"),
        ("symphony-dt reproduce", "rerun and compare"),
    ]
    x = 76
    widths = [330, 250, 260, 220, 280, 300]
    for k, ((head, s), w) in enumerate(zip(steps, widths, strict=True)):
        mono = head.startswith("symphony-dt")
        p.append(box(x, 1140, w, 92))
        p.append(
            t(
                x + 16,
                1176,
                head,
                16,
                weight=400 if mono else 700,
                family="SF Mono, Menlo, Consolas, monospace" if mono else FONT,
            )
        )
        p.append(t(x + 16, 1206, s, 15, fill=MUTED, family="SF Mono, Menlo, Consolas, monospace" if mono else FONT))
        if k < len(steps) - 1:
            p.append(arrow(x + w + 4, 1186, x + w + 24, 1186))
        x += w + 28
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="symphony-dt method overview"><title>symphony-dt method overview</title>'
        f"{''.join(p)}</svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text, width in (
        ("twin-concept", twin_concept(), 3000),
        ("method-overview", method_overview(), 3000),
    ):
        (OUT / f"{name}.svg").write_text(text)
        cairosvg.svg2png(bytestring=text.encode(), write_to=str(OUT / f"{name}.png"), output_width=width)


if __name__ == "__main__":
    main()
