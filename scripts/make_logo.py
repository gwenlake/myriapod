"""Emit assets/logo.svg and assets/banner.svg from one mascot generator."""
import math

def bez(t, p0, p1, p2, p3):
    mt = 1 - t
    return (mt**3*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t**3*p3[0],
            mt**3*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t**3*p3[1])

def dbez(t, p0, p1, p2, p3):
    mt = 1 - t
    return (3*mt*mt*(p1[0]-p0[0]) + 6*mt*t*(p2[0]-p1[0]) + 3*t*t*(p3[0]-p2[0]),
            3*mt*mt*(p1[1]-p0[1]) + 6*mt*t*(p2[1]-p1[1]) + 3*t*t*(p3[1]-p2[1]))

def rot(v, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return (v[0]*c - v[1]*s, v[0]*s + v[1]*c)

def hx(c): return "#%02X%02X%02X" % c
def lerp(c1, c2, t): return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))

P = [(46, 176), (120, 232), (212, 60), (300, 96)]
N = 10
C1, C2 = (108, 92, 231), (34, 211, 178)
LEG = "#FFB020"
HX, HY, HR = 305.0, 92.0, 30.0

def mascot(grad_id):
    segs = []
    for i in range(N):
        t = i / (N - 1) * 0.92
        x, y = bez(t, *P)
        dx, dy = dbez(t, *P); n = math.hypot(dx, dy)
        tan = (dx / n, dy / n); nor = (-tan[1], tan[0])
        r = 9.5 + 12.5 * (i / (N - 1)) ** 0.8
        segs.append((x, y, r, tan, nor, hx(lerp(C1, C2, i / (N - 1)))))

    legs = []
    for i, (x, y, r, tan, nor, col) in enumerate(segs):
        if i == 0:
            continue
        for k, side in enumerate((-15, 15)):
            d = rot(nor, side)
            ax, ay = x + d[0] * (r * 0.7), y + d[1] * (r * 0.7)
            L = 13 + r * 0.55 + (2.5 if k else 0)
            kx, ky = ax + d[0] * L, ay + d[1] * L
            fx, fy = kx + tan[0] * 11, ky + tan[1] * 11
            legs.append(f'<path d="M{ax:.1f} {ay:.1f} L{kx:.1f} {ky:.1f} L{fx:.1f} {fy:.1f}"/>')

    spine = "M%.1f %.1f C%.1f %.1f, %.1f %.1f, %.1f %.1f" % (
        P[0][0], P[0][1], P[1][0], P[1][1], P[2][0], P[2][1], P[3][0], P[3][1])

    o = []
    o.append(f'<g fill="none" stroke="{LEG}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round">')
    o += ["  " + l for l in legs]
    o.append('</g>')
    o.append(f'<g fill="none" stroke="{LEG}" stroke-width="6" stroke-linecap="round">')
    o.append(f'  <path d="M{HX-6:.0f} {HY-24:.0f} C{HX-2:.0f} {HY-48:.0f}, {HX+10:.0f} {HY-54:.0f}, {HX+16:.0f} {HY-62:.0f}"/>')
    o.append(f'  <path d="M{HX+14:.0f} {HY-20:.0f} C{HX+30:.0f} {HY-38:.0f}, {HX+42:.0f} {HY-38:.0f}, {HX+54:.0f} {HY-42:.0f}"/>')
    o.append('</g>')
    o.append('<g fill="#FF6B6B">')
    o.append(f'  <circle cx="{HX+17:.0f}" cy="{HY-64:.0f}" r="6.5"/>')
    o.append(f'  <circle cx="{HX+56:.0f}" cy="{HY-43:.0f}" r="6.5"/>')
    o.append('</g>')
    o.append(f'<path d="{spine}" fill="none" stroke="url(#{grad_id})" stroke-width="26" stroke-linecap="round" opacity="0.95"/>')
    for x, y, r, tan, nor, col in segs:
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{col}"/>')
        o.append(f'<circle cx="{x - nor[0]*r*0.3:.1f}" cy="{y - nor[1]*r*0.3:.1f}" r="{r*0.42:.1f}" fill="#FFFFFF" opacity="0.22"/>')
    o.append(f'<circle cx="{HX:.0f}" cy="{HY:.0f}" r="{HR:.0f}" fill="{hx(C2)}"/>')
    o.append(f'<circle cx="{HX-4:.0f}" cy="{HY+8:.0f}" r="{HR*0.68:.0f}" fill="#FFFFFF" opacity="0.18"/>')
    for ex, ey in ((HX - 9, HY - 4), (HX + 13, HY - 6)):
        o.append(f'<circle cx="{ex:.0f}" cy="{ey:.0f}" r="9" fill="#FFFFFF"/>')
        o.append(f'<circle cx="{ex+2:.0f}" cy="{ey+1:.0f}" r="4.6" fill="#12303B"/>')
        o.append(f'<circle cx="{ex+3.6:.0f}" cy="{ey-1.2:.0f}" r="1.6" fill="#FFFFFF"/>')
    o.append(f'<path d="M{HX-4:.0f} {HY+13:.0f} q9 9 18 -1" fill="none" stroke="#12303B" stroke-width="4" stroke-linecap="round"/>')
    o.append(f'<circle cx="{HX-16:.0f}" cy="{HY+11:.0f}" r="5" fill="#FF9AA2" opacity="0.75"/>')
    return o

GRAD = ('<linearGradient id="%s" x1="0" y1="1" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{hx(C1)}"/><stop offset="1" stop-color="{hx(C2)}"/></linearGradient>')

# ---- logo.svg ----------------------------------------------------------
logo = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 372 260" width="372" height="260" role="img" aria-label="myriapod">',
        '  <title>myriapod</title>',
        '  <defs>', '    ' + GRAD % "spine", '  </defs>']
logo += ["  " + l for l in mascot("spine")]
logo.append('</svg>')
open("assets/logo.svg", "w").write("\n".join(logo) + "\n")

# ---- banner.svg --------------------------------------------------------
W, H = 1200, 400
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

# a small task graph in the background: one root, two ranks of workers
import random
random.seed(7)
nodes, edges = [], []
ranks = [(150, [200]), (330, [120, 200, 280]), (510, [90, 165, 240, 315]), (690, [130, 210, 290])]
prev = None
for rx, ys in ranks:
    cur = [(rx, y) for y in ys]
    if prev:
        for i, (x2, y2) in enumerate(cur):
            x1, y1 = prev[i % len(prev)]
            edges.append((x1, y1, x2, y2))
    nodes += cur
    prev = cur

b = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="myriapod — planner/worker agent swarms over a dynamic task tree">',
     '  <title>myriapod</title>',
     '  <defs>',
     '    ' + GRAD % "spine",
     '    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
     '<stop offset="0" stop-color="#0C1322"/><stop offset="0.55" stop-color="#141E36"/>'
     '<stop offset="1" stop-color="#101A2E"/></linearGradient>',
     '    <linearGradient id="word" x1="0" y1="0" x2="1" y2="0">'
     f'<stop offset="0" stop-color="{hx(C1)}"/><stop offset="0.5" stop-color="#4FB6F0"/>'
     f'<stop offset="1" stop-color="{hx(C2)}"/></linearGradient>',
     '  </defs>',
     f'  <rect x="0" y="0" width="{W}" height="{H}" rx="28" fill="url(#bg)"/>']

# background graph
b.append('  <g stroke="#3B82F6" stroke-width="2" opacity="0.22" fill="none">')
for x1, y1, x2, y2 in edges:
    b.append(f'    <path d="M{x1} {y1} C{(x1+x2)/2} {y1}, {(x1+x2)/2} {y2}, {x2} {y2}"/>')
b.append('  </g>')
b.append('  <g fill="#5EEAD4" opacity="0.28">')
for x, y in nodes:
    b.append(f'    <circle cx="{x}" cy="{y}" r="7"/>')
b.append('  </g>')

# mascot, right side
b.append('  <g transform="translate(690, 62) scale(1.28)">')
b += ["    " + l for l in mascot("spine")]
b.append('  </g>')

# wordmark
b.append(f'  <text x="86" y="214" font-family="{FONT}" font-size="112" font-weight="800" '
         'letter-spacing="-3" fill="url(#word)">myriapod</text>')
b.append(f'  <text x="92" y="262" font-family="{FONT}" font-size="26" font-weight="500" '
         'fill="#9FB0C9">One planner. A thousand little legs.</text>')
b.append(f'  <text x="92" y="306" font-family="{FONT}" font-size="19" font-weight="400" '
         'fill="#6B7C96">Planner / worker agent swarms over a dynamic task tree</text>')
b.append('</svg>')
open("assets/banner.svg", "w").write("\n".join(b) + "\n")
print("written")
