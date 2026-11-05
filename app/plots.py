"""Les figures de l'app, faites avec matplotlib.

Chaque dessin se fait sur un axe (draw_*), et les fonctions sans draw_ renvoient directement un SVG pour la page.
"""

import io
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'écran sur le serveur
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

# Space Mono, la police de mon portfolio, aussi dans les figures
for f in (Path(__file__).parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
# svg.fonttype path, le texte devient des formes, donc la police s'affiche partout pareil
plt.rcParams.update({"svg.fonttype": "path", "font.family": "Space Mono", "font.size": 9})

INK = "#000000"
MUTED = "#6b6b6b"
RULE = "#d9d9d9"
SHEET = "#fff4f2"
GRID_MINOR = "#f8d6d0"
GRID_MAJOR = "#eca69d"
RED = "#c4161c"
PALE_RED = "#f6d3d4"
GREY = "#9a9a9a"
PALE_GREY = "#ececec"
HATCH = "#3a3a3a"
BLUE = "#2d5da1"

# disposition classique 3 x 4 (2.5 s par colonne) + une bande de rythme de 10 s en DII
LAYOUT = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
STATE_WORD = {"present": "Present", "absent": "Absent", "uncertain": "Can't tell", "atypical": "Unusual"}
STATE_COLOR = {"present": RED, "absent": MUTED, "uncertain": INK, "atypical": BLUE}


def to_svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    # largeur 100% pour que l'image s'adapte à l'écran, le viewBox garde les proportions
    return re.sub(r'(<svg[^>]*?) width="[^"]+" height="[^"]+"', r'\1 width="100%"', svg, count=1)


# --- ECG ---

ECG_W, ECG_H = 262, 118  # taille du papier en mm


def draw_ecg(ax, signal, leads, fs, saliency=None, win=50):
    # ECG 12 dérivations à 25 mm/s et 10 mm/mV, comme sur le papier
    # 1 unité du graphique = 1 mm de papier, c'est plus simple pour placer la grille
    X0, ROW, Y0 = 10, 28, 17
    mm = 25 / fs
    ax.set_xlim(0, ECG_W)
    ax.set_ylim(ECG_H, 0)
    ax.set_facecolor(SHEET)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # grille fine tous les 1 mm et grille épaisse tous les 5 mm
    for step, color, lw in ((1, GRID_MINOR, 0.3), (5, GRID_MAJOR, 0.6)):
        xs, ys = np.arange(0, ECG_W + 1, step), np.arange(0, ECG_H + 1, step)
        segs = [[(x, 0), (x, ECG_H)] for x in xs] + [[(0, y), (ECG_W, y)] for y in ys]
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=0))

    rows = [(Y0 + i * ROW, cols) for i, cols in enumerate(LAYOUT)] + [(Y0 + 3 * ROW + 2, "II")]
    n = signal.shape[1]

    # zones bleues = fenêtres qui font le plus baisser le score quand on les efface
    if saliency is not None:
        n_win = n // win
        wmm = win * mm

        def paint(lead, w, y):
            v = saliency[leads.index(lead) * n_win + w]
            if v >= 0.05:
                ax.add_patch(Rectangle((X0 + w * wmm, y - ROW / 2), wmm, ROW, color=BLUE, alpha=0.38 * v, lw=0, zorder=1))

        for y, cols in rows:
            if isinstance(cols, str):
                for w in range(n_win):
                    paint(cols, w, y)
            else:
                for ci, lead in enumerate(cols):
                    for w in range(ci * n_win // 4, (ci + 1) * n_win // 4):
                        paint(lead, w, y)

    def trace(lead, a, b, y):
        s = signal[leads.index(lead), a:b]
        ax.plot(X0 + np.arange(a, b) * mm, y - 10 * s, color=INK, lw=0.7, solid_joinstyle="round", zorder=2)

    def label(txt, x, y):
        ax.text(x, y, txt, fontsize=7, color=INK, va="bottom", zorder=3,
                bbox={"facecolor": SHEET, "edgecolor": "none", "pad": 0.6})

    for y, cols in rows:
        # signal de calibration au début de chaque ligne (1 mV, 200 ms)
        ax.plot([2, 3.5, 3.5, 8.5, 8.5, 10], [y, y, y - 10, y - 10, y, y], color=INK, lw=0.7, zorder=2)
        if isinstance(cols, str):
            trace(cols, 0, n, y)
            label(cols, X0 + 1, y - 9)
        else:
            seg = n // 4
            for ci, lead in enumerate(cols):
                trace(lead, ci * seg, (ci + 1) * seg, y)
                label(lead, X0 + ci * seg * mm + 1, y - 9)
                if ci:
                    x = X0 + ci * seg * mm
                    ax.plot([x, x], [y - 3, y + 3], color=INK, lw=0.7, zorder=2)


def ecg_sheet(signal, leads, fs, saliency=None):
    fig = plt.figure(figsize=(ECG_W / 25.4, ECG_H / 25.4), facecolor=SHEET)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_ecg(ax, signal, leads, fs, saliency)
    return to_svg(fig)


# --- réponse du modèle ---

def draw_readout(ax, probs, bounds, states, names, truth=None):
    # un bloc par diagnostic
    # en haut le nom et la réponse, en bas la barre de proba de 0 à 1 coupée en zones
    # gris = absent, hachuré = je ne sais pas, rouge = présent, le rond = ce patient
    # les zones sont pâles, seule celle où tombe le patient est en couleur pleine
    k = len(probs)
    BW = 0.84  # largeur de la barre, le reste sert pour le chiffre
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(k, 0)
    ax.axis("off")
    for i in range(k):
        top, y, h = i + 0.28, i + 0.64, 0.15
        ax.axhline(i, color=RULE, lw=0.8)
        ax.text(0, top, names[i], fontsize=11, color=INK, va="center")
        ax.text(1, top, STATE_WORD[states[i]], fontsize=11, color=STATE_COLOR[states[i]], ha="right", va="center",
                fontweight="normal" if states[i] == "absent" else "bold")
        if truth is not None:
            # ce qu'a écrit le cardiologue, discret, sous la réponse du modèle
            ax.text(1, i + 0.9, "annotated" if truth[i] else "not annotated", fontsize=7.5,
                    color=RED if truth[i] else MUTED, ha="right", va="center")

        a, b = bounds[i]
        lo, hi = max(0.0, min(a, b)), min(1.0, max(a, b))
        p = probs[i]
        zone = "absent" if p < lo else ("present" if p > hi else "middle")
        ax.add_patch(Rectangle((0, y - h / 2), lo * BW, h, lw=0, facecolor=GREY if zone == "absent" else PALE_GREY))
        ax.add_patch(Rectangle((hi * BW, y - h / 2), (1 - hi) * BW, h, lw=0, facecolor=RED if zone == "present" else PALE_RED))
        if hi > lo:
            strong = zone == "middle"
            if a <= b:
                style = {"facecolor": "white", "hatch": "//////", "lw": 0.6, "edgecolor": HATCH if strong else "#c8c8c8"}
            else:
                style = {"facecolor": "white", "lw": 0.8, "edgecolor": BLUE if strong else "#b9c8e0"}
            ax.add_patch(Rectangle((lo * BW, y - h / 2), (hi - lo) * BW, h, **style))
        ax.scatter([p * BW], [y], s=70, facecolor="white", edgecolor=INK, linewidth=1.6, zorder=3)
        ax.text(1, y, f"{p:.2f}", fontsize=10, color=MUTED, ha="right", va="center")
    ax.axhline(k - 0.002, color=RULE, lw=0.8)


def readout(probs, bounds, states, names, truth=None):
    k = len(probs)
    fig = plt.figure(figsize=(5.6, 0.9 * k + 0.05), facecolor="white")
    ax = fig.add_axes([0.01, 0, 0.98, 1])
    draw_readout(ax, probs, bounds, states, names, truth)
    return to_svg(fig)


# --- position de l'ECG parmi la calibration ---

def position(cal, p, bounds, name):
    # les deux groupes de la calibration pour ce diagnostic, les zones, et où tombe cet ECG
    a, b = bounds
    fig, ax = plt.subplots(figsize=(4.6, 3.1), facecolor="white")
    bins = np.linspace(0, 1, 31)
    ax.hist(cal["neg"], bins=bins, color="#d6d6d6", label=f"without {name.lower()}")
    ax.hist(1 - cal["pos"], bins=bins, histtype="step", color=RED, lw=1.5, label=f"with {name.lower()}")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    lo, hi = max(0.0, min(a, b)), min(1.0, max(a, b))
    if hi > lo:
        ax.axvspan(lo, hi, facecolor="none", edgecolor=HATCH, hatch="////", lw=0, alpha=0.3, zorder=0)
    for x in (lo, hi):
        ax.axvline(x, color=MUTED, lw=0.8)
    ax.axvline(p, color=INK, lw=1.8)
    ax.text(p, 1.03, "this ECG", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("model probability", color=MUTED, fontsize=8)
    ax.set_ylabel("calibration ECGs", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.tight_layout()
    return to_svg(fig)
