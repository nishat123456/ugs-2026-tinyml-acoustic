"""
make_poster.py — USM Undergraduate Symposium 2026
Generates a 36" × 48" portrait academic poster as a high-res PNG.

Run from this directory:
    python3 make_poster.py

Output: poster_ugs2026.png  (3600 × 4800 px @ 100 dpi)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'tinyml-acoustic', 'figures')
RES     = os.path.join(BASE, 'tinyml-acoustic', 'results', 'metrics.json')
OUT     = os.path.join(BASE, 'poster_ugs2026.png')

# ── Load metrics ─────────────────────────────────────────────────────────────
with open(RES) as f:
    data = json.load(f)
model   = data['model']
systems = data['systems']
sysA, sysB, sysRand, sysC1, sysC2 = systems

# ── Colours ──────────────────────────────────────────────────────────────────
C_BG      = '#0F1923'   # deep navy background
C_HEADER  = '#162533'   # slightly lighter navy for header
C_PANEL   = '#1C2E3F'   # panel background
C_ACCENT  = '#27AE60'   # USM green
C_GOLD    = '#F4C430'   # callout gold
C_RED     = '#E74C3C'
C_BLUE    = '#3498DB'
C_GREY    = '#95A5A6'
C_TEXT    = '#ECF0F1'   # near-white body text
C_MUTED   = '#BDC3C7'   # muted text

def load_img(fname):
    path = os.path.join(FIG_DIR, fname)
    return mpimg.imread(path)

def hline(ax, y, xmin=0, xmax=1, color='white', lw=4, clip_on=False):
    """Horizontal line in axes fraction coordinates."""
    ax.plot([xmin, xmax], [y, y], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=clip_on)

def vline(ax, x, ymin=0, ymax=1, color='white', lw=1.5, clip_on=False):
    """Vertical line in axes fraction coordinates."""
    ax.plot([x, x], [ymin, ymax], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=clip_on)

def panel(ax, color=C_PANEL, alpha=1.0):
    """Fill axes background."""
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_visible(False)

def section_title(ax, text, y=0.97, fontsize=22, color=C_ACCENT):
    ax.text(0.03, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', color=color,
            va='top', ha='left')

def body_text(ax, text, x=0.03, y=0.0, fontsize=14, color=C_TEXT, **kw):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, color=color, va='top', ha='left',
            wrap=True, **kw)

# ─────────────────────────────────────────────────────────────────────────────
# CANVAS
# ─────────────────────────────────────────────────────────────────────────────
DPI = 100
fig = plt.figure(figsize=(36, 48), dpi=DPI, facecolor=C_BG)

# Master grid: header | body | footer
master = gridspec.GridSpec(
    3, 1,
    figure=fig,
    height_ratios=[3.2, 38, 6.8],
    hspace=0.0,
    left=0.015, right=0.985,
    top=0.995, bottom=0.005,
)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
ax_hdr = fig.add_subplot(master[0])
ax_hdr.set_facecolor(C_HEADER)
for spine in ax_hdr.spines.values():
    spine.set_visible(False)
ax_hdr.set_xticks([]); ax_hdr.set_yticks([])

# Green top bar
hline(ax_hdr, y=1.0, color=C_ACCENT, lw=10)

ax_hdr.text(0.5, 0.74,
    'Event-Triggered Acoustic Monitoring via\nCircular Buffer Simulation: A TinyML Framework',
    transform=ax_hdr.transAxes, fontsize=44, fontweight='bold',
    color=C_TEXT, ha='center', va='top', linespacing=1.25)

ax_hdr.text(0.5, 0.24,
    'M M Nishat  ·  Department of Computer Science, University of Southern Mississippi  ·  Hattiesburg, MS',
    transform=ax_hdr.transAxes, fontsize=22, color=C_MUTED,
    ha='center', va='top')

ax_hdr.text(0.5, 0.06,
    'USM Undergraduate Research Symposium  ·  April 18, 2026',
    transform=ax_hdr.transAxes, fontsize=18, color=C_ACCENT,
    ha='center', va='top', fontstyle='italic')

# ─────────────────────────────────────────────────────────────────────────────
# BODY  — 3 columns
# ─────────────────────────────────────────────────────────────────────────────
body = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=master[1],
    width_ratios=[1, 1.35, 1],
    wspace=0.025,
)

# padding constant for inner subplots
PAD = 0.02

# ── LEFT COLUMN ──────────────────────────────────────────────────────────────
left = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=body[0],
    height_ratios=[2.6, 3.5, 3.9], hspace=0.03,
)

# LEFT-1: Introduction
ax_intro = fig.add_subplot(left[0])
panel(ax_intro)
ax_intro.set_xticks([]); ax_intro.set_yticks([])
section_title(ax_intro, '01  INTRODUCTION', fontsize=22)
intro_text = (
    "Continuous audio recording on embedded edge devices wastes flash\n"
    "storage and power — most recorded segments contain no actionable\n"
    "acoustic event. Circular buffers offer a classical solution:\n"
    "maintain a rolling pre-roll window so that when a trigger fires,\n"
    "context is already captured.\n\n"
    "Two questions are rarely answered in prior work:\n"
    "  1.  How much of the system's capture rate is due to intelligent\n"
    "       detection vs. coincidental storage volume?\n"
    "  2.  Where does the buffer size trade-off plateau?\n\n"
    "This work contributes a reproducible simulation framework that\n"
    "answers both — with a random baseline at matched storage budget\n"
    "as a credibility check, and a full buffer ablation study (B=0,1,2)."
)
ax_intro.text(0.03, 0.82, intro_text, transform=ax_intro.transAxes,
              fontsize=14.5, color=C_TEXT, va='top', linespacing=1.55)

# Motivation callout box
ax_intro.add_patch(FancyBboxPatch((0.03, 0.04), 0.94, 0.20,
    boxstyle='round,pad=0.01', facecolor='#1A3A2A', edgecolor=C_ACCENT,
    linewidth=2, transform=ax_intro.transAxes, clip_on=False))
ax_intro.text(0.50, 0.155, 'Real-World Impact', transform=ax_intro.transAxes,
              fontsize=14, fontweight='bold', color=C_ACCENT, ha='center', va='top')
ax_intro.text(0.50, 0.105,
    'Low-power forest sensors for illegal logging detection\n& biodiversity monitoring where bandwidth is critical.',
    transform=ax_intro.transAxes, fontsize=13, color=C_TEXT, ha='center', va='top',
    linespacing=1.4)

# LEFT-2: Architecture
ax_arch = fig.add_subplot(left[1])
panel(ax_arch)
ax_arch.set_xticks([]); ax_arch.set_yticks([])
section_title(ax_arch, '02  SYSTEM ARCHITECTURE', fontsize=22)
img_arch = load_img('fig7_architecture.png')
ax_arch.imshow(img_arch, aspect='auto',
               extent=[0.01, 0.99, 0.01, 0.87], transform=ax_arch.transAxes,
               clip_on=True)

# LEFT-3: Methodology
ax_meth = fig.add_subplot(left[2])
panel(ax_meth)
ax_meth.set_xticks([]); ax_meth.set_yticks([])
section_title(ax_meth, '03  METHODOLOGY', fontsize=22)

meth = (
    "Dataset:  ESC-50 — 10 classes (400 clips, 5s each)\n"
    "          3 event classes · 7 non-event classes\n"
    "          30% / 70% event / non-event split\n\n"
    "Features: 122-dim vector per clip\n"
    "  • MFCC mean + std  (80-dim)\n"
    "  • Delta-MFCC mean  (40-dim)\n"
    "  • RMS energy mean + std  (2-dim)\n"
    "  SR=22050 Hz · n_mfcc=40 · hop=512\n\n"
    "Classifier:\n"
    "  RandomForest  n=200, depth=12\n"
    "  class_weight='balanced'  τ=0.35\n\n"
    "Five Systems Evaluated:\n"
    "  A  Store-All (baseline)\n"
    "  B  Detect-Only (B=0)\n"
    "  C  Random baseline (same budget as B)\n"
    "  D  Circular Buffer B=1  ← proposed\n"
    "  E  Circular Buffer B=2\n\n"
    "Metrics: DRR · ECR · FPR · PPV\n"
    "(formal definitions in paper)"
)
ax_meth.text(0.03, 0.89, meth, transform=ax_meth.transAxes,
             fontsize=14, color=C_TEXT, va='top', linespacing=1.6,
             fontfamily='monospace')

# ── CENTRE COLUMN ─────────────────────────────────────────────────────────────
centre = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=body[1],
    height_ratios=[5, 3], hspace=0.03,
)

# CENTRE-1: Hero figure
ax_hero = fig.add_subplot(centre[0])
panel(ax_hero)
ax_hero.set_xticks([]); ax_hero.set_yticks([])
section_title(ax_hero, '04  KEY RESULT — STORAGE vs. EVENT CAPTURE TRADE-OFF', fontsize=22)
img_hero = load_img('fig5_hero_tradeoff.png')
ax_hero.imshow(img_hero, aspect='auto',
               extent=[0.01, 0.99, 0.01, 0.88], transform=ax_hero.transAxes,
               clip_on=True)

# CENTRE-2: Comparison table
ax_tbl = fig.add_subplot(centre[1])
panel(ax_tbl)
ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
section_title(ax_tbl, '05  FULL SYSTEM COMPARISON', fontsize=22)
img_tbl = load_img('fig6_comparison_table.png')
ax_tbl.imshow(img_tbl, aspect='auto',
              extent=[0.01, 0.99, 0.02, 0.84], transform=ax_tbl.transAxes,
              clip_on=True)

# ── RIGHT COLUMN ──────────────────────────────────────────────────────────────
right = gridspec.GridSpecFromSubplotSpec(
    3, 1, subplot_spec=body[2],
    height_ratios=[3.2, 3.2, 3.6], hspace=0.03,
)

# RIGHT-1: Confusion matrix
ax_cm = fig.add_subplot(right[0])
panel(ax_cm)
ax_cm.set_xticks([]); ax_cm.set_yticks([])
section_title(ax_cm, '06  CLASSIFIER PERFORMANCE', fontsize=22)
img_cm = load_img('fig1_confusion_matrix.png')
ax_cm.imshow(img_cm, aspect='auto',
             extent=[0.01, 0.99, 0.01, 0.84], transform=ax_cm.transAxes,
             clip_on=True)

# RIGHT-2: Model metrics bar
ax_mmet = fig.add_subplot(right[1])
panel(ax_mmet)
ax_mmet.set_xticks([]); ax_mmet.set_yticks([])
img_mmet = load_img('fig2_model_metrics.png')
ax_mmet.imshow(img_mmet, aspect='auto',
               extent=[0.01, 0.99, 0.01, 0.99], transform=ax_mmet.transAxes,
               clip_on=True)

# RIGHT-3: Conclusions + Future Work
ax_conc = fig.add_subplot(right[2])
panel(ax_conc)
ax_conc.set_xticks([]); ax_conc.set_yticks([])
section_title(ax_conc, '07  CONCLUSIONS', fontsize=22)

conc = (
    "1.  Intelligent detection vs. random (same budget):\n"
    "    +62.5pp ECR  [95.0% vs 32.5%] — quantifies\n"
    "    the intelligence value of the classifier.\n\n"
    "2.  B=1 is the Pareto knee:\n"
    "    +1.7pp ECR over Detect-Only at just -23.3pp\n"
    "    DRR cost. B=2 gains only +0.8pp more ECR.\n\n"
    "3.  Framework is reproducible:\n"
    "    ESC-50 + sklearn + formal metric definitions.\n"
    "    All temporal assumptions made explicit."
)
ax_conc.text(0.03, 0.87, conc, transform=ax_conc.transAxes,
             fontsize=14, color=C_TEXT, va='top', linespacing=1.6)

section_title(ax_conc, 'FUTURE WORK', y=0.42, fontsize=18, color=C_GOLD)
future = (
    "→  Deploy on Arduino Nano 33 BLE Sense / ESP32;\n"
    "    measure on-device inference latency.\n"
    "→  Evaluate on a continuous real acoustic stream\n"
    "    (concept drift, polyphonic audio, variable SNR).\n"
    "→  Replace RF with quantized DS-CNN for better F1\n"
    "    while remaining MCU-deployable."
)
ax_conc.text(0.03, 0.37, future, transform=ax_conc.transAxes,
             fontsize=13.5, color=C_MUTED, va='top', linespacing=1.6)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER — Key finding callout + acknowledgments
# ─────────────────────────────────────────────────────────────────────────────
ax_ftr = fig.add_subplot(master[2])
ax_ftr.set_facecolor('#0A1520')
for spine in ax_ftr.spines.values():
    spine.set_visible(False)
ax_ftr.set_xticks([]); ax_ftr.set_yticks([])

# Green divider line
hline(ax_ftr, y=1.0, color=C_ACCENT, lw=6)

# Big callout numbers
callouts = [
    (0.13, f"{model['accuracy']:.0%}",    'Classifier Accuracy'),
    (0.34, '+62.5pp',                      'Intelligence Gap\n(Detector vs Random, same budget)'),
    (0.55, f"{sysC1['ECR']:.0%}",          'Event Capture Rate\n(Circular Buffer B=1)'),
    (0.76, f"{sysC1['DRR']:.0%}",          'Data Reduction\n(Circular Buffer B=1)'),
]
for (x, val, label) in callouts:
    ax_ftr.text(x, 0.82, val, transform=ax_ftr.transAxes,
                fontsize=48, fontweight='bold', color=C_GOLD,
                ha='center', va='top')
    ax_ftr.text(x, 0.42, label, transform=ax_ftr.transAxes,
                fontsize=14, color=C_MUTED, ha='center', va='top',
                linespacing=1.3)

# Dividers between callouts
for xd in [0.235, 0.455, 0.665]:
    vline(ax_ftr, x=xd, ymin=0.25, ymax=0.95, color='#2C3E50', lw=1.5)

# Acknowledgment
ax_ftr.text(0.985, 0.15,
    'ESC-50 dataset: Piczak (2015)  ·  Classifier: scikit-learn Random Forest  ·  '
    'nishat12sikdar@gmail.com',
    transform=ax_ftr.transAxes, fontsize=12, color='#7F8C8D',
    ha='right', va='top')

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=C_BG)
plt.close()
print(f"Poster saved → {OUT}")
print(f"Size: 36\" × 48\" at {DPI} dpi  ({36*DPI} × {48*DPI} px)")
