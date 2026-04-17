"""
make_handout.py — USM Undergraduate Symposium 2026
Generates a single-page printable handout as a PNG (Letter: 8.5" × 11").

Run from this directory:
    python3 make_handout.py

Output: handout_ugs2026.png  (850 × 1100 px @ 100 dpi, prints to Letter)
"""

import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

BASE    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'tinyml-acoustic', 'figures')
RES     = os.path.join(BASE, 'tinyml-acoustic', 'results', 'metrics.json')
OUT     = os.path.join(BASE, 'handout_ugs2026.png')

with open(RES) as f:
    data = json.load(f)
model   = data['model']
systems = data['systems']
sysA, sysB, sysRand, sysC1, sysC2 = systems

# Colours — white/light for printing
C_BG     = '#FFFFFF'
C_ACCENT = '#1A7C3E'   # dark green — prints well
C_BLUE   = '#1A5276'
C_RED    = '#922B21'
C_HEADER = '#1A252F'
C_BODY   = '#212121'
C_MUTED  = '#555555'
C_PANEL  = '#F4F6F7'
C_GOLD   = '#7D6608'

DPI = 100
fig = plt.figure(figsize=(8.5, 11), dpi=DPI, facecolor=C_BG)

gs = gridspec.GridSpec(
    5, 2,
    figure=fig,
    height_ratios=[1.1, 2.8, 2.8, 2.4, 0.9],
    width_ratios=[1, 1],
    hspace=0.10, wspace=0.07,
    left=0.04, right=0.96,
    top=0.97, bottom=0.03,
)

def no_ticks(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def hline(ax, y, xmin=0, xmax=1, color='black', lw=3, clip_on=False):
    ax.plot([xmin, xmax], [y, y], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=clip_on)

def vline(ax, x, ymin=0, ymax=1, color='black', lw=1, clip_on=False):
    ax.plot([x, x], [ymin, ymax], color=color, linewidth=lw,
            transform=ax.transAxes, clip_on=clip_on)

# ── HEADER (full width) ───────────────────────────────────────────────────────
ax_hdr = fig.add_subplot(gs[0, :])
ax_hdr.set_facecolor(C_HEADER)
no_ticks(ax_hdr)

hline(ax_hdr, y=1.0, color=C_ACCENT, lw=5)

ax_hdr.text(0.5, 0.80,
    'Event-Triggered Acoustic Monitoring via Circular Buffer Simulation',
    transform=ax_hdr.transAxes, fontsize=11.5, fontweight='bold',
    color='white', ha='center', va='top')
ax_hdr.text(0.5, 0.42,
    'M M Nishat  ·  University of Southern Mississippi  ·  USM Undergraduate Symposium 2026',
    transform=ax_hdr.transAxes, fontsize=8, color='#BDC3C7', ha='center', va='top')
ax_hdr.text(0.5, 0.10,
    'nishat12sikdar@gmail.com',
    transform=ax_hdr.transAxes, fontsize=7.5, color=C_ACCENT,
    ha='center', va='top', fontstyle='italic')

# ── ABSTRACT (full width) ─────────────────────────────────────────────────────
ax_abs = fig.add_subplot(gs[1, :])
ax_abs.set_facecolor(C_PANEL)
no_ticks(ax_abs)

ax_abs.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
    boxstyle='square,pad=0', facecolor=C_PANEL, edgecolor='#CACFD2',
    linewidth=1, transform=ax_abs.transAxes))

ax_abs.text(0.02, 0.97, 'ABSTRACT', transform=ax_abs.transAxes,
            fontsize=9, fontweight='bold', color=C_ACCENT, va='top')

abstract = (
    "Continuous audio recording on resource-constrained edge devices wastes storage and power by persisting non-event "
    "segments. We present a simulation framework evaluating five acoustic monitoring strategies — Store-All, Detect-Only, "
    "Random baseline, and Circular Buffer at B=1 and B=2 — using the ESC-50 environmental sound dataset. A Random Forest "
    "classifier (MFCC + delta-MFCC + RMS, 122-dim) achieves 82.5% accuracy on a 10-class subset.\n\n"
    "Key finding: at the same storage budget as Detect-Only (DRR=69.5%), a random-save baseline captures only 32.5% of "
    "events — while Detect-Only captures 95.0% (+62.5pp gap). Adding a 1-clip circular buffer raises ECR to 96.7% at "
    "DRR=46.2%. A buffer ablation study identifies B=1 as the Pareto-optimal point: B=2 gains only +0.8pp ECR at a "
    "-15.2pp DRR cost."
)
ax_abs.text(0.02, 0.82, abstract, transform=ax_abs.transAxes,
            fontsize=8.5, color=C_BODY, va='top', linespacing=1.55,
            wrap=True)

# Key numbers bar
kn = [
    (f"{model['accuracy']:.0%}", 'Classifier Accuracy'),
    ('+62.5pp', 'Intelligence Gap'),
    (f"{sysC1['ECR']:.0%}", 'ECR  (B=1)'),
    (f"{sysC1['DRR']:.0%}", 'DRR  (B=1)'),
]
for i, (val, lbl) in enumerate(kn):
    x = 0.125 + i * 0.25
    ax_abs.text(x, 0.18, val, transform=ax_abs.transAxes,
                fontsize=14, fontweight='bold', color=C_BLUE, ha='center', va='top')
    ax_abs.text(x, 0.05, lbl, transform=ax_abs.transAxes,
                fontsize=7.5, color=C_MUTED, ha='center', va='top')

# ── HERO FIGURE ───────────────────────────────────────────────────────────────
ax_hero = fig.add_subplot(gs[2, 0])
ax_hero.set_facecolor(C_BG)
no_ticks(ax_hero)
ax_hero.text(0.02, 0.99, 'Storage Efficiency vs. Event Capture Trade-off',
             transform=ax_hero.transAxes, fontsize=8.5, fontweight='bold',
             color=C_ACCENT, va='top')
img = mpimg.imread(os.path.join(FIG_DIR, 'fig5_hero_tradeoff.png'))
ax_hero.imshow(img, aspect='auto',
               extent=[0.01, 0.99, 0.01, 0.93], transform=ax_hero.transAxes,
               clip_on=True)

# ── COMPARISON TABLE ─────────────────────────────────────────────────────────
ax_tbl = fig.add_subplot(gs[2, 1])
ax_tbl.set_facecolor(C_BG)
no_ticks(ax_tbl)
ax_tbl.text(0.02, 0.99, 'System Comparison — All Metrics',
            transform=ax_tbl.transAxes, fontsize=8.5, fontweight='bold',
            color=C_ACCENT, va='top')
img2 = mpimg.imread(os.path.join(FIG_DIR, 'fig6_comparison_table.png'))
ax_tbl.imshow(img2, aspect='auto',
              extent=[0.01, 0.99, 0.04, 0.93], transform=ax_tbl.transAxes,
              clip_on=True)

# ── CLASSIFIER METRICS ────────────────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[3, 0])
ax_cm.set_facecolor(C_BG)
no_ticks(ax_cm)
ax_cm.text(0.02, 0.99, 'Confusion Matrix',
           transform=ax_cm.transAxes, fontsize=8.5, fontweight='bold',
           color=C_ACCENT, va='top')
img3 = mpimg.imread(os.path.join(FIG_DIR, 'fig1_confusion_matrix.png'))
ax_cm.imshow(img3, aspect='auto',
             extent=[0.01, 0.99, 0.01, 0.93], transform=ax_cm.transAxes,
             clip_on=True)

# ── METHODOLOGY + CONCLUSIONS ─────────────────────────────────────────────────
ax_mc = fig.add_subplot(gs[3, 1])
ax_mc.set_facecolor(C_PANEL)
no_ticks(ax_mc)

ax_mc.text(0.03, 0.98, 'METHODOLOGY', transform=ax_mc.transAxes,
           fontsize=8.5, fontweight='bold', color=C_ACCENT, va='top')
meth = (
    "ESC-50 · 10 classes · 400 clips · 122-dim features\n"
    "(MFCC + Δ-MFCC + RMS)  ·  RF n=200  ·  τ=0.35\n"
    "5 systems: Store-All · Detect-Only · Random ·\n"
    "Circ. Buffer B=1 (proposed) · Circ. Buffer B=2"
)
ax_mc.text(0.03, 0.84, meth, transform=ax_mc.transAxes,
           fontsize=8, color=C_BODY, va='top', linespacing=1.5)

hline(ax_mc, y=0.56, xmin=0.03, xmax=0.97, color='#CACFD2', lw=0.8)

ax_mc.text(0.03, 0.53, 'CONCLUSIONS', transform=ax_mc.transAxes,
           fontsize=8.5, fontweight='bold', color=C_BLUE, va='top')
conc = (
    "1. +62.5pp ECR gap proves detector adds real value.\n"
    "2. B=1 is Pareto-optimal — B=2 offers diminishing returns.\n"
    "3. Framework is reproducible with formal metric definitions.\n\n"
    "Future: on-device latency · continuous stream · DS-CNN."
)
ax_mc.text(0.03, 0.40, conc, transform=ax_mc.transAxes,
           fontsize=8, color=C_BODY, va='top', linespacing=1.55)

hline(ax_mc, y=0.10, xmin=0.03, xmax=0.97, color='#CACFD2', lw=0.8)
ax_mc.text(0.03, 0.08, 'ESC-50: Piczak (2015)  ·  sklearn Random Forest',
           transform=ax_mc.transAxes, fontsize=7, color=C_MUTED, va='top')

# ── FOOTER ────────────────────────────────────────────────────────────────────
ax_ftr = fig.add_subplot(gs[4, :])
ax_ftr.set_facecolor(C_HEADER)
no_ticks(ax_ftr)

hline(ax_ftr, y=1.0, color=C_ACCENT, lw=3)

ax_ftr.text(0.5, 0.65,
    'TinyML · Acoustic Event Detection · Edge Computing · Environmental Monitoring · Circular Buffer · ESC-50',
    transform=ax_ftr.transAxes, fontsize=7.5, color='#BDC3C7',
    ha='center', va='top')
ax_ftr.text(0.5, 0.18,
    'University of Southern Mississippi  ·  Department of Computer Science  ·  nishat12sikdar@gmail.com',
    transform=ax_ftr.transAxes, fontsize=7, color='#7F8C8D',
    ha='center', va='top')

# ── SAVE ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=C_BG)
plt.close()
print(f"Handout saved → {OUT}")
print(f"Size: 8.5\" × 11\" Letter at {DPI} dpi  ({int(8.5*DPI)} × {11*DPI} px)")
