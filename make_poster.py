"""
make_poster.py  —  USM Undergraduate Symposium 2026
36" × 48" portrait, white background, print-ready at 100 dpi.

Run:
    tinyml-acoustic/venv/bin/python3 make_poster.py
Output:
    poster_ugs2026.png  (3600 × 4800 px)
"""

import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
FIGS    = os.path.join(BASE, 'tinyml-acoustic', 'figures')
RES     = os.path.join(BASE, 'tinyml-acoustic', 'results')
OUT     = os.path.join(BASE, 'poster_ugs2026.png')

with open(os.path.join(RES, 'winner_metrics.json')) as f:
    winner = json.load(f)['RandomForest']

rf       = winner  # alias
adaptive = rf['Proposed_Adaptive']
periodic = rf['Fixed_Periodic']
p_val    = rf['_p_value']
d_val    = rf['_cohen_d']
gap_pp   = adaptive['ECR'] - rf['Pure_Random']['ECR']

# ── Palette ──────────────────────────────────────────────────────────────────
BG       = '#FFFFFF'
OFFWHITE = '#F7F9F9'
GREEN    = '#1A7C3E'
DKGREEN  = '#145A2E'
LTGREEN  = '#EAF4EE'
NAVY     = '#17202A'
MUTED    = '#5D6D7E'
BODY     = '#1C2833'
RULE     = '#D5D8DC'
GOLD     = '#7D6608'
LTGOLD   = '#FEF9E7'
WHITE    = '#FFFFFF'

DPI = 100

# ── Helpers ──────────────────────────────────────────────────────────────────
def load(fname):
    return mpimg.imread(os.path.join(FIGS, fname))

def clean(ax, bg=BG):
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def show_img(ax, fname, x0=0.01, x1=0.99, y0=0.02, y1=0.99):
    """Place image using data coordinates (xlim/ylim = 0..1)."""
    ax.imshow(load(fname), aspect='auto', extent=[x0, x1, y0, y1],
              origin='upper', zorder=2)

def hline(ax, y, x0=0, x1=1, color=RULE, lw=1.5):
    ax.plot([x0, x1], [y, y], color=color, lw=lw,
            transform=ax.transAxes, clip_on=False, zorder=10)

def vline(ax, x, y0=0, y1=1, color=RULE, lw=1.5):
    ax.plot([x, x], [y0, y1], color=color, lw=lw,
            transform=ax.transAxes, clip_on=False, zorder=10)

def sec(ax, text, y=0.97, size=52, color=GREEN, x=0.03):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=size, fontweight='bold', color=color, va='top', zorder=5)

def para(ax, text, x=0.04, y=0.88, size=28, color=BODY, ls=1.6, **kw):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=size, color=color, va='top', linespacing=ls, zorder=5, **kw)

def pill(ax, label, x, y, w=0.28, h=0.09, fc=LTGREEN, ec=GREEN, tsize=26, tc=DKGREEN):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.01', facecolor=fc, edgecolor=ec,
        linewidth=2, transform=ax.transAxes, clip_on=False, zorder=4))
    ax.text(x + w/2, y + h*0.55, label, transform=ax.transAxes,
            fontsize=tsize, fontweight='bold', color=tc,
            ha='center', va='center', zorder=5)

# ─────────────────────────────────────────────────────────────────────────────
# CANVAS
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(36, 48), dpi=DPI, facecolor=BG)

# 4 rows: header | top-body | bottom-body | footer
master = gridspec.GridSpec(
    4, 1, figure=fig,
    height_ratios=[4.4, 21.0, 15.1, 7.5],
    hspace=0.0,
    left=0.016, right=0.984,
    top=0.997, bottom=0.003,
)

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
ax_h = fig.add_subplot(master[0])
ax_h.set_facecolor(NAVY)
ax_h.set_xticks([]); ax_h.set_yticks([])
for s in ax_h.spines.values(): s.set_visible(False)

# top accent stripe
ax_h.plot([0, 1], [1, 1], color=GREEN, lw=16,
          transform=ax_h.transAxes, clip_on=False)

ax_h.text(0.5, 0.82,
    'Event-Triggered Acoustic Monitoring via\nCircular Buffer Simulation: A TinyML Framework',
    transform=ax_h.transAxes,
    fontsize=84, fontweight='bold', color=WHITE,
    ha='center', va='top', linespacing=1.18)

ax_h.text(0.5, 0.22,
    'M M Nishat   ·   Dikshant Aryal   ·   Sujjal Chapagain   ·   Dept. of Computer Science, University of Southern Mississippi',
    transform=ax_h.transAxes,
    fontsize=36, color='#AAB7B8', ha='center', va='top')

ax_h.text(0.025, 0.08,
    'USM Undergraduate Research Symposium  ·  April 18, 2026',
    transform=ax_h.transAxes,
    fontsize=30, color=GREEN, ha='left', va='top', fontstyle='italic')

ax_h.text(0.975, 0.08, 'nishat12sikdar@gmail.com',
    transform=ax_h.transAxes,
    fontsize=30, color='#7F8C8D', ha='right', va='top')

# ═════════════════════════════════════════════════════════════════════════════
# TOP BODY  —  2 columns: Introduction/Architecture  |  Buffer Heatmap (hero)
# ═════════════════════════════════════════════════════════════════════════════
row1 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=master[1],
    width_ratios=[1, 1.55], wspace=0.018,
)

# ── LEFT: Introduction + Architecture ────────────────────────────────────────
left1 = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=row1[0],
    height_ratios=[1.05, 1], hspace=0.018,
)

ax_intro = fig.add_subplot(left1[0])
clean(ax_intro)
sec(ax_intro, '01   INTRODUCTION', size=50)

intro_txt = (
    "Acoustic monitoring on edge devices must balance two\n"
    "competing pressures: continuous recording captures\n"
    "everything but overflows limited flash storage, while\n"
    "detection-only logging loses the surrounding context\n"
    "critical for downstream analysis.\n\n"
    "Circular buffers are a classical fix — keep a rolling\n"
    "pre-event window so context is already saved when a\n"
    "trigger fires. But two questions stay unanswered:\n\n"
    "   1.  How much ECR is intelligence vs. storage luck?\n"
    "   2.  What are the optimal pre/post-event buffer sizes?"
)
para(ax_intro, intro_txt, size=29, y=0.84)

# green callout box
ax_intro.add_patch(FancyBboxPatch(
    (0.03, 0.04), 0.94, 0.20,
    boxstyle='round,pad=0.015',
    facecolor=LTGREEN, edgecolor=GREEN, linewidth=2.5,
    transform=ax_intro.transAxes, clip_on=False, zorder=4))
ax_intro.text(0.50, 0.22,
    'Target Application: Forest Edge Sensors',
    transform=ax_intro.transAxes,
    fontsize=30, fontweight='bold', color=DKGREEN, ha='center', va='top', zorder=5)
ax_intro.text(0.50, 0.12,
    'Illegal logging detection & biodiversity monitoring\n'
    'where storage and bandwidth are severely constrained.',
    transform=ax_intro.transAxes,
    fontsize=26, color=BODY, ha='center', va='top', linespacing=1.4, zorder=5)

ax_arch = fig.add_subplot(left1[1])
clean(ax_arch)
sec(ax_arch, '02   SYSTEM ARCHITECTURE', size=50)
show_img(ax_arch, 'fig4_architecture.png', x0=0.02, x1=0.98, y0=0.03, y1=0.88)

# ── RIGHT: Buffer Heatmap (hero) ──────────────────────────────────────────────
ax_heat = fig.add_subplot(row1[1])
clean(ax_heat, bg=OFFWHITE)
sec(ax_heat, '03   BUFFER CONFIGURATION SWEEP  —  ECR HEATMAP', size=50)

ax_heat.text(0.04, 0.90,
    'Pre-event buffer (rows) × post-event buffer (cols) swept 0–4 s independently.\n'
    'Each cell = Event Capture Rate (%). Brighter yellow = more events preserved.',
    transform=ax_heat.transAxes,
    fontsize=27, color=MUTED, va='top', linespacing=1.5, zorder=5)

show_img(ax_heat, 'figB_buffer_heatmap.png', x0=0.03, x1=0.97, y0=0.18, y1=0.86)

# gold callout at bottom of heatmap panel
ax_heat.add_patch(FancyBboxPatch(
    (0.03, 0.03), 0.94, 0.145,
    boxstyle='round,pad=0.015',
    facecolor=LTGOLD, edgecolor='#C9A227', linewidth=2.5,
    transform=ax_heat.transAxes, clip_on=False, zorder=4))
ax_heat.text(0.50, 0.165,
    'Key Finding: Pre-event buffer drives ECR gains.'
    '  Post-event saturates past 1 s.',
    transform=ax_heat.transAxes,
    fontsize=31, fontweight='bold', color=GOLD, ha='center', va='top', zorder=5)
ax_heat.text(0.50, 0.095,
    'Optimal region (pre ≥ 1 s, post ≥ 1 s) achieves ECR ≥ 97.7 %  '
    'with manageable storage cost.',
    transform=ax_heat.transAxes,
    fontsize=27, color=BODY, ha='center', va='top', zorder=5)

# thin rule between rows
hline(ax_heat, y=0.0, color=RULE, lw=2)

# ═════════════════════════════════════════════════════════════════════════════
# BOTTOM BODY  —  Methods+Classifier  |  Pareto  |  Intelligence Gap
# ═════════════════════════════════════════════════════════════════════════════
row2 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=master[2],
    width_ratios=[1, 1.1, 1.1], wspace=0.018,
)

# ── METHODS + CLASSIFIER ─────────────────────────────────────────────────────
mc = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=row2[0],
    height_ratios=[1.05, 0.95], hspace=0.018,
)

ax_meth = fig.add_subplot(mc[0])
clean(ax_meth)
sec(ax_meth, '04   METHODOLOGY', size=50)

meth_txt = (
    "Dataset:  ESC-50  ·  10 classes  ·  400 clips (5 s)\n"
    "  3 event  ·  7 non-event  ·  30/70 split\n\n"
    "Features:  122-dim per clip\n"
    "  MFCC mean + std (80)  ·  Δ-MFCC (40)  ·  RMS (2)\n\n"
    "Classifier:  Random Forest\n"
    "  n=200  ·  depth=12  ·  balanced  ·  τ=0.35\n\n"
    "Stream:  Poisson arrivals, SNR=10 dB, 30% overlap\n"
    "  (PlaylistEngine — physically grounded sim)\n\n"
    "Systems evaluated:\n"
    "  Store-All  ·  Detect-Only  ·  Fixed Periodic\n"
    "  Pure Random  ·  Proposed Adaptive\n\n"
    "Metrics:  ECR  ·  DRR  ·  FPR  ·  Cohen's d"
)
para(ax_meth, meth_txt, size=25, y=0.88, ls=1.5)

ax_clf = fig.add_subplot(mc[1])
clean(ax_clf)
sec(ax_clf, '05   CLASSIFIER', size=50)
# confusion matrix left, model metrics right
show_img(ax_clf, 'fig1_confusion_matrix.png',  x0=0.02, x1=0.52, y0=0.03, y1=0.86)
show_img(ax_clf, 'fig2_model_metrics.png',      x0=0.50, x1=0.98, y0=0.03, y1=0.86)

# ── PARETO FRONTIER ───────────────────────────────────────────────────────────
ax_pareto = fig.add_subplot(row2[1])
clean(ax_pareto, bg=OFFWHITE)
sec(ax_pareto, '06   PARETO FRONTIER', size=50)

ax_pareto.text(0.04, 0.90,
    'Each point = one system. Error bars = 95 % CI\n'
    'over repeated stream simulations (RF, 10 dB SNR).',
    transform=ax_pareto.transAxes,
    fontsize=26, color=MUTED, va='top', linespacing=1.5, zorder=5)

show_img(ax_pareto, 'figA_pareto_frontier.png', x0=0.02, x1=0.98, y0=0.05, y1=0.86)

ax_pareto.text(0.50, 0.04,
    'Adaptive reaches top-right optimal region.\n'
    'Fixed Periodic & Random cluster at bottom.',
    transform=ax_pareto.transAxes,
    fontsize=25, color=BODY, ha='center', va='bottom',
    linespacing=1.45, zorder=5)

# ── INTELLIGENCE GAP  +  SYSTEM COMPARISON ───────────────────────────────────
right2 = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=row2[2],
    height_ratios=[1, 1], hspace=0.018,
)

ax_gap = fig.add_subplot(right2[0])
clean(ax_gap, bg=OFFWHITE)
sec(ax_gap, '07   INTELLIGENCE GAP', size=50)
ax_gap.text(0.04, 0.90,
    'Proposed Adaptive vs. Pure Random at matched\n'
    'storage budget (p = 0.0004,  d = 2.59).',
    transform=ax_gap.transAxes,
    fontsize=26, color=MUTED, va='top', linespacing=1.5, zorder=5)
show_img(ax_gap, 'figH_statistical_analysis.png', x0=0.02, x1=0.98, y0=0.06, y1=0.84)

ax_comp = fig.add_subplot(right2[1])
clean(ax_comp, bg=OFFWHITE)
sec(ax_comp, '08   SYSTEM COMPARISON', size=50)
ax_comp.text(0.04, 0.90,
    'Circular Buffer 1s/1s captures ~94% of events;\n'
    'Fixed Periodic (same storage) captures only ~10%.',
    transform=ax_comp.transAxes,
    fontsize=26, color=MUTED, va='top', linespacing=1.5, zorder=5)
show_img(ax_comp, 'fig2_drr_ecr_comparison.png', x0=0.02, x1=0.98, y0=0.06, y1=0.84)

# ═════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════════════════
ax_ft = fig.add_subplot(master[3])
ax_ft.set_facecolor(NAVY)
ax_ft.set_xticks([]); ax_ft.set_yticks([])
for s in ax_ft.spines.values(): s.set_visible(False)

# top rule
ax_ft.plot([0, 1], [1, 1], color=GREEN, lw=10,
           transform=ax_ft.transAxes, clip_on=False)

# ── 4 stat callouts (left 75%) ────────────────────────────────────────────────
detect = rf['Detect-Only']

stats = [
    (f"{adaptive['ECR']:.0f}%",          'ECR — Proposed Adaptive\n(realistic streaming)'),
    (f"{adaptive['DRR']:.0f}%",          'Data Reduction Ratio\n(Proposed Adaptive)'),
    (f"+{gap_pp:.1f}pp",                 'ECR over Pure Random\n(same storage budget)'),
    (f"p={p_val:.4f}",              f'Statistical Significance\n(Cohen\'s d = {d_val:.2f})'),
]

for i, (val, lbl) in enumerate(stats):
    cx = 0.085 + i * 0.185
    ax_ft.text(cx, 0.90, val,
               transform=ax_ft.transAxes,
               fontsize=72, fontweight='bold', color='#F4C430',
               ha='center', va='top')
    ax_ft.text(cx, 0.46, lbl,
               transform=ax_ft.transAxes,
               fontsize=23, color='#BDC3C7',
               ha='center', va='top', linespacing=1.35)

for xd in [0.185, 0.370, 0.555]:
    ax_ft.plot([xd, xd], [0.12, 0.95], color='#2C3E50', lw=1.5,
               transform=ax_ft.transAxes, clip_on=False)

# ── Conclusions + Future Work (right 25%) ────────────────────────────────────
ax_ft.text(0.760, 0.95, 'CONCLUSIONS',
    transform=ax_ft.transAxes,
    fontsize=30, fontweight='bold', color=GREEN, va='top')
ax_ft.text(0.760, 0.83,
    '1. Buffer size is an optimizable variable — not a knob.\n'
    '2. Pre-event ≥ 1 s + post-event ≥ 1 s = Pareto-optimal.\n'
    '3. Adaptive system outperforms Fixed Periodic and\n'
    '    Pure Random with p < 0.001 (d = 2.59).\n'
    '4. Results hold under realistic streaming conditions\n'
    '    (Poisson arrivals, 10 dB SNR, 30% overlap).',
    transform=ax_ft.transAxes,
    fontsize=21, color='#BDC3C7', va='top', linespacing=1.45)

ax_ft.text(0.760, 0.24,
    'FUTURE WORK',
    transform=ax_ft.transAxes,
    fontsize=26, fontweight='bold', color='#7F8C8D', va='top')
ax_ft.text(0.760, 0.14,
    'On-device latency (Arduino/ESP32)  ·  DS-CNN replace RF  ·  Real forest deploy',
    transform=ax_ft.transAxes,
    fontsize=19, color='#566573', va='top')

ax_ft.text(0.025, 0.07,
    '[1] Piczak, K.J. (2015). ESC: Dataset for Environmental Sound Classification. ACM Multimedia.  '
    '[2] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.  '
    '[3] Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.  '
    '[4] McFee et al. (2015). librosa: Audio and Music Signal Analysis in Python. SciPy Conf.',
    transform=ax_ft.transAxes,
    fontsize=19, color='#566573', va='top')

# ═════════════════════════════════════════════════════════════════════════════
plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved → {OUT}")
print(f"36\" × 48\" @ {DPI} dpi  ({36*DPI} × {48*DPI} px)")
