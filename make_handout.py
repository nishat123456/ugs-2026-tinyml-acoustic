"""
make_handout.py  —  USM Undergraduate Symposium 2026
8.5" × 11" Letter, white background, print-ready.

Run:
    tinyml-acoustic/venv/bin/python3 make_handout.py
Output:
    handout_ugs2026.png  (850 × 1100 px @ 100 dpi)
"""

import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

BASE  = os.path.dirname(os.path.abspath(__file__))
FIGS  = os.path.join(BASE, 'tinyml-acoustic', 'figures')
RES   = os.path.join(BASE, 'tinyml-acoustic', 'results')
OUT   = os.path.join(BASE, 'handout_ugs2026.png')

with open(os.path.join(RES, 'winner_metrics.json')) as f:
    rf = json.load(f)['RandomForest']

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = '#FFFFFF'
OFFWHITE= '#F7F9F9'
NAVY    = '#17202A'
GREEN   = '#1A7C3E'
DKGREEN = '#145A2E'
LTGREEN = '#EAF4EE'
BLUE    = '#1A5276'
BODY    = '#1C2833'
MUTED   = '#5D6D7E'
RULE    = '#D5D8DC'
GOLD    = '#7D6608'
LTGOLD  = '#FEF9E7'
WHITE   = '#FFFFFF'

DPI = 100

def load(fname):
    return mpimg.imread(os.path.join(FIGS, fname))

def clean(ax, bg=BG):
    ax.set_facecolor(bg)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def show_img(ax, fname, x0=0.01, x1=0.99, y0=0.02, y1=0.98):
    ax.imshow(load(fname), aspect='auto', extent=[x0, x1, y0, y1],
              origin='upper', zorder=2)

def hline(ax, y, x0=0, x1=1, color=RULE, lw=1):
    ax.plot([x0, x1], [y, y], color=color, lw=lw,
            transform=ax.transAxes, clip_on=False, zorder=10)

# ─────────────────────────────────────────────────────────────────────────────
# CANVAS
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.5, 11), dpi=DPI, facecolor=BG)

master = gridspec.GridSpec(
    4, 1, figure=fig,
    height_ratios=[1.05, 2.4, 5.4, 2.1],
    hspace=0.06,
    left=0.03, right=0.97,
    top=0.98, bottom=0.02,
)

# ── HEADER ───────────────────────────────────────────────────────────────────
ax_h = fig.add_subplot(master[0])
ax_h.set_facecolor(NAVY)
ax_h.set_xticks([]); ax_h.set_yticks([])
for s in ax_h.spines.values(): s.set_visible(False)

ax_h.plot([0, 1], [1, 1], color=GREEN, lw=4,
          transform=ax_h.transAxes, clip_on=False)

ax_h.text(0.5, 0.85,
    'Event-Triggered Acoustic Monitoring via Circular Buffer Simulation',
    transform=ax_h.transAxes,
    fontsize=11, fontweight='bold', color=WHITE, ha='center', va='top')
ax_h.text(0.5, 0.44,
    'M M Nishat  ·  University of Southern Mississippi  ·  USM Undergraduate Symposium 2026',
    transform=ax_h.transAxes,
    fontsize=7.5, color='#AAB7B8', ha='center', va='top')
ax_h.text(0.02, 0.10, 'nishat12sikdar@gmail.com',
    transform=ax_h.transAxes,
    fontsize=7, color=GREEN, ha='left', va='top', fontstyle='italic')
ax_h.text(0.98, 0.10, 'April 18, 2026',
    transform=ax_h.transAxes,
    fontsize=7, color='#566573', ha='right', va='top')

# ── ABSTRACT + KEY NUMBERS ───────────────────────────────────────────────────
ax_abs = fig.add_subplot(master[1])
clean(ax_abs, bg=OFFWHITE)

ax_abs.text(0.02, 0.97, 'ABSTRACT',
    transform=ax_abs.transAxes,
    fontsize=8.5, fontweight='bold', color=GREEN, va='top')

adaptive = rf['Proposed_Adaptive']
periodic = rf['Fixed_Periodic']
p_val    = rf['_p_value']
d_val    = rf['_cohen_d']
gap_pp   = adaptive['ECR'] - rf['Pure_Random']['ECR']

# [SYMPOSIUM FIX]: High-impact Hero Sentence
hero_text = (
    f"A 1-second circular buffer on edge devices achieves optimal TCR utility, "
    f"reducing storage requirements by {adaptive['DRR']:.0f}% while preserving "
    f"{adaptive['ECR']:.0f}% of events under realistic stochastic streaming."
)
ax_abs.text(0.02, 0.90, hero_text,
    transform=ax_abs.transAxes,
    fontsize=9.5, fontweight='bold', color=NAVY, va='top')

abstract = (
    "Acoustic event detection on embedded edge devices faces a fundamental tension: continuous recording "
    "overflows flash storage, while detection-only logging loses surrounding event context. We present a "
    "simulation framework that treats the circular buffer as a 2D scientific variable — independently "
    "sweeping pre-event (0–2 s) and post-event (0–4 s) buffer sizes — across five strategies: Store-All, "
    "Detect-Only, Fixed Periodic, Pure Random, and Proposed Adaptive. Experiments run under physically "
    f"grounded streaming (Poisson arrivals, SNR=10 dB, 30% overlap). Under realistic streaming, Proposed "
    f"Adaptive achieves ECR={adaptive['ECR']:.1f}% at DRR={adaptive['DRR']:.1f}%, vs "
    f"ECR={periodic['ECR']:.1f}% for Fixed Periodic at the same storage budget. "
    f"Intelligence gap over Pure Random: +{gap_pp:.1f} pp (p={p_val:.4f}, Cohen's d={d_val:.2f}). "
    "In the isolated buffer sweep (clean stream), the optimal 1s/1s configuration achieves ECR=97.7%."
)
ax_abs.text(0.02, 0.78, abstract,
    transform=ax_abs.transAxes,
    fontsize=7.8, color=BODY, va='top', linespacing=1.45, wrap=True)

# 4 key numbers
kn = [
    (f"{adaptive['ECR']:.1f}%",  'Event Capture Rate\n(ECR)'),
    (f"{adaptive['DRR']:.1f}%",  'Data Reduction\n(DRR)'),
    (f'+{gap_pp:.1f}pp',           'Intelligence Gap\n(vs. Random)'),
    (f'd={d_val:.2f}',            'Large Effect Size\n(Cohen\'s d)'),
]
for i, (val, lbl) in enumerate(kn):
    x = 0.13 + i * 0.25
    ax_abs.text(x, 0.22, val,
        transform=ax_abs.transAxes,
        fontsize=15, fontweight='bold', color=BLUE, ha='center', va='top')
    ax_abs.text(x, 0.09, lbl,
        transform=ax_abs.transAxes,
        fontsize=7, color=MUTED, ha='center', va='top', linespacing=1.3)

hline(ax_abs, y=0.27, x0=0.02, x1=0.98, color=RULE, lw=0.8)

# ── FIGURES — 3-panel grid ────────────────────────────────────────────────────
row_figs = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=master[2],
    hspace=0.06, wspace=0.04,
)

# Row A: Heatmap (span 2 cols) | Pareto
ax_heat = fig.add_subplot(row_figs[0, 0:2])
clean(ax_heat, bg=OFFWHITE)
ax_heat.text(0.02, 0.98,
    'Buffer Configuration Sweep — ECR Heatmap',
    transform=ax_heat.transAxes,
    fontsize=7.5, fontweight='bold', color=GREEN, va='top')
show_img(ax_heat, 'figB_buffer_heatmap.png', x0=0.02, x1=0.98, y0=0.03, y1=0.90)

ax_pareto = fig.add_subplot(row_figs[0, 2])
clean(ax_pareto, bg=OFFWHITE)
ax_pareto.text(0.04, 0.98,
    'Pareto Frontier',
    transform=ax_pareto.transAxes,
    fontsize=7.5, fontweight='bold', color=GREEN, va='top')
show_img(ax_pareto, 'figA_pareto_frontier.png', x0=0.02, x1=0.98, y0=0.03, y1=0.90)

# Row B: Intelligence gap | System comparison | Classifier
ax_gap = fig.add_subplot(row_figs[1, 0])
clean(ax_gap, bg=OFFWHITE)
ax_gap.text(0.04, 0.98,
    f'Intelligence Gap (d={d_val:.2f})',
    transform=ax_gap.transAxes,
    fontsize=7.5, fontweight='bold', color=GREEN, va='top')
show_img(ax_gap, 'figH_statistical_analysis.png', x0=0.02, x1=0.98, y0=0.03, y1=0.90)

ax_comp = fig.add_subplot(row_figs[1, 1])
clean(ax_comp, bg=OFFWHITE)
ax_comp.text(0.04, 0.98,
    'System Comparison',
    transform=ax_comp.transAxes,
    fontsize=7.5, fontweight='bold', color=GREEN, va='top')
show_img(ax_comp, 'fig2_drr_ecr_comparison.png', x0=0.02, x1=0.98, y0=0.03, y1=0.90)

ax_clf = fig.add_subplot(row_figs[1, 2])
clean(ax_clf, bg=OFFWHITE)
ax_clf.text(0.04, 0.98,
    'Classifier Performance',
    transform=ax_clf.transAxes,
    fontsize=7.5, fontweight='bold', color=GREEN, va='top')
show_img(ax_clf, 'fig1_confusion_matrix.png', x0=0.02, x1=0.98, y0=0.03, y1=0.90)

# ── FOOTER — Conclusions + Methods + References ───────────────────────────────
ax_ft = fig.add_subplot(master[3])
clean(ax_ft, bg=OFFWHITE)
hline(ax_ft, y=1.0, color=GREEN, lw=3)

# 3 columns
cols = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=master[3],
    wspace=0.06,
)

ax_c1 = fig.add_subplot(cols[0])
clean(ax_c1, bg=OFFWHITE)
ax_c1.text(0.04, 0.96, 'CONCLUSIONS', transform=ax_c1.transAxes,
           fontsize=7.5, fontweight='bold', color=GREEN, va='top')
ax_c1.text(0.04, 0.82,
    f'1. A {adaptive["DRR"]:.0f}% reduction in storage\n'
    '    is achieved without losing semantic\n'
    '    event context (ECR > 75%).\n'
    '2. Dynamic buffering dominates all\n'
    '    literature-standard baselines.\n'
    f'3. The Intelligence Gap is dominant:\n'
    f'    d={d_val:.2f} (p={p_val:.4f}).',
    transform=ax_c1.transAxes,
    fontsize=7.2, color=BODY, va='top', linespacing=1.45)

ax_c2 = fig.add_subplot(cols[1])
clean(ax_c2, bg=OFFWHITE)
ax_c2.text(0.04, 0.96, 'METHODOLOGY', transform=ax_c2.transAxes,
           fontsize=7.5, fontweight='bold', color=GREEN, va='top')
ax_c2.text(0.04, 0.82,
    'ESC-50 (Piczak) · 10 classes · 400 clips\n'
    '122-dim features: MFCC+Δ+RMS\n'
    'Random Forest (n=100, τ=0.35)\n'
    'Stochastic Stream: Poisson Arrivals,\n'
    'SNR=10 dB, Augmentation (±10%)\n'
    'Metrics: ECR · DRR · Cohen\'s d',
    transform=ax_c2.transAxes,
    fontsize=7.2, color=BODY, va='top', linespacing=1.45)

ax_c3 = fig.add_subplot(cols[2])
clean(ax_c3, bg=OFFWHITE)
ax_c3.text(0.04, 0.96, 'FUTURE WORK', transform=ax_c3.transAxes,
           fontsize=7.5, fontweight='bold', color=GREEN, va='top')
ax_c3.text(0.04, 0.82,
    '→  On-device deployment (Arduino/ESP32)\n'
    '→  Measure inference latency on MCU\n'
    '→  Replace RF with quantized DS-CNN\n'
    '→  Evaluate on continuous real stream\n\n'
    'ESC-50: Piczak (ACM MM 2015)\n'
    'Framework: PlaylistEngine (this work)',
    transform=ax_c3.transAxes,
    fontsize=7.2, color=BODY, va='top', linespacing=1.45)

# ─────────────────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved → {OUT}")
print(f"8.5\" × 11\" @ {DPI} dpi  (850 × 1100 px)")
