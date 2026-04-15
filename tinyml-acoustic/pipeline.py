"""
TinyML Acoustic Monitoring — Simulation Framework
USM Undergraduate Symposium 2026 / arXiv submission
M M Nishat

Contribution:
  We propose a system-level evaluation framework for event-triggered acoustic
  sensing that quantifies the trade-off between data storage efficiency (DRR)
  and event preservation (ECR) under simulated streaming constraints.

  Key finding: A 1-clip circular buffer recovers near-full event capture
  with only half the storage of continuous recording — and vastly outperforms
  a random-save baseline at the same storage budget.

NOTE ON TEMPORAL ASSUMPTION:
  ESC-50 consists of non-continuous, independently recorded clips. To approximate
  edge-device behavior, we treat clips as a sequential stream and simulate
  circular buffer context using adjacent sample indices. This is an explicit
  simulation assumption — results represent system-level trade-offs under
  idealized conditions, not a deployment evaluation.

Pipeline:
  Dataset (ESC-50) → MFCC Feature Extraction → TinyML Classifier
  → Event Detection → Multi-System Comparison:
      [A] Store-All Baseline
      [B] Detect-Only (no buffer)
      [C] Random Baseline (same storage as B, random selection)
      [D] Circular Buffer, B=1 (proposed)
      [E] Circular Buffer, B=2 (extended)
  → Buffer Ablation Study: B = 0, 1, 2 clips
  → Formal Metrics: ECR, DRR, FPR, PPV
"""

import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, 'data', 'ESC-50-master')
FIG    = os.path.join(BASE, 'figures')
RES    = os.path.join(BASE, 'results')
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

TARGET_CATEGORIES = {
    'chainsaw':       'event',
    'hand_saw':       'event',
    'engine':         'event',
    'crickets':       'non_event',
    'frog':           'non_event',
    'wind':           'non_event',
    'rain':           'non_event',
    'thunderstorm':   'non_event',
    'insects':        'non_event',
    'water_drops':    'non_event',
}

SR         = 22050
DURATION   = 5
N_MFCC     = 40
N_MELS     = 40
HOP_LENGTH = 512
THRESHOLD  = 0.35   # decision boundary — tuned for high recall

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset():
    print("\n[1/5] Loading ESC-50 dataset...")
    meta  = pd.read_csv(os.path.join(DATA, 'meta', 'esc50.csv'))
    audio = os.path.join(DATA, 'audio')
    records = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc='  Filtering'):
        cat = row['category']
        if cat not in TARGET_CATEGORIES:
            continue
        fp = os.path.join(audio, row['filename'])
        if os.path.exists(fp):
            records.append({'filename': row['filename'], 'filepath': fp,
                            'category': cat, 'label': TARGET_CATEGORIES[cat],
                            'fold': row['fold']})
    df = pd.DataFrame(records)
    print(f"  {len(df)} clips  |  event={( df.label=='event').sum()}  non_event={(df.label=='non_event').sum()}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: MFCC + DELTA FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(filepath):
    """
    122-dim feature vector: MFCC mean(40) + MFCC std(40) + delta-MFCC mean(40) + RMS mean(1) + RMS std(1).
    Compact enough to run on ARM Cortex-M4 class microcontroller.
    """
    y, sr = librosa.load(filepath, sr=SR, duration=DURATION)
    if len(y) < SR * DURATION:
        y = np.pad(y, (0, SR * DURATION - len(y)))
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc)
    rms        = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
                           delta_mfcc.mean(axis=1), rms.mean(axis=1), rms.std(axis=1)])

def extract_all(df):
    print("\n[2/5] Extracting features...")
    feats, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='  MFCC+delta'):
        feats.append(extract_features(row['filepath']))
        labels.append(1 if row['label'] == 'event' else 0)
    X, y = np.array(feats), np.array(labels)
    print(f"  Feature matrix: {X.shape}")
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X, y):
    print("\n[3/5] Training classifier...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_tr_s, y_tr)

    y_prob = clf.predict_proba(X_te_s)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    metrics = {
        'accuracy':  round(accuracy_score(y_te, y_pred), 4),
        'precision': round(precision_score(y_te, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_te, y_pred, zero_division=0), 4),
        'f1':        round(f1_score(y_te, y_pred, zero_division=0), 4),
    }
    cm = confusion_matrix(y_te, y_pred)
    print(f"  Accuracy={metrics['accuracy']}  Precision={metrics['precision']}  "
          f"Recall={metrics['recall']}  F1={metrics['f1']}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['non_event','event'])}")
    return clf, scaler, cm, metrics

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: THREE-SYSTEM SIMULATION + FORMAL METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_system_metrics(name, saved_indices, y_true, total_clips, y_pred_all):
    """
    Formal metric definitions (consistent with paper notation):

    DRR  = Data Reduction Ratio      = (N_total - N_saved) / N_total
    ECR  = Event Capture Rate        = TP / (TP + FN)   (fraction of true events saved)
    FPR  = False Positive Rate       = FP / (FP + TN)   (fraction of non-events saved unnecessarily)
    PPV  = Positive Predictive Value = TP / (TP + FP)   (= Precision at system level)
    """
    total_seconds   = total_clips * DURATION
    saved_seconds   = len(saved_indices) * DURATION

    # Which clips in saved_indices are true events?
    tp = sum(1 for i in saved_indices if y_true[i] == 1)
    fp = sum(1 for i in saved_indices if y_true[i] == 0)
    fn = sum(1 for i in range(total_clips) if y_true[i] == 1 and i not in saved_indices)
    tn = sum(1 for i in range(total_clips) if y_true[i] == 0 and i not in saved_indices)

    true_events = int(y_true.sum())
    DRR = (total_seconds - saved_seconds) / total_seconds
    ECR = tp / true_events if true_events > 0 else 0
    FPR = fp / (fp + tn) if (fp + tn) > 0 else 0
    PPV = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        'system':         name,
        'saved_clips':    len(saved_indices),
        'saved_seconds':  saved_seconds,
        'total_seconds':  total_seconds,
        'DRR':            round(DRR, 4),
        'ECR':            round(ECR, 4),
        'FPR':            round(FPR, 4),
        'PPV':            round(PPV, 4),
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
    }


def buffer_indices(y_pred, total_clips, B):
    """Compute saved clip indices for circular buffer with pre/post context = B clips."""
    saved = set()
    for i, p in enumerate(y_pred):
        if p == 1:
            for offset in range(-B, B + 1):
                idx = i + offset
                if 0 <= idx < total_clips:
                    saved.add(idx)
    return saved


def run_simulation(df, clf, scaler, X, y_true):
    print("\n[4/5] Running multi-system simulation + buffer ablation...")
    total_clips = len(df)
    X_s         = scaler.transform(X)
    y_prob      = clf.predict_proba(X_s)[:, 1]
    y_pred      = (y_prob >= THRESHOLD).astype(int)
    rng         = np.random.default_rng(42)

    # ── Baseline A: Store Everything ─────────────────────────────────────────
    A_indices = set(range(total_clips))
    sysA = compute_system_metrics('Baseline A: Store-All', A_indices, y_true, total_clips, y_pred)

    # ── Baseline B: Detect-Only (B=0, no context buffer) ─────────────────────
    B_indices = buffer_indices(y_pred, total_clips, B=0)
    sysB = compute_system_metrics('Baseline B: Detect-Only (B=0)', B_indices, y_true, total_clips, y_pred)

    # ── Random Baseline: same storage budget as Detect-Only ──────────────────
    # Randomly saves |B_indices| clips. If the system only matches random
    # performance, it provides no intelligence value.
    n_random = len(B_indices)
    rand_indices = set(rng.choice(total_clips, size=n_random, replace=False).tolist())
    sysRand = compute_system_metrics('Random Baseline (same budget)', rand_indices, y_true, total_clips, y_pred)

    # ── Buffer Ablation: B=1 and B=2 ─────────────────────────────────────────
    C1_indices = buffer_indices(y_pred, total_clips, B=1)
    sysC1 = compute_system_metrics('Circular Buffer B=1 (Proposed)', C1_indices, y_true, total_clips, y_pred)

    C2_indices = buffer_indices(y_pred, total_clips, B=2)
    sysC2 = compute_system_metrics('Circular Buffer B=2', C2_indices, y_true, total_clips, y_pred)

    results = [sysA, sysB, sysRand, sysC1, sysC2]

    print(f"\n  {'System':<38} {'DRR':>6} {'ECR':>6} {'FPR':>6} {'PPV':>6} {'Saved':>7}")
    print("  " + "-"*74)
    for r in results:
        print(f"  {r['system']:<38} {r['DRR']:>6.2%} {r['ECR']:>6.2%} {r['FPR']:>6.2%} "
              f"{r['PPV']:>6.2%} {r['saved_clips']:>5}clips")

    return results, y_pred

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def generate_figures(cm, model_metrics, sim_results, df):
    print("\n[5/5] Generating figures...")
    sns.set_theme(style='whitegrid', font_scale=1.05)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    sysA, sysB, sysRand, sysC1, sysC2 = sim_results

    # ── Fig 1: Confusion Matrix ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
                xticklabels=['Non-Event','Event'], yticklabels=['Non-Event','Event'], ax=ax)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title('Confusion Matrix — RF Classifier', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig1_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 2: Classifier Performance Metrics ────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ['Accuracy','Precision','Recall','F1']
    vals  = [model_metrics[k.lower()] for k in names]
    cols  = ['#2ECC71','#3498DB','#E74C3C','#9B59B6']
    bars  = ax.bar(names, vals, color=cols, edgecolor='white', linewidth=1.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.3f}',
                ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_title('Classifier Performance Metrics', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig2_model_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 3: DRR Comparison (primary systems) ───────────────────────────────
    plot_systems = [sysA, sysB, sysC1]
    plot_labels  = ['Store-All\n(Baseline A)', 'Detect-Only\n(Baseline B)', 'Circular Buffer B=1\n(Proposed)']
    plot_colors  = ['#95A5A6', '#3498DB', '#27AE60']
    fig, ax = plt.subplots(figsize=(7, 4))
    drr_vals = [r['DRR']*100 for r in plot_systems]
    bars = ax.bar(plot_labels, drr_vals, color=plot_colors, edgecolor='white', linewidth=1.5)
    for b, v in zip(bars, drr_vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_ylabel('Data Reduction Ratio — DRR (%)', fontweight='bold')
    ax.set_title('DRR: Fraction of Audio Discarded', fontweight='bold')
    ax.set_ylim(0, 115)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig3_drr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 4: ECR Comparison (primary systems) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ecr_vals = [r['ECR']*100 for r in plot_systems]
    bars = ax.bar(plot_labels, ecr_vals, color=plot_colors, edgecolor='white', linewidth=1.5)
    for b, v in zip(bars, ecr_vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f'{v:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    ax.set_ylabel('Event Capture Rate — ECR (%)', fontweight='bold')
    ax.set_title('ECR: Fraction of True Events Captured', fontweight='bold')
    ax.set_ylim(0, 115)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig4_ecr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 5: HERO FIGURE — DRR vs ECR Trade-off with Ablation Curve ────────
    # This is the key "insight" figure for the poster and presentation.
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Ideal region shading
    ax.axhspan(85, 103, xmin=0.35, xmax=1.0, alpha=0.08, color='#27AE60')
    ax.text(75, 95, 'Ideal region\n(high ECR, high DRR)', fontsize=8.5,
            color='#27AE60', alpha=0.8, ha='center')

    # Buffer ablation curve: B=0, B=1, B=2 — connected line shows the trade-off frontier
    ablation = [sysB, sysC1, sysC2]
    abl_drr  = [r['DRR']*100 for r in ablation]
    abl_ecr  = [r['ECR']*100 for r in ablation]
    ax.plot(abl_drr, abl_ecr, color='#27AE60', linewidth=2, linestyle='--',
            zorder=3, label='Buffer ablation curve')
    for i, (r, b_val) in enumerate(zip(ablation, [0, 1, 2])):
        ax.scatter(r['DRR']*100, r['ECR']*100, s=200, color='#27AE60',
                   zorder=5, edgecolors='white', linewidths=1.5)
        offset_x = 6 if b_val < 2 else -48
        ax.annotate(f'Buffer B={b_val}', (r['DRR']*100, r['ECR']*100),
                    textcoords='offset points', xytext=(offset_x, 5),
                    fontsize=9.5, color='#1E8449', fontweight='bold')

    # Store-All baseline (anchor point)
    ax.scatter(sysA['DRR']*100, sysA['ECR']*100, s=200, color='#95A5A6',
               zorder=5, edgecolors='white', linewidths=1.5, label='Store-All (B-A)')
    ax.annotate('Store-All\n(Baseline A)', (sysA['DRR']*100, sysA['ECR']*100),
                textcoords='offset points', xytext=(6, -18), fontsize=9)

    # Random baseline (critical credibility check)
    ax.scatter(sysRand['DRR']*100, sysRand['ECR']*100, s=200, color='#E74C3C',
               zorder=5, edgecolors='white', linewidths=1.5, marker='X',
               label='Random baseline (same storage budget as B=0)')
    ax.annotate(f'Random\n(ECR={sysRand["ECR"]:.0%})', (sysRand['DRR']*100, sysRand['ECR']*100),
                textcoords='offset points', xytext=(6, -22), fontsize=9, color='#C0392B')

    # Key insight annotation box
    b0_ecr  = sysB['ECR'] * 100
    b1_ecr  = sysC1['ECR'] * 100
    b1_drr  = sysC1['DRR'] * 100
    rand_ecr = sysRand['ECR'] * 100
    insight = (f'Detector vs Random (same budget):\n'
               f'+{b0_ecr - rand_ecr:.0f}pp ECR  [{b0_ecr:.0f}% vs {rand_ecr:.0f}%]\n'
               f'Buffer B=1: +{b1_ecr - b0_ecr:.1f}pp ECR, {b1_drr:.0f}% storage kept')
    ax.annotate(insight,
                xy=(sysC1['DRR']*100, sysC1['ECR']*100),
                xytext=(sysC1['DRR']*100 - 35, sysC1['ECR']*100 - 28),
                fontsize=9, color='#145A32',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#EAFAF1', edgecolor='#27AE60', linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))

    ax.set_xlabel('Data Reduction Ratio — DRR (%)  [higher = less storage]', fontweight='bold', fontsize=11)
    ax.set_ylabel('Event Capture Rate — ECR (%)  [higher = fewer missed events]', fontweight='bold', fontsize=11)
    ax.set_title('Storage Efficiency vs. Event Preservation Trade-off\nBuffer Ablation + Random Baseline Comparison',
                 fontweight='bold', fontsize=12)
    ax.set_xlim(-5, 108); ax.set_ylim(15, 108)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig5_hero_tradeoff.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 6: Full Comparison Table (all 5 systems) ─────────────────────────
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis('off')
    col_labels = ['System', 'DRR', 'ECR', 'FPR', 'PPV', 'Saved Clips']
    # Short names for the table
    short_names = ['Store-All (B-A)', 'Detect-Only B=0 (B-B)', 'Random Baseline',
                   'Circ. Buffer B=1 (Proposed)', 'Circ. Buffer B=2']
    table_data = [
        [short_names[i], f"{r['DRR']:.1%}", f"{r['ECR']:.1%}",
         f"{r['FPR']:.1%}", f"{r['PPV']:.1%}", str(r['saved_clips'])]
        for i, r in enumerate(sim_results)
    ]
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.1, 2.0)
    # Header row
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    # Highlight proposed (row 4)
    for j in range(len(col_labels)):
        tbl[4, j].set_facecolor('#D5F5E3')
    # Highlight random baseline (row 3) in light red
    for j in range(len(col_labels)):
        tbl[3, j].set_facecolor('#FDEDEC')
    ax.set_title('Full System Comparison — All Formal Metrics', fontweight='bold', pad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig6_comparison_table.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fig 7: System Architecture Diagram ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis('off')
    boxes = [
        (0.02, 0.55, 0.13, 0.25, 'Audio Dataset\n(ESC-50)',         '#3498DB', 'white'),
        (0.20, 0.55, 0.13, 0.25, 'Feature\nExtraction\n(MFCC+Δ)',   '#8E44AD', 'white'),
        (0.38, 0.55, 0.13, 0.25, 'TinyML\nClassifier\n(RF, n=200)', '#E67E22', 'white'),
        (0.56, 0.55, 0.13, 0.25, 'Event\nDetected?',                '#2C3E50', 'white'),
        (0.74, 0.72, 0.13, 0.20, 'Circular\nBuffer\n(Save Packet)', '#27AE60', 'white'),
        (0.74, 0.35, 0.13, 0.20, 'Discard\n(Low-Power\nSleep)',     '#95A5A6', 'white'),
        (0.74, 0.05, 0.22, 0.20, 'Event Packet\nPre(5s)+Event(5s)+Post(5s)', '#C0392B', 'white'),
    ]
    for (x, y, w, h, txt, fc, tc) in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.02',
            facecolor=fc, edgecolor='white', linewidth=2, transform=ax.transAxes, clip_on=False))
        ax.text(x+w/2, y+h/2, txt, transform=ax.transAxes, ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=tc, multialignment='center')
    ap = dict(arrowstyle='->', color='#2C3E50', lw=2)
    for (x1,y1,x2,y2) in [(0.15,0.675,0.20,0.675),(0.33,0.675,0.38,0.675),
                           (0.51,0.675,0.56,0.675),(0.69,0.78,0.74,0.80),
                           (0.69,0.57,0.74,0.42),(0.805,0.72,0.805,0.25)]:
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), xycoords='axes fraction',
                    textcoords='axes fraction', arrowprops=ap)
    ax.text(0.70, 0.83, 'YES', transform=ax.transAxes, fontsize=9, fontweight='bold', color='#27AE60')
    ax.text(0.70, 0.55, 'NO',  transform=ax.transAxes, fontsize=9, fontweight='bold', color='#95A5A6')
    ax.set_title('System Architecture — Event-Triggered Acoustic Monitoring',
                 fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig7_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  7 figures saved to: {FIG}/")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: SAVE + PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def save_results(model_metrics, sim_results):
    out = {'model': model_metrics, 'systems': sim_results}
    with open(os.path.join(RES, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)

    sysA, sysB, sysRand, sysC1, sysC2 = sim_results
    print("\n" + "="*75)
    print("  RESULTS SUMMARY  (for paper / poster)")
    print("="*75)
    print(f"  Classifier:   Acc={model_metrics['accuracy']:.1%}  "
          f"P={model_metrics['precision']:.1%}  "
          f"R={model_metrics['recall']:.1%}  "
          f"F1={model_metrics['f1']:.1%}")
    print()
    print(f"  {'System':<38}  {'DRR':>7}  {'ECR':>7}  {'FPR':>7}  {'PPV':>7}")
    print("  " + "-"*70)
    for r in sim_results:
        print(f"  {r['system']:<38}  {r['DRR']:>7.1%}  {r['ECR']:>7.1%}  {r['FPR']:>7.1%}  {r['PPV']:>7.1%}")
    print()
    print(f"  KEY FINDING:")
    print(f"    Intelligence gap (B=0 vs Random, same {sysB['saved_clips']}-clip budget):  "
          f"ECR {sysB['ECR']:.1%} vs {sysRand['ECR']:.1%}  "
          f"(+{(sysB['ECR']-sysRand['ECR'])*100:.1f}pp)")
    print(f"    Circular Buffer B=1 adds:            "
          f"+{(sysC1['ECR']-sysB['ECR'])*100:.1f}pp ECR over Detect-Only, "
          f"at cost of {sysC1['saved_clips']-sysB['saved_clips']} additional clips saved")
    print(f"    Storage reduction (B=1):             {sysC1['DRR']:.1%}  ({sysC1['saved_clips']} clips saved vs {sysA['saved_clips']})")
    print("="*75)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*65)
    print("  Event-Triggered Acoustic Monitoring — Simulation Framework")
    print("  M M Nishat  |  USM 2026")
    print("="*65)
    df              = load_dataset()
    X, y            = extract_all(df)
    clf, scaler, cm, model_m = train_model(X, y)
    sim_results, _  = run_simulation(df, clf, scaler, X, y)
    generate_figures(cm, model_m, sim_results, df)
    save_results(model_m, sim_results)
    print("\nDone. figures/ and results/metrics.json updated.")
