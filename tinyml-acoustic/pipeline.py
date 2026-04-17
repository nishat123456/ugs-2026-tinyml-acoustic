"""
TinyML Acoustic Monitoring — Systems Framework
USM Undergraduate Symposium 2026 / arXiv submission
M M Nishat

========================================================================
CONTRIBUTION STATEMENT:
We do NOT propose a new ML model. We propose a system-level evaluation 
framework for temporal context retention in event-triggered acoustic 
monitoring under resource constraints. 
========================================================================
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA_ESC50 = os.path.join(BASE, 'data', 'ESC-50-master')
DATA_US8K  = os.path.join(BASE, 'data', 'UrbanSound8K')
FIG    = os.path.join(BASE, 'figures')
RES    = os.path.join(BASE, 'results')
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

TARGET_CATEGORIES = {
    'chainsaw':       'event',    'hand_saw':       'event',
    'engine':         'event',    'crickets':       'non_event',
    'frog':           'non_event','wind':           'non_event',
    'rain':           'non_event','thunderstorm':   'non_event',
    'insects':        'non_event','water_drops':    'non_event',
}

SR         = 22050
WINDOW_S   = 1.0
HOP_S      = 0.5
WINDOW_LEN = int(SR * WINDOW_S)
HOP_LEN    = int(SR * HOP_S)

N_MFCC     = 40
N_MELS     = 40
HOP_LENGTH_MFCC = 512
THRESHOLD  = 0.35

N_RUNS = 5
STREAM_MINUTES = 15
DATASET_TO_USE = 'ESC-50' # Options: 'ESC-50', 'US8K'

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_esc50():
    print("\n[1/7] Loading ESC-50 dataset...")
    meta  = pd.read_csv(os.path.join(DATA_ESC50, 'meta', 'esc50.csv'))
    audio = os.path.join(DATA_ESC50, 'audio')
    records = []
    
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc='  Filtering ESC-50'):
        cat = row['category']
        if cat not in TARGET_CATEGORIES: continue
        fp = os.path.join(audio, row['filename'])
        if os.path.exists(fp):
            records.append({'filename': row['filename'], 'filepath': fp,
                            'category': cat, 'label': TARGET_CATEGORIES[cat], 'fold': row['fold']})
    df = pd.DataFrame(records)
    df_train = df[df['fold'].isin([1, 2, 3])]
    df_test  = df[df['fold'].isin([4, 5])]
    print(f"  ESC-50 Train: {len(df_train)} clips | Stream/Test: {len(df_test)} clips")
    return df_train, df_test

def load_us8k():
    print("\n[1/7] Loading UrbanSound8K dataset...")
    csv_path = os.path.join(DATA_US8K, 'metadata', 'UrbanSound8K.csv')
    audio_dir = os.path.join(DATA_US8K, 'audio')
    
    if not os.path.exists(csv_path):
        print("  [WARNING] UrbanSound8K dataset not found! Please download it to data/UrbanSound8K/")
        return pd.DataFrame(), pd.DataFrame()
        
    meta = pd.read_csv(csv_path)
    records = []
    
    # Event classes: siren (8), drilling (4), jackhammer (7), gun_shot (6)
    us8k_events = [4, 6, 7, 8]
    
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc='  Filtering US8K'):
        fold = row['fold']
        fp = os.path.join(audio_dir, f"fold{fold}", row['slice_file_name'])
        label = 'event' if row['classID'] in us8k_events else 'non_event'
        
        # We only take clips that actually exist on disk
        if os.path.exists(fp):
            records.append({
                'filename': row['slice_file_name'], 'filepath': fp,
                'category': row['class'], 'label': label, 'fold': fold
            })
            
    df = pd.DataFrame(records)
    # Split across 10 folds
    df_train = df[df['fold'].isin([1, 2, 3, 4, 5, 6])]
    df_test  = df[df['fold'].isin([7, 8, 9, 10])]
    print(f"  US8K Train: {len(df_train)} clips | Stream/Test: {len(df_test)} clips")
    return df_train, df_test

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2/3: FEATURES & MODEL
# ─────────────────────────────────────────────────────────────────────────────
def extract_features_array(y, sr):
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS, hop_length=HOP_LENGTH_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    rms        = librosa.feature.rms(y=y, hop_length=HOP_LENGTH_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta_mfcc.mean(axis=1), rms.mean(axis=1), rms.std(axis=1)])

def extract_train_features(df):
    print("\n[2/7] Extracting frame features for training...")
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='  Train Features'):
        audio_y, sr = librosa.load(row['filepath'], sr=SR, duration=5.0)
        label = 1 if row['label'] == 'event' else 0
        num_windows = (len(audio_y) - WINDOW_LEN) // HOP_LEN + 1
        if num_windows <= 0: continue
        for i in range(num_windows):
            start = i * HOP_LEN
            segment = audio_y[start:start+WINDOW_LEN]
            X.append(extract_features_array(segment, sr))
            y.append(label)
    return np.array(X), np.array(y)

def train_classifier(X, y):
    print(f"\n[3/7] Training Edge Classifier on {X.shape[0]} base frames...")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_s, y)
    y_pred = clf.predict(X_s)
    mod_metrics = {
        'accuracy':  accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall':    recall_score(y, y_pred, zero_division=0),
        'f1':        f1_score(y, y_pred, zero_division=0)
    }
    print(f"  Frame-level -> Acc: {mod_metrics['accuracy']:.3f} | F1: {mod_metrics['f1']:.3f}")
    return clf, scaler, mod_metrics

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_realistic_stream(df_test, seed, minutes=15, background_db=-20):
    np.random.seed(seed)
    samples = int(minutes * 60 * SR)
    stream_y = np.zeros(samples, dtype=np.float32)
    stream_labels = np.zeros(samples, dtype=int)
    events = df_test[df_test['label'] == 'event']
    noises = df_test[df_test['label'] == 'non_event']
    
    b_idx = 0
    while b_idx < samples:
        noise_row = noises.sample(1).iloc[0]
        y_n, _ = librosa.load(noise_row['filepath'], sr=SR, duration=5.0)
        end_idx = min(samples, b_idx + len(y_n))
        stream_y[b_idx:end_idx] += y_n[:end_idx - b_idx] * (10**(background_db/20))
        b_idx += len(y_n)
        
    curr_time = np.random.uniform(10, 30)
    while curr_time < (minutes * 60 - 5):
        event_row = events.sample(1).iloc[0]
        start_idx = int(curr_time * SR)
        y_e, _ = librosa.load(event_row['filepath'], sr=SR, duration=5.0)
        end_idx = min(samples, start_idx + len(y_e))
        stream_y[start_idx:end_idx] += y_e[:end_idx - start_idx]
        stream_labels[start_idx:end_idx] = 1
        curr_time += 5.0 + np.random.uniform(20, 100)
        
    if np.abs(stream_y).max() > 0: stream_y /= np.abs(stream_y).max()
    return stream_y, stream_labels

def run_streaming_inference(stream_y, stream_labels, clf, scaler, custom_thresh=THRESHOLD):
    num_windows = (len(stream_y) - WINDOW_LEN) // HOP_LEN + 1
    y_true_w = np.zeros(num_windows, dtype=int)
    X_stream = []
    for i in range(num_windows):
        start = i * HOP_LEN
        X_stream.append(extract_features_array(stream_y[start:start+WINDOW_LEN], SR))
        if stream_labels[start:start+WINDOW_LEN].mean() > 0.5:
            y_true_w[i] = 1
    X_s = scaler.transform(np.array(X_stream))
    y_prob = clf.predict_proba(X_s)[:, 1]
    y_pred = (y_prob >= custom_thresh).astype(int)
    return y_prob, y_pred, y_true_w, num_windows

def get_buffered_indices(y_pred, y_prob, num_windows, pre_w, post_w, adaptive=False):
    saved = set()
    for i in range(num_windows):
        if y_pred[i] == 1:
            curr_post = post_w
            if adaptive and y_prob[i] >= 0.75: curr_post = int(post_w * 1.5)
            for offset in range(-pre_w, curr_post + 1):
                if 0 <= i + offset < num_windows: saved.add(i + offset)
    return saved

def compute_sys_metrics(name, saved_set, y_true_w, num_windows):
    total_saved = len(saved_set)
    DRR = (num_windows - total_saved) / num_windows if num_windows > 0 else 0
    true_idx = np.where(y_true_w == 1)[0]
    total_true = len(true_idx)
    ECR = sum(1 for idx in true_idx if idx in saved_set) / total_true if total_true > 0 else 0.0
    return {'system': name, 'DRR': DRR, 'ECR': ECR, 'saved': total_saved}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4, 5, 6: EVALUATION PROTOCOL (MULTI-RUN)
# ─────────────────────────────────────────────────────────────────────────────
def run_stability_experiments(df_ts, clf, scaler):
    print(f"\n[4/7] Running {N_RUNS}x {STREAM_MINUTES}-min simulation streams (Multiple Seeds)...")
    
    # Track stats for aggregation
    stats = {}
    config_keys = []
    
    # Store matrix for Heatmap (Average over runs)
    pre_ops = [0, 1, 2, 3, 4]  # 0, 0.5, 1.0, 1.5, 2.0s
    post_ops = [0, 2, 4, 6, 8] # 0, 1.0, 2.0, 3.0, 4.0s
    hm_ecr = np.zeros((len(pre_ops), len(post_ops)))
    hm_drr = np.zeros((len(pre_ops), len(post_ops)))
    
    # Sensitivity analysis
    thresh_ops = [0.2, 0.3, 0.4, 0.5, 0.6]
    sens_drr = {t: [] for t in thresh_ops}
    sens_ecr = {t: [] for t in thresh_ops}
    
    for seed in range(42, 42 + N_RUNS):
        st_y, st_labels = generate_realistic_stream(df_ts, seed, minutes=STREAM_MINUTES)
        y_prob, y_pred, y_true_w, nw = run_streaming_inference(st_y, st_labels, clf, scaler)
        
        configs = []
        # Store-All (Upper bound ECR, Zero DRR)
        configs.append(compute_sys_metrics('Store-All', set(range(nw)), y_true_w, nw))
        
        # Detect-Only (Lower bound storage for ML)
        b0_set = get_buffered_indices(y_pred, y_prob, nw, 0, 0)
        sysB = compute_sys_metrics('Detect-Only', b0_set, y_true_w, nw)
        configs.append(sysB)
        
        # Periodic (Fairness / Budget matched to Detect-Only)
        k_saved = sysB['saved']
        step = max(1, nw // k_saved) if k_saved > 0 else nw
        configs.append(compute_sys_metrics('Fixed_Periodic', set(range(0, nw, step)[:k_saved]), y_true_w, nw))

        # Pure Random Baseline (Matches Detect-Only budget but arbitrary indices)
        if k_saved > 0:
            rand_set = set(np.random.choice(nw, size=min(k_saved, nw), replace=False))
        else:
            rand_set = set()
        configs.append(compute_sys_metrics('Pure_Random', rand_set, y_true_w, nw))
        
        # Sweeps (Pareto Frontier Contexts)
        for (pw, pow) in [(1,1), (2,2), (2,4), (4,6)]:
            lbl = f'Buffer_{pw*HOP_S}s_{pow*HOP_S}s'
            idx_set = get_buffered_indices(y_pred, y_prob, nw, pw, pow)
            configs.append(compute_sys_metrics(lbl, idx_set, y_true_w, nw))
            
        # Ablation: Proposed Fixed VS Proposed Adaptive (both base 1s pre / 2s post)
        f_set = get_buffered_indices(y_pred, y_prob, nw, 2, 4)
        configs.append(compute_sys_metrics('Proposed_Fixed(1s/2s)', f_set, y_true_w, nw))
        
        a_set = get_buffered_indices(y_pred, y_prob, nw, 2, 4, adaptive=True)
        configs.append(compute_sys_metrics('Proposed_Adaptive', a_set, y_true_w, nw))

        for c in configs:
            k = c['system']
            if k not in stats: stats[k] = {'DRR': [], 'ECR': []}
            stats[k]['DRR'].append(c['DRR'])
            stats[k]['ECR'].append(c['ECR'])
            if k not in config_keys: config_keys.append(k)
            
        # Accumulate Heatmap
        for r_idx, pre_w in enumerate(pre_ops):
            for c_idx, post_w in enumerate(post_ops):
                s_set = get_buffered_indices(y_pred, y_prob, nw, pre_w, post_w)
                res = compute_sys_metrics('tmp', s_set, y_true_w, nw)
                hm_ecr[r_idx, c_idx] += res['ECR'] / N_RUNS
                hm_drr[r_idx, c_idx] += res['DRR'] / N_RUNS
                
        # Sensitivity (Thresholds vs Fixed Buffer)
        for thresh in thresh_ops:
            _, yp_sens, _, _ = run_streaming_inference(st_y, st_labels, clf, scaler, custom_thresh=thresh)
            ss = get_buffered_indices(yp_sens, y_prob, nw, 2, 4, adaptive=False)
            res = compute_sys_metrics('tmp', ss, y_true_w, nw)
            sens_drr[thresh].append(res['DRR'])
            sens_ecr[thresh].append(res['ECR'])

    # Aggregate Main Stats
    agg_stats = {}
    for k in config_keys:
        agg_stats[k] = {
            'DRR': np.mean(stats[k]['DRR']) * 100, 'DRR_std': np.std(stats[k]['DRR']) * 100,
            'ECR': np.mean(stats[k]['ECR']) * 100, 'ECR_std': np.std(stats[k]['ECR']) * 100
        }
    
    # Aggregate Sensitivity
    agg_sens = {
        'thresh': thresh_ops,
        'DRR': [np.mean(sens_drr[t]) * 100 for t in thresh_ops],
        'ECR': [np.mean(sens_ecr[t]) * 100 for t in thresh_ops]
    }
    
    return agg_stats, config_keys, hm_ecr, hm_drr, agg_sens, pre_ops, post_ops

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: PRINT & VISUALIZE
# ─────────────────────────────────────────────────────────────────────────────
def print_and_visualize(stats, config_keys, hm_ecr, agg_sens, pre_ops, post_ops):
    print("\n[5/7] Baseline & Contribution Explanations:")
    print("  * Store-All: represents the absolute upper bound of ECR but 0 data reduction.")
    print("  * Detect-Only: represents the rigid lower bound of the ML framework context.")
    print("  * Fixed Periodic: acts as a strict fairness baseline; evaluates if intelligence ")
    print("    outperforms blind sampling matched to the identical memory budget.")
    print("  * Proposed Fixed vs Adaptive: Ablation measuring context-awareness gains.")

    print("\n[6/7] Aggregated System Results (Mean ± Std over 5 runs)")
    print("  " + "-"*80)
    for k in config_keys:
        drr, d_std = stats[k]['DRR'], stats[k]['DRR_std']
        ecr, e_std = stats[k]['ECR'], stats[k]['ECR_std']
        print(f"  {k:<25} | DRR: {drr:>6.2f}% ±{d_std:>4.2f} | ECR: {ecr:>6.2f}% ±{e_std:>4.2f}")
    
    print("\n[7/7] Generating final visualizations...")
    sns.set_theme(style='whitegrid', font_scale=1.0)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # 1. Pareto Frontier
    fig, ax = plt.subplots(figsize=(8, 6))
    drrs = [stats[k]['DRR'] for k in config_keys if 'Buffer' in k or 'Detect' in k]
    ecrs = [stats[k]['ECR'] for k in config_keys if 'Buffer' in k or 'Detect' in k]
    lbls = [k for k in config_keys if 'Buffer' in k or 'Detect' in k]
    
    sort_idx = np.argsort(drrs)[::-1]
    ax.plot(np.array(drrs)[sort_idx], np.array(ecrs)[sort_idx], color='#27AE60', linestyle='--', zorder=3)
    
    for k in config_keys:
        col = '#95A5A6' if k == 'Store-All' else '#E74C3C' if k == 'Fixed_Periodic' else '#3498DB' if 'Detect' in k else '#9B59B6' if 'Adaptive' in k else '#27AE60'
        ax.scatter(stats[k]['DRR'], stats[k]['ECR'], s=200, color=col, edgecolors='white', zorder=5)
        # Error bars
        ax.errorbar(stats[k]['DRR'], stats[k]['ECR'], xerr=stats[k]['DRR_std'], yerr=stats[k]['ECR_std'], fmt='none', color='black', alpha=0.3, zorder=4)
        sh = k.replace('Buffer_', '').replace('Proposed_', '').replace('_', ' ')
        ax.annotate(sh, (stats[k]['DRR'], stats[k]['ECR']), xytext=(0, 10), textcoords='offset points', fontsize=9, ha='center', color=col, fontweight='bold')

    ax.set_xlabel('Data Reduction Ratio — DRR (%)', fontweight='bold')
    ax.set_ylabel('Event Capture Rate — ECR (%)', fontweight='bold')
    ax.set_title('Pareto Frontier (Mean ± Std over 5 seeds)', fontweight='bold')
    ax.set_xlim(-5, 105); ax.set_ylim(0, 105)
    sns.despine()
    plt.savefig(os.path.join(FIG, 'figA_pareto_stability.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Buffer Sweep Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(hm_ecr * 100, annot=True, fmt='.1f', cmap='viridis', 
                xticklabels=[f"{x*HOP_S:.1f}s" for x in post_ops], 
                yticklabels=[f"{y*HOP_S:.1f}s" for y in pre_ops], ax=ax)
    ax.set_xlabel('Post-Event Buffer', fontweight='bold')
    ax.set_ylabel('Pre-Event Buffer', fontweight='bold')
    ax.set_title('ECR Heatmap: Buffer Configuration Sweep (%)', fontweight='bold')
    plt.savefig(os.path.join(FIG, 'figB_buffer_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Sensitivity Analysis
    fig, ax1 = plt.subplots(figsize=(6, 4))
    color = '#E74C3C'
    ax1.set_xlabel('Classification Threshold', fontweight='bold')
    ax1.set_ylabel('ECR (%)', color=color, fontweight='bold')
    ax1.plot(agg_sens['thresh'], agg_sens['ECR'], marker='o', color=color, linewidth=2, label='ECR')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color2 = '#3498DB'
    ax2.set_ylabel('DRR (%)', color=color2, fontweight='bold')
    ax2.plot(agg_sens['thresh'], agg_sens['DRR'], marker='s', color=color2, linewidth=2, label='DRR')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.suptitle('Sensitivity Analysis: Impact of Base Threshold', fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(FIG, 'figC_sensitivity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("="*75)
    print("  System-Level Evaluation Framework for Temporal Context Retention")
    print("  Note: NO new ML algorithm proposed. Investigating systems tradeoffs.")
    print("="*75)
    
    if DATASET_TO_USE == 'ESC-50':
        df_tr, df_ts = load_esc50()
    else:
        df_tr, df_ts = load_us8k()
        
    if df_tr.empty:
        print("Exiting pipeline due to missing data.")
        exit(1)
    X_tr, y_tr = extract_train_features(df_tr)
    clf, scaler, mod_metrics = train_classifier(X_tr, y_tr)
    
    agg_stats, config_keys, hm_ecr, hm_drr, agg_sens, pre_op, post_op = run_stability_experiments(df_ts, clf, scaler)
    print_and_visualize(agg_stats, config_keys, hm_ecr, agg_sens, pre_op, post_op)
    
    with open(os.path.join(RES, 'stability_metrics.json'), 'w') as f:
        json.dump({'agg_stats': agg_stats, 'sensitivity': agg_sens}, f, indent=2)
    print("\nOutputs saved to figures/ and results/.")
