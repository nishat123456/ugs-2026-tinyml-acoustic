"""
Evaluating Event-Driven TinyML Acoustic Monitoring:
A Systems-Level Framework for Quantifying Context Retention Trade-offs
USM Undergraduate Symposium 2026
M M Nishat
"""

import os, json, warnings, joblib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import bootstrap, mannwhitneyu
from tqdm import tqdm

from playlist_engine import PlaylistEngine
from physics_model import PhysicsModel

# ─────────────────────────────────────────────────────────────────────────────
# COLLISION & SYSTEM CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA_ESC50 = os.path.join(BASE, 'data', 'ESC-50-master')
DATA_US8K  = os.path.join(BASE, 'data', 'UrbanSound8K')
DATA_BIRDCLEF = os.path.join(BASE, 'data', 'BirdCLEF')
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
THRESHOLD  = 2.5 # Sigma multiplier for Energy Detector

N_RUNS = 10     
BOOTSTRAP_ITERS = 1000
STREAM_MINUTES = 15
DATASET_TO_USE = 'ESC-50' 
CROSS_DATASET   = False    
COMPARE_MODELS  = True     

# Simulation Constraints
SNR_DB_LIST = [10] 
OVERLAP_PROB = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# DATA & FEATURES
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
    return df_train, df_test

def load_us8k():
    print("\n[1/7] Loading UrbanSound8K dataset...")
    csv_path = os.path.join(DATA_US8K, 'metadata', 'UrbanSound8K.csv')
    if not os.path.exists(csv_path): return pd.DataFrame(), pd.DataFrame()
    meta = pd.read_csv(csv_path)
    records = []
    us8k_events = [4, 6, 7, 8]
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc='  Filtering US8K'):
        fold = row['fold']
        fp = os.path.join(DATA_US8K, 'audio', f"fold{fold}", row['slice_file_name'])
        label = 'event' if row['classID'] in us8k_events else 'non_event'
        if os.path.exists(fp):
            records.append({'filename': row['slice_file_name'], 'filepath': fp, 'category': row['class'], 'label': label, 'fold': fold})
    df = pd.DataFrame(records)
    df_train = df[df['fold'].isin([1, 2, 3, 4, 5, 6])]
    df_test  = df[df['fold'].isin([7, 8, 9, 10])]
    return df_train, df_test

def load_birdclef():
    print("\n[1/7] Loading BirdCLEF dataset abstraction...")
    audio_dir = os.path.join(DATA_BIRDCLEF, 'train_audio')
    if not os.path.exists(audio_dir): return pd.DataFrame(), pd.DataFrame()
    target_birds = ['aeacan', 'amecro', 'comrav']
    records = []
    for species in tqdm(os.listdir(audio_dir), desc='  Scanning BirdCLEF'):
        s_path = os.path.join(audio_dir, species)
        if not os.path.isdir(s_path): continue
        label = 'event' if species in target_birds else 'non_event'
        files = [f for f in os.listdir(s_path) if f.endswith('.ogg') or f.endswith('.wav')][:50]
        for f in files:
            records.append({'filename': f, 'filepath': os.path.join(s_path, f), 'category': species, 'label': label})
    df = pd.DataFrame(records)
    if df.empty: return df, df
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.6)
    return df.iloc[:split_idx], df.iloc[split_idx:]

def extract_features_array(y, sr):
    mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS, hop_length=HOP_LENGTH_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    rms        = librosa.feature.rms(y=y, hop_length=HOP_LENGTH_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta_mfcc.mean(axis=1), rms.mean(axis=1), rms.std(axis=1)])

def extract_train_features(df):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='  Train Features'):
        audio_y, sr = librosa.load(row['filepath'], sr=SR, duration=5.0)
        label = 1 if row['label'] == 'event' else 0
        num_windows = (len(audio_y) - WINDOW_LEN) // HOP_LEN + 1
        if num_windows <= 0: continue
        for i in range(num_windows):
            X.append(extract_features_array(audio_y[i*HOP_LEN:i*HOP_LEN+WINDOW_LEN], sr))
            y.append(label)
    return np.array(X), np.array(y)

def train_classifier(X, y):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(X_s, y)
    
    # [DEEP RIGOR]: Empirical Model Measurement
    model_path = os.path.join(RES, 'rf_model.joblib')
    joblib.dump(clf, model_path)
    emp_size_kb = os.path.getsize(model_path) / 1024.0
    print(f"  Empirical Flash Size (Joblib): {emp_size_kb:.1f} KB")
    
    return clf, scaler, emp_size_kb

def train_linear_model(X, y):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(X_s, y)
    
    model_path = os.path.join(RES, 'lr_model.joblib')
    joblib.dump(clf, model_path)
    emp_size_kb = os.path.getsize(model_path) / 1024.0
    return clf, scaler, emp_size_kb

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def run_streaming_inference(stream_y, stream_labels, clf, scaler, custom_thresh=0.35):
    num_windows = (len(stream_y) - WINDOW_LEN) // HOP_LEN + 1
    y_true_w = np.zeros(num_windows, dtype=int)
    X_stream = []
    for i in range(num_windows):
        start = i * HOP_LEN
        X_stream.append(extract_features_array(stream_y[start:start+WINDOW_LEN], SR))
        if stream_labels[start:start+WINDOW_LEN].mean() > 0.5: y_true_w[i] = 1
    X_s = scaler.transform(np.array(X_stream))
    y_prob = clf.predict_proba(X_s)[:, 1]
    y_pred = (y_prob >= custom_thresh).astype(int)
    return y_prob, y_pred, y_true_w, num_windows

def get_energy_triggers(stream_y, sr, num_windows, sigma_mult=3.0):
    """
    [DEEP RIGOR]: Literature Baseline - Energy-Based Detector (RMS Threshold)
    """
    rms = librosa.feature.rms(y=stream_y, frame_length=WINDOW_LEN, hop_length=HOP_LEN)[0]
    # Calibrate to first 2 seconds of background noise
    bg_rms = rms[:int(2.0/HOP_S)]
    thresh = np.mean(bg_rms) + sigma_mult * np.std(bg_rms)
    triggers = (rms[:num_windows] > thresh).astype(int)
    return triggers

def get_buffered_indices(y_pred, y_prob, num_windows, pre_w, post_w, adaptive=False):
    saved = set()
    for i in range(num_windows):
        if y_pred[i] == 1:
            curr_post = post_w
            if adaptive and y_prob is not None and y_prob[i] >= 0.75: 
                curr_post = int(post_w * 1.5)
            for offset in range(-pre_w, curr_post + 1):
                if 0 <= i + offset < num_windows: saved.add(i + offset)
    return saved

def calculate_cohens_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0: return 0
    return (np.mean(x1) - np.mean(x2)) / pooled_sd

def compute_sys_metrics(name, saved_set, y_pred, y_true_w, num_windows, onsets=None, model_type='RandomForest', emp_size=0):
    total_saved = len(saved_set)
    DRR = (num_windows - total_saved) / num_windows if num_windows > 0 else 0
    
    captured_events = 0
    if onsets:
        for onset_s in onsets:
            onset_win = int(onset_s / HOP_S)
            event_captured = False
            for w in range(onset_win, min(num_windows, onset_win + 10)):
                if w in saved_set:
                    event_captured = True
                    break
            if event_captured: captured_events += 1
    
    ECR = (captured_events / len(onsets)) * 100 if onsets else 0.0
    
    # Latency Calculation
    latency_ms = []
    if onsets and y_pred is not None:
        for onset_s in onsets:
            onset_win = int(onset_s / HOP_S)
            for w in range(onset_win, min(num_windows, onset_win + 10)):
                if y_pred[w] == 1:
                    latency_ms.append(max(0, (w * HOP_S) - onset_s) * 1000)
                    break
    
    phys = PhysicsModel(sparsity_gamma=0.5, overhead_factor=1.2)
    n_features = N_MFCC * 3 + 2 
    sram_kb = phys.calculate_sram_kb(total_saved, n_features)
    
    # Empirical Flash measurement if available, else analytical
    flash_kb = emp_size if emp_size > 0 else phys.calculate_flash_kb(model_type, {'n_estimators': 100, 'max_depth': 12, 'n_features': n_features})
    macs     = phys.calculate_compute_proxy(model_type, {'n_estimators': 100, 'max_depth': 12, 'n_features': n_features})
    
    return {'system': name, 'DRR': DRR*100, 'ECR': ECR, 
            'latency': np.mean(latency_ms) if latency_ms else 0, 
            'sram_kb': sram_kb, 'flash_kb': flash_kb, 'macs': macs, 'saved': total_saved}

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────
def run_repetitive_simulations(df_ts, clf, scaler, model_type='RandomForest', emp_size=0):
    print(f"\n[4/7] Running {N_RUNS}x repetitive simulations (Augmentation-Enabled)...")
    stats = {}
    config_keys = []
    
    ps = PlaylistEngine(sr=SR)
    
    for seed in range(42, 42 + N_RUNS):
        schedule = ps.create_physically_grounded_schedule(df_ts, seed=seed, snr_db=SNR_DB_LIST[0], overlap_prob=OVERLAP_PROB)
        st_y, st_labels, onsets = ps.generate_stream_from_schedule(schedule)
        y_prob, y_pred, y_true_w, nw = run_streaming_inference(st_y, st_labels, clf, scaler)
        
        configs = []
        configs.append(compute_sys_metrics('Store-All', set(range(nw)), y_pred, y_true_w, nw, onsets, model_type, emp_size))
        
        # Energy-Detector literature baseline
        ed_triggers = get_energy_triggers(st_y, SR, nw, sigma_mult=THRESHOLD)
        ed_set = get_buffered_indices(ed_triggers, None, nw, 0, 0)
        configs.append(compute_sys_metrics('ED_Baseline', ed_set, ed_triggers, y_true_w, nw, onsets, 'SimpleDetector', 0))

        b0_set = get_buffered_indices(y_pred, y_prob, nw, 0, 0)
        sysB = compute_sys_metrics('Detect-Only', b0_set, y_pred, y_true_w, nw, onsets, model_type, emp_size)
        configs.append(sysB)
        
        k_saved = sysB['saved']
        step = max(1, nw // k_saved) if k_saved > 0 else nw
        per_set = set(list(range(0, nw, step))[:k_saved])
        configs.append(compute_sys_metrics('Fixed_Periodic', per_set, None, y_true_w, nw, onsets, model_type, emp_size))
        
        rand_set = set(np.random.choice(nw, size=min(k_saved, nw), replace=False)) if k_saved > 0 else set()
        configs.append(compute_sys_metrics('Pure_Random', rand_set, None, y_true_w, nw, onsets, model_type, emp_size))
        
        a_set = get_buffered_indices(y_pred, y_prob, nw, 2, 4, adaptive=True)
        configs.append(compute_sys_metrics('Proposed_Adaptive', a_set, y_pred, y_true_w, nw, onsets, model_type, emp_size))

        for c in configs:
            k = c['system']
            if k not in stats: stats[k] = {'DRR': [], 'ECR': [], 'latency': [], 'sram_kb': [], 'flash_kb': [], 'macs': []}
            for metr in ['DRR', 'ECR', 'latency', 'sram_kb', 'flash_kb', 'macs']: stats[k][metr].append(c[metr])
            if k not in config_keys: config_keys.append(k)
            
    agg_stats = {}
    print("  Aggregating results with 95% Confidence Intervals...")
    for k in config_keys:
        res_ecr = bootstrap((stats[k]['ECR'],), np.mean, confidence_level=0.95, n_resamples=BOOTSTRAP_ITERS)
        res_drr = bootstrap((stats[k]['DRR'],), np.mean, confidence_level=0.95, n_resamples=BOOTSTRAP_ITERS)
        agg_stats[k] = {
            'DRR': np.mean(stats[k]['DRR']), 
            'DRR_low': res_drr.confidence_interval.low, 'DRR_high': res_drr.confidence_interval.high,
            'ECR': np.mean(stats[k]['ECR']), 
            'ECR_low': res_ecr.confidence_interval.low, 'ECR_high': res_ecr.confidence_interval.high,
            'latency': np.mean(stats[k]['latency']), 
            'sram_kb': np.mean(stats[k]['sram_kb']), 
            'flash_kb': np.mean(stats[k]['flash_kb']),
            'macs': np.mean(stats[k]['macs'])
        }
    
    cohen_d = calculate_cohens_d(stats['Proposed_Adaptive']['ECR'], stats['Pure_Random']['ECR'])
    _, p = mannwhitneyu(stats['Proposed_Adaptive']['ECR'], stats['Pure_Random']['ECR'])
    agg_stats['_p_value'] = p
    agg_stats['_cohen_d'] = cohen_d
    return agg_stats, config_keys

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def print_and_visualize(all_results):
    sns.set_theme(style='ticks', font_scale=1.0)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    primary = all_results['RandomForest']['agg_stats']
    keys = all_results['RandomForest']['keys']
    
    # Fig A: Pareto Frontier (Systems Trade-offs)
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in keys:
        col = '#7F8C8D' if k == 'Store-All' else '#F39C12' if k == 'ED_Baseline' else '#C0392B' if k == 'Fixed_Periodic' else '#2980B9' if 'Detect' in k else '#8E44AD' if 'Adaptive' in k else '#27AE60'
        ax.scatter(primary[k]['DRR'], primary[k]['ECR'], s=150, color=col, edgecolors='white', zorder=5)
        ax.errorbar(primary[k]['DRR'], primary[k]['ECR'], 
                    xerr=[[primary[k]['DRR'] - primary[k]['DRR_low']], [primary[k]['DRR_high'] - primary[k]['DRR']]],
                    yerr=[[primary[k]['ECR'] - primary[k]['ECR_low']], [primary[k]['ECR_high'] - primary[k]['ECR']]],
                    fmt='none', color='black', alpha=0.3)
        ax.annotate(k.replace('Proposed_', '').replace('Fixed_',''), (primary[k]['DRR'], primary[k]['ECR']), xytext=(0, 10), textcoords='offset points', fontsize=9, ha='center', fontweight='bold')

    ax.set_xlabel('Data Reduction Ratio — DRR (%)')
    ax.set_ylabel('Event Capture Rate — ECR (%)')
    ax.set_title('Fig A: System Pareto Frontier of Buffering Policies (Augmented)')
    sns.despine()
    plt.savefig(os.path.join(FIG, 'figA_pareto_frontier.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Fig H: Significance & Effect Size
    fig, ax = plt.subplots(figsize=(6, 4))
    comp = ['Pure Random', 'Adaptive (Prop.)']
    means = [primary['Pure_Random']['ECR'], primary['Proposed_Adaptive']['ECR']]
    err = [[primary['Pure_Random']['ECR'] - primary['Pure_Random']['ECR_low'], primary['Proposed_Adaptive']['ECR'] - primary['Proposed_Adaptive']['ECR_low']],
           [primary['Pure_Random']['ECR_high'] - primary['Pure_Random']['ECR'], primary['Proposed_Adaptive']['ECR_high'] - primary['Proposed_Adaptive']['ECR']]]
    ax.bar(comp, means, yerr=err, capsize=10, color=['#95A5A6', '#8E44AD'], alpha=0.8)
    d, p = primary['_cohen_d'], primary['_p_value']
    ax.set_title(f"Fig H: Quantifying the Intelligence Gap (p={p:.4f}, d={d:.2f})")
    ax.set_ylabel("Event Capture Rate (ECR %)")
    sns.despine()
    plt.savefig(os.path.join(FIG, 'figH_statistical_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("\n" + "="*75)
    print("  Evaluating Event-Driven TinyML Acoustic Monitoring")
    print("  Continuous Augmented Streaming | Empirical Systems Evaluation")
    print("="*75)
    
    if DATASET_TO_USE == 'ESC-50': df_tr, df_ts = load_esc50()
    elif DATASET_TO_USE == 'US8K': df_tr, df_ts = load_us8k()
    else: df_tr, df_ts = load_birdclef()
    
    if df_tr.empty: exit(1)

    print("\n[LEAKAGE AUDIT] Verifying deterministic set disjointness...")
    tr_files, ts_files = set(df_tr['filepath']), set(df_ts['filepath'])
    if len(tr_files.intersection(ts_files)) > 0: exit("  [FAILURE] Data leakage detected.")
    else: print("  [SUCCESS] Train/Test sets strictly isolated.")

    X_tr, y_tr = extract_train_features(df_tr)
    all_results = {}
    
    # Reference Classifier (Random Forest)
    clf_rf, sc_rf, emp_rf = train_classifier(X_tr, y_tr)
    agg_rf, keys_rf = run_repetitive_simulations(df_ts, clf_rf, sc_rf, 'RandomForest', emp_rf)
    all_results['RandomForest'] = {'agg_stats': agg_rf, 'keys': keys_rf}

    if COMPARE_MODELS:
        # Linear Comparison (Baseline Complexity)
        clf_lr, sc_lr, emp_lr = train_linear_model(X_tr, y_tr)
        agg_lr, keys_lr = run_repetitive_simulations(df_ts, clf_lr, sc_lr, 'LinearBenchmark', emp_lr)
        all_results['LinearBenchmark'] = {'agg_stats': agg_lr, 'keys': keys_lr}

    print_and_visualize(all_results)
    with open(os.path.join(RES, 'evaluation_summary.json'), 'w') as f:
        json.dump({m: v['agg_stats'] for m, v in all_results.items()}, f, indent=2)
    print("\n[COMPLETE] Empirical footprints and augmented results generated.")
