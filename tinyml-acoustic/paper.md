# Event-Triggered Acoustic Monitoring via Circular Buffer Simulation: A TinyML Framework for Edge Deployment

**M M Nishat**
Department of Computer Science, University of Southern Mississippi
Hattiesburg, MS, USA

---

## Abstract

Continuous audio recording on resource-constrained edge devices wastes storage and power by persisting non-event segments that carry no actionable signal. We present a simulation framework that evaluates five acoustic monitoring strategies — Store-All, Detect-Only, Random baseline, and Circular Buffer at B=1 and B=2 — using the ESC-50 environmental sound dataset. Our Random Forest classifier (MFCC + delta-MFCC + RMS, 122-dim) achieves 82.5% accuracy on a 10-class subset. The **key finding**: at the same storage budget as Detect-Only (DRR=69.5%), a random-save baseline captures only 32.5% of events — while Detect-Only captures 95.0%. Adding a 1-clip circular buffer raises ECR to 96.7% at DRR=46.2%, and a buffer ablation study shows B=1 is the "knee of the curve": B=2 gains only +0.8pp ECR at a -15.2pp DRR cost. The framework quantifies, for the first time at this scale, how much of the system's performance is attributable to intelligent detection versus storage volume — with all temporal assumptions made explicit.

---

## 1. Introduction

Acoustic event detection (AED) on embedded hardware faces a fundamental tension: high-fidelity continuous recording preserves all context but overwhelms limited flash storage, while detection-only logging discards surrounding context that is often critical for downstream analysis (e.g., determining onset, offset, and acoustic neighborhood of a chainsaw detection).

Circular buffers are a classical solution in embedded systems — they maintain a rolling window of audio so that when a trigger fires, pre-event context is already in memory. However, two critical questions are rarely answered:

1. **How much of the detected system's ECR is "intelligence" vs. coincidental storage?** (The random baseline answers this.)
2. **Where is the diminishing-returns point on buffer size?** (The ablation study answers this.)

**Contribution.** We contribute:
1. A reproducible simulation framework implementing five systems with formal metric definitions.
2. A random baseline at matched storage budget — demonstrating +64.2pp ECR attributable to intelligent detection.
3. A buffer ablation study (B=0, 1, 2) identifying B=1 as the Pareto-optimal operating point.
4. Explicit documentation of the temporal assumption introduced by treating ESC-50 clips as a sequential stream.
5. A single hero figure (DRR vs. ECR trade-off) summarizing all findings in one view.

**Real-world impact:** This approach can be directly extended to low-power environmental sensors deployed in forests for illegal logging detection and biodiversity monitoring, where bandwidth and storage are severely constrained.

---

## 2. Related Work

**Acoustic event detection.** Early systems used mel-spectrogram CNNs [Salamon & Bello 2017; Piczak 2015]. Recent TinyML work targets MCUs with sub-1MB models [Zhang et al. 2022]. We use a Random Forest over MFCC features — deliberately shallow to reflect deployment constraints on devices like the Arduino Nano 33 BLE Sense or ESP32.

**Edge audio buffering.** Circular buffers for audio pre-roll are standard in consumer voice assistants (e.g., always-on keyword detection keeps 1–2s of pre-roll). Our work formalizes this in a conservation-vs-capture framework with ablation and credibility testing.

**ESC-50.** The ESC-50 dataset [Piczak 2015] contains 2000 5-second clips across 50 environmental sound classes at 22050 Hz. We use 10 classes (400 clips) and treat them as a synthetic sequential stream — an assumption described fully in Section 4.1.

---

## 3. Problem Formulation

Let $\mathcal{C} = \{c_1, c_2, \ldots, c_N\}$ be a stream of $N$ audio clips, each of duration $\delta$ seconds. Each clip is labeled $y_i \in \{0, 1\}$ where $y_i = 1$ denotes an acoustic event of interest.

**Goal:** Retain a subset $\mathcal{S} \subseteq \mathcal{C}$ that maximizes event coverage while minimizing $|\mathcal{S}|$.

**Formal Metrics:**

$$\text{DRR} = \frac{N - |\mathcal{S}|}{N} \quad \text{(storage efficiency)}$$

$$\text{ECR} = \frac{TP}{TP + FN} \quad \text{(event preservation)}$$

$$\text{FPR} = \frac{FP}{FP + TN} \quad \text{(non-event contamination)}$$

$$\text{PPV} = \frac{TP}{TP + FP} \quad \text{(precision of saved set)}$$

The fundamental tension: a system that saves everything achieves ECR=1 but DRR=0. A system that saves nothing achieves DRR=1 but ECR=0. The research question is: **where on this frontier does intelligent detection + buffering operate, and how much better is it than random?**

---

## 4. Methodology

### 4.1 Dataset and Temporal Assumption

We select 10 categories from ESC-50: three event classes (chainsaw, hand\_saw, engine) and seven ambient classes (crickets, frog, wind, rain, thunderstorm, insects, water\_drops). This yields 400 clips: 120 event, 280 non-event (30%/70% split).

**Temporal Assumption (explicit):** ESC-50 clips are independently recorded and non-continuous. For simulation purposes, we treat them as a sequential stream of 5s segments. This is a controlled benchmark assumption — we do not claim these results reflect real acoustic ecology. The contribution is the evaluation framework and the relative comparisons between systems, not the absolute numbers on ESC-50.

### 4.2 Feature Extraction

For each clip we extract a 122-dimensional feature vector:

| Feature | Dimension |
|---------|-----------|
| MFCC mean (40 coefficients) | 40 |
| MFCC std (40 coefficients) | 40 |
| Delta-MFCC mean | 40 |
| RMS energy mean | 1 |
| RMS energy std | 1 |
| **Total** | **122** |

Parameters: SR=22050 Hz, n\_mfcc=40, hop\_length=512, n\_fft=2048.

### 4.3 Classifier

```
RandomForestClassifier(
    n_estimators=200, max_depth=12,
    class_weight='balanced', random_state=42
)
```

`class_weight='balanced'` corrects for the 30/70 event/non-event imbalance. Decision threshold $\tau = 0.35$ (lowered from default 0.50) favors recall over precision — consistent with safety-oriented deployment where missing an event is more costly than a false alarm.

### 4.4 Five-System Simulation

**Baseline A — Store-All:** All clips saved. DRR=0, ECR=1 by definition.

**Baseline B — Detect-Only (B=0):** Only classifier-positive clips saved. No context.

**Random Baseline:** Saves the same number of clips as Detect-Only ($|\mathcal{S}| = 122$), chosen uniformly at random (seed=42). This is the critical credibility check: if our system only matches random performance at the same storage budget, it provides no intelligence value.

**Circular Buffer B=1 (Proposed):** On detection at clip $i$, saves clips $\{i-1, i, i+1\}$. Minimum event packet = 15s.

**Circular Buffer B=2:** On detection at clip $i$, saves clips $\{i-2, i-1, i, i+1, i+2\}$. Tests diminishing returns.

---

## 5. Results

### 5.1 Classifier Performance

| Metric | Value |
|--------|-------|
| Accuracy | 82.5% |
| Precision | 69.2% |
| Recall (per-clip) | 75.0% |
| F1-score | 72.0% |

### 5.2 System-Level Comparison

| System | DRR | ECR | FPR | PPV | Saved Clips |
|--------|-----|-----|-----|-----|-------------|
| Store-All (B-A) | 0.0% | 100.0% | 100.0% | 30.0% | 400 |
| Detect-Only B=0 (B-B) | 69.5% | 95.0% | 2.9% | 93.4% | 122 |
| **Random Baseline** | **69.5%** | **32.5%** | 29.6% | 32.0% | 122 |
| **Circular Buffer B=1 (Proposed)** | **46.2%** | **96.7%** | 35.4% | 53.9% | 215 |
| Circular Buffer B=2 | 31.0% | 97.5% | 56.8% | 42.4% | 276 |

### 5.3 The Intelligence Gap (Key Finding)

At DRR=69.5%, the random baseline achieves ECR=32.5%. Detect-Only achieves ECR=95.0% at the same budget — a **+62.5pp gap** attributable entirely to intelligent detection. This is the strongest result in the paper: the system is not accidentally good.

### 5.4 Buffer Ablation — The Pareto Knee

| B | DRR | ECR | ECR gain over B=0 | DRR cost vs B=0 |
|---|-----|-----|------|------|
| 0 (Detect-Only) | 69.5% | 95.0% | — | — |
| 1 (Proposed) | 46.2% | 96.7% | +1.7pp | -23.3pp |
| 2 | 31.0% | 97.5% | +2.5pp | -38.5pp |

B=1 is the Pareto-optimal point: it recovers the most ECR per unit of DRR sacrificed. Going from B=1 to B=2 gains only +0.8pp ECR at a -15.2pp DRR cost — strongly diminishing returns. **B=1 is the recommended operating point.**

The hero figure (Fig. 5) visualizes this finding: the buffer ablation curve bends sharply at B=1, with B=2 providing minimal upward movement for substantial leftward displacement.

---

## 6. System Architecture

The deployed TinyML pipeline follows four stages:

1. **Capture:** Continuous audio at SR=22050 Hz via onboard PDM microphone.
2. **Preprocess:** Frame into 5s clips; extract 122-dim feature vector.
3. **Infer:** Random Forest forward pass; apply threshold $\tau = 0.35$.
4. **Buffer Logic:** On positive detection at clip $i$: flush pre-roll buffer + current clip + post-roll ($B$ clips each side); transmit event packet.

The circular buffer operates at clip granularity (5s), not sample granularity — significantly reducing the memory overhead for embedded implementation.

---

## 7. Limitations

1. **Non-continuous stream assumption.** ESC-50 clips are not a real acoustic deployment. True monitoring would face concept drift, polyphonic audio, and variable SNR.
2. **No on-device latency measurement.** All results are from offline simulation. Inference time on MCU-class hardware requires profiling.
3. **Fixed threshold.** $\tau=0.35$ was hand-tuned. Adaptive thresholding for acoustic background shifts is not implemented.
4. **10-class subset.** Generalizing to all 50 ESC-50 categories or domain-specific classes requires retraining.

---

## 8. Conclusion

We presented a simulation framework with five systems and a buffer ablation study quantifying the ECR-DRR trade-off for event-triggered acoustic monitoring. The critical result: at the same storage budget, intelligent detection achieves 95.0% ECR vs. 32.5% for random selection — a +62.5pp gap that quantifies the "intelligence value" of the classifier. Adding a 1-clip circular buffer raises ECR to 96.7% with B=1 identified as the Pareto knee of the ablation curve.

**Future work:** (1) Deploy on embedded hardware and measure inference latency; (2) evaluate on a continuous real-world acoustic stream; (3) replace Random Forest with a quantized DS-CNN for improved F1 while remaining MCU-deployable.

---

## References

- Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. *ACM Multimedia*.
- Salamon, J., & Bello, J. P. (2017). Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification. *IEEE Signal Processing Letters*.
- Zhang, Y., et al. (2022). Benchmarking TinyML Systems: Challenges and Direction. *arXiv:2003.04821*.
- Warden, P., & Situnayake, D. (2019). *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. O'Reilly Media.

---

*Submitted to: USM Undergraduate Research Symposium, April 18, 2026*
