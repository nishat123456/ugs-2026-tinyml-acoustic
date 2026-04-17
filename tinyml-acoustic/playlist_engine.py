import numpy as np
import pandas as pd
import librosa
import os

class PlaylistEngine:
    """
    Simulated streaming environment for evaluating acoustic systems.
    Supports deterministic schedules with calibrated SNR mixing, Poisson arrivals,
    and stochastic augmentation (pitch/stretch).
    """
    def __init__(self, sr=22050):
        self.sr = sr

    def compute_rms(self, y):
        return np.sqrt(np.mean(y**2))

    def apply_augmentation(self, y, sr, seed):
        """
        Applies stochastic time-stretch and pitch-shift to a clip.
        ±10% range for academic realism.
        """
        np.random.seed(seed)
        # Time stretch: 0.9 to 1.1
        rate = np.random.uniform(0.9, 1.1)
        y_aug = librosa.effects.time_stretch(y=y, rate=rate)
        
        # Pitch shift: -1.5 to 1.5 semitones
        n_steps = np.random.uniform(-1.5, 1.5)
        y_aug = librosa.effects.pitch_shift(y=y_aug, sr=sr, n_steps=n_steps)
        
        return y_aug

    def create_physically_grounded_schedule(self, df_test, stream_minutes=15, 
                                          snr_db=10, overlap_prob=0.3, seed=42):
        """
        Generates a deterministic schedule using a Poisson arrival process.
        """
        np.random.seed(seed)
        events = df_test[df_test['label'] == 'event']
        noises = df_test[df_test['label'] == 'non_event']
        
        schedule = []
        
        # 1. Background Noise schedule
        curr_b = 0
        while curr_b < (stream_minutes * 60):
            noise_row = noises.sample(1, random_state=int(seed+curr_b)).iloc[0]
            schedule.append({
                'type': 'background',
                'filepath': noise_row['filepath'],
                'start_sec': float(curr_b),
                'label': 'non_event'
            })
            curr_b += 5.0 
            
        # 2. Event schedule (Poisson Process Arrivals)
        curr_e = 5.0
        while curr_e < (stream_minutes * 60 - 10):
            iat = np.random.exponential(scale=50.0) 
            curr_e += iat
            if curr_e >= (stream_minutes * 60 - 10): break
            
            event_row = events.sample(1).iloc[0]
            schedule.append({
                'type': 'event',
                'filepath': event_row['filepath'],
                'start_sec': float(curr_e),
                'label': 'event',
                'snr_db': snr_db,
                'aug_seed': int(seed + curr_e) # Seed for augmentation diversity
            })
            
            # Interference/Overlap modeling
            if np.random.rand() < overlap_prob:
                offset = np.random.uniform(0.5, 2.0)
                second_event = events.sample(1).iloc[0]
                schedule.append({
                    'type': 'event',
                    'filepath': second_event['filepath'],
                    'start_sec': float(curr_e + offset),
                    'label': 'event',
                    'snr_db': snr_db,
                    'aug_seed': int(seed + curr_e + 1000)
                })
            
        return schedule

    def generate_stream_from_schedule(self, schedule, stream_minutes=15):
        """
        Constructs a continuous audio stream using power-calibrated SNR mixing 
        and augmented events.
        """
        samples_total = int(stream_minutes * 60 * self.sr)
        stream_y = np.zeros(samples_total, dtype=np.float32)
        stream_labels = np.zeros(samples_total, dtype=int)
        onsets = []
        
        bg_items = [i for i in schedule if i['type'] == 'background']
        ev_items = [i for i in schedule if i['type'] == 'event']
        
        # 1. Background Overlay
        for item in bg_items:
            y, _ = librosa.load(item['filepath'], sr=self.sr, duration=5.0)
            start_idx = int(item['start_sec'] * self.sr)
            end_idx = min(samples_total, start_idx + len(y))
            stream_y[start_idx:end_idx] += y[:end_idx - start_idx]
            
        stream_y += np.random.normal(0, 0.001, samples_total) # Noise floor
        
        # 2. Event Overlay (Augmented + SNR Calibrated)
        for item in ev_items:
            y_e, _ = librosa.load(item['filepath'], sr=self.sr, duration=5.0)
            
            # [DEEP RIGOR]: Stochastic Augmentation
            y_e = self.apply_augmentation(y_e, self.sr, item['aug_seed'])
            
            start_idx = int(item['start_sec'] * self.sr)
            end_idx = min(samples_total, start_idx + len(y_e))
            
            y_n = stream_y[start_idx:end_idx]
            rms_n = self.compute_rms(y_n)
            rms_e = self.compute_rms(y_e)
            
            if rms_n > 0 and rms_e > 0:
                snr_lin = 10**(item['snr_db'] / 10.0)
                target_rms_e = rms_n * np.sqrt(snr_lin)
                scale = target_rms_e / rms_e
                stream_y[start_idx:end_idx] += y_e[:end_idx - start_idx] * scale
            else:
                stream_y[start_idx:end_idx] += y_e[:end_idx - start_idx]
                
            stream_labels[start_idx:end_idx] = 1
            onsets.append(item['start_sec'])
                
        if np.abs(stream_y).max() > 0: stream_y /= np.abs(stream_y).max()
        return stream_y, stream_labels, onsets
