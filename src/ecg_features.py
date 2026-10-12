"""
Modul Pemrosesan Sinyal ECG & Ekstraksi 4 Fitur QRST
Sesuai Arsitektur Blok Diagram:
ECG -> Bandpass Filter -> Detektor R & Shift Register -> Detektor Q, S, T -> Ekstraksi 4 Fitur (x1, x2, x3, x4)
"""

import numpy as np
import scipy.signal as signal
import os
from typing import Tuple, List, Dict, Optional


def parse_diagnosis_from_header(header_file: str) -> str:
    """Membaca alasan admisi / diagnosis medis dari komentar di file .hea."""
    if not os.path.exists(header_file):
        return "Unknown"
    with open(header_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Reason for admission:" in line:
                diag = line.split(":", 1)[1].strip()
                return diag
    return "Unknown"


def map_diagnosis_to_class(diagnosis: str) -> str:
    """Memetakan diagnosis medis ke 3 kelas target: Normal, Infark Miokard, Unknown."""
    d_lower = diagnosis.lower()
    if "healthy" in d_lower or "control" in d_lower:
        return "Normal"
    elif "myocardial infarction" in d_lower or "infarction" in d_lower:
        return "Infark Miokard"
    else:
        return "Unknown"


def bandpass_filter(data: np.ndarray, fs: float = 1000.0, lowcut: float = 0.5, highcut: float = 45.0, order: int = 3) -> np.ndarray:
    """
    Butterworth Bandpass Filter untuk menghilangkan baseline wander (drift frekuensi rendah)
    dan derau frekuensi tinggi (powerline 50/60 Hz & EMG).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)
    return filtered


def detect_r_peaks(ecg_signal: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """
    Deteksi puncak R (R-peak) terinspirasi Pan-Tompkins:
    1. Diferensiasi (turunan)
    2. Kuadrat amplitudo
    3. Moving window integrator
    4. Deteksi puncak lokal dengan refractory period (~250-300 ms).
    """
    # 1. Turunan
    diff = np.diff(ecg_signal, prepend=ecg_signal[0])
    
    # 2. Kuadrat
    squared = diff ** 2
    
    # 3. Integrator jendela geser (Shift Register window / Moving average ~150 ms)
    win_size = int(0.15 * fs)
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode='same')
    
    # 4. Threshold adaptif
    threshold = 0.35 * np.max(integrated)
    min_dist = int(0.4 * fs)  # minimal jarak antar R-peak (~400 ms, maks 150 bpm)
    
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_dist)
    
    # Koreksi letak puncak R ke puncak gelombang sinyal asli dalam window +/- 60 ms
    corrected_r_peaks = []
    search_win = int(0.06 * fs)
    for p in peaks:
        left = max(0, p - search_win)
        right = min(len(ecg_signal), p + search_win)
        if right > left:
            true_r = left + np.argmax(ecg_signal[left:right])
            corrected_r_peaks.append(true_r)
            
    return np.array(corrected_r_peaks, dtype=int)


def extract_qrst_fiducials(ecg_signal: np.ndarray, r_peak: int, fs: float = 1000.0) -> Dict[str, int]:
    """
    Menemukan titik fidusial Q, R, S, T untuk sebuah detak jantung (beat).
    Menggunakan buffer jendela geser (Shift Register) relatif terhadap R-peak:
    - Q: Titik minimum lokal sebelum R (dalam rentang 20 ms s.d. 80 ms sebelum R)
    - S: Titik minimum lokal setelah R (dalam rentang 20 ms s.d. 80 ms setelah R)
    - T: Titik ekstremum / maksimum lokal gelombang T (dalam rentang 120 ms s.d. 450 ms setelah S)
    """
    # Rentang pencarian (sampel)
    q_search_start = max(0, r_peak - int(0.08 * fs))
    q_search_end = max(0, r_peak - int(0.01 * fs))
    
    s_search_start = min(len(ecg_signal), r_peak + int(0.01 * fs))
    s_search_end = min(len(ecg_signal), r_peak + int(0.09 * fs))
    
    # Detektor Q: minimum sebelum R
    if q_search_end > q_search_start:
        q_idx = q_search_start + np.argmin(ecg_signal[q_search_start:q_search_end])
    else:
        q_idx = max(0, r_peak - int(0.03 * fs))
        
    # Detektor S: minimum setelah R
    if s_search_end > s_search_start:
        s_idx = s_search_start + np.argmin(ecg_signal[s_search_start:s_search_end])
    else:
        s_idx = min(len(ecg_signal) - 1, r_peak + int(0.03 * fs))
        
    # Detektor T: puncak gelombang T setelah S
    t_search_start = min(len(ecg_signal), s_idx + int(0.08 * fs))
    t_search_end = min(len(ecg_signal), s_idx + int(0.40 * fs))
    
    if t_search_end > t_search_start:
        # Puncak gelombang T bisa positif atau terbalik (T-wave inversion pada MI)
        window_t = ecg_signal[t_search_start:t_search_end]
        max_t = np.max(window_t)
        min_t = np.min(window_t)
        # Cari deviasi terbesar dari garis isoelektrik
        if abs(max_t) >= abs(min_t):
            t_idx = t_search_start + np.argmax(window_t)
        else:
            t_idx = t_search_start + np.argmin(window_t)
    else:
        t_idx = min(len(ecg_signal) - 1, s_idx + int(0.20 * fs))
        
    return {
        "Q": int(q_idx),
        "R": int(r_peak),
        "S": int(s_idx),
        "T": int(t_idx)
    }


def compute_4_qrst_features(ecg_signal: np.ndarray, fiducials: Dict[str, int], fs: float = 1000.0) -> np.ndarray:
    """
    Ekstraksi 4 Fitur QRST Utama [x1, x2, x3, x4] untuk masukan ANN:
    1. x1: Amplitudo Gelombang Q (V_Q) atau kedalaman Q relatif terhadap baseline.
           Gelombang Q patologis (Q dalam & lebar) adalah tanda khas Infark Miokard lama/akut.
    2. x2: Amplitudo Puncak Gelombang R (V_R).
           Penurunan amplitudo R (loss of R-wave height) menandakan nekrosis miokard.
    3. x3: Elevasi/Deviasi Segmen ST (V_ST), diukur pada J-point (S + 60ms) relatif terhadap baseline.
           Elevasi segmen ST (STEMI) atau depresi ST adalah penanda utama Infark Miokard akut.
    4. x4: Amplitudo Puncak Gelombang T (V_T).
           Inversi gelombang T (T negatif) atau gelombang T hiperakut adalah ciri khas infark & iskemia.
    """
    q_idx = fiducials["Q"]
    r_idx = fiducials["R"]
    s_idx = fiducials["S"]
    t_idx = fiducials["T"]
    
    # Baseline isoelektrik diambil ~50-80ms sebelum gelombang Q (segmen PR)
    pr_start = max(0, q_idx - int(0.08 * fs))
    pr_end = max(0, q_idx - int(0.02 * fs))
    if pr_end > pr_start:
        v_iso = np.median(ecg_signal[pr_start:pr_end])
    else:
        v_iso = ecg_signal[max(0, q_idx - 10)]
        
    # Nilai 4 Fitur:
    # x1: Amplitudo Q relatif ke baseline
    v_q = ecg_signal[q_idx] - v_iso
    
    # x2: Amplitudo R relatif ke baseline
    v_r = ecg_signal[r_idx] - v_iso
    
    # x3: ST segment level (diukur 60ms setelah S-point) relatif ke baseline
    st_idx = min(len(ecg_signal) - 1, s_idx + int(0.06 * fs))
    v_st = ecg_signal[st_idx] - v_iso
    
    # x4: Amplitudo puncak T relatif ke baseline
    v_t = ecg_signal[t_idx] - v_iso
    
    return np.array([v_q, v_r, v_st, v_t], dtype=np.float32)


def process_lead_signal(ecg_raw: np.ndarray, fs: float = 1000.0) -> Tuple[np.ndarray, List[Dict[str, int]], np.ndarray]:
    """
    Pipeline lengkap dari sinyal mentah ke 4 fitur untuk setiap detak:
    1. Filter Bandpass
    2. Deteksi R-peaks
    3. Deteksi titik fidusial Q, R, S, T
    4. Ekstraksi matriks fitur [N_beats, 4]
    """
    filtered = bandpass_filter(ecg_raw, fs=fs)
    r_peaks = detect_r_peaks(filtered, fs=fs)
    
    fiducials_list = []
    features_list = []
    
    for r in r_peaks:
        # Pastikan margin beat cukup di awal dan akhir sinyal
        if r < int(0.2 * fs) or r > len(filtered) - int(0.5 * fs):
            continue
        fid = extract_qrst_fiducials(filtered, r, fs=fs)
        feats = compute_4_qrst_features(filtered, fid, fs=fs)
        fiducials_list.append(fid)
        features_list.append(feats)
        
    features_matrix = np.array(features_list, dtype=np.float32) if features_list else np.empty((0, 4), dtype=np.float32)
    return filtered, fiducials_list, features_matrix
