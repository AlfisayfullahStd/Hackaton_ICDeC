"""
Script Pelatihan ANN (Multilayer Perceptron) untuk Klasifikasi ECG:
Input: 4 Fitur QRST (x1: Q-amp, x2: R-amp, x3: ST-level, x4: T-amp)
Output: 3 Kelas (0: Normal, 1: Infark Miokard, 2: Unknown)
Sesuai Arsitektur Blok Diagram & Kesiapan Hardware FPGA (RTL Export).
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Pastikan path modul src/ dapat diimpor dengan aman
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_features import process_lead_signal, map_diagnosis_to_class, parse_diagnosis_from_header

CLASS_NAMES = ["Normal", "Infark Miokard", "Unknown"]
CLASS_TO_IDX = {"Normal": 0, "Infark Miokard": 1, "Unknown": 2}


class ECG_ANN(nn.Module):
    """
    Arsitektur Artificial Neural Network (MLP):
    - Input Layer: 4 neuron (x1, x2, x3, x4)
    - Hidden Layer: 8 neuron (h1, h2, ..., hz) dengan aktivasi ReLU
    - Output Layer: 3 neuron (Normal, Infark Miokard, Unknown)
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8, num_classes: int = 3):
        super(ECG_ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        out = self.fc2(h)
        return out


def collect_dataset_from_local(data_dir: str = "data/ptbdb", lead_name: str = "ii") -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Membaca semua record yang telah diunduh di direktori data/ptbdb/,
    mengekstrak 4 fitur QRST untuk setiap detak jantung,
    dan memetakan label ke Normal (0), Infark Miokard (1), atau Unknown (2).
    """
    records_info = []
    hea_files = glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True)
    
    all_features = []
    all_labels = []
    
    print(f"Ditemukan {len(hea_files)} file header (.hea) di {data_dir}...")
    
    for hea_path in hea_files:
        base_record = hea_path[:-4]  # buang ekstensi .hea
        # Ambil nama relatif
        rel_path = os.path.relpath(base_record, data_dir).replace("\\", "/")
        
        diag = parse_diagnosis_from_header(hea_path)
        class_label = map_diagnosis_to_class(diag)
        label_idx = CLASS_TO_IDX[class_label]
        
        try:
            # Membaca hanya lead standar (lead ii)
            h = wfdb.rdheader(base_record)
            lead_idx = 1  # default lead ii
            if lead_name in h.sig_name:
                lead_idx = h.sig_name.index(lead_name)
            
            rec = wfdb.rdrecord(base_record, channels=[lead_idx])
            sig = rec.p_signal[:, 0]
            fs = rec.fs
            
            _, _, feats = process_lead_signal(sig, fs=fs)
            if len(feats) > 0:
                all_features.append(feats)
                all_labels.append(np.full(len(feats), label_idx, dtype=int))
                records_info.append({
                    "record": rel_path,
                    "diagnosis": diag,
                    "class": class_label,
                    "beats_count": len(feats)
                })
                print(f"  [OK] {rel_path} | {diag} -> {class_label} ({len(feats)} detak)")
        except Exception as e:
            print(f"  [Skip] {rel_path}: {e}")
            
    df_info = pd.DataFrame(records_info)
    
    if len(all_features) > 0:
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
    else:
        # Fallback jika belum ada data lokal: sintesis data berbasis distribusi medis
        print("Peringatan: Belum ada record lokal yang berhasil diproses, menggunakan generator sintetis representatif.")
        X, y = generate_synthetic_data(n_samples_per_class=200)
        
    return X, y, df_info


def generate_synthetic_data(n_samples_per_class: int = 250) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generator fitur fisiologis representatif (Q, R, ST, T) berdasarkan parameter klinis:
    - Normal: Q kecil (-0.08mV), R tinggi (1.1mV), ST isoelektrik (0.02mV), T positif (0.35mV)
    - Infark Miokard: Q patologis (-0.25mV), R menurun (0.25mV), ST elevasi/depresi (0.35mV / -0.2mV), T terbalik (-0.3mV)
    - Unknown: pola campuran / aritmia / LVH
    """
    np.random.seed(42)
    # Normal
    X_norm = np.column_stack([
        np.random.normal(-0.08, 0.03, n_samples_per_class),
        np.random.normal(1.15, 0.15, n_samples_per_class),
        np.random.normal(0.02, 0.03, n_samples_per_class),
        np.random.normal(0.35, 0.06, n_samples_per_class)
    ])
    y_norm = np.zeros(n_samples_per_class, dtype=int)
    
    # Infark Miokard (Elevasi ST / Q dalam / T inversi)
    X_mi = np.column_stack([
        np.random.normal(-0.22, 0.08, n_samples_per_class),
        np.random.normal(0.40, 0.18, n_samples_per_class),
        np.random.normal(0.28, 0.12, n_samples_per_class),
        np.random.normal(-0.25, 0.10, n_samples_per_class)
    ])
    y_mi = np.ones(n_samples_per_class, dtype=int)
    
    # Unknown (Aritmia, Kardiomiopati, n/a)
    X_unk = np.column_stack([
        np.random.normal(-0.05, 0.05, n_samples_per_class),
        np.random.normal(0.75, 0.25, n_samples_per_class),
        np.random.normal(-0.08, 0.07, n_samples_per_class),
        np.random.normal(0.10, 0.15, n_samples_per_class)
    ])
    y_unk = np.full(n_samples_per_class, 2, dtype=int)
    
    X = np.vstack([X_norm, X_mi, X_unk])
    y = np.concatenate([y_norm, y_mi, y_unk])
    return X.astype(np.float32), y


def export_weights_to_hardware(model: ECG_ANN, scaler: StandardScaler, output_dir: str = "models", fractional_bits: int = 12):
    """
    Mengekspor bobot dan bias ke format Fixed-Point (Q4.12) untuk implementasi RTL FPGA:
    1. C Header (`weights.h`) untuk testbench / microcontroller
    2. Verilog RAM Memory file (`weights.mem`)
    3. JSON file parameter lengkap
    """
    os.makedirs(output_dir, exist_ok=True)
    scale_factor = 1 << fractional_bits
    
    w1 = model.fc1.weight.detach().cpu().numpy()  # [8, 4]
    b1 = model.fc1.bias.detach().cpu().numpy()    # [8]
    w2 = model.fc2.weight.detach().cpu().numpy()  # [3, 8]
    b2 = model.fc2.bias.detach().cpu().numpy()    # [3]
    
    # Kuantisasi ke Fixed-Point 16-bit
    w1_fp = np.round(w1 * scale_factor).astype(np.int16)
    b1_fp = np.round(b1 * scale_factor).astype(np.int16)
    w2_fp = np.round(w2 * scale_factor).astype(np.int16)
    b2_fp = np.round(b2 * scale_factor).astype(np.int16)
    
    # Simpan JSON
    export_dict = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "fractional_bits": fractional_bits,
        "W1_float": w1.tolist(),
        "b1_float": b1.tolist(),
        "W2_float": w2.tolist(),
        "b2_float": b2.tolist(),
        "W1_fixed": w1_fp.tolist(),
        "b1_fixed": b1_fp.tolist(),
        "W2_fixed": w2_fp.tolist(),
        "b2_fixed": b2_fp.tolist(),
    }
    with open(os.path.join(output_dir, "ann_model_params.json"), "w") as f:
        json.dump(export_dict, f, indent=2)
        
    # Simpan C Header (weights.h)
    header_path = os.path.join(output_dir, "weights.h")
    with open(header_path, "w") as f:
        f.write("/* Auto-generated ANN Weights for ECG Classifier (FPGA / MCU) */\n")
        f.write(f"#define Q_FRACTIONAL_BITS {fractional_bits}\n")
        f.write("#define INPUT_DIM 4\n#define HIDDEN_DIM 8\n#define NUM_CLASSES 3\n\n")
        
        f.write(f"const short W1[HIDDEN_DIM][INPUT_DIM] = {{\n")
        for row in w1_fp:
            f.write("    {" + ", ".join(f"{v}" for v in row) + "},\n")
        f.write("};\n\n")
        
        f.write(f"const short b1[HIDDEN_DIM] = {{" + ", ".join(f"{v}" for v in b1_fp) + "};\n\n")
        
        f.write(f"const short W2[NUM_CLASSES][HIDDEN_DIM] = {{\n")
        for row in w2_fp:
            f.write("    {" + ", ".join(f"{v}" for v in row) + "},\n")
        f.write("};\n\n")
        
        f.write(f"const short b2[NUM_CLASSES] = {{" + ", ".join(f"{v}" for v in b2_fp) + "};\n")
        
    # Simpan Verilog mem file (weights.mem)
    mem_path = os.path.join(output_dir, "weights.mem")
    with open(mem_path, "w") as f:
        f.write("// Verilog ROM Initialization (16-bit hex, 2's complement)\n")
        all_weights = list(w1_fp.flatten()) + list(b1_fp) + list(w2_fp.flatten()) + list(b2_fp)
        for val in all_weights:
            u16 = int(val) & 0xFFFF
            f.write(f"{u16:04X}\n")
            
    print(f"\n[Hardware Export Q4.12 Berhasil]")
    print(f"  -> JSON: {os.path.join(output_dir, 'ann_model_params.json')}")
    print(f"  -> C Header: {header_path}")
    print(f"  -> Verilog ROM: {mem_path}")


def export_int8_model(model: ECG_ANN, scaler: StandardScaler, X_ref: np.ndarray, output_dir: str = "models") -> Dict:
    """
    Kuantisasi Post-Training (PTQ) model ANN ke format INT8 murni:
    - Masukan x: int8 [-128, 127]
    - Bobot W1: int8 [-128, 127]
    - Bias b1: int32 (skala input * bobot)
    - Hidden Layer: ReLU + kuantisasi int8 [0, 127]
    - Bobot W2: int8 [-128, 127]
    - Bias b2: int32 (skala hidden * bobot)
    - Output Skor/Logits: int8 [-128, 127] untuk masing-masing kelas (Normal, Infark Miokard, Unknown)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ambil bobot float32
    w1 = model.fc1.weight.detach().cpu().numpy() # [8, 4]
    b1 = model.fc1.bias.detach().cpu().numpy()   # [8]
    w2 = model.fc2.weight.detach().cpu().numpy() # [3, 8]
    b2 = model.fc2.bias.detach().cpu().numpy()   # [3]
    
    # 1. Skala kuantisasi simetris
    scale_x = float(np.max(np.abs(X_ref)) / 127.0) if np.max(np.abs(X_ref)) > 0 else 1.0
    scale_w1 = float(np.max(np.abs(w1)) / 127.0)
    scale_w2 = float(np.max(np.abs(w2)) / 127.0)
    
    w1_int8 = np.clip(np.round(w1 / scale_w1), -128, 127).astype(np.int8)
    w2_int8 = np.clip(np.round(w2 / scale_w2), -128, 127).astype(np.int8)
    
    # Skala bias layer 1
    scale_b1 = scale_x * scale_w1
    b1_int32 = np.round(b1 / scale_b1).astype(np.int32)
    
    # Estimasi skala aktivasi tersembunyi menggunakan data referensi
    X_ref_int8 = np.clip(np.round(X_ref / scale_x), -128, 127).astype(np.int32)
    acc1_ref = np.dot(X_ref_int8, w1_int8.T.astype(np.int32)) + b1_int32
    h1_relu_ref = np.maximum(0, acc1_ref)
    max_h1 = float(np.max(h1_relu_ref))
    scale_h1 = (max_h1 / 127.0) if max_h1 > 0 else 1.0
    
    # Skala bias layer 2
    scale_b2 = scale_h1 * scale_w2
    b2_int32 = np.round(b2 / scale_b2).astype(np.int32)
    
    # Estimasi skala output logits INT8
    h1_int8_ref = np.clip(np.round(h1_relu_ref / scale_h1), 0, 127).astype(np.int32)
    acc2_ref = np.dot(h1_int8_ref, w2_int8.T.astype(np.int32)) + b2_int32
    max_out = float(np.max(np.abs(acc2_ref)))
    scale_out = (max_out / 127.0) if max_out > 0 else 1.0
    
    # Simpan JSON parameter INT8
    int8_params = {
        "format": "INT8_SYMMETRIC",
        "description": "ECG ANN Classifier Model in INT8 for FPGA / MCU",
        "input_dim": 4,
        "hidden_dim": 8,
        "num_classes": 3,
        "classes": CLASS_NAMES,
        "scales": {
            "scale_x": scale_x,
            "scale_w1": scale_w1,
            "scale_b1": scale_b1,
            "scale_h1": scale_h1,
            "scale_w2": scale_w2,
            "scale_b2": scale_b2,
            "scale_out": scale_out
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "W1_int8": w1_int8.tolist(),
        "b1_int32": b1_int32.tolist(),
        "W2_int8": w2_int8.tolist(),
        "b2_int32": b2_int32.tolist()
    }
    json_path = os.path.join(output_dir, "ann_model_int8_params.json")
    with open(json_path, "w") as f:
        json.dump(int8_params, f, indent=2)
        
    # Simpan C Header (weights_int8.h)
    c_header_path = os.path.join(output_dir, "weights_int8.h")
    with open(c_header_path, "w") as f:
        f.write("/* Auto-generated INT8 Quantized Weights & Inference for ECG Classifier (FPGA / MCU) */\n")
        f.write("#ifndef WEIGHTS_INT8_H\n#define WEIGHTS_INT8_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#define INPUT_DIM 4\n#define HIDDEN_DIM 8\n#define NUM_CLASSES 3\n\n")
        f.write(f"// Skala Kuantisasi\n")
        f.write(f"#define SCALE_X_INV   {1.0/scale_x:.6f}f\n")
        f.write(f"#define SCALE_H1_INV  {1.0/scale_h1:.6f}f\n")
        f.write(f"#define SCALE_OUT_INV {1.0/scale_out:.6f}f\n\n")
        
        f.write("// Bobot Layer 1 [8][4] (INT8)\n")
        f.write("static const int8_t W1_INT8[HIDDEN_DIM][INPUT_DIM] = {\n")
        for row in w1_int8:
            f.write("    {" + ", ".join(f"{int(v)}" for v in row) + "},\n")
        f.write("};\n\n")
        
        f.write("// Bias Layer 1 [8] (INT32)\n")
        f.write("static const int32_t b1_INT32[HIDDEN_DIM] = {" + ", ".join(f"{int(v)}" for v in b1_int32) + "};\n\n")
        
        f.write("// Bobot Layer 2 [3][8] (INT8)\n")
        f.write("static const int8_t W2_INT8[NUM_CLASSES][HIDDEN_DIM] = {\n")
        for row in w2_int8:
            f.write("    {" + ", ".join(f"{int(v)}" for v in row) + "},\n")
        f.write("};\n\n")
        
        f.write("// Bias Layer 2 [3] (INT32)\n")
        f.write("static const int32_t b2_INT32[NUM_CLASSES] = {" + ", ".join(f"{int(v)}" for v in b2_int32) + "};\n\n")
        
        f.write("""/**
 * Inferensi INT8 murni: Mengembalikan output skor logits INT8 untuk 3 kelas:
 * out_scores[0]: Skor Normal
 * out_scores[1]: Skor Infark Miokard
 * out_scores[2]: Skor Unknown
 * Mengembalikan indeks kelas dengan skor tertinggi (0, 1, atau 2).
 */
static inline uint8_t predict_ecg_int8(const int8_t x_in[INPUT_DIM], int8_t out_scores[NUM_CLASSES]) {
    int32_t h_acc[HIDDEN_DIM];
    int8_t h_int8[HIDDEN_DIM];
    
    // Layer 1: H = ReLU(W1 * x + b1)
    for (int i = 0; i < HIDDEN_DIM; i++) {
        int32_t acc = b1_INT32[i];
        for (int j = 0; j < INPUT_DIM; j++) {
            acc += (int32_t)W1_INT8[i][j] * (int32_t)x_in[j];
        }
        if (acc < 0) acc = 0; // ReLU
        // Re-kuantisasi ke INT8 [0, 127]
        int32_t scaled_h = (int32_t)(acc * SCALE_H1_INV);
        if (scaled_h > 127) scaled_h = 127;
        h_int8[i] = (int8_t)scaled_h;
    }
    
    // Layer 2: Out = W2 * H + b2
    uint8_t best_class = 0;
    int32_t max_score = -2147483647;
    for (int k = 0; k < NUM_CLASSES; k++) {
        int32_t acc = b2_INT32[k];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            acc += (int32_t)W2_INT8[k][i] * (int32_t)h_int8[i];
        }
        // Skalakan output menjadi skor INT8 [-128, 127]
        int32_t scaled_out = (int32_t)(acc * SCALE_OUT_INV);
        if (scaled_out > 127) scaled_out = 127;
        if (scaled_out < -128) scaled_out = -128;
        out_scores[k] = (int8_t)scaled_out;
        
        if (scaled_out > max_score) {
            max_score = scaled_out;
            best_class = k;
        }
    }
    return best_class;
}

#endif // WEIGHTS_INT8_H
""")

    # Simpan Verilog Memory File 8-bit hex (weights_int8.mem)
    mem_path = os.path.join(output_dir, "weights_int8.mem")
    with open(mem_path, "w") as f:
        f.write("// Verilog ROM 8-bit Hex (2 hex chars per byte)\n")
        f.write("// W1 [8x4 = 32 bytes]\n")
        for val in w1_int8.flatten():
            f.write(f"{(int(val) & 0xFF):02X}\n")
        f.write("// W2 [3x8 = 24 bytes]\n")
        for val in w2_int8.flatten():
            f.write(f"{(int(val) & 0xFF):02X}\n")
            
    print(f"\n[Model INT8 Hardware Export Berhasil]")
    print(f"  -> JSON INT8: {json_path}")
    print(f"  -> C Header INT8: {c_header_path}")
    print(f"  -> Verilog ROM 8-bit: {mem_path}")
    
    return int8_params


def train_and_evaluate(X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 16, lr: float = 0.005):
    """
    Pelatihan dan evaluasi lengkap model ANN PyTorch.
    """
    print(f"\nTotal data detak ECG: {len(X)} sampel")
    for idx, cname in enumerate(CLASS_NAMES):
        count = np.sum(y == idx)
        print(f"  Kelas {idx} ({cname}): {count} sampel ({count/len(y)*100:.1f}%)")
        
    # Split Train / Val / Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Standarisasi
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # DataLoader
    train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_s, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Model
    model = ECG_ANN(input_dim=4, hidden_dim=8, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print("\n--- Memulai Pelatihan ANN ---")
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(by)
            preds = torch.argmax(out, dim=1)
            correct += (preds == by).sum().item()
            total += len(by)
            
        train_acc = correct / total
        
        # Validasi
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == by).sum().item()
                val_total += len(by)
        val_acc = val_correct / val_total
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Loss: {total_loss/total:.4f} - Train Acc: {train_acc*100:.2f}% - Val Acc: {val_acc*100:.2f}%")
            
    # Evaluasi Test Set
    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test_s, dtype=torch.float32)
        test_out = model(test_x)
        test_preds = torch.argmax(test_out, dim=1).numpy()
        
    print("\n=== HASIL EVALUASI TEST SET ===")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4))
    
    cm = confusion_matrix(y_test, test_preds)
    print("Confusion Matrix:\n", cm)
    
    # Simpan model PyTorch
    torch.save(model.state_dict(), "models/ecg_ann_model.pth")
    print("Model weights disimpan di: models/ecg_ann_model.pth")
    
    # Ekspor bobot Fixed-Point Q4.12 untuk FPGA RTL
    export_weights_to_hardware(model, scaler, output_dir="models", fractional_bits=12)
    
    # Ekspor model INT8 murni (input, bobot, dan output INT8)
    int8_params = export_int8_model(model, scaler, X_train_s, output_dir="models")
    
    # Evaluasi INT8 pada Test Set
    scale_x = int8_params["scales"]["scale_x"]
    scale_w1 = int8_params["scales"]["scale_w1"]
    scale_h1 = int8_params["scales"]["scale_h1"]
    scale_w2 = int8_params["scales"]["scale_w2"]
    scale_out = int8_params["scales"]["scale_out"]
    
    w1_i8 = np.array(int8_params["W1_int8"], dtype=np.int32)
    b1_i32 = np.array(int8_params["b1_int32"], dtype=np.int32)
    w2_i8 = np.array(int8_params["W2_int8"], dtype=np.int32)
    b2_i32 = np.array(int8_params["b2_int32"], dtype=np.int32)
    
    # Forward pass INT8
    X_test_i8 = np.clip(np.round(X_test_s / scale_x), -128, 127).astype(np.int32)
    acc1 = np.dot(X_test_i8, w1_i8.T) + b1_i32
    h1_relu = np.maximum(0, acc1)
    h1_i8 = np.clip(np.round(h1_relu / scale_h1), 0, 127).astype(np.int32)
    acc2 = np.dot(h1_i8, w2_i8.T) + b2_i32
    out_scores_i8 = np.clip(np.round(acc2 / scale_out), -128, 127).astype(np.int8)
    preds_int8 = np.argmax(out_scores_i8, axis=1)
    
    acc_int8 = accuracy_score(y_test, preds_int8)
    print(f"\n=== EVALUASI MODEL INT8 PADA TEST SET ===")
    print(f"Akurasi Model INT8: {acc_int8*100:.2f}% (Float32: {accuracy_score(y_test, test_preds)*100:.2f}%)")
    print(classification_report(y_test, preds_int8, target_names=CLASS_NAMES, digits=4))
    
    return model, scaler


if __name__ == "__main__":
    X, y, df = collect_dataset_from_local("data/ptbdb")
    # Jika kelas belum seimbang (karena unduhan di latar belakang masih berjalan), tambahkan data sintetis pendukung
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 3 or min(counts) < 30:
        print("Menambahkan data augmentasi agar ketiga kelas seimbang dan representatif...")
        X_syn, y_syn = generate_synthetic_data(n_samples_per_class=150)
        X = np.vstack([X, X_syn])
        y = np.concatenate([y, y_syn])
        
    train_and_evaluate(X, y, epochs=80, batch_size=16, lr=0.005)
