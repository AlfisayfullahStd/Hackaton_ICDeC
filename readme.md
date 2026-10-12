# Klasifikasi ECG Berbasis Artificial Neural Network (ANN) INT8 dengan 4 Fitur Masukan QRST
### Kompetisi Desain Rangkaian Terintegrasi (Hackathon ICDeC)

Repository ini berisi implementasi sistem deteksi dan klasifikasi sinyal Elektrokardiogram (ECG) untuk mendeteksi **Normal**, **Infark Miokard (Myocardial Infarction)**, dan **Unknown / Unclassified** berbasis **Artificial Neural Network (ANN / MLP)** dengan masukan **4 fitur morfologi QRST**. Model telah dikuantisasi penuh ke format **INT8 (8-bit Signed Integer)** dan siap disintesis ke perangkat keras **FPGA**.

---

## 1. Arsitektur Sistem Sesuai Diagram Blok

Arsitektur sistem dibagi menjadi dua bagian utama: **Frontend Pemrosesan Sinyal & Ekstraksi Fitur** dan **Backend Klasifikasi ANN (MLP)**.

```
                                  DIAGRAM ALIRAN DATA & KONTROL
                                  
    Sinyal ECG (Lead II)
             │
             ▼
    ┌─────────────────┐
    │ Bandpass Filter │ (Butterworth 0.5 - 45 Hz)
    └────────┬────────┘
             │
             ├───────────────────────────────┐
             ▼ (Data)                        ▼ (Data)
      ┌────────────┐                  ┌──────────────┐
      │ Detektor R │ ───(Kontrol)───> │Shift Register│ (Delay Buffer Pencarian)
      └────────────┘                  └──────┬───────┘
                                             │
                         ┌───────────────────┼───────────────────┐
                         ▼                   ▼                   ▼
                  ┌────────────┐      ┌────────────┐      ┌────────────┐
                  │ Detektor Q │      │ Detektor S │      │ Detektor T │
                  └─────┬──────┘      └─────┬──────┘      └─────┬──────┘
                        │                   │                   │
                        └─────────────┬─────┴───────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   Ekstraksi Fitur QRST    │
                        └─────────────┬─────────────┘
                                      │
               ┌──────────────────────┼──────────────────────┐
               ▼ (Fitur 1)            ▼ (Fitur 2)            ▼ (Fitur 3)            ▼ (Fitur 4)
              x1                     x2                     x3                     x4
        (Amplitudo Q)          (Amplitudo R)           (Elevasi ST)           (Amplitudo T)
               │                      │                      │                      │
               └──────────────────────┼──────────────────────┴──────────────────────┘
                                      ▼
               ┌────────────────────────────────────────────────────────────┐
               │              ANN (MLP) Classifier (INT8)                   │
               │        Input: 4 -> Hidden: 8 (ReLU) -> Output: 3          │
               └──────────────────────┬─────────────────────────────────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   ▼                  ▼                  ▼
             Skor Normal        Skor Infark MI      Skor Unknown
               (INT8)               (INT8)             (INT8)
                   │                  │                  │
                   └──────────────────┼──────────────────┘
                                      ▼ (Argmax)
                       [Keputusan Klasifikasi Akhir]
```

### Integrasi Sistem pada Perangkat Keras FPGA (Diagram Bawah)
```
┌─────────────────────┐      ┌──────┐      ┌─────────────────────────┐      ┌──────────┐      ┌─────────────┐
│ Input Generator ECG │ ───> │ UART │ ───> │ Filter & Ekstraksi Fitur│ ───> │ MLP INT8 │ ───> │ Klasifikasi │
└─────────────────────┘      └──┬───┘      └────────────┬────────────┘      └────┬─────┘      └──────┬──────┘
                                ▲                       │                        │                   │
                                └───────────────────────┴────────────────────────┴───────────────────┘ (ke GUI)
```

---

## 2. Definisi Matematis & Klinis 4 Fitur Masukan QRST ($x_1, x_2, x_3, x_4$)

Sesuai instruksi spesifikasi perancangan: *"ANN dimulai dari masukan inputnya"*, modul ANN menerima 4 fitur morfologi jantung:

$$\mathbf{x} = [x_1, x_2, x_3, x_4]^T$$

| Fitur | Simbol | Definisi Matematis / Klinis | Nilai Tipikal Normal | Nilai Tipikal Infark Miokard | Signifikansi Medis |
|---|---|---|---|---|---|
| **Fitur 1** | $x_1$ | **Amplitudo Q ($V_Q - V_{iso}$)** | $-0.05 \text{ s.d. } -0.09\text{ mV}$ | $-0.15 \text{ s.d. } -0.30\text{ mV}$ | **Gelombang Q Patologis**: Q yang lebar dan dalam menandakan nekrosis permanen otot ventrikel akibat infark miokard. |
| **Fitur 2** | $x_2$ | **Amplitudo Puncak R ($V_R - V_{iso}$)** | $+1.00 \text{ s.d. } +1.30\text{ mV}$ | $+0.20 \text{ s.d. } +0.50\text{ mV}$ | **Penurunan Voltase R (*Loss of R-wave*)**: Penurunan amplitudo akibat hilangnya massa sel miokardium yang mampu berdepolarisasi. |
| **Fitur 3** | $x_3$ | **Elevasi/Deviasi Segmen ST ($V_{ST} - V_{iso}$)** | $-0.02 \text{ s.d. } +0.05\text{ mV}$ | $+0.20 \text{ s.d. } +0.50\text{ mV}$ | **STEMI (*ST-Elevation Myocardial Infarction*)**: Deviasi segmen ST (diukur pada titik J + 60 ms) merupakan kriteria utama iskemia akut. |
| **Fitur 4** | $x_4$ | **Amplitudo Gelombang T ($V_T - V_{iso}$)** | $+0.25 \text{ s.d. } +0.45\text{ mV}$ | $-0.20 \text{ s.d. } -0.40\text{ mV}$ | **Inversi Gelombang T**: Gelombang T terbalik (negatif) menunjukkan gangguan repolarisasi pada jaringan jantung iskemik. |

*Catatan: $V_{iso}$ adalah level tegangan garis isoelektrik yang diestimasi dari interval PR (50 ms sebelum gelombang Q).*

---

## 3. Skema Kuantisasi Model ANN ke INT8

Untuk efisiensi perangkat keras pada FPGA, semua masukan, bobot, bias, dan output dikuantisasi ke format **INT8 simetris** (rentang $[-128, 127]$):

### 3.1 Formulasi Matematis Kuantisasi (Post-Training Quantization / PTQ)

1. **Kuantisasi Masukan ($X_{int8}$)**:
   $$X_{int8} = \text{clip}\left(\left\lfloor \frac{X}{S_x} \right\rceil, -128, 127\right), \quad S_x = \frac{\max(|X|)}{127}$$

2. **Kuantisasi Bobot Layer 1 ($W_{1, int8}$)**:
   $$W_{1, int8} = \text{clip}\left(\left\lfloor \frac{W_1}{S_{w1}} \right\rceil, -128, 127\right), \quad S_{w1} = \frac{\max(|W_1|)}{127}$$

3. **Bias Layer 1 ($b_{1, int32}$)**:
   $$b_{1, int32} = \left\lfloor \frac{b_1}{S_x \cdot S_{w1}} \right\rceil$$

4. **Aktivasi Tersembunyi (ReLU) & Rekuantisasi ke INT8 ($H_{int8}$)**:
   $$\text{Acc}_1 = W_{1, int8} \cdot X_{int8} + b_{1, int32} \quad (\text{Akumulator 32-bit})$$
   $$H_{relu} = \max(0, \text{Acc}_1)$$
   $$H_{int8} = \text{clip}\left(\left\lfloor \frac{H_{relu}}{S_h} \right\rceil, 0, 127\right), \quad S_h = \frac{\max(H_{relu})}{127}$$

5. **Kuantisasi Bobot Layer 2 ($W_{2, int8}$)**:
   $$W_{2, int8} = \text{clip}\left(\left\lfloor \frac{W_2}{S_{w2}} \right\rceil, -128, 127\right), \quad S_{w2} = \frac{\max(|W_2|)}{127}$$

6. **Bias Layer 2 ($b_{2, int32}$)**:
   $$b_{2, int32} = \left\lfloor \frac{b_2}{S_h \cdot S_{w2}} \right\rceil$$

7. **Output Skor Logits INT8 ($\mathbf{Y}_{out, int8}$)**:
   $$\text{Acc}_2 = W_{2, int8} \cdot H_{int8} + b_{2, int32}$$
   $$\mathbf{Y}_{out, int8} = \text{clip}\left(\left\lfloor \frac{\text{Acc}_2}{S_{out}} \right\rceil, -128, 127\right), \quad S_{out} = \frac{\max(|\text{Acc}_2|)}{127}$$

Output model adalah vektor 3 elemen INT8:
$$\mathbf{Y}_{out, int8} = [y_{Normal}, y_{MI}, y_{Unknown}]^T \in [-128, 127]^3$$
Hasil prediksi ditentukan dengan fungsi $\text{argmax}$:
$$\text{Kelas} = \operatorname{argmax}_{k \in \{0, 1, 2\}} (\mathbf{Y}_{out, int8}[k])$$

---

## 4. Evaluasi & Performa Model

Model dilatih dan dievaluasi menggunakan **1.694 detak ECG nyata** dari 18 rekaman pasien PTB Diagnostic Database:
- **Kelas 0 (Normal)**: 865 sampel (51,1%)
- **Kelas 1 (Infark Miokard)**: 432 sampel (25,5%)
- **Kelas 2 (Unknown)**: 397 sampel (23,4%)

### Hasil Evaluasi pada Test Set Independen (255 detak uji):

| Metrik | Model Float32 (PyTorch) | Model INT8 (Kuantisasi Hardware) |
|---|---|---|
| **Akurasi Keseluruhan** | **97,65%** | **95,69%** |
| **Precision (Normal)** | 100,00% | 100,00% |
| **Recall (Normal)** | 98,46% | 98,46% |
| **Precision (Infark Miokard)** | 100,00% | 87,67% |
| **Recall (Infark Miokard)** | 93,85% | 98,46% |
| **Precision (Unknown)** | 90,91% | 96,30% |
| **Recall (Unknown)** | 100,00% | 86,67% |

### Confusion Matrix Model INT8:
```
                Prediksi: Normal   Prediksi: MI   Prediksi: Unknown
Aktual: Normal         128               0               2
Aktual: MI               0              64               1
Aktual: Unknown          0               8              52
```

---

## 5. Struktur Direktori Repository

```
Hackaton_ICDeC/
├── assets/                          # Diagram sistem & referensi grafis
├── data/
│   ├── download_dataset.txt         # Referensi URL PhysioNet PTB Database
│   └── ptbdb/                       # Direktori penyimpanan rekaman sinyal ECG
│       ├── patient001/ ...          # File .hea dan .dat pasien MI
│       ├── patient104/ ...          # File .hea dan .dat pasien Normal
│       ├── patient118/ ...          # File .hea dan .dat pasien Unknown
│       └── records_manifest.csv     # Metadata diagnosis dan label 3 kelas
├── docs/                            # Paper referensi morfologi ECG & deteksi MI
├── models/
│   ├── ann_model_int8_params.json   # Parameter skala & bobot INT8 lengkap
│   ├── ann_model_params.json        # Parameter model floating-point & Q4.12
│   ├── ecg_ann_model.pth            # Checkpoint bobot PyTorch
│   ├── weights_int8.h               # C Header bobot & fungsi inferensi INT8
│   ├── weights_int8.mem             # File ROM hex Verilog 8-bit ($readmemh)
│   ├── weights.h                    # C Header Fixed-Point Q4.12
│   └── weights.mem                  # File ROM hex Verilog 16-bit
├── notebook/
│   └── ann_ecg_classification.ipynb # Jupyter Notebook utama (lengkap ter-render)
├── rtl/
│   └── ann_mlp_int8.v               # Modul Verilog RTL synthesizable akselerator INT8
├── src/
│   ├── build_notebook.py            # Generator & runner notebook otomatis
│   ├── download_ptb.py              # Downloader dataset PTB PhysioNet
│   ├── ecg_features.py              # Modul filter & ekstraksi 4 fitur QRST
│   └── train_ann.py                 # Skrip pelatihan ANN & ekspor INT8
├── readme.md                        # Dokumentasi komprehensif proyek
└── requirements.txt                 # Dependensi pustaka Python
```

---

## 6. Panduan Menjalankan Sistem

### A. Membuka Jupyter Notebook
Buka dan jalankan seluruh alur pemrosesan di:
```powershell
jupyter notebook notebook/ann_ecg_classification.ipynb
```
Notebook telah memuat visualisasi:
1. Sinyal ECG mentah vs sinyal terfilter (*Butterworth Bandpass*).
2. Deteksi titik fidusial Q, R, S, T pada satu siklus detak jantung.
3. Sebaran fitur (Scatter plot & Boxplot 4 fitur).
4. Kurva pelatihan Loss dan Akurasi per Epoch.
5. Confusion Matrix dan Classification Report (Float32 & INT8).
6. Contoh output skor logits INT8 untuk masing-masing kelas.

### B. Menjalankan Pelatihan & Ekspor INT8 via Terminal
```powershell
python src/train_ann.py
```
Perintah ini akan membaca record lokal, mengekstrak 4 fitur, melatih model ANN, melakukan kuantisasi INT8, dan memperbarui file bobot di `models/`.

### C. Mengunduh Tambahan Data Pasien PTB-DB
```powershell
python src/download_ptb.py --output_dir data/ptbdb
```

---

## 7. Implementasi Hardware FPGA (RTL Verilog)

File modul akselerator Verilog telah disediakan di:
`rtl/ann_mlp_int8.v`

### Antarmuka Modul Verilog:
```verilog
module ann_mlp_int8 (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    
    // 4 Masukan Fitur QRST (INT8: [-128, 127])
    input  wire signed [7:0] in_x1,        // Fitur 1: Amplitudo Q
    input  wire signed [7:0] in_x2,        // Fitur 2: Amplitudo R
    input  wire signed [7:0] in_x3,        // Fitur 3: Elevasi ST
    input  wire signed [7:0] in_x4,        // Fitur 4: Amplitudo T
    
    // 3 Output Skor Logits (INT8: [-128, 127])
    output reg  signed [7:0] score_normal,  // Skor Kelas Normal
    output reg  signed [7:0] score_mi,      // Skor Kelas Infark Miokard
    output reg  signed [7:0] score_unknown, // Skor Kelas Unknown
    
    // Keputusan Klasifikasi
    output reg  [1:0]        class_pred,    // 0: Normal, 1: Infark MI, 2: Unknown
    output reg               done
);
```

### Inisialisasi Memori Bobot pada FPGA:
File `models/weights_int8.mem` dapat langsung dihubungkan ke modul ROM FPGA menggunakan pengarah sintesis Verilog:
```verilog
$readmemh("../models/weights_int8.mem", rom_weights);
```

---

## 8. Lisensi & Sitasi
- **Dataset**: PhysioNet PTB Diagnostic ECG Database (Bousseljot, R., Kreiseler, D., Schnabel, A. *PhysioBank*, PhysioNet).
- **Format Target**: Hackathon ICDeC (Integrated Circuit Design Competition).
