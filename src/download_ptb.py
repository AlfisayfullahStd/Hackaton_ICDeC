"""
Script untuk mengunduh dataset PTB Diagnostic ECG Database dari PhysioNet.
Mendukung unduhan mode cepat (subset seimbang: Normal, Infark Miokard, Unknown)
maupun mode unduhan lengkap.
"""

import os
import urllib.request
import argparse
import pandas as pd
from typing import List, Dict

BASE_URL = "https://physionet.org/files/ptbdb/1.0.0/"

# Default representative records for quick experimentation
DEFAULT_BALANCED_RECORDS = {
    "Normal": [
        "patient104/s0306lre",
        "patient117/s0291lre",
        "patient117/s0292lre",
        "patient121/s0311lre",
        "patient122/s0312lre",
        "patient131/s0273lre",
    ],
    "Infark Miokard": [
        "patient001/s0010_re",
        "patient120/s0331lre",
        "patient128/s0182_re",
        "patient135/s0334lre",
        "patient138/s0005_re",
        "patient139/s0223_re",
    ],
    "Unknown": [
        "patient118/s0183_re",  # Heart failure (NYHA 3)
        "patient119/s0001_re",  # n/a (unclassified)
        "patient123/s0224_re",  # Heart failure (NYHA 4)
        "patient125/s0006_re",  # n/a (unclassified)
        "patient126/s0154_re",  # Palpitation
        "patient285/s0544_re",  # n/a (unclassified)
    ]
}


def download_file(url: str, dest_path: str):
    """Mengunduh sebuah file jika belum ada secara lokal."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return
    print(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)


def download_record(record_path: str, output_dir: str):
    """Mengunduh file .hea dan .dat untuk sebuah record."""
    for ext in [".hea", ".dat"]:
        url = f"{BASE_URL}{record_path}{ext}"
        dest = os.path.join(output_dir, f"{record_path}{ext}")
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"Peringatan: Gagal mengunduh {url}: {e}")


def parse_diagnosis_from_header(header_file: str) -> str:
    """Membaca diagnosis dari komentar di file .hea."""
    if not os.path.exists(header_file):
        return "Unknown"
    with open(header_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Reason for admission:" in line:
                diag = line.split(":", 1)[1].strip()
                return diag
    return "Unknown"


def map_diagnosis_to_class(diagnosis: str) -> str:
    """Memetakan alasan admisi medis ke 3 kelas target arsitektur."""
    d_lower = diagnosis.lower()
    if "healthy" in d_lower or "control" in d_lower:
        return "Normal"
    elif "myocardial infarction" in d_lower or "infarction" in d_lower:
        return "Infark Miokard"
    else:
        return "Unknown"


def download_balanced_dataset(output_dir: str = "data/ptbdb") -> pd.DataFrame:
    """Mengunduh kumpulan record seimbang untuk 3 kelas."""
    manifest = []
    print("=== Mulai Mengunduh Subset Seimbang PTB-DB ===")
    for expected_class, rec_list in DEFAULT_BALANCED_RECORDS.items():
        print(f"\n--- Mengunduh Kelas: {expected_class} ({len(rec_list)} records) ---")
        for rec in rec_list:
            download_record(rec, output_dir)
            hea_path = os.path.join(output_dir, f"{rec}.hea")
            diag = parse_diagnosis_from_header(hea_path)
            mapped = map_diagnosis_to_class(diag)
            manifest.append({
                "record": rec,
                "raw_diagnosis": diag,
                "label": mapped
            })
            print(f"  [OK] {rec}: {diag} => Label: {mapped}")

    df_manifest = pd.DataFrame(manifest)
    manifest_csv = os.path.join(output_dir, "records_manifest.csv")
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"\nManifest berhasil disimpan di: {manifest_csv}")
    print(df_manifest["label"].value_counts())
    return df_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloader PTB Diagnostic Database")
    parser.add_argument("--output_dir", type=str, default="data/ptbdb", help="Direktori penyimpanan data")
    args = parser.parse_args()

    download_balanced_dataset(args.output_dir)
