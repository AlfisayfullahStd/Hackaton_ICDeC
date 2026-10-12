/* Auto-generated INT8 Quantized Weights & Inference for ECG Classifier (FPGA / MCU) */
#ifndef WEIGHTS_INT8_H
#define WEIGHTS_INT8_H

#include <stdint.h>

#define INPUT_DIM 4
#define HIDDEN_DIM 8
#define NUM_CLASSES 3

// Skala Kuantisasi
#define SCALE_X_INV   22.459542f
#define SCALE_H1_INV  0.008631f
#define SCALE_OUT_INV 0.010019f

// Bobot Layer 1 [8][4] (INT8)
static const int8_t W1_INT8[HIDDEN_DIM][INPUT_DIM] = {
    {-30, -103, 55, -99},
    {-100, 52, 53, 18},
    {-103, 43, 59, -127},
    {-108, -84, -34, 3},
    {74, 89, -51, -15},
    {-62, 26, 39, -34},
    {48, 69, 30, 39},
    {-5, -60, -64, 87},
};

// Bias Layer 1 [8] (INT32)
static const int32_t b1_INT32[HIDDEN_DIM] = {-187, 1460, 1419, -242, 322, 2181, 991, -932};

// Bobot Layer 2 [3][8] (INT8)
static const int8_t W2_INT8[NUM_CLASSES][HIDDEN_DIM] = {
    {-127, 21, -123, 26, 16, 68, 18, -88},
    {-9, -79, 64, 48, -36, -10, 30, 68},
    {55, 43, -10, -22, 26, -36, -66, -6},
};

// Bias Layer 2 [3] (INT32)
static const int32_t b2_INT32[NUM_CLASSES] = {0, 0, 0};

/**
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
