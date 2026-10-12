// ============================================================================
// Modul: ann_mlp_int8.v
// Deskripsi: Akselerator Hardware Artificial Neural Network (MLP) INT8
// Arsitektur: 4 Masukan Fitur QRST -> 8 Neuron Hidden (ReLU) -> 3 Output Kelas (INT8)
// Target: Sintesis FPGA (Hackathon ICDeC)
// Kelas: 2'b00: Normal, 2'b01: Infark Miokard, 2'b10: Unknown
// ============================================================================

`timescale 1ns / 1ps

module ann_mlp_int8 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    
    // 4 Masukan Fitur QRST (INT8: [-128, 127])
    input  wire signed [7:0] in_x1, // Fitur 1: Q-wave amplitude
    input  wire signed [7:0] in_x2, // Fitur 2: R-wave peak amplitude
    input  wire signed [7:0] in_x3, // Fitur 3: ST-segment elevation/deviation
    input  wire signed [7:0] in_x4, // Fitur 4: T-wave amplitude
    
    // 3 Output Skor Logits (INT8: [-128, 127])
    output reg  signed [7:0] score_normal,
    output reg  signed [7:0] score_mi,
    output reg  signed [7:0] score_unknown,
    
    // Output Keputusan Klasifikasi
    output reg  [1:0]  class_pred, // 0: Normal, 1: Infark Miokard, 2: Unknown
    output reg         done
);

    // State Machine
    localparam IDLE      = 2'd0;
    localparam LAYER1    = 2'd1;
    localparam LAYER2    = 2'd2;
    localparam FINISH    = 2'd3;

    reg [1:0] state;
    reg [3:0] step;

    // Register masukan
    reg signed [7:0] x[0:3];

    // Hidden layer registers (8 neuron, aktivasi ReLU -> nilai >= 0)
    reg signed [7:0] h[0:7];

    // ROM Memori Bobot INT8 (Inisialisasi dari models/weights_int8.mem)
    reg signed [7:0] rom_w1 [0:31]; // 8 x 4 = 32
    reg signed [7:0] rom_w2 [0:23]; // 3 x 8 = 24

    initial begin
        // Nilai bobot dapat dibaca dari file .mem atau diinisialisasi langsung
        $readmemh("../models/weights_int8.mem", rom_w1, 0, 31);
        $readmemh("../models/weights_int8.mem", rom_w2, 32, 55);
    end

    // Bias Layer 1 (INT32) & Layer 2 (INT32)
    // Nilai diambil dari ann_model_int8_params.json
    wire signed [31:0] b1 [0:7];
    wire signed [31:0] b2 [0:2];

    assign b1[0] = -32'd187;
    assign b1[1] =  32'd1460;
    assign b1[2] =  32'd1419;
    assign b1[3] = -32'd242;
    assign b1[4] =  32'd322;
    assign b1[5] =  32'd2181;
    assign b1[6] =  32'd991;
    assign b1[7] = -32'd932;

    assign b2[0] =  32'd0;
    assign b2[1] =  32'd0;
    assign b2[2] =  32'd0;

    // Akumulator 32-bit
    reg signed [31:0] acc;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            step          <= 4'd0;
            done          <= 1'b0;
            class_pred    <= 2'd0;
            score_normal  <= 8'd0;
            score_mi      <= 8'd0;
            score_unknown <= 8'd0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        x[0]  <= in_x1;
                        x[1]  <= in_x2;
                        x[2]  <= in_x3;
                        x[3]  <= in_x4;
                        step  <= 4'd0;
                        state <= LAYER1;
                    end
                end

                LAYER1: begin
                    // Komputasi neuron hidden ke-`step`: H[step] = ReLU(W1[step] * x + b1[step])
                    acc = b1[step] + 
                          (rom_w1[step*4 + 0] * x[0]) +
                          (rom_w1[step*4 + 1] * x[1]) +
                          (rom_w1[step*4 + 2] * x[2]) +
                          (rom_w1[step*4 + 3] * x[3]);

                    // Fungsi Aktivasi ReLU
                    if (acc < 0) begin
                        h[step] <= 8'd0;
                    end else begin
                        // Skalasi (Right-shift / scaling ~ 1/116)
                        if ((acc >>> 7) > 32'd127)
                            h[step] <= 8'd127;
                        else
                            h[step] <= acc[14:7];
                    end

                    if (step == 4'd7) begin
                        step  <= 4'd0;
                        state <= LAYER2;
                    end else begin
                        step <= step + 1'b1;
                    end
                end

                LAYER2: begin
                    // Komputasi 3 neuron keluaran: Out[k] = W2[k] * H + b2[k]
                    acc = b2[step] +
                          (rom_w2[step*8 + 0] * h[0]) +
                          (rom_w2[step*8 + 1] * h[1]) +
                          (rom_w2[step*8 + 2] * h[2]) +
                          (rom_w2[step*8 + 3] * h[3]) +
                          (rom_w2[step*8 + 4] * h[4]) +
                          (rom_w2[step*8 + 5] * h[5]) +
                          (rom_w2[step*8 + 6] * h[6]) +
                          (rom_w2[step*8 + 7] * h[7]);

                    // Simpan skor INT8
                    if (step == 4'd0) score_normal  <= acc[14:7];
                    if (step == 4'd1) score_mi      <= acc[14:7];
                    if (step == 4'd2) score_unknown <= acc[14:7];

                    if (step == 4'd2) begin
                        state <= FINISH;
                    end else begin
                        step <= step + 1'b1;
                    end
                end

                FINISH: begin
                    // Argmax untuk menentukan kelas
                    if (score_normal >= score_mi && score_normal >= score_unknown)
                        class_pred <= 2'b00; // Normal
                    else if (score_mi >= score_normal && score_mi >= score_unknown)
                        class_pred <= 2'b01; // Infark Miokard
                    else
                        class_pred <= 2'b10; // Unknown

                    done  <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
