/* Auto-generated Fixed-Point ANN Weights for FPGA / MCU */
#define Q_FRACTIONAL_BITS 12
#define INPUT_DIM 4
#define HIDDEN_DIM 8
#define NUM_CLASSES 3

const short W1[HIDDEN_DIM][INPUT_DIM] = {
    {897, 3289, -198, 668},
    {191, -2383, 3899, 2884},
    {-4330, -282, 2897, -6083},
    {-1984, 7344, 470, -7395},
    {8481, -2387, -4580, 4085},
    {-4764, -7187, 10312, -5324},
    {5018, 2628, 7157, 836},
    {2401, -10063, -2679, 10564},
};

const short b1[HIDDEN_DIM] = {6478, 5190, 3982, 4881, -3183, 2960, 4042, -11388};

const short W2[NUM_CLASSES][HIDDEN_DIM] = {
    {2739, 3406, -5096, -11117, -4977, -19159, 4457, -15979},
    {-5536, -6492, 4106, 802, -7723, 2563, 1567, 19953},
    {1778, -1040, -4117, 1231, 8630, 5657, -8142, -6734},
};

const short b2[NUM_CLASSES] = {2589, -1176, -526};
