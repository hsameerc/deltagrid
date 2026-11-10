#ifndef DELTAGRID_MATH_FXP_H
#define DELTAGRID_MATH_FXP_H

#include <stdint.h>

// =============================================================================
// Q16.16 Fixed-Point Arithmetic Definitions (MISRA C++ / Embedded MCU)
// =============================================================================
#define DG_FXP_FRAC_BITS 16
#define DG_FXP_ONE       (1 << DG_FXP_FRAC_BITS)
#define DG_FXP_HALF      (1 << (DG_FXP_FRAC_BITS - 1))

#define DG_FLOAT_TO_FXP(x) ((int32_t)((x) * (float)DG_FXP_ONE))
#define DG_FXP_TO_FLOAT(x) ((float)(x) / (float)DG_FXP_ONE)

static inline int32_t dg_fxp_mul(int32_t a, int32_t b) {
    int64_t prod = (int64_t)a * (int64_t)b;
    return (int32_t)((prod + DG_FXP_HALF) >> DG_FXP_FRAC_BITS);
}

void dg_matmul_fxp(
    const int32_t* x,
    const int32_t* W,
    const int32_t* b,
    int32_t* out,
    int out_dim,
    int in_dim
);

int32_t dg_sigmoid_fxp(int32_t x);
int32_t dg_silu_fxp(int32_t x);
int32_t dg_sin_fxp(float angle_rad);
int32_t dg_cos_fxp(float angle_rad);

void dg_swiglu_fxp(const int32_t* x_2d, int32_t* out_d, int d_model);

void dg_layernorm_fxp(
    const int32_t* x,
    int dim,
    const int32_t* weight,
    const int32_t* bias,
    int32_t* out
);

void dg_l2_normalize_fxp(
    const int32_t* x,
    int32_t* out,
    int dim
);

void dg_topk_descending_fxp(
    const int32_t* scores,
    int N,
    int K,
    int32_t* out_vals,
    int32_t* out_indices
);

#endif // DELTAGRID_MATH_FXP_H

