#ifndef DELTAGRID_MATH_H
#define DELTAGRID_MATH_H

#include "deltagrid_internal.h"
#include <cmath>
#include <stdint.h>

// Scalar non-linearities matching PyTorch
static inline float dg_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float dg_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// Vector / Matrix Math Primitives
void dg_matmul_bias(
    const float* x,
    const DGWeightArray& W,
    const float* b,
    float* out,
    int out_dim,
    int in_dim
);

void dg_matmul_f32(
    const float* x,
    const float* W,
    const float* b,
    float* out,
    int out_dim,
    int in_dim
);

void dg_layernorm(
    const float* x,
    int dim,
    const float* weight,
    const float* bias,
    float* out
);

void dg_l2_normalize(
    const float* x,
    float* out,
    int dim,
    float eps = 1e-12f
);

void dg_glu(
    const float* x_2d,
    float* out_d,
    int d_model
);

void dg_topk_descending(
    const float* scores,
    int N,
    int K,
    float* out_vals,
    int32_t* out_indices
);

#endif // DELTAGRID_MATH_H

