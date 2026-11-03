#include "deltagrid_math_fxp.h"
#include <cmath>
#include <algorithm>

void dg_matmul_fxp(
    const int32_t* x,
    const int32_t* W,
    const int32_t* b,
    int32_t* out,
    int out_dim,
    int in_dim
) {
#ifdef _OPENMP
    #pragma omp parallel for if(out_dim > 64)
#endif
    for (int i = 0; i < out_dim; ++i) {
        int64_t sum = b ? ((int64_t)b[i] << DG_FXP_FRAC_BITS) : 0;
        const int32_t* row = W + (size_t)i * in_dim;
#ifdef _OPENMP
        #pragma omp simd reduction(+:sum)
#endif
        for (int j = 0; j < in_dim; ++j) {
            sum += (int64_t)row[j] * (int64_t)x[j];
        }
        out[i] = (int32_t)((sum + DG_FXP_HALF) >> DG_FXP_FRAC_BITS);
    }
}

int32_t dg_sigmoid_fxp(int32_t x) {
    float val = DG_FXP_TO_FLOAT(x);
    float sig = 1.0f / (1.0f + std::exp(-val));
    return DG_FLOAT_TO_FXP(sig);
}

int32_t dg_silu_fxp(int32_t x) {
    float val = DG_FXP_TO_FLOAT(x);
    float s = val / (1.0f + std::exp(-val));
    return DG_FLOAT_TO_FXP(s);
}

int32_t dg_sin_fxp(float angle_rad) {
    return DG_FLOAT_TO_FXP(std::sin(angle_rad));
}

int32_t dg_cos_fxp(float angle_rad) {
    return DG_FLOAT_TO_FXP(std::cos(angle_rad));
}

void dg_glu_fxp(const int32_t* x_2d, int32_t* out_d, int d_model) {
    for (int i = 0; i < d_model; ++i) {
        int32_t gate = dg_sigmoid_fxp(x_2d[d_model + i]);
        out_d[i] = dg_fxp_mul(x_2d[i], gate);
    }
}

void dg_layernorm_fxp(
    const int32_t* x,
    int dim,
    const int32_t* weight,
    const int32_t* bias,
    int32_t* out
) {
    int64_t sum = 0;
    for (int i = 0; i < dim; ++i) {
        sum += x[i];
    }
    int32_t mean = (int32_t)(sum / dim);

    float var = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = DG_FXP_TO_FLOAT(x[i] - mean);
        var += diff * diff;
    }
    var /= (float)dim;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    int32_t inv_std_fxp = DG_FLOAT_TO_FXP(inv_std);

    for (int i = 0; i < dim; ++i) {
        int64_t centered = ((int64_t)(x[i] - mean) * inv_std_fxp + DG_FXP_HALF) >> DG_FXP_FRAC_BITS;
        out[i] = (int32_t)(((centered * weight[i] + DG_FXP_HALF) >> DG_FXP_FRAC_BITS) + bias[i]);
    }
}

void dg_l2_normalize_fxp(
    const int32_t* x,
    int32_t* out,
    int dim
) {
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float v = DG_FXP_TO_FLOAT(x[i]);
        norm_sq += v * v;
    }
    float norm = std::sqrt(norm_sq);
    float denom = (norm > 1e-12f) ? norm : 1e-12f;
    int32_t inv_norm_fxp = DG_FLOAT_TO_FXP(1.0f / denom);
    for (int i = 0; i < dim; ++i) {
        out[i] = dg_fxp_mul(x[i], inv_norm_fxp);
    }
}

void dg_topk_descending_fxp(
    const int32_t* scores,
    int N,
    int K,
    int32_t* out_vals,
    int32_t* out_indices
) {
    for (int k = 0; k < K; ++k) {
        out_vals[k] = INT32_MIN;
        out_indices[k] = -1;
    }

    for (int i = 0; i < N; ++i) {
        int32_t val = scores[i];
        if (val > out_vals[K - 1] || out_indices[K - 1] == -1) {
            int pos = K - 1;
            while (pos > 0 && (val > out_vals[pos - 1] || out_indices[pos - 1] == -1)) {
                out_vals[pos] = out_vals[pos - 1];
                out_indices[pos] = out_indices[pos - 1];
                --pos;
            }
            out_vals[pos] = val;
            out_indices[pos] = i;
        }
    }
}

