#include "deltagrid_math.h"
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>

// =============================================================================
// Low-Level Vector & Matrix Math Kernels
// =============================================================================

void dg_matmul_bias(
    const float* x,
    const DGWeightArray& W,
    const float* b,
    float* out,
    int out_dim,
    int in_dim
) {
    if (W.q8 != nullptr) {
#ifdef _OPENMP
        #pragma omp parallel for if(out_dim > 64)
#endif
        for (int i = 0; i < out_dim; ++i) {
            float bias_val = b ? b[i] : 0.0f;
            const int8_t* row = W.q8 + (size_t)i * in_dim;
            float sum = 0.0f;
#ifdef _OPENMP
            #pragma omp simd reduction(+:sum)
#endif
            for (int j = 0; j < in_dim; ++j) {
                sum += (float)row[j] * x[j];
            }
            out[i] = bias_val + sum * W.scale;
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(out_dim > 64)
#endif
        for (int i = 0; i < out_dim; ++i) {
            float bias_val = b ? b[i] : 0.0f;
            const float* row = W.f32 + (size_t)i * in_dim;
            float sum = 0.0f;
#ifdef _OPENMP
            #pragma omp simd reduction(+:sum)
#endif
            for (int j = 0; j < in_dim; ++j) {
                sum += row[j] * x[j];
            }
            out[i] = bias_val + sum;
        }
    }
}

void dg_matmul_f32(
    const float* x,
    const float* W,
    const float* b,
    float* out,
    int out_dim,
    int in_dim
) {
    for (int i = 0; i < out_dim; ++i) {
        float sum = b ? b[i] : 0.0f;
        const float* row = W + (size_t)i * in_dim;
#ifdef _OPENMP
        #pragma omp simd reduction(+:sum)
#endif
        for (int j = 0; j < in_dim; ++j) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

void dg_layernorm(
    const float* x,
    int dim,
    const float* weight,
    const float* bias,
    float* out
) {
    float mean = 0.0f;
    for (int i = 0; i < dim; ++i) {
        mean += x[i];
    }
    mean /= (float)dim;

    float var = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= (float)dim;

    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < dim; ++i) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

void dg_l2_normalize(
    const float* x,
    float* out,
    int dim,
    float eps
) {
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        norm_sq += x[i] * x[i];
    }
    float norm = std::sqrt(norm_sq);
    float denom = (norm > eps) ? norm : eps;
    float inv_norm = 1.0f / denom;
    for (int i = 0; i < dim; ++i) {
        out[i] = x[i] * inv_norm;
    }
}

void dg_glu(
    const float* x_2d,
    float* out_d,
    int d_model
) {
    for (int i = 0; i < d_model; ++i) {
        out_d[i] = x_2d[i] * dg_sigmoid(x_2d[d_model + i]);
    }
}

void dg_swiglu(
    const float* x_2d,
    float* out_d,
    int d_model
) {
    for (int i = 0; i < d_model; ++i) {
        float a = x_2d[i];
        float b = x_2d[d_model + i];
        out_d[i] = dg_silu(a) * b;
    }
}

void dg_topk_descending(
    const float* scores,
    int N,
    int K,
    float* out_vals,
    int32_t* out_indices
) {
    // Initialize top-K with -infinity
    for (int k = 0; k < K; ++k) {
        out_vals[k] = -1e38f;
        out_indices[k] = -1;
    }

    for (int i = 0; i < N; ++i) {
        float val = scores[i];
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

// =============================================================================
// Core Tier 2 Associative Memory Step
// =============================================================================

static void dg_memory_layer_step(
    DGRuntimeContext* ctx,
    const DGMemoryLayerWeights& mem,
    const float* h_in,
    float* mem_state,
    float* mem_out
) {
    const int D            = ctx->header.d_model;
    const int H            = ctx->header.num_heads;
    const int Dh           = ctx->head_dim;
    const int actual_slots = ctx->header.actual_slots;
    const int active_slots = ctx->active_slots;
    const int num_axes     = ctx->header.num_axes;
    const int top_k        = ctx->header.top_k;
    const bool dyn_decay   = (ctx->header.flags & DG_FLAG_DYNAMIC_DECAY) != 0;
    const bool has_slot_p  = (ctx->header.flags & DG_FLAG_HAS_SLOT_PROJ) != 0;

    // 1. Fused projection: Q, K, V, and beta_raw
    dg_matmul_bias(h_in, mem.proj_qkv_beta, nullptr, ctx->packed_qkvb_buf, 3 * D + H, D);

    const float* q_raw    = ctx->packed_qkvb_buf;
    const float* k_raw    = ctx->packed_qkvb_buf + D;
    const float* v_raw    = ctx->packed_qkvb_buf + 2 * D;
    const float* beta_raw = ctx->packed_qkvb_buf + 3 * D;

    std::memcpy(ctx->q_buf, q_raw, D * sizeof(float));
    std::memcpy(ctx->v_buf, v_raw, D * sizeof(float));

    for (int h = 0; h < H; ++h) {
        dg_l2_normalize(k_raw + h * Dh, ctx->k_buf + h * Dh, Dh, 1e-12f);
        ctx->beta_buf[h] = dg_sigmoid(beta_raw[h] + mem.beta_bias[h]);
    }

    // 2. Decay gate alpha
    if (dyn_decay) {
        dg_matmul_bias(h_in, mem.proj_alpha_weight, mem.proj_alpha_bias, ctx->alpha_buf, H, D);
        for (int h = 0; h < H; ++h) {
            ctx->alpha_buf[h] = 0.99f + 0.01f * dg_sigmoid(ctx->alpha_buf[h]);
        }
    } else {
        for (int h = 0; h < H; ++h) {
            ctx->alpha_buf[h] = mem.decay[h];
        }
    }

    // 3. Per-Head Coordinate Routing & Closed-Loop Householder Delta Update
#ifdef _OPENMP
    #pragma omp parallel for if(H > 1)
#endif
    for (int h = 0; h < H; ++h) {
        const float* q_h = ctx->q_buf + h * Dh;
        const float* k_h = ctx->k_buf + h * Dh;
        const float* v_h = ctx->v_buf + h * Dh;
        float* yt_h      = ctx->yt_buf + h * Dh;
        float* state_h   = mem_state + (size_t)h * actual_slots * Dh;

        float beta_h   = ctx->beta_buf[h];
        float alpha_h  = ctx->alpha_buf[h];
        float q_scale  = mem.q_scale[h];

        float* wk_h      = ctx->wk_buf + (size_t)h * active_slots;
        float* wq_h      = ctx->wq_buf + (size_t)h * active_slots;
        int32_t* idx_h   = ctx->active_idx_buf + (size_t)h * active_slots;

        if (top_k > 0 && num_axes > 0) {
            // -----------------------------------------------------------------
            // Mode 3: Sparse Delta-PKM
            // -----------------------------------------------------------------
            float* sk_h = ctx->sk_buf + (size_t)h * ctx->max_axis_dim;
            float* sq_h = ctx->sq_buf + (size_t)h * ctx->max_axis_dim;

            for (int a = 0; a < num_axes; ++a) {
                int M_a = ctx->grid_dim[a];
                float* uk_a    = ctx->uk_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                float* uq_a    = ctx->uq_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                int32_t* idx_a = ctx->idx_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;

                dg_matmul_f32(k_h, mem.axes_proj_k[a], nullptr, sk_h, M_a, Dh);
                dg_topk_descending(sk_h, M_a, top_k, uk_a, idx_a);
                dg_l2_normalize(uk_a, uk_a, top_k, 1e-12f);

                dg_matmul_f32(q_h, mem.axes_proj_q[a], nullptr, sq_h, M_a, Dh);
                for (int j = 0; j < top_k; ++j) {
                    uq_a[j] = sq_h[idx_a[j]];
                }
                dg_l2_normalize(uq_a, uq_a, top_k, 1e-12f);
            }

            // Kronecker product across axes (in-place reverse expansion)
            const float* uk_0    = ctx->uk_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;
            const float* uq_0    = ctx->uq_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;
            const int32_t* idx_0 = ctx->idx_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;

            for (int j = 0; j < top_k; ++j) {
                wk_h[j]  = uk_0[j];
                wq_h[j]  = uq_0[j];
                idx_h[j] = idx_0[j];
            }
            int cur_len = top_k;
            for (int a = 1; a < num_axes; ++a) {
                int M_a = ctx->grid_dim[a];
                const float* uk_a    = ctx->uk_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                const float* uq_a    = ctx->uq_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                const int32_t* idx_a = ctx->idx_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;

                for (int i = cur_len - 1; i >= 0; --i) {
                    float wk_i    = wk_h[i];
                    float wq_i    = wq_h[i];
                    int32_t idx_i = idx_h[i];
                    for (int j = top_k - 1; j >= 0; --j) {
                        int dst = i * top_k + j;
                        wk_h[dst]  = wk_i * uk_a[j];
                        wq_h[dst]  = wq_i * uq_a[j];
                        idx_h[dst] = idx_i * M_a + idx_a[j];
                    }
                }
                cur_len *= top_k;
            }

            for (int m = 0; m < active_slots; ++m) {
                wq_h[m] *= q_scale;
            }

            // Read & Predict over sparse active slots
            float pred_v[512]; // head_dim <= 512
            for (int d = 0; d < Dh; ++d) {
                yt_h[d]   = 0.0f;
                pred_v[d] = 0.0f;
            }

            for (int m = 0; m < active_slots; ++m) {
                int32_t slot = idx_h[m];
                const float* s_row = state_h + (size_t)slot * Dh;
                float wq_m = wq_h[m];
                float wk_m = wk_h[m];
                for (int d = 0; d < Dh; ++d) {
                    yt_h[d]   += wq_m * s_row[d];
                    pred_v[d] += wk_m * s_row[d];
                }
            }

            // Error residual & Householder Delta Write to active slots
            float rt[512];
            for (int d = 0; d < Dh; ++d) {
                rt[d] = v_h[d] - pred_v[d];
            }

            for (int m = 0; m < active_slots; ++m) {
                int32_t slot = idx_h[m];
                float* s_row = state_h + (size_t)slot * Dh;
                float bw = beta_h * wk_h[m];
                for (int d = 0; d < Dh; ++d) {
                    s_row[d] = alpha_h * s_row[d] + bw * rt[d];
                }
            }
        } else {
            // -----------------------------------------------------------------
            // Mode 1 (Standard Dense) & Mode 2 (Factorized Kronecker Dense)
            // -----------------------------------------------------------------
            if (num_axes > 0) {
                for (int a = 0; a < num_axes; ++a) {
                    int M_a = ctx->grid_dim[a];
                    float* uk_a = ctx->uk_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                    float* uq_a = ctx->uq_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;

                    dg_matmul_f32(k_h, mem.axes_proj_k[a], nullptr, uk_a, M_a, Dh);
                    dg_l2_normalize(uk_a, uk_a, M_a, 1e-12f);
                    dg_matmul_f32(q_h, mem.axes_proj_q[a], nullptr, uq_a, M_a, Dh);
                }

                int M_0 = ctx->grid_dim[0];
                const float* uk_0 = ctx->uk_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;
                const float* uq_0 = ctx->uq_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;
                for (int j = 0; j < M_0; ++j) {
                    wk_h[j] = uk_0[j];
                    wq_h[j] = uq_0[j];
                }
                int cur_len = M_0;
                for (int a = 1; a < num_axes; ++a) {
                    int M_a = ctx->grid_dim[a];
                    const float* uk_a = ctx->uk_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                    const float* uq_a = ctx->uq_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                    for (int i = cur_len - 1; i >= 0; --i) {
                        float wk_i = wk_h[i];
                        float wq_i = wq_h[i];
                        for (int j = M_a - 1; j >= 0; --j) {
                            int dst = i * M_a + j;
                            wk_h[dst] = wk_i * uk_a[j];
                            wq_h[dst] = wq_i * uq_a[j];
                        }
                    }
                    cur_len *= M_a;
                }
            } else if (has_slot_p) {
                dg_matmul_f32(k_h, mem.slot_proj_k, nullptr, wk_h, actual_slots, Dh);
                dg_l2_normalize(wk_h, wk_h, actual_slots, 1e-12f);
                dg_matmul_f32(q_h, mem.slot_proj_q, nullptr, wq_h, actual_slots, Dh);
            } else {
                for (int s = 0; s < actual_slots; ++s) {
                    wk_h[s] = k_h[s];
                    wq_h[s] = q_h[s];
                }
            }

            for (int s = 0; s < actual_slots; ++s) {
                wq_h[s] *= q_scale;
            }

            // Read & Predict over all slots
            float pred_v[512];
            for (int d = 0; d < Dh; ++d) {
                yt_h[d]   = 0.0f;
                pred_v[d] = 0.0f;
            }

            for (int s = 0; s < actual_slots; ++s) {
                const float* s_row = state_h + (size_t)s * Dh;
                float wq_s = wq_h[s];
                float wk_s = wk_h[s];
                for (int d = 0; d < Dh; ++d) {
                    yt_h[d]   += wq_s * s_row[d];
                    pred_v[d] += wk_s * s_row[d];
                }
            }

            float rt[512];
            for (int d = 0; d < Dh; ++d) {
                rt[d] = v_h[d] - pred_v[d];
            }

            for (int s = 0; s < actual_slots; ++s) {
                float* s_row = state_h + (size_t)s * Dh;
                float bw = beta_h * wk_h[s];
                for (int d = 0; d < Dh; ++d) {
                    s_row[d] = alpha_h * s_row[d] + bw * rt[d];
                }
            }
        }
    }

    // 4. Output projection & LayerNorm
    dg_matmul_bias(ctx->yt_buf, mem.out_proj, nullptr, ctx->mem_proj_buf, D, D);
    dg_layernorm(ctx->mem_proj_buf, D, mem.norm.weight, mem.norm.bias, mem_out);
}

// =============================================================================
// Complete 4-Tier Block Step Execution
// =============================================================================

void dg_step_all_blocks(DGRuntimeContext* ctx) {
    const int L       = ctx->header.num_layers;
    const int D       = ctx->header.d_model;
    const int K       = ctx->header.kernel_size;
    const int cycles  = ctx->header.thinking_cycles;
    const int offset  = ctx->step_count;
    const size_t state_floats = (size_t)ctx->header.num_heads * ctx->header.actual_slots * ctx->head_dim;

    for (int l = 0; l < L; ++l) {
        const DGBlockWeights& blk = ctx->blocks[l];
        DGLayerState& st          = ctx->states[l];

        // ---------------------------------------------------------------------
        // Tier 0: Biological Harmonic Clock
        // ---------------------------------------------------------------------
        float c_scale = blk.clock_scale;
        int half_d = D / 2;
        for (int k = 0; k < half_d; ++k) {
            float angle = (float)offset * blk.div_term[k];
            ctx->x_in_buf[2 * k]     = ctx->hidden_buf[2 * k]     + std::sin(angle) * c_scale;
            ctx->x_in_buf[2 * k + 1] = ctx->hidden_buf[2 * k + 1] + std::cos(angle) * c_scale;
        }

        // ---------------------------------------------------------------------
        // Tier 1: Causal Depthwise Conv1D (k = K)
        // ---------------------------------------------------------------------
        int hist_len = K - 1;
        for (int c = 0; c < D; ++c) {
            const float* w_c = blk.conv1d_weight + c * K;
            float* hist_c    = st.conv_state + c * hist_len;

            float sum = blk.conv1d_bias[c];
            for (int j = 0; j < hist_len; ++j) {
                sum += w_c[j] * hist_c[j];
            }
            sum += w_c[hist_len] * ctx->x_in_buf[c];

            ctx->c_out_buf[c] = ctx->x_in_buf[c] + dg_silu(sum);

            // Shift Conv1D circular history state
            for (int j = 0; j < hist_len - 1; ++j) {
                hist_c[j] = hist_c[j + 1];
            }
            if (hist_len > 0) {
                hist_c[hist_len - 1] = ctx->x_in_buf[c];
            }
        }
        dg_layernorm(ctx->c_out_buf, D, blk.norm_conv.weight, blk.norm_conv.bias, ctx->h_conv_buf);

        // ---------------------------------------------------------------------
        // Tier 2: Associative DeltaGrid Memory Layer
        // ---------------------------------------------------------------------
        if (blk.memory.has_memory) {
            dg_memory_layer_step(ctx, blk.memory, ctx->h_conv_buf, st.mem_state, ctx->mem_out_buf);
            float s_mem = blk.memory.scale_mem;
            for (int i = 0; i < D; ++i) {
                ctx->x_in_buf[i] += ctx->mem_out_buf[i] * s_mem;
            }
        }

        // ---------------------------------------------------------------------
        // Tier 3: Non-Linear SwiGLU Logic Reasoning Engine
        // ---------------------------------------------------------------------
        dg_matmul_bias(ctx->x_in_buf, blk.ffn_fc1_weight, blk.ffn_fc1_bias, ctx->ffn_fc1_buf, 2 * D, D);
        dg_swiglu(ctx->ffn_fc1_buf, ctx->ffn_glu_buf, D);
        dg_matmul_bias(ctx->ffn_glu_buf, blk.ffn_fc2_weight, blk.ffn_fc2_bias, ctx->ffn_fc2_buf, D, D);
        dg_layernorm(ctx->ffn_fc2_buf, D, blk.norm_ffn.weight, blk.norm_ffn.bias, ctx->ffn_norm_buf);

        float s_logic = blk.scale_logic;
        for (int i = 0; i < D; ++i) {
            ctx->x_in_buf[i] += ctx->ffn_norm_buf[i] * s_logic;
        }

        // ---------------------------------------------------------------------
        // Cognitive Thinking Loops (Working-Memory Scratchpad Recurrence)
        // ---------------------------------------------------------------------
        if (cycles > 0 && blk.memory.has_memory && ctx->think_mem_buf != nullptr) {
            std::memcpy(ctx->think_mem_buf, st.mem_state, state_floats * sizeof(float));
            float s_mem = blk.memory.scale_mem;
            for (int cyc = 0; cyc < cycles; ++cyc) {
                dg_memory_layer_step(ctx, blk.memory, ctx->x_in_buf, ctx->think_mem_buf, ctx->mem_out_buf);
                for (int i = 0; i < D; ++i) {
                    ctx->x_in_buf[i] += ctx->mem_out_buf[i] * s_mem;
                }
                dg_matmul_bias(ctx->x_in_buf, blk.ffn_fc1_weight, blk.ffn_fc1_bias, ctx->ffn_fc1_buf, 2 * D, D);
                dg_swiglu(ctx->ffn_fc1_buf, ctx->ffn_glu_buf, D);
                dg_matmul_bias(ctx->ffn_glu_buf, blk.ffn_fc2_weight, blk.ffn_fc2_bias, ctx->ffn_fc2_buf, D, D);
                dg_layernorm(ctx->ffn_fc2_buf, D, blk.norm_ffn.weight, blk.norm_ffn.bias, ctx->ffn_norm_buf);
                for (int i = 0; i < D; ++i) {
                    ctx->x_in_buf[i] += ctx->ffn_norm_buf[i] * s_logic;
                }
            }
        }

        // Propagate to next block
        std::memcpy(ctx->hidden_buf, ctx->x_in_buf, D * sizeof(float));
    }

    ctx->step_count += 1;
}

// =============================================================================
// Weight Loading, Context Lifecycle, and State Persistence
// =============================================================================

static void load_f32_array(std::ifstream& f, float*& arr, size_t n) {
    if (n == 0) {
        arr = nullptr;
        return;
    }
    arr = (float*)std::malloc(n * sizeof(float));
    f.read(reinterpret_cast<char*>(arr), n * sizeof(float));
}

static void load_weight_array(std::ifstream& f, DGWeightArray& W, size_t n, int precision_mode) {
    W.f32 = nullptr;
    W.q8  = nullptr;
    W.scale = 1.0f;
    if (n == 0) return;

    if (precision_mode == DG_PREC_INT8) {
        f.read(reinterpret_cast<char*>(&W.scale), sizeof(float));
        W.q8 = (int8_t*)std::malloc(n * sizeof(int8_t));
        f.read(reinterpret_cast<char*>(W.q8), n * sizeof(int8_t));
    } else {
        W.f32 = (float*)std::malloc(n * sizeof(float));
        f.read(reinterpret_cast<char*>(W.f32), n * sizeof(float));
    }
}

static void free_weight_array(DGWeightArray& W) {
    if (W.f32) { std::free(W.f32); W.f32 = nullptr; }
    if (W.q8)  { std::free(W.q8);  W.q8  = nullptr; }
}

DGRuntimeContext* dg_load_context_from_file(const char* weights_path, int expected_model_type) {
    if (!is_deltagrid_unlocked || weights_path == nullptr) return nullptr;

    std::ifstream f(weights_path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[DeltaGrid Engine] Failed to open weights file: " << weights_path << "\n";
        return nullptr;
    }

    DeltaGridBinHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(DeltaGridBinHeader));
    if (!f.good() || hdr.magic != DELTAGRID_BIN_MAGIC) {
        std::cerr << "[DeltaGrid Engine] Invalid binary magic header in: " << weights_path << "\n";
        return nullptr;
    }
    if (expected_model_type >= 0 && hdr.model_type != expected_model_type) {
        std::cerr << "[DeltaGrid Engine] Model type mismatch in: " << weights_path << "\n";
        return nullptr;
    }
    if (hdr.precision_mode != DG_PREC_FP32 && hdr.precision_mode != DG_PREC_INT8) {
        std::cerr << "[DeltaGrid Engine] Unsupported float/int8 precision mode: " << hdr.precision_mode << "\n";
        return nullptr;
    }

    DGRuntimeContext* ctx = (DGRuntimeContext*)std::calloc(1, sizeof(DGRuntimeContext));
    ctx->header    = hdr;
    ctx->head_dim  = hdr.d_model / hdr.num_heads;
    ctx->step_count = 0;

    const int L      = hdr.num_layers;
    const int D      = hdr.d_model;
    const int H      = hdr.num_heads;
    const int Dh     = ctx->head_dim;
    const int K      = hdr.kernel_size;
    const int pmode  = hdr.precision_mode;
    const bool dyn_decay  = (hdr.flags & DG_FLAG_DYNAMIC_DECAY) != 0;
    const bool has_slot_p = (hdr.flags & DG_FLAG_HAS_SLOT_PROJ) != 0;
    const bool tied_head  = (hdr.flags & DG_FLAG_TIED_LM_HEAD) != 0;

    if (hdr.num_axes > 0) {
        ctx->grid_dim = (int32_t*)std::malloc(hdr.num_axes * sizeof(int32_t));
        f.read(reinterpret_cast<char*>(ctx->grid_dim), hdr.num_axes * sizeof(int32_t));
        int max_ax = 0;
        int act_slots = 1;
        for (int a = 0; a < hdr.num_axes; ++a) {
            if (ctx->grid_dim[a] > max_ax) max_ax = ctx->grid_dim[a];
            act_slots *= (hdr.top_k > 0) ? hdr.top_k : ctx->grid_dim[a];
        }
        ctx->max_axis_dim    = max_ax;
        ctx->max_axis_active = (hdr.top_k > 0) ? hdr.top_k : max_ax;
        ctx->active_slots    = act_slots;
    } else {
        ctx->grid_dim        = nullptr;
        ctx->max_axis_dim    = 0;
        ctx->max_axis_active = 0;
        ctx->active_slots    = hdr.actual_slots;
    }

    // 1. Input Encoder / Token Embedding
    if (hdr.model_type == DG_MODEL_CONTINUOUS) {
        load_weight_array(f, ctx->input_weight, (size_t)D * hdr.input_or_vocab, pmode);
        load_f32_array(f, ctx->input_bias, D);
    } else {
        load_weight_array(f, ctx->input_weight, (size_t)hdr.input_or_vocab * D, pmode);
        ctx->input_bias = nullptr;
    }

    // 2. 4-Tier Blocks
    ctx->blocks = (DGBlockWeights*)std::calloc(L, sizeof(DGBlockWeights));
    ctx->states = (DGLayerState*)std::calloc(L, sizeof(DGLayerState));

    for (int l = 0; l < L; ++l) {
        DGBlockWeights& blk = ctx->blocks[l];
        int32_t has_mem = 1;
        f.read(reinterpret_cast<char*>(&has_mem), sizeof(int32_t));
        blk.memory.has_memory = has_mem;

        // Tier 0: Harmonic Clock
        load_f32_array(f, blk.div_term, D / 2);
        f.read(reinterpret_cast<char*>(&blk.clock_scale), sizeof(float));

        // Tier 1: Causal Depthwise Conv1D
        load_f32_array(f, blk.conv1d_weight, (size_t)D * K);
        load_f32_array(f, blk.conv1d_bias, D);
        load_f32_array(f, blk.norm_conv.weight, D);
        load_f32_array(f, blk.norm_conv.bias, D);

        // Tier 2: Associative Memory
        if (has_mem) {
            f.read(reinterpret_cast<char*>(&blk.memory.scale_mem), sizeof(float));
            load_weight_array(f, blk.memory.proj_qkv_beta, (size_t)(3 * D + H) * D, pmode);
            load_f32_array(f, blk.memory.q_scale, H);
            load_f32_array(f, blk.memory.beta_bias, H);
            load_f32_array(f, blk.memory.decay, H);

            if (dyn_decay) {
                load_weight_array(f, blk.memory.proj_alpha_weight, (size_t)H * D, pmode);
                load_f32_array(f, blk.memory.proj_alpha_bias, H);
            }

            if (hdr.num_axes > 0) {
                blk.memory.axes_proj_k = (float**)std::calloc(hdr.num_axes, sizeof(float*));
                blk.memory.axes_proj_q = (float**)std::calloc(hdr.num_axes, sizeof(float*));
                for (int a = 0; a < hdr.num_axes; ++a) {
                    int M_a = ctx->grid_dim[a];
                    load_f32_array(f, blk.memory.axes_proj_k[a], (size_t)M_a * Dh);
                    load_f32_array(f, blk.memory.axes_proj_q[a], (size_t)M_a * Dh);
                }
            } else if (has_slot_p) {
                load_f32_array(f, blk.memory.slot_proj_k, (size_t)hdr.actual_slots * Dh);
                load_f32_array(f, blk.memory.slot_proj_q, (size_t)hdr.actual_slots * Dh);
            }

            load_weight_array(f, blk.memory.out_proj, (size_t)D * D, pmode);
            load_f32_array(f, blk.memory.norm.weight, D);
            load_f32_array(f, blk.memory.norm.bias, D);
        }

        // Tier 3: GLU Logic Reasoning Engine
        f.read(reinterpret_cast<char*>(&blk.scale_logic), sizeof(float));
        load_weight_array(f, blk.ffn_fc1_weight, (size_t)(2 * D) * D, pmode);
        load_f32_array(f, blk.ffn_fc1_bias, 2 * D);
        load_weight_array(f, blk.ffn_fc2_weight, (size_t)D * D, pmode);
        load_f32_array(f, blk.ffn_fc2_bias, D);
        load_f32_array(f, blk.norm_ffn.weight, D);
        load_f32_array(f, blk.norm_ffn.bias, D);

        // Allocate Recurrent States (zero-initialized)
        int hist_len = (K > 1) ? (K - 1) : 1;
        ctx->states[l].conv_state = (float*)std::calloc((size_t)D * hist_len, sizeof(float));
        ctx->states[l].mem_state  = (float*)std::calloc((size_t)H * hdr.actual_slots * Dh, sizeof(float));
    }

    // 3. Final Readout Layer
    load_f32_array(f, ctx->final_norm.weight, D);
    load_f32_array(f, ctx->final_norm.bias, D);

    if (hdr.model_type == DG_MODEL_CONTINUOUS) {
        load_weight_array(f, ctx->head_weight, (size_t)hdr.output_dim * D, pmode);
        load_f32_array(f, ctx->head_bias, hdr.output_dim);
        ctx->head_is_tied = false;

        if (hdr.forecast_horizon > 0) {
            load_weight_array(f, ctx->forecast_fc1_weight, (size_t)D * D, pmode);
            load_f32_array(f, ctx->forecast_fc1_bias, D);
            load_weight_array(
                f,
                ctx->forecast_fc2_weight,
                (size_t)hdr.forecast_horizon * hdr.output_dim * D,
                pmode
            );
            load_f32_array(f, ctx->forecast_fc2_bias, (size_t)hdr.forecast_horizon * hdr.output_dim);
        }
    } else {
        if (tied_head) {
            ctx->head_weight  = ctx->input_weight;
            ctx->head_bias    = nullptr;
            ctx->head_is_tied = true;
        } else {
            load_weight_array(f, ctx->head_weight, (size_t)hdr.output_dim * D, pmode);
            ctx->head_bias    = nullptr;
            ctx->head_is_tied = false;
        }
    }

    // 4. Preallocate Scratch Buffers
    ctx->hidden_buf       = (float*)std::calloc(D, sizeof(float));
    ctx->x_in_buf         = (float*)std::calloc(D, sizeof(float));
    ctx->c_out_buf        = (float*)std::calloc(D, sizeof(float));
    ctx->h_conv_buf       = (float*)std::calloc(D, sizeof(float));
    ctx->packed_qkvb_buf  = (float*)std::calloc(3 * D + H, sizeof(float));
    ctx->q_buf            = (float*)std::calloc(D, sizeof(float));
    ctx->k_buf            = (float*)std::calloc(D, sizeof(float));
    ctx->v_buf            = (float*)std::calloc(D, sizeof(float));
    ctx->beta_buf         = (float*)std::calloc(H, sizeof(float));
    ctx->alpha_buf        = (float*)std::calloc(H, sizeof(float));
    ctx->yt_buf           = (float*)std::calloc(D, sizeof(float));
    ctx->mem_proj_buf     = (float*)std::calloc(D, sizeof(float));
    ctx->mem_out_buf      = (float*)std::calloc(D, sizeof(float));
    ctx->ffn_fc1_buf      = (float*)std::calloc(2 * D, sizeof(float));
    ctx->ffn_glu_buf      = (float*)std::calloc(D, sizeof(float));
    ctx->ffn_fc2_buf      = (float*)std::calloc(D, sizeof(float));
    ctx->ffn_norm_buf     = (float*)std::calloc(D, sizeof(float));
    ctx->final_norm_buf   = (float*)std::calloc(D, sizeof(float));
    ctx->forecast_hid_buf = (float*)std::calloc(D, sizeof(float));

    ctx->wk_buf         = (float*)std::calloc((size_t)H * ctx->active_slots, sizeof(float));
    ctx->wq_buf         = (float*)std::calloc((size_t)H * ctx->active_slots, sizeof(float));
    ctx->active_idx_buf = (int32_t*)std::calloc((size_t)H * ctx->active_slots, sizeof(int32_t));

    if (hdr.num_axes > 0) {
        ctx->sk_buf       = (float*)std::calloc((size_t)H * ctx->max_axis_dim, sizeof(float));
        ctx->sq_buf       = (float*)std::calloc((size_t)H * ctx->max_axis_dim, sizeof(float));
        ctx->uk_axes_buf  = (float*)std::calloc((size_t)H * hdr.num_axes * ctx->max_axis_active, sizeof(float));
        ctx->uq_axes_buf  = (float*)std::calloc((size_t)H * hdr.num_axes * ctx->max_axis_active, sizeof(float));
        ctx->idx_axes_buf = (int32_t*)std::calloc((size_t)H * hdr.num_axes * ctx->max_axis_active, sizeof(int32_t));
    }

    if (hdr.thinking_cycles > 0) {
        ctx->think_mem_buf = (float*)std::calloc((size_t)H * hdr.actual_slots * Dh, sizeof(float));
    }

    return ctx;
}

void dg_reset_runtime_state(DGRuntimeContext* ctx) {
    if (!ctx) return;
    const int L  = ctx->header.num_layers;
    const int D  = ctx->header.d_model;
    const int H  = ctx->header.num_heads;
    const int Dh = ctx->head_dim;
    const int K  = ctx->header.kernel_size;
    int hist_len = (K > 1) ? (K - 1) : 1;

    ctx->step_count = 0;
    for (int l = 0; l < L; ++l) {
        std::memset(ctx->states[l].conv_state, 0, (size_t)D * hist_len * sizeof(float));
        std::memset(ctx->states[l].mem_state,  0, (size_t)H * ctx->header.actual_slots * Dh * sizeof(float));
    }
}

int dg_save_runtime_state(DGRuntimeContext* ctx, const char* filepath) {
    if (!ctx || !filepath) return -1;
    std::ofstream f(filepath, std::ios::binary);
    if (!f.is_open()) return -1;

    int32_t magic = DELTAGRID_STATE_MAGIC;
    int32_t L     = ctx->header.num_layers;
    int32_t D     = ctx->header.d_model;
    int32_t H     = ctx->header.num_heads;
    int32_t slots = ctx->header.actual_slots;
    int32_t K     = ctx->header.kernel_size;
    int32_t step  = ctx->step_count;

    f.write(reinterpret_cast<const char*>(&magic), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&L),     sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&D),     sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&H),     sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&slots), sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&K),     sizeof(int32_t));
    f.write(reinterpret_cast<const char*>(&step),  sizeof(int32_t));

    int hist_len = (K > 1) ? (K - 1) : 1;
    size_t conv_bytes = (size_t)D * hist_len * sizeof(float);
    size_t mem_bytes  = (size_t)H * slots * ctx->head_dim * sizeof(float);

    for (int l = 0; l < L; ++l) {
        f.write(reinterpret_cast<const char*>(ctx->states[l].conv_state), conv_bytes);
        f.write(reinterpret_cast<const char*>(ctx->states[l].mem_state),  mem_bytes);
    }

    return f.good() ? 0 : -1;
}

int dg_load_runtime_state(DGRuntimeContext* ctx, const char* filepath) {
    if (!ctx || !filepath) return -1;
    std::ifstream f(filepath, std::ios::binary);
    if (!f.is_open()) return -1;

    int32_t magic = 0, L = 0, D = 0, H = 0, slots = 0, K = 0, step = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&L),     sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&D),     sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&H),     sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&slots), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&K),     sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&step),  sizeof(int32_t));

    if (!f.good() || magic != DELTAGRID_STATE_MAGIC) return -1;
    if (L != ctx->header.num_layers || D != ctx->header.d_model ||
        H != ctx->header.num_heads  || slots != ctx->header.actual_slots ||
        K != ctx->header.kernel_size) {
        return -1;
    }

    ctx->step_count = step;
    int hist_len = (K > 1) ? (K - 1) : 1;
    size_t conv_bytes = (size_t)D * hist_len * sizeof(float);
    size_t mem_bytes  = (size_t)H * slots * ctx->head_dim * sizeof(float);

    for (int l = 0; l < L; ++l) {
        f.read(reinterpret_cast<char*>(ctx->states[l].conv_state), conv_bytes);
        f.read(reinterpret_cast<char*>(ctx->states[l].mem_state),  mem_bytes);
    }

    return f.good() ? 0 : -1;
}

void dg_free_runtime_context(DGRuntimeContext* ctx) {
    if (!ctx) return;

    const int L = ctx->header.num_layers;
    if (ctx->grid_dim) std::free(ctx->grid_dim);

    free_weight_array(ctx->input_weight);
    if (ctx->input_bias) std::free(ctx->input_bias);

    if (ctx->blocks) {
        for (int l = 0; l < L; ++l) {
            DGBlockWeights& blk = ctx->blocks[l];
            if (blk.div_term)          std::free(blk.div_term);
            if (blk.conv1d_weight)     std::free(blk.conv1d_weight);
            if (blk.conv1d_bias)       std::free(blk.conv1d_bias);
            if (blk.norm_conv.weight)  std::free(blk.norm_conv.weight);
            if (blk.norm_conv.bias)    std::free(blk.norm_conv.bias);

            if (blk.memory.has_memory) {
                free_weight_array(blk.memory.proj_qkv_beta);
                if (blk.memory.q_scale)   std::free(blk.memory.q_scale);
                if (blk.memory.beta_bias) std::free(blk.memory.beta_bias);
                if (blk.memory.decay)     std::free(blk.memory.decay);
                free_weight_array(blk.memory.proj_alpha_weight);
                if (blk.memory.proj_alpha_bias) std::free(blk.memory.proj_alpha_bias);

                if (blk.memory.axes_proj_k) {
                    for (int a = 0; a < ctx->header.num_axes; ++a) {
                        if (blk.memory.axes_proj_k[a]) std::free(blk.memory.axes_proj_k[a]);
                        if (blk.memory.axes_proj_q[a]) std::free(blk.memory.axes_proj_q[a]);
                    }
                    std::free(blk.memory.axes_proj_k);
                    std::free(blk.memory.axes_proj_q);
                }
                if (blk.memory.slot_proj_k) std::free(blk.memory.slot_proj_k);
                if (blk.memory.slot_proj_q) std::free(blk.memory.slot_proj_q);

                free_weight_array(blk.memory.out_proj);
                if (blk.memory.norm.weight) std::free(blk.memory.norm.weight);
                if (blk.memory.norm.bias)   std::free(blk.memory.norm.bias);
            }

            free_weight_array(blk.ffn_fc1_weight);
            if (blk.ffn_fc1_bias)     std::free(blk.ffn_fc1_bias);
            free_weight_array(blk.ffn_fc2_weight);
            if (blk.ffn_fc2_bias)     std::free(blk.ffn_fc2_bias);
            if (blk.norm_ffn.weight)  std::free(blk.norm_ffn.weight);
            if (blk.norm_ffn.bias)    std::free(blk.norm_ffn.bias);
        }
        std::free(ctx->blocks);
    }

    if (ctx->states) {
        for (int l = 0; l < L; ++l) {
            if (ctx->states[l].conv_state) std::free(ctx->states[l].conv_state);
            if (ctx->states[l].mem_state)  std::free(ctx->states[l].mem_state);
        }
        std::free(ctx->states);
    }

    if (ctx->final_norm.weight) std::free(ctx->final_norm.weight);
    if (ctx->final_norm.bias)   std::free(ctx->final_norm.bias);

    if (!ctx->head_is_tied) {
        free_weight_array(ctx->head_weight);
    }
    if (ctx->head_bias) std::free(ctx->head_bias);

    free_weight_array(ctx->forecast_fc1_weight);
    if (ctx->forecast_fc1_bias) std::free(ctx->forecast_fc1_bias);
    free_weight_array(ctx->forecast_fc2_weight);
    if (ctx->forecast_fc2_bias) std::free(ctx->forecast_fc2_bias);

    if (ctx->hidden_buf)       std::free(ctx->hidden_buf);
    if (ctx->x_in_buf)         std::free(ctx->x_in_buf);
    if (ctx->c_out_buf)        std::free(ctx->c_out_buf);
    if (ctx->h_conv_buf)       std::free(ctx->h_conv_buf);
    if (ctx->packed_qkvb_buf)  std::free(ctx->packed_qkvb_buf);
    if (ctx->q_buf)            std::free(ctx->q_buf);
    if (ctx->k_buf)            std::free(ctx->k_buf);
    if (ctx->v_buf)            std::free(ctx->v_buf);
    if (ctx->beta_buf)         std::free(ctx->beta_buf);
    if (ctx->alpha_buf)        std::free(ctx->alpha_buf);
    if (ctx->yt_buf)           std::free(ctx->yt_buf);
    if (ctx->mem_proj_buf)     std::free(ctx->mem_proj_buf);
    if (ctx->mem_out_buf)      std::free(ctx->mem_out_buf);
    if (ctx->ffn_fc1_buf)      std::free(ctx->ffn_fc1_buf);
    if (ctx->ffn_glu_buf)      std::free(ctx->ffn_glu_buf);
    if (ctx->ffn_fc2_buf)      std::free(ctx->ffn_fc2_buf);
    if (ctx->ffn_norm_buf)     std::free(ctx->ffn_norm_buf);
    if (ctx->final_norm_buf)   std::free(ctx->final_norm_buf);
    if (ctx->forecast_hid_buf) std::free(ctx->forecast_hid_buf);

    if (ctx->wk_buf)         std::free(ctx->wk_buf);
    if (ctx->wq_buf)         std::free(ctx->wq_buf);
    if (ctx->active_idx_buf) std::free(ctx->active_idx_buf);
    if (ctx->sk_buf)         std::free(ctx->sk_buf);
    if (ctx->sq_buf)         std::free(ctx->sq_buf);
    if (ctx->uk_axes_buf)    std::free(ctx->uk_axes_buf);
    if (ctx->uq_axes_buf)    std::free(ctx->uq_axes_buf);
    if (ctx->idx_axes_buf)   std::free(ctx->idx_axes_buf);
    if (ctx->think_mem_buf)  std::free(ctx->think_mem_buf);

    std::free(ctx);
}

