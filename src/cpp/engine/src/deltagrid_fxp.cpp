// =============================================================================
// HSRU DeltaGrid Q16.16 Fixed-Point C++ Edge Engine (Embedded MCUs / IEC 62304)
// =============================================================================
//
// Overview:
// ---------
// Provides a 100% integer-based runtime for DeltaGridContinuous models on
// microcontrollers (e.g., ARM Cortex-M, RISC-V, DSPs) that lack hardware
// Floating-Point Units (FPUs) or require strict deterministic integer bounds.
//
// Q16.16 Fixed-Point Representation:
// ----------------------------------
//   - Every real number x is stored as a signed 32-bit integer:
//       x_fxp = round(x * 2^16) = round(x * 65536)
//   - Range: [-32768.0, +32767.9999847]
//   - Resolution: 1 / 65536 ≈ 0.00001526 (1.53e-5)
//   - Multiplication of two Q16.16 numbers produces a Q32.32 intermediate in
//     64-bit integer space (int64_t), which is rounded by adding 2^15 (DG_FXP_HALF)
//     and shifted right by 16 bits (>> 16) back to Q16.16.
//
// Supported DeltaGrid Modes in Q16.16:
// ------------------------------------
//   1. Mode 1 (Standard Dense): S in Z^{H x Slots x Dh}
//   2. Mode 2 (Factorized Kronecker Dense): Multi-axis Kronecker routing
//   3. Mode 3 (Sparse Delta-PKM Top-K): Exponential virtual slots with Top-K routing
//   4. Multi-Horizon Trajectory Forecasting Head (deltagrid_step_forecast_fxp)
//   5. Cognitive Working-Memory Recurrence (thinking_cycles > 0)
//   6. Deterministic State Checkpointing (deltagrid_save_state_fxp / load_state_fxp)
// =============================================================================

#ifndef DELTAGRID_EXPORTS
#define DELTAGRID_EXPORTS
#endif

#include "deltagrid.h"
#include "deltagrid_internal.h"
#include "deltagrid_math_fxp.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>

// -----------------------------------------------------------------------------
// Internal Q16.16 Weight & State Structures
// -----------------------------------------------------------------------------

struct DGFxpLayerNorm {
    int32_t* weight; // [d_model] in Q16.16
    int32_t* bias;   // [d_model] in Q16.16
};

struct DGFxpMemoryLayer {
    int has_memory;
    int32_t scale_mem;        // Q16.16 scalar residual scale

    int32_t* proj_qkv_beta;   // [(3*D + H) * D] fused Q, K, V, and beta projection
    int32_t* q_scale;         // [H] per-head query routing scale
    int32_t* beta_bias;       // [H] per-head write gate bias
    int32_t* decay;           // [H] per-head static retention factor alpha

    // Optional data-dependent dynamic decay gate: alpha_t = 0.99 + 0.01 * sigmoid(W_a x + b_a)
    int32_t* proj_alpha_w;    // [H * D]
    int32_t* proj_alpha_b;    // [H]

    // Coordinate routing projections
    int32_t** axes_proj_k;    // [num_axes][grid_dim[a] * Dh] for Kronecker / Sparse PKM
    int32_t** axes_proj_q;    // [num_axes][grid_dim[a] * Dh]
    int32_t*  slot_proj_k;    // [actual_slots * Dh] when actual_slots != Dh in Dense mode
    int32_t*  slot_proj_q;    // [actual_slots * Dh]

    // Output projection & LayerNorm
    int32_t* out_proj;        // [D * D]
    DGFxpLayerNorm norm;      // [D]
};

struct DGFxpBlock {
    // Tier 0: Biological Harmonic Clock
    int32_t* div_term;        // [D / 2] frequency table
    int32_t  clock_scale;     // Q16.16 clock amplitude

    // Tier 1: Causal Depthwise Conv1D (k = kernel_size)
    int32_t* conv1d_weight;   // [D * K]
    int32_t* conv1d_bias;     // [D]
    DGFxpLayerNorm norm_conv;

    // Tier 2: Associative DeltaGrid Memory
    DGFxpMemoryLayer memory;

    // Tier 3: Non-Linear GLU Logic Reasoning Engine
    int32_t  scale_logic;
    int32_t* ffn_fc1_weight;  // [2 * D * D]
    int32_t* ffn_fc1_bias;    // [2 * D]
    int32_t* ffn_fc2_weight;  // [D * D]
    int32_t* ffn_fc2_bias;    // [D]
    DGFxpLayerNorm norm_ffn;
};

struct DGFxpLayerState {
    int32_t* conv_state;      // [D * (K - 1)] causal shift register
    int32_t* mem_state;       // [H * actual_slots * Dh] fast-weight matrix S
};

struct DGFxpContext {
    DeltaGridBinHeader header;
    int head_dim;
    int active_slots;
    int max_axis_dim;
    int max_axis_active;
    int step_count;

    int32_t* grid_dim;        // [num_axes] if num_axes > 0

    // Continuous sensory encoder
    int32_t* input_weight;    // [D * input_dim]
    int32_t* input_bias;      // [D]

    // Stacked 4-tier blocks & states
    DGFxpBlock*      blocks;
    DGFxpLayerState* states;

    // Readout head
    DGFxpLayerNorm final_norm;
    int32_t* head_weight;     // [output_dim * D]
    int32_t* head_bias;       // [output_dim]

    // Optional Multi-Horizon Trajectory Forecast Head
    int32_t* forecast_fc1_weight; // [D * D]
    int32_t* forecast_fc1_bias;   // [D]
    int32_t* forecast_fc2_weight; // [forecast_horizon * output_dim * D]
    int32_t* forecast_fc2_bias;   // [forecast_horizon * output_dim]

    // Preallocated Scratch Buffers (strictly O(1) RAM, zero malloc during step)
    int32_t* hidden_buf;
    int32_t* x_in_buf;
    int32_t* c_out_buf;
    int32_t* h_conv_buf;
    int32_t* packed_qkvb_buf;
    int32_t* q_buf;
    int32_t* k_buf;
    int32_t* v_buf;
    int32_t* beta_buf;
    int32_t* alpha_buf;
    int32_t* yt_buf;
    int32_t* mem_proj_buf;
    int32_t* mem_out_buf;
    int32_t* ffn_fc1_buf;
    int32_t* ffn_glu_buf;
    int32_t* ffn_fc2_buf;
    int32_t* ffn_norm_buf;
    int32_t* final_norm_buf;
    int32_t* forecast_hid_buf;

    // Coordinate routing scratch buffers
    int32_t* wk_buf;
    int32_t* wq_buf;
    int32_t* active_idx_buf;
    int32_t* sk_buf;
    int32_t* sq_buf;
    int32_t* uk_axes_buf;
    int32_t* uq_axes_buf;
    int32_t* idx_axes_buf;

    // Scratch memory state for cognitive thinking_cycles > 0
    int32_t* think_mem_buf;
};

static void load_i32_array(std::ifstream& f, int32_t*& arr, size_t n) {
    if (n == 0) {
        arr = nullptr;
        return;
    }
    arr = (int32_t*)std::malloc(n * sizeof(int32_t));
    f.read(reinterpret_cast<char*>(arr), n * sizeof(int32_t));
}

// -----------------------------------------------------------------------------
// Internal Tier 2 Q16.16 Memory Layer Step (Dense, Kronecker & Sparse PKM)
// -----------------------------------------------------------------------------

static void dg_memory_layer_step_fxp(
    DGFxpContext* ctx,
    const DGFxpMemoryLayer& mem,
    const int32_t* h_in,
    int32_t* mem_state,
    int32_t* mem_out
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

    // 1. Project Q, K, V, and beta_raw in a single fused MAC call
    dg_matmul_fxp(h_in, mem.proj_qkv_beta, nullptr, ctx->packed_qkvb_buf, 3 * D + H, D);

    std::memcpy(ctx->q_buf, ctx->packed_qkvb_buf, D * sizeof(int32_t));
    std::memcpy(ctx->v_buf, ctx->packed_qkvb_buf + 2 * D, D * sizeof(int32_t));

    for (int h = 0; h < H; ++h) {
        dg_l2_normalize_fxp(ctx->packed_qkvb_buf + D + h * Dh, ctx->k_buf + h * Dh, Dh);
        ctx->beta_buf[h] = dg_sigmoid_fxp(ctx->packed_qkvb_buf[3 * D + h] + mem.beta_bias[h]);
    }

    // 2. Compute retention factor alpha_t
    if (dyn_decay) {
        dg_matmul_fxp(h_in, mem.proj_alpha_w, mem.proj_alpha_b, ctx->alpha_buf, H, D);
        int32_t c_099 = DG_FLOAT_TO_FXP(0.99f);
        int32_t c_001 = DG_FLOAT_TO_FXP(0.01f);
        for (int h = 0; h < H; ++h) {
            ctx->alpha_buf[h] = c_099 + dg_fxp_mul(c_001, dg_sigmoid_fxp(ctx->alpha_buf[h]));
        }
    } else {
        for (int h = 0; h < H; ++h) {
            ctx->alpha_buf[h] = mem.decay[h];
        }
    }

    // 3. Per-head Coordinate Routing & Householder Delta Update
    for (int h = 0; h < H; ++h) {
        const int32_t* q_h = ctx->q_buf + h * Dh;
        const int32_t* k_h = ctx->k_buf + h * Dh;
        const int32_t* v_h = ctx->v_buf + h * Dh;
        int32_t* yt_h      = ctx->yt_buf + h * Dh;
        int32_t* state_h   = mem_state + (size_t)h * actual_slots * Dh;

        int32_t beta_h  = ctx->beta_buf[h];
        int32_t alpha_h = ctx->alpha_buf[h];
        int32_t q_scale = mem.q_scale[h];

        int32_t* wk_h  = ctx->wk_buf + (size_t)h * active_slots;
        int32_t* wq_h  = ctx->wq_buf + (size_t)h * active_slots;
        int32_t* idx_h = ctx->active_idx_buf + (size_t)h * active_slots;

        if (top_k > 0 && num_axes > 0) {
            // -----------------------------------------------------------------
            // Mode 3: Sparse Delta-PKM Top-K Coordinate Routing in Q16.16
            // -----------------------------------------------------------------
            int32_t* sk_h = ctx->sk_buf + (size_t)h * ctx->max_axis_dim;
            int32_t* sq_h = ctx->sq_buf + (size_t)h * ctx->max_axis_dim;

            for (int a = 0; a < num_axes; ++a) {
                int M_a = ctx->grid_dim[a];
                int32_t* uk_a  = ctx->uk_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                int32_t* uq_a  = ctx->uq_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                int32_t* idx_a = ctx->idx_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;

                dg_matmul_fxp(k_h, mem.axes_proj_k[a], nullptr, sk_h, M_a, Dh);
                dg_topk_descending_fxp(sk_h, M_a, top_k, uk_a, idx_a);
                dg_l2_normalize_fxp(uk_a, uk_a, top_k);

                dg_matmul_fxp(q_h, mem.axes_proj_q[a], nullptr, sq_h, M_a, Dh);
                for (int j = 0; j < top_k; ++j) {
                    uq_a[j] = sq_h[idx_a[j]];
                }
                dg_l2_normalize_fxp(uq_a, uq_a, top_k);
            }

            // Kronecker product across active axis coordinates
            const int32_t* uk_0  = ctx->uk_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;
            const int32_t* uq_0  = ctx->uq_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;
            const int32_t* idx_0 = ctx->idx_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;

            for (int j = 0; j < top_k; ++j) {
                wk_h[j]  = uk_0[j];
                wq_h[j]  = uq_0[j];
                idx_h[j] = idx_0[j];
            }
            int cur_len = top_k;
            for (int a = 1; a < num_axes; ++a) {
                int M_a = ctx->grid_dim[a];
                const int32_t* uk_a  = ctx->uk_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                const int32_t* uq_a  = ctx->uq_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                const int32_t* idx_a = ctx->idx_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;

                for (int i = cur_len - 1; i >= 0; --i) {
                    int32_t wk_i  = wk_h[i];
                    int32_t wq_i  = wq_h[i];
                    int32_t idx_i = idx_h[i];
                    for (int j = top_k - 1; j >= 0; --j) {
                        int dst = i * top_k + j;
                        wk_h[dst]  = dg_fxp_mul(wk_i, uk_a[j]);
                        wq_h[dst]  = dg_fxp_mul(wq_i, uq_a[j]);
                        idx_h[dst] = idx_i * M_a + idx_a[j];
                    }
                }
                cur_len *= top_k;
            }

            for (int m = 0; m < active_slots; ++m) {
                wq_h[m] = dg_fxp_mul(wq_h[m], q_scale);
            }

            int64_t yt_acc[512]   = {0};
            int64_t pred_acc[512] = {0};

            for (int m = 0; m < active_slots; ++m) {
                int32_t slot = idx_h[m];
                const int32_t* s_row = state_h + (size_t)slot * Dh;
                int64_t wq_m = wq_h[m];
                int64_t wk_m = wk_h[m];
                for (int d = 0; d < Dh; ++d) {
                    yt_acc[d]   += wq_m * s_row[d];
                    pred_acc[d] += wk_m * s_row[d];
                }
            }

            int32_t rt[512];
            for (int d = 0; d < Dh; ++d) {
                yt_h[d] = (int32_t)((yt_acc[d] + DG_FXP_HALF) >> DG_FXP_FRAC_BITS);
                int32_t pred_d = (int32_t)((pred_acc[d] + DG_FXP_HALF) >> DG_FXP_FRAC_BITS);
                rt[d] = v_h[d] - pred_d;
            }

            for (int m = 0; m < active_slots; ++m) {
                int32_t slot = idx_h[m];
                int32_t* s_row = state_h + (size_t)slot * Dh;
                int32_t bw = dg_fxp_mul(beta_h, wk_h[m]);
                for (int d = 0; d < Dh; ++d) {
                    s_row[d] = dg_fxp_mul(alpha_h, s_row[d]) + dg_fxp_mul(bw, rt[d]);
                }
            }
        } else {
            // -----------------------------------------------------------------
            // Mode 1 (Standard Dense) & Mode 2 (Factorized Kronecker Dense)
            // -----------------------------------------------------------------
            if (num_axes > 0) {
                for (int a = 0; a < num_axes; ++a) {
                    int M_a = ctx->grid_dim[a];
                    int32_t* uk_a = ctx->uk_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                    int32_t* uq_a = ctx->uq_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;

                    dg_matmul_fxp(k_h, mem.axes_proj_k[a], nullptr, uk_a, M_a, Dh);
                    dg_l2_normalize_fxp(uk_a, uk_a, M_a);
                    dg_matmul_fxp(q_h, mem.axes_proj_q[a], nullptr, uq_a, M_a, Dh);
                }

                int M_0 = ctx->grid_dim[0];
                const int32_t* uk_0 = ctx->uk_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;
                const int32_t* uq_0 = ctx->uq_axes_buf + ((size_t)h * num_axes + 0) * ctx->max_axis_active;
                for (int j = 0; j < M_0; ++j) {
                    wk_h[j] = uk_0[j];
                    wq_h[j] = uq_0[j];
                }
                int cur_len = M_0;
                for (int a = 1; a < num_axes; ++a) {
                    int M_a = ctx->grid_dim[a];
                    const int32_t* uk_a = ctx->uk_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                    const int32_t* uq_a = ctx->uq_axes_buf + ((size_t)h * num_axes + a) * ctx->max_axis_active;
                    for (int i = cur_len - 1; i >= 0; --i) {
                        int32_t wk_i = wk_h[i];
                        int32_t wq_i = wq_h[i];
                        for (int j = M_a - 1; j >= 0; --j) {
                            int dst = i * M_a + j;
                            wk_h[dst] = dg_fxp_mul(wk_i, uk_a[j]);
                            wq_h[dst] = dg_fxp_mul(wq_i, uq_a[j]);
                        }
                    }
                    cur_len *= M_a;
                }
            } else if (has_slot_p) {
                dg_matmul_fxp(k_h, mem.slot_proj_k, nullptr, wk_h, actual_slots, Dh);
                dg_l2_normalize_fxp(wk_h, wk_h, actual_slots);
                dg_matmul_fxp(q_h, mem.slot_proj_q, nullptr, wq_h, actual_slots, Dh);
            } else {
                for (int s = 0; s < actual_slots; ++s) {
                    wk_h[s] = k_h[s];
                    wq_h[s] = q_h[s];
                }
            }

            for (int s = 0; s < actual_slots; ++s) {
                wq_h[s] = dg_fxp_mul(wq_h[s], q_scale);
            }

            int64_t yt_acc[512]   = {0};
            int64_t pred_acc[512] = {0};

            for (int s = 0; s < actual_slots; ++s) {
                const int32_t* s_row = state_h + (size_t)s * Dh;
                int64_t wq_s = wq_h[s];
                int64_t wk_s = wk_h[s];
                for (int d = 0; d < Dh; ++d) {
                    yt_acc[d]   += wq_s * s_row[d];
                    pred_acc[d] += wk_s * s_row[d];
                }
            }

            int32_t rt[512];
            for (int d = 0; d < Dh; ++d) {
                yt_h[d] = (int32_t)((yt_acc[d] + DG_FXP_HALF) >> DG_FXP_FRAC_BITS);
                int32_t pred_d = (int32_t)((pred_acc[d] + DG_FXP_HALF) >> DG_FXP_FRAC_BITS);
                rt[d] = v_h[d] - pred_d;
            }

            for (int s = 0; s < actual_slots; ++s) {
                int32_t* s_row = state_h + (size_t)s * Dh;
                int32_t bw = dg_fxp_mul(beta_h, wk_h[s]);
                for (int d = 0; d < Dh; ++d) {
                    s_row[d] = dg_fxp_mul(alpha_h, s_row[d]) + dg_fxp_mul(bw, rt[d]);
                }
            }
        }
    }

    // 4. Readout projection & LayerNorm
    dg_matmul_fxp(ctx->yt_buf, mem.out_proj, nullptr, ctx->mem_proj_buf, D, D);
    dg_layernorm_fxp(ctx->mem_proj_buf, D, mem.norm.weight, mem.norm.bias, mem_out);
}

static void dg_step_all_blocks_fxp(DGFxpContext* ctx) {
    const int L      = ctx->header.num_layers;
    const int D      = ctx->header.d_model;
    const int K      = ctx->header.kernel_size;
    const int cycles = ctx->header.thinking_cycles;
    const int offset = ctx->step_count;
    const size_t state_i32s = (size_t)ctx->header.num_heads * ctx->header.actual_slots * ctx->head_dim;

    for (int l = 0; l < L; ++l) {
        const DGFxpBlock& blk = ctx->blocks[l];
        DGFxpLayerState& st   = ctx->states[l];

        // Tier 0: Biological Harmonic Clock
        int half_d = D / 2;
        for (int k = 0; k < half_d; ++k) {
            float angle = (float)offset * DG_FXP_TO_FLOAT(blk.div_term[k]);
            int32_t s_val = dg_sin_fxp(angle);
            int32_t c_val = dg_cos_fxp(angle);
            ctx->x_in_buf[2 * k]     = ctx->hidden_buf[2 * k]     + dg_fxp_mul(s_val, blk.clock_scale);
            ctx->x_in_buf[2 * k + 1] = ctx->hidden_buf[2 * k + 1] + dg_fxp_mul(c_val, blk.clock_scale);
        }

        // Tier 1: Causal Depthwise Conv1D
        int hist_len = K - 1;
        for (int c = 0; c < D; ++c) {
            const int32_t* w_c = blk.conv1d_weight + c * K;
            int32_t* hist_c    = st.conv_state + c * hist_len;
            int64_t sum = (int64_t)blk.conv1d_bias[c] << DG_FXP_FRAC_BITS;
            for (int j = 0; j < hist_len; ++j) {
                sum += (int64_t)w_c[j] * (int64_t)hist_c[j];
            }
            sum += (int64_t)w_c[hist_len] * (int64_t)ctx->x_in_buf[c];
            int32_t conv_val = (int32_t)((sum + DG_FXP_HALF) >> DG_FXP_FRAC_BITS);

            ctx->c_out_buf[c] = ctx->x_in_buf[c] + dg_silu_fxp(conv_val);

            for (int j = 0; j < hist_len - 1; ++j) {
                hist_c[j] = hist_c[j + 1];
            }
            if (hist_len > 0) {
                hist_c[hist_len - 1] = ctx->x_in_buf[c];
            }
        }
        dg_layernorm_fxp(ctx->c_out_buf, D, blk.norm_conv.weight, blk.norm_conv.bias, ctx->h_conv_buf);

        // Tier 2: Associative DeltaGrid Memory
        if (blk.memory.has_memory) {
            dg_memory_layer_step_fxp(ctx, blk.memory, ctx->h_conv_buf, st.mem_state, ctx->mem_out_buf);
            for (int i = 0; i < D; ++i) {
                ctx->x_in_buf[i] += dg_fxp_mul(ctx->mem_out_buf[i], blk.memory.scale_mem);
            }
        }

        // Tier 3: GLU Logic Reasoning Engine
        dg_matmul_fxp(ctx->x_in_buf, blk.ffn_fc1_weight, blk.ffn_fc1_bias, ctx->ffn_fc1_buf, 2 * D, D);
        dg_glu_fxp(ctx->ffn_fc1_buf, ctx->ffn_glu_buf, D);
        dg_matmul_fxp(ctx->ffn_glu_buf, blk.ffn_fc2_weight, blk.ffn_fc2_bias, ctx->ffn_fc2_buf, D, D);
        dg_layernorm_fxp(ctx->ffn_fc2_buf, D, blk.norm_ffn.weight, blk.norm_ffn.bias, ctx->ffn_norm_buf);
        for (int i = 0; i < D; ++i) {
            ctx->x_in_buf[i] += dg_fxp_mul(ctx->ffn_norm_buf[i], blk.scale_logic);
        }

        // Cognitive Thinking Loops (Working-Memory Recurrence)
        if (cycles > 0 && blk.memory.has_memory && ctx->think_mem_buf != nullptr) {
            std::memcpy(ctx->think_mem_buf, st.mem_state, state_i32s * sizeof(int32_t));
            for (int cyc = 0; cyc < cycles; ++cyc) {
                dg_memory_layer_step_fxp(ctx, blk.memory, ctx->x_in_buf, ctx->think_mem_buf, ctx->mem_out_buf);
                for (int i = 0; i < D; ++i) {
                    ctx->x_in_buf[i] += dg_fxp_mul(ctx->mem_out_buf[i], blk.memory.scale_mem);
                }
                dg_matmul_fxp(ctx->x_in_buf, blk.ffn_fc1_weight, blk.ffn_fc1_bias, ctx->ffn_fc1_buf, 2 * D, D);
                dg_glu_fxp(ctx->ffn_fc1_buf, ctx->ffn_glu_buf, D);
                dg_matmul_fxp(ctx->ffn_glu_buf, blk.ffn_fc2_weight, blk.ffn_fc2_bias, ctx->ffn_fc2_buf, D, D);
                dg_layernorm_fxp(ctx->ffn_fc2_buf, D, blk.norm_ffn.weight, blk.norm_ffn.bias, ctx->ffn_norm_buf);
                for (int i = 0; i < D; ++i) {
                    ctx->x_in_buf[i] += dg_fxp_mul(ctx->ffn_norm_buf[i], blk.scale_logic);
                }
            }
        }

        std::memcpy(ctx->hidden_buf, ctx->x_in_buf, D * sizeof(int32_t));
    }

    ctx->step_count += 1;
}

extern "C" {

DELTAGRID_API void* deltagrid_init_context_fxp(const char* weights_path) {
    if (!is_deltagrid_unlocked || weights_path == nullptr) return nullptr;

    std::ifstream f(weights_path, std::ios::binary);
    if (!f.is_open()) return nullptr;

    DeltaGridBinHeader hdr;
    f.read(reinterpret_cast<char*>(&hdr), sizeof(DeltaGridBinHeader));
    if (!f.good() || hdr.magic != DELTAGRID_BIN_MAGIC || hdr.precision_mode != DG_PREC_FXP) {
        return nullptr;
    }

    DGFxpContext* ctx = (DGFxpContext*)std::calloc(1, sizeof(DGFxpContext));
    ctx->header     = hdr;
    ctx->head_dim   = hdr.d_model / hdr.num_heads;
    ctx->step_count = 0;

    const int L      = hdr.num_layers;
    const int D      = hdr.d_model;
    const int H      = hdr.num_heads;
    const int Dh     = ctx->head_dim;
    const int K      = hdr.kernel_size;
    const int slots  = hdr.actual_slots;
    const bool dyn_decay  = (hdr.flags & DG_FLAG_DYNAMIC_DECAY) != 0;
    const bool has_slot_p = (hdr.flags & DG_FLAG_HAS_SLOT_PROJ) != 0;

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
        ctx->active_slots    = slots;
    }

    load_i32_array(f, ctx->input_weight, (size_t)D * hdr.input_or_vocab);
    load_i32_array(f, ctx->input_bias, D);

    ctx->blocks = (DGFxpBlock*)std::calloc(L, sizeof(DGFxpBlock));
    ctx->states = (DGFxpLayerState*)std::calloc(L, sizeof(DGFxpLayerState));

    for (int l = 0; l < L; ++l) {
        DGFxpBlock& blk = ctx->blocks[l];
        int32_t has_mem = 1;
        f.read(reinterpret_cast<char*>(&has_mem), sizeof(int32_t));
        blk.memory.has_memory = has_mem;

        load_i32_array(f, blk.div_term, D / 2);
        f.read(reinterpret_cast<char*>(&blk.clock_scale), sizeof(int32_t));

        load_i32_array(f, blk.conv1d_weight, (size_t)D * K);
        load_i32_array(f, blk.conv1d_bias, D);
        load_i32_array(f, blk.norm_conv.weight, D);
        load_i32_array(f, blk.norm_conv.bias, D);

        if (has_mem) {
            f.read(reinterpret_cast<char*>(&blk.memory.scale_mem), sizeof(int32_t));
            load_i32_array(f, blk.memory.proj_qkv_beta, (size_t)(3 * D + H) * D);
            load_i32_array(f, blk.memory.q_scale, H);
            load_i32_array(f, blk.memory.beta_bias, H);
            load_i32_array(f, blk.memory.decay, H);

            if (dyn_decay) {
                load_i32_array(f, blk.memory.proj_alpha_w, (size_t)H * D);
                load_i32_array(f, blk.memory.proj_alpha_b, H);
            }

            if (hdr.num_axes > 0) {
                blk.memory.axes_proj_k = (int32_t**)std::calloc(hdr.num_axes, sizeof(int32_t*));
                blk.memory.axes_proj_q = (int32_t**)std::calloc(hdr.num_axes, sizeof(int32_t*));
                for (int a = 0; a < hdr.num_axes; ++a) {
                    int M_a = ctx->grid_dim[a];
                    load_i32_array(f, blk.memory.axes_proj_k[a], (size_t)M_a * Dh);
                    load_i32_array(f, blk.memory.axes_proj_q[a], (size_t)M_a * Dh);
                }
            } else if (has_slot_p) {
                load_i32_array(f, blk.memory.slot_proj_k, (size_t)slots * Dh);
                load_i32_array(f, blk.memory.slot_proj_q, (size_t)slots * Dh);
            }

            load_i32_array(f, blk.memory.out_proj, (size_t)D * D);
            load_i32_array(f, blk.memory.norm.weight, D);
            load_i32_array(f, blk.memory.norm.bias, D);
        }

        f.read(reinterpret_cast<char*>(&blk.scale_logic), sizeof(int32_t));
        load_i32_array(f, blk.ffn_fc1_weight, (size_t)(2 * D) * D);
        load_i32_array(f, blk.ffn_fc1_bias, 2 * D);
        load_i32_array(f, blk.ffn_fc2_weight, (size_t)D * D);
        load_i32_array(f, blk.ffn_fc2_bias, D);
        load_i32_array(f, blk.norm_ffn.weight, D);
        load_i32_array(f, blk.norm_ffn.bias, D);

        int hist_len = (K > 1) ? (K - 1) : 1;
        ctx->states[l].conv_state = (int32_t*)std::calloc((size_t)D * hist_len, sizeof(int32_t));
        ctx->states[l].mem_state  = (int32_t*)std::calloc((size_t)H * slots * Dh, sizeof(int32_t));
    }

    load_i32_array(f, ctx->final_norm.weight, D);
    load_i32_array(f, ctx->final_norm.bias, D);
    load_i32_array(f, ctx->head_weight, (size_t)hdr.output_dim * D);
    load_i32_array(f, ctx->head_bias, hdr.output_dim);

    if (hdr.forecast_horizon > 0) {
        load_i32_array(f, ctx->forecast_fc1_weight, (size_t)D * D);
        load_i32_array(f, ctx->forecast_fc1_bias, D);
        load_i32_array(f, ctx->forecast_fc2_weight, (size_t)hdr.forecast_horizon * hdr.output_dim * D);
        load_i32_array(f, ctx->forecast_fc2_bias, (size_t)hdr.forecast_horizon * hdr.output_dim);
    }

    ctx->hidden_buf       = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->x_in_buf         = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->c_out_buf        = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->h_conv_buf       = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->packed_qkvb_buf  = (int32_t*)std::calloc(3 * D + H, sizeof(int32_t));
    ctx->q_buf            = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->k_buf            = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->v_buf            = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->beta_buf         = (int32_t*)std::calloc(H, sizeof(int32_t));
    ctx->alpha_buf        = (int32_t*)std::calloc(H, sizeof(int32_t));
    ctx->yt_buf           = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->mem_proj_buf     = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->mem_out_buf      = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->ffn_fc1_buf      = (int32_t*)std::calloc(2 * D, sizeof(int32_t));
    ctx->ffn_glu_buf      = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->ffn_fc2_buf      = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->ffn_norm_buf     = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->final_norm_buf   = (int32_t*)std::calloc(D, sizeof(int32_t));
    ctx->forecast_hid_buf = (int32_t*)std::calloc(D, sizeof(int32_t));

    ctx->wk_buf         = (int32_t*)std::calloc((size_t)H * ctx->active_slots, sizeof(int32_t));
    ctx->wq_buf         = (int32_t*)std::calloc((size_t)H * ctx->active_slots, sizeof(int32_t));
    ctx->active_idx_buf = (int32_t*)std::calloc((size_t)H * ctx->active_slots, sizeof(int32_t));

    if (hdr.num_axes > 0) {
        ctx->sk_buf       = (int32_t*)std::calloc((size_t)H * ctx->max_axis_dim, sizeof(int32_t));
        ctx->sq_buf       = (int32_t*)std::calloc((size_t)H * ctx->max_axis_dim, sizeof(int32_t));
        ctx->uk_axes_buf  = (int32_t*)std::calloc((size_t)H * hdr.num_axes * ctx->max_axis_active, sizeof(int32_t));
        ctx->uq_axes_buf  = (int32_t*)std::calloc((size_t)H * hdr.num_axes * ctx->max_axis_active, sizeof(int32_t));
        ctx->idx_axes_buf = (int32_t*)std::calloc((size_t)H * hdr.num_axes * ctx->max_axis_active, sizeof(int32_t));
    }

    if (hdr.thinking_cycles > 0) {
        ctx->think_mem_buf = (int32_t*)std::calloc((size_t)H * slots * Dh, sizeof(int32_t));
    }

    return static_cast<void*>(ctx);
}

DELTAGRID_API void deltagrid_step_fxp(void* ctx_ptr, const int32_t* input, int32_t* output) {
    if (!ctx_ptr || !input || !output) return;
    DGFxpContext* ctx = static_cast<DGFxpContext*>(ctx_ptr);

    const int D     = ctx->header.d_model;
    const int in_d  = ctx->header.input_or_vocab;
    const int out_d = ctx->header.output_dim;

    dg_matmul_fxp(input, ctx->input_weight, ctx->input_bias, ctx->hidden_buf, D, in_d);
    dg_step_all_blocks_fxp(ctx);
    dg_layernorm_fxp(ctx->hidden_buf, D, ctx->final_norm.weight, ctx->final_norm.bias, ctx->final_norm_buf);
    dg_matmul_fxp(ctx->final_norm_buf, ctx->head_weight, ctx->head_bias, output, out_d, D);
}

DELTAGRID_API void deltagrid_step_forecast_fxp(
    void* ctx_ptr,
    const int32_t* input,
    int32_t* output,
    int32_t* forecast_out
) {
    if (!ctx_ptr || !input || !output) return;
    DGFxpContext* ctx = static_cast<DGFxpContext*>(ctx_ptr);

    const int D       = ctx->header.d_model;
    const int in_d    = ctx->header.input_or_vocab;
    const int out_d   = ctx->header.output_dim;
    const int horizon = ctx->header.forecast_horizon;

    dg_matmul_fxp(input, ctx->input_weight, ctx->input_bias, ctx->hidden_buf, D, in_d);
    dg_step_all_blocks_fxp(ctx);
    dg_layernorm_fxp(ctx->hidden_buf, D, ctx->final_norm.weight, ctx->final_norm.bias, ctx->final_norm_buf);
    dg_matmul_fxp(ctx->final_norm_buf, ctx->head_weight, ctx->head_bias, output, out_d, D);

    if (forecast_out != nullptr) {
        if (horizon > 0 && ctx->forecast_fc1_weight != nullptr) {
            dg_matmul_fxp(ctx->final_norm_buf, ctx->forecast_fc1_weight, ctx->forecast_fc1_bias, ctx->forecast_hid_buf, D, D);
            for (int i = 0; i < D; ++i) {
                ctx->forecast_hid_buf[i] = dg_silu_fxp(ctx->forecast_hid_buf[i]);
            }
            dg_matmul_fxp(ctx->forecast_hid_buf, ctx->forecast_fc2_weight, ctx->forecast_fc2_bias, forecast_out, horizon * out_d, D);
        } else {
            for (int i = 0; i < out_d; ++i) {
                forecast_out[i] = output[i];
            }
        }
    }
}

DELTAGRID_API void deltagrid_reset_state_fxp(void* ctx_ptr) {
    if (!ctx_ptr) return;
    DGFxpContext* ctx = static_cast<DGFxpContext*>(ctx_ptr);
    const int L     = ctx->header.num_layers;
    const int D     = ctx->header.d_model;
    const int H     = ctx->header.num_heads;
    const int Dh    = ctx->head_dim;
    const int K     = ctx->header.kernel_size;
    const int slots = ctx->header.actual_slots;
    int hist_len = (K > 1) ? (K - 1) : 1;

    ctx->step_count = 0;
    for (int l = 0; l < L; ++l) {
        std::memset(ctx->states[l].conv_state, 0, (size_t)D * hist_len * sizeof(int32_t));
        std::memset(ctx->states[l].mem_state,  0, (size_t)H * slots * Dh * sizeof(int32_t));
    }
}

DELTAGRID_API int deltagrid_save_state_fxp(void* ctx_ptr, const char* filepath) {
    if (!ctx_ptr || !filepath) return -1;
    DGFxpContext* ctx = static_cast<DGFxpContext*>(ctx_ptr);
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
    size_t conv_bytes = (size_t)D * hist_len * sizeof(int32_t);
    size_t mem_bytes  = (size_t)H * slots * ctx->head_dim * sizeof(int32_t);

    for (int l = 0; l < L; ++l) {
        f.write(reinterpret_cast<const char*>(ctx->states[l].conv_state), conv_bytes);
        f.write(reinterpret_cast<const char*>(ctx->states[l].mem_state),  mem_bytes);
    }

    return f.good() ? 0 : -1;
}

DELTAGRID_API int deltagrid_load_state_fxp(void* ctx_ptr, const char* filepath) {
    if (!ctx_ptr || !filepath) return -1;
    DGFxpContext* ctx = static_cast<DGFxpContext*>(ctx_ptr);
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
    size_t conv_bytes = (size_t)D * hist_len * sizeof(int32_t);
    size_t mem_bytes  = (size_t)H * slots * ctx->head_dim * sizeof(int32_t);

    for (int l = 0; l < L; ++l) {
        f.read(reinterpret_cast<char*>(ctx->states[l].conv_state), conv_bytes);
        f.read(reinterpret_cast<char*>(ctx->states[l].mem_state),  mem_bytes);
    }

    return f.good() ? 0 : -1;
}

DELTAGRID_API float deltagrid_compute_normalized_mse_fxp(
    const float* actual,
    const float* predicted,
    int dim
) {
    if (!actual || !predicted || dim <= 0) return 0.0f;
    float sum_sq_err = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = actual[i] - predicted[i];
        sum_sq_err += diff * diff;
    }
    return sum_sq_err / (float)dim;
}

DELTAGRID_API void deltagrid_release_context_fxp(void* ctx_ptr) {
    if (!ctx_ptr) return;
    DGFxpContext* ctx = static_cast<DGFxpContext*>(ctx_ptr);
    const int L = ctx->header.num_layers;

    if (ctx->grid_dim)     std::free(ctx->grid_dim);
    if (ctx->input_weight) std::free(ctx->input_weight);
    if (ctx->input_bias)   std::free(ctx->input_bias);

    if (ctx->blocks) {
        for (int l = 0; l < L; ++l) {
            DGFxpBlock& blk = ctx->blocks[l];
            if (blk.div_term)         std::free(blk.div_term);
            if (blk.conv1d_weight)    std::free(blk.conv1d_weight);
            if (blk.conv1d_bias)      std::free(blk.conv1d_bias);
            if (blk.norm_conv.weight) std::free(blk.norm_conv.weight);
            if (blk.norm_conv.bias)   std::free(blk.norm_conv.bias);

            if (blk.memory.has_memory) {
                if (blk.memory.proj_qkv_beta) std::free(blk.memory.proj_qkv_beta);
                if (blk.memory.q_scale)       std::free(blk.memory.q_scale);
                if (blk.memory.beta_bias)     std::free(blk.memory.beta_bias);
                if (blk.memory.decay)         std::free(blk.memory.decay);
                if (blk.memory.proj_alpha_w)  std::free(blk.memory.proj_alpha_w);
                if (blk.memory.proj_alpha_b)  std::free(blk.memory.proj_alpha_b);

                if (blk.memory.axes_proj_k) {
                    for (int a = 0; a < ctx->header.num_axes; ++a) {
                        if (blk.memory.axes_proj_k[a]) std::free(blk.memory.axes_proj_k[a]);
                        if (blk.memory.axes_proj_q[a]) std::free(blk.memory.axes_proj_q[a]);
                    }
                    std::free(blk.memory.axes_proj_k);
                    std::free(blk.memory.axes_proj_q);
                }
                if (blk.memory.slot_proj_k)   std::free(blk.memory.slot_proj_k);
                if (blk.memory.slot_proj_q)   std::free(blk.memory.slot_proj_q);
                if (blk.memory.out_proj)      std::free(blk.memory.out_proj);
                if (blk.memory.norm.weight)   std::free(blk.memory.norm.weight);
                if (blk.memory.norm.bias)     std::free(blk.memory.norm.bias);
            }

            if (blk.ffn_fc1_weight)  std::free(blk.ffn_fc1_weight);
            if (blk.ffn_fc1_bias)    std::free(blk.ffn_fc1_bias);
            if (blk.ffn_fc2_weight)  std::free(blk.ffn_fc2_weight);
            if (blk.ffn_fc2_bias)    std::free(blk.ffn_fc2_bias);
            if (blk.norm_ffn.weight) std::free(blk.norm_ffn.weight);
            if (blk.norm_ffn.bias)   std::free(blk.norm_ffn.bias);
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

    if (ctx->final_norm.weight)   std::free(ctx->final_norm.weight);
    if (ctx->final_norm.bias)     std::free(ctx->final_norm.bias);
    if (ctx->head_weight)         std::free(ctx->head_weight);
    if (ctx->head_bias)           std::free(ctx->head_bias);
    if (ctx->forecast_fc1_weight) std::free(ctx->forecast_fc1_weight);
    if (ctx->forecast_fc1_bias)   std::free(ctx->forecast_fc1_bias);
    if (ctx->forecast_fc2_weight) std::free(ctx->forecast_fc2_weight);
    if (ctx->forecast_fc2_bias)   std::free(ctx->forecast_fc2_bias);

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

}
