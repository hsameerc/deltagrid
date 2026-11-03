// =============================================================================
// HSRU DeltaGrid Internal Engine Structures & Binary Specification
// =============================================================================
//
// Binary File Layout ('DLG1' Container Format):
// ---------------------------------------------
//   +-----------------------------------------------------------------------+
//   | 1. 64-Byte Header (DeltaGridBinHeader: 16 x int32_t)                  |
//   |    - magic ('DLG1'), version (1), model_type (0=Cont, 1=LM)           |
//   |    - precision_mode (0=FP32, 1=INT8, 2=FXP Q16.16)                    |
//   |    - num_layers, d_model, num_heads, kernel_size, thinking_cycles     |
//   |    - input_or_vocab, output_dim, forecast_horizon                     |
//   |    - num_axes, top_k, actual_slots, flags                             |
//   +-----------------------------------------------------------------------+
//   | 2. Optional Multi-Axis Coordinate Dimensions (if num_axes > 0)        |
//   |    - int32_t grid_dim[num_axes]                                       |
//   +-----------------------------------------------------------------------+
//   | 3. Input Feature Encoder (Continuous) OR Token Embedding Table (LM)   |
//   +-----------------------------------------------------------------------+
//   | 4. Stacked 4-Tier DeltaGrid Blocks (for l = 0 .. num_layers - 1)      |
//   |    - Tier 0: Biological Harmonic Clock (div_term, clock_scale)        |
//   |    - Tier 1: Causal Depthwise Conv1D (weights, bias, norm_conv)       |
//   |    - Tier 2: Associative DeltaGrid Memory (proj_qkv_beta, routing,    |
//   |              decay/proj_alpha, out_proj, norm)                        |
//   |    - Tier 3: GLU Logic Reasoning Engine (fc1, fc2, norm_ffn)          |
//   +-----------------------------------------------------------------------+
//   | 5. Pre-Readout LayerNorm & Readout Head (+ optional Forecast Head)    |
//   +-----------------------------------------------------------------------+
// =============================================================================

#ifndef DELTAGRID_INTERNAL_H
#define DELTAGRID_INTERNAL_H

#include "deltagrid.h"
#include "deltagrid_llm.h"
#include <stdint.h>
#include <stddef.h>

extern bool is_deltagrid_unlocked;

// Magic header for DeltaGrid binary weight files ('DLG1' in little-endian)
constexpr int32_t DELTAGRID_BIN_MAGIC   = 0x31474C44;
// Magic header for saved recurrent state snapshots ('DLGS' in little-endian)
constexpr int32_t DELTAGRID_STATE_MAGIC = 0x53474C44;
constexpr int32_t DELTAGRID_BIN_VERSION = 1;

// Model Architecture Identifiers
enum DeltaGridModelType {
    DG_MODEL_CONTINUOUS = 0, // DeltaGridContinuous (telemetry, robotics, physics)
    DG_MODEL_LM         = 1  // DeltaGridLM (autoregressive causal language model)
};

// Numerical Precision Identifiers
enum DeltaGridPrecisionMode {
    DG_PREC_FP32 = 0, // IEEE-754 32-bit Floating Point
    DG_PREC_INT8 = 1, // Symmetric 8-bit Integer Weights with FP32 Scale (Q8_0 style)
    DG_PREC_FXP  = 2  // Q16.16 Fixed-Point 32-bit Integer (FPU-free MCU runtime)
};

// Header Feature Bitflags
constexpr int32_t DG_FLAG_DYNAMIC_DECAY = (1 << 0); // Data-dependent forget gate alpha_t
constexpr int32_t DG_FLAG_HAS_SLOT_PROJ = (1 << 1); // Linear slot expansion (actual_slots != head_dim)
constexpr int32_t DG_FLAG_TIED_LM_HEAD  = (1 << 2); // LM readout head shares embedding weight buffer

// 64-byte Binary Header (16 x int32_t, 64-bit aligned)
struct DeltaGridBinHeader {
    int32_t magic;            // DELTAGRID_BIN_MAGIC (0x31474C44)
    int32_t version;          // Format version (1)
    int32_t model_type;       // 0 = Continuous, 1 = LM
    int32_t precision_mode;   // 0 = FP32, 1 = INT8 (Q8), 2 = FXP (Q16.16)
    int32_t num_layers;       // Number of stacked 4-tier DeltaGrid blocks
    int32_t d_model;          // Latent embedding dimension D
    int32_t num_heads;        // Number of associative memory heads H
    int32_t kernel_size;      // Tier 1 Conv1D kernel size K (default 4)
    int32_t thinking_cycles;  // Cognitive recurrence cycles per step (default 0)
    int32_t input_or_vocab;   // input_dim (Continuous) or vocab_size (LM)
    int32_t output_dim;       // output_dim (Continuous) or vocab_size (LM)
    int32_t forecast_horizon; // 0 if None, >0 if multi-horizon trajectory head present
    int32_t num_axes;         // 0 if Standard Dense, >=2 if Kronecker / Sparse PKM grid_dim
    int32_t top_k;            // 0 if Dense, >=1 if Sparse PKM Top-K coordinate routing
    int32_t actual_slots;     // Total memory slots per head (Dh, num_slots, or prod(grid_dim))
    int32_t flags;            // Bitmask of DG_FLAG_*
};

// Dual-Precision Weight Array (FP32 or Symmetric INT8 with Dequantization Scale)
struct DGWeightArray {
    float*  f32;   // Non-null when precision_mode == DG_PREC_FP32
    int8_t* q8;    // Non-null when precision_mode == DG_PREC_INT8
    float   scale; // Symmetric dequantization factor: w_f32[i] = q8[i] * scale
};

// Standard LayerNorm Parameters (y = (x - mean) / sqrt(var + 1e-5) * weight + bias)
struct DGLayerNormWeights {
    float* weight; // [d_model]
    float* bias;   // [d_model]
};

// Tier 2: Associative DeltaGrid Memory Layer Weights
struct DGMemoryLayerWeights {
    int has_memory;
    float scale_mem;

    // Fused projection computing [q, k, v, beta_raw] in a single matrix-vector product
    DGWeightArray proj_qkv_beta; // [3 * d_model + num_heads, d_model]
    float* q_scale;              // [num_heads] learnable query scaling
    float* beta_bias;            // [num_heads] write gate bias
    float* decay;                // [num_heads] static decay factor alpha

    // Optional data-dependent dynamic decay gate: alpha_t = 0.99 + 0.01 * sigmoid(proj_alpha(x))
    DGWeightArray proj_alpha_weight; // [num_heads, d_model]
    float* proj_alpha_bias;          // [num_heads]

    // Coordinate routing weights:
    //   - Mode 2 (Kronecker Dense) & Mode 3 (Sparse Delta-PKM):
    //     axes_proj_k[a] and axes_proj_q[a] have shape [grid_dim[a], head_dim]
    float** axes_proj_k;
    float** axes_proj_q;

    //   - Mode 1 (Standard Dense with slot expansion actual_slots != head_dim):
    //     slot_proj_k and slot_proj_q have shape [actual_slots, head_dim]
    float* slot_proj_k;
    float* slot_proj_q;

    // Output readout projection & LayerNorm
    DGWeightArray out_proj;      // [d_model, d_model]
    DGLayerNormWeights norm;     // [d_model]
};

// Complete 4-Tier DeltaGridBlock Weights
struct DGBlockWeights {
    // Tier 0: Biological Multi-Frequency Harmonic Clock
    float* div_term;             // [d_model / 2] frequencies: exp(-2k/D * ln(10000))
    float  clock_scale;          // Learnable phase injection scale (default 0.1)

    // Tier 1: Causal Depthwise Conv1D (local n-gram & transient binding)
    float* conv1d_weight;        // [d_model * kernel_size] depthwise filters
    float* conv1d_bias;          // [d_model]
    DGLayerNormWeights norm_conv;

    // Tier 2: Associative Fast-Weight Memory Layer
    DGMemoryLayerWeights memory;

    // Tier 3: Non-Linear GLU Logic Reasoning Engine
    //   ffn(x) = fc2( GLU( fc1(x) ) )
    float  scale_logic;
    DGWeightArray ffn_fc1_weight; // [2 * d_model, d_model]
    float*        ffn_fc1_bias;   // [2 * d_model]
    DGWeightArray ffn_fc2_weight; // [d_model, d_model]
    float*        ffn_fc2_bias;   // [d_model]
    DGLayerNormWeights norm_ffn;
};

// Persistent Recurrent State per Layer (Strictly Constant O(1) Footprint)
struct DGLayerState {
    float* conv_state; // [d_model * (kernel_size - 1)] rolling causal Conv1D window
    float* mem_state;  // [num_heads * actual_slots * head_dim] fast-weight matrix S
};

// Unified Runtime Context for both DeltaGridContinuous and DeltaGridLM
struct DGRuntimeContext {
    DeltaGridBinHeader header;
    int head_dim;        // d_model / num_heads
    int active_slots;    // Active slots touched per step (top_k^num_axes for PKM, else actual_slots)
    int max_axis_dim;    // max_a(grid_dim[a])
    int max_axis_active; // top_k (if Sparse PKM) or max_axis_dim (if Kronecker Dense)
    int step_count;      // Monotonic sequence counter for Tier 0 Harmonic Clock

    int32_t* grid_dim;   // [num_axes] coordinate dimensions if num_axes > 0

    // Input encoder: either feature_proj (Continuous) or token embed table (LM)
    DGWeightArray input_weight; // [d_model, input_dim] or [vocab_size, d_model]
    float*        input_bias;   // [d_model] (Continuous only; nullptr for LM)

    // Stacked 4-tier blocks and their recurrent states
    DGBlockWeights* blocks;     // [num_layers]
    DGLayerState*   states;     // [num_layers]

    // Pre-readout LayerNorm
    DGLayerNormWeights final_norm;

    // Readout head: either linear head (Continuous) or lm_head (LM)
    DGWeightArray head_weight;  // [output_dim, d_model] or [vocab_size, d_model]
    float*        head_bias;    // [output_dim] (Continuous only; nullptr for LM)
    bool          head_is_tied; // true if head_weight shares input_weight pointers

    // Multi-Horizon Trajectory Forecast Head (Continuous only, when forecast_horizon > 0)
    DGWeightArray forecast_fc1_weight; // [d_model, d_model]
    float*        forecast_fc1_bias;   // [d_model]
    DGWeightArray forecast_fc2_weight; // [forecast_horizon * output_dim, d_model]
    float*        forecast_fc2_bias;   // [forecast_horizon * output_dim]

    // Preallocated Scratch Buffers (Allocated once at init -> Zero malloc during step!)
    float* hidden_buf;       // [d_model] current layer latent vector
    float* x_in_buf;         // [d_model] clock-anchored block residual stream
    float* c_out_buf;        // [d_model] Conv1D + SiLU residual sum
    float* h_conv_buf;       // [d_model] LayerNormed Conv1D output feeding Tier 2 Memory
    float* packed_qkvb_buf;  // [3 * d_model + num_heads] raw proj_qkv_beta output
    float* q_buf;            // [d_model] query vectors [num_heads, head_dim]
    float* k_buf;            // [d_model] L2-normalized key vectors [num_heads, head_dim]
    float* v_buf;            // [d_model] value vectors [num_heads, head_dim]
    float* beta_buf;         // [num_heads] sigmoid write gates in (0, 1)
    float* alpha_buf;        // [num_heads] retention decay factors in (0.99, 1.0)
    float* yt_buf;           // [d_model] associative memory readout [num_heads, head_dim]
    float* mem_proj_buf;     // [d_model] out_proj(yt_buf)
    float* mem_out_buf;      // [d_model] norm(out_proj(yt_buf))
    float* ffn_fc1_buf;      // [2 * d_model] GLU pre-activation
    float* ffn_glu_buf;      // [d_model] GLU gated activation
    float* ffn_fc2_buf;      // [d_model] FFN linear projection output
    float* ffn_norm_buf;     // [d_model] LayerNormed FFN output
    float* final_norm_buf;   // [d_model] pre-head LayerNorm output
    float* forecast_hid_buf; // [d_model] multi-horizon forecast hidden state

    // Per-head coordinate routing scratch buffers
    float*   wk_buf;         // [num_heads * active_slots] Kronecker / routed key weights
    float*   wq_buf;         // [num_heads * active_slots] Kronecker / routed query weights
    int32_t* active_idx_buf; // [num_heads * active_slots] active slot indices for Sparse PKM
    float*   sk_buf;         // [num_heads * max_axis_dim] per-axis key logits
    float*   sq_buf;         // [num_heads * max_axis_dim] per-axis query logits
    float*   uk_axes_buf;    // [num_heads * num_axes * max_axis_active] normalized axis keys
    float*   uq_axes_buf;    // [num_heads * num_axes * max_axis_active] normalized axis queries
    int32_t* idx_axes_buf;   // [num_heads * num_axes * max_axis_active] Top-K axis indices

    // Scratch memory state for cognitive thinking_cycles > 0
    float*   think_mem_buf;  // [num_heads * actual_slots * head_dim]
};

// Shared Internal Core Routines (implemented in deltagrid_math.cpp)
DGRuntimeContext* dg_load_context_from_file(const char* weights_path, int expected_model_type);
void dg_free_runtime_context(DGRuntimeContext* ctx);
void dg_reset_runtime_state(DGRuntimeContext* ctx);
int  dg_save_runtime_state(DGRuntimeContext* ctx, const char* filepath);
int  dg_load_runtime_state(DGRuntimeContext* ctx, const char* filepath);
void dg_step_all_blocks(DGRuntimeContext* ctx);

#endif // DELTAGRID_INTERNAL_H
