#ifndef DELTA_KERNEL_H
#define DELTA_KERNEL_H

// =============================================================================
//   HSRU DeltaGrid Recurrent Associative Memory CUDA Kernel Header
// =============================================================================
// Architecture & Hardware Specification:
//   - Pure Recurrent Associative State: S in R^{B x H x Slots x Dh}
//   - 100% On-Chip SRAM Residency: State fits entirely within Streaming Multiprocessor
//     (SM) Shared Memory (4 KB - 32 KB per head), completely eliminating DRAM roundtrips.
//   - Closed-Loop Error-Erasing Delta Rule:
//         Read:   y_t = S_{t-1} @ w_q,t
//         Error:  r_t = v_t - S_{t-1} @ w_k,t
//         Write:  S_t = alpha * S_{t-1} + beta_t * (r_t outer w_k,t^T)
// =============================================================================

#include <torch/extension.h>

/**
 * @brief Launches the fused Shared-Memory CUDA forward kernel for full sequences.
 *
 * Each thread block is assigned to a single (batch, head) pair: GridDim = [B * H].
 * State matrix S is kept resident in Shared Memory (SRAM) across all timesteps t in [0, T-1].
 *
 * @tparam scalar_t Precision type (float, double, at::Half, at::BFloat16).
 * @param q Pointer to query tensor [B, H, T, Dh].
 * @param k Pointer to key tensor [B, H, T, Dh].
 * @param v Pointer to value tensor [B, H, T, Dh].
 * @param beta Pointer to scalar write gate tensor [B, H, T].
 * @param wk Pointer to slot key weight tensor [B, H, T, Slots].
 * @param wq Pointer to slot query weight tensor [B, H, T, Slots].
 * @param decay Pointer to per-head retention decay alpha [H].
 * @param s_init Optional pointer to initial recurrent state [B, H, Slots, Dh] (nullptr for zero state).
 * @param out Output pointer for readout representations [B, H, T, Dh].
 * @param s_final Output pointer for final recurrent state [B, H, Slots, Dh].
 * @param err_out Optional output pointer for saving error residual r_t [B, H, T, Dh] (for backward pass).
 * @param B Batch size.
 * @param H Number of associative heads.
 * @param T Sequence length.
 * @param Slots Number of memory slots per head (Dh or grid dimension).
 * @param Dh Hidden dimension per head.
 */
template <typename scalar_t>
void launch_delta_grid_forward(
    const scalar_t* q,           // [B, H, T, Dh]
    const scalar_t* k,           // [B, H, T, Dh]
    const scalar_t* v,           // [B, H, T, Dh]
    const scalar_t* beta,        // [B, H, T]
    const scalar_t* wk,          // [B, H, T, Slots]
    const scalar_t* wq,          // [B, H, T, Slots]
    const scalar_t* decay,       // [H] or [B, H, T]
    const scalar_t* s_init,      // [B, H, Slots, Dh] (can be nullptr)
    scalar_t*       out,         // [B, H, T, Dh]
    scalar_t*       s_final,     // [B, H, Slots, Dh]
    float*          err_out,     // [B, H, T, Dh] float32 exact residual (can be nullptr)
    float*          s_final_f,   // [B, H, Slots, Dh] float32 exact final state (can be nullptr)
    int B, int H, int T, int Slots, int Dh,
    bool dynamic_decay = false
);

/**
 * @brief Launches the ultra-low-latency single-step CUDA kernel for autoregressive decoding.
 *
 * Executes a single recurrent step in constant O(1) time and constant O(1) RAM.
 *
 * @tparam scalar_t Precision type (float, double, at::Half, at::BFloat16).
 * @param q Pointer to single-step query [B, H, Dh].
 * @param k Pointer to single-step key [B, H, Dh].
 * @param v Pointer to single-step value [B, H, Dh].
 * @param beta Pointer to single-step write gate [B, H].
 * @param wk Pointer to single-step slot key weights [B, H, Slots].
 * @param wq Pointer to single-step slot query weights [B, H, Slots].
 * @param decay Pointer to per-head retention decay alpha [H].
 * @param s_prev Pointer to preceding recurrent state [B, H, Slots, Dh].
 * @param out Output pointer for transformed step representation [B, H, Dh].
 * @param s_next Output pointer for updated recurrent state [B, H, Slots, Dh].
 * @param B Batch size.
 * @param H Number of associative heads.
 * @param Slots Number of memory slots per head.
 * @param Dh Hidden dimension per head.
 */
template <typename scalar_t>
void launch_delta_grid_step(
    const scalar_t* q,           // [B, H, Dh]
    const scalar_t* k,           // [B, H, Dh]
    const scalar_t* v,           // [B, H, Dh]
    const scalar_t* beta,        // [B, H]
    const scalar_t* wk,          // [B, H, Slots]
    const scalar_t* wq,          // [B, H, Slots]
    const scalar_t* decay,       // [H] or [B, H]
    const scalar_t* s_prev,      // [B, H, Slots, Dh]
    scalar_t*       out,         // [B, H, Dh]
    scalar_t*       s_next,      // [B, H, Slots, Dh]
    int B, int H, int Slots, int Dh,
    int v_s0 = 0, int v_s1 = 0,
    int beta_s0 = 0, int beta_s1 = 0,
    int wk_s0 = 0, int wk_s1 = 0,
    int wq_s0 = 0, int wq_s1 = 0,
    bool dynamic_decay = false,
    int decay_s0 = 0, int decay_s1 = 0
);

/**
 * @brief Launches the fused Shared-Memory CUDA backward kernel for analytical gradient propagation.
 *
 * Traverses the sequence backward in time from t = T-1 down to 0 inside Shared Memory.
 * Reconstructs prior states S_{t-1} on the fly from S_{final} and saved residuals r_t,
 * eliminating the need to store intermediate states in DRAM.
 *
 * @tparam scalar_t Precision type (float, double, at::Half, at::BFloat16).
 * @param grad_out Pointer to incoming upstream gradients [B, H, T, Dh].
 * @param grad_s_final Optional pointer to incoming state gradient [B, H, Slots, Dh].
 * @param v Pointer to forward values [B, H, T, Dh].
 * @param beta Pointer to forward write gates [B, H, T].
 * @param wk Pointer to forward slot key weights [B, H, T, Slots].
 * @param wq Pointer to forward slot query weights [B, H, T, Slots].
 * @param decay Pointer to per-head decay factors [H].
 * @param s_final Pointer to final forward state [B, H, Slots, Dh].
 * @param err_saved Pointer to forward saved residuals r_t [B, H, T, Dh].
 * @param grad_v Output gradient pointer for v [B, H, T, Dh].
 * @param grad_beta Output gradient pointer for beta [B, H, T].
 * @param grad_wk Output gradient pointer for wk [B, H, T, Slots].
 * @param grad_wq Output gradient pointer for wq [B, H, T, Slots].
 * @param grad_decay Output gradient pointer for decay alpha [H].
 * @param grad_s_init Optional output gradient pointer for initial state [B, H, Slots, Dh].
 * @param B Batch size.
 * @param H Number of associative heads.
 * @param T Sequence length.
 * @param Slots Number of memory slots per head.
 * @param Dh Hidden dimension per head.
 */
template <typename scalar_t>
void launch_delta_grid_backward(
    const scalar_t* grad_out,     // [B, H, T, Dh]
    const scalar_t* grad_s_final, // [B, H, Slots, Dh] (can be nullptr)
    const scalar_t* v,            // [B, H, T, Dh]
    const scalar_t* beta,         // [B, H, T]
    const scalar_t* wk,           // [B, H, T, Slots]
    const scalar_t* wq,           // [B, H, T, Slots]
    const scalar_t* decay,        // [H] or [B, H, T]
    const float*    s_final,      // [B, H, Slots, Dh] exact float32 final state
    const float*    err_saved,    // [B, H, T, Dh] exact float32 residuals
    scalar_t*       grad_v,       // [B, H, T, Dh]
    scalar_t*       grad_beta,    // [B, H, T]
    scalar_t*       grad_wk,      // [B, H, T, Slots]
    scalar_t*       grad_wq,      // [B, H, T, Slots]
    scalar_t*       grad_decay,   // [H]
    scalar_t*       grad_s_init,  // [B, H, Slots, Dh] (can be nullptr)
    float*          s_workspace,  // Optional [B, H, Slots, Dh] workspace when Slots*Dh > 4096
    float*          g_workspace,  // Optional [B, H, Slots, Dh] workspace when Slots*Dh > 4096
    int B, int H, int T, int Slots, int Dh,
    bool dynamic_decay = false
);

/**
 * @brief Launches the fused Sparse Delta-PKM single-step kernel.
 *
 * Coalesced gather of active slots from 1M-slot global state, fused Read + Key Prediction +
 * Delta Error + Write in registers (8 KB), and coalesced scatter writeback to global state.
 */
template <typename scalar_t>
void launch_delta_grid_pkm_step(
    const scalar_t* v,
    const scalar_t* beta,
    const scalar_t* wk,
    const scalar_t* wq,
    const int64_t*  active_idx,
    const scalar_t* decay,
    scalar_t*       state_1m,
    scalar_t*       out,
    int B, int H, int Dh, int ActiveSlots, int64_t TotalSlots
);

/**
 * @brief Launches the fused Sparse Delta-PKM forward sequence kernel.
 *
 * Traverses sequence T entirely on-chip within GPU registers, performing active slot
 * coalesced gather, query output projection, delta error prediction, and scatter update.
 */
template <typename scalar_t>
void launch_delta_grid_pkm_forward(
    const scalar_t* v,           // [B, H, T, Dh]
    const scalar_t* beta,        // [B, H, T]
    const scalar_t* wk,          // [B, H, T, ActiveSlots]
    const scalar_t* wq,          // [B, H, T, ActiveSlots]
    const int64_t*  active_idx,  // [B, H, T, ActiveSlots]
    const scalar_t* decay,       // [H] or [B, H, T]
    scalar_t*       state_1m,    // [B, H, TotalSlots, Dh]
    scalar_t*       out,         // [B, H, T, Dh]
    scalar_t*       saved_s_act, // [B, H, T, ActiveSlots, Dh] (can be nullptr)
    int B, int H, int T, int Dh, int ActiveSlots, int64_t TotalSlots,
    bool dynamic_decay, bool save_for_backward
);

/**
 * @brief Launches the fused Sparse Delta-PKM analytical backward kernel.
 *
 * Traverses sequence T in reverse order inside GPU registers, accumulating exact analytical
 * VJP gradients for v, beta, wk, wq, decay, and previous states without CPU-GPU synchronization.
 */
template <typename scalar_t>
void launch_delta_grid_pkm_backward(
    const scalar_t* grad_out,    // [B, H, T, Dh]
    const scalar_t* v,           // [B, H, T, Dh]
    const scalar_t* beta,        // [B, H, T]
    const scalar_t* wk,          // [B, H, T, ActiveSlots]
    const scalar_t* wq,          // [B, H, T, ActiveSlots]
    const int64_t*  active_idx,  // [B, H, T, ActiveSlots]
    const scalar_t* decay,       // [H] or [B, H, T]
    const scalar_t* saved_s_act, // [B, H, T, ActiveSlots, Dh]
    scalar_t*       grad_state,  // [B, H, TotalSlots, Dh]
    scalar_t*       grad_v,      // [B, H, T, Dh]
    scalar_t*       grad_beta,   // [B, H, T]
    scalar_t*       grad_wk,     // [B, H, T, ActiveSlots]
    scalar_t*       grad_wq,     // [B, H, T, ActiveSlots]
    scalar_t*       grad_decay,  // [B, H, T] or [H] (can be nullptr)
    int B, int H, int T, int Dh, int ActiveSlots, int64_t TotalSlots,
    bool dynamic_decay
);

#endif // DELTA_KERNEL_H
