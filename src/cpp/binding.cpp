// =============================================================================
//   HSRU DeltaGrid PyTorch C++ / CUDA Extension Bindings
// =============================================================================
// Connects PyTorch autograd tensors to high-performance CUDA Shared Memory kernels.
//
// Key Responsibilities:
//   1. Tensor validation: Verifies CUDA placement and enforces memory contiguity (.contiguous()).
//   2. Type dispatch: Dispatches across float32, float64, float16 (Half), and bfloat16.
//   3. Autograd tape support: Saves residual errors r_t for analytical reverse-time backpropagation.
//   4. Zero-copy execution: Interacts directly with GPU memory buffers via data_ptr<scalar_t>().
// =============================================================================

#include <torch/extension.h>
#include <vector>
#include "delta_kernel.h"

// Enforce CUDA device residency for input tensors
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

/**
 * @brief PyTorch binding for DeltaGrid forward sequence pass.
 *
 * Traverses sequence T using on-chip Shared Memory recurrence.
 *
 * @param q Query tensor [B, H, T, Dh].
 * @param k Key tensor [B, H, T, Dh].
 * @param v Value tensor [B, H, T, Dh].
 * @param beta Scalar write gate tensor [B, H, T].
 * @param wk Slot key weights [B, H, T, Slots].
 * @param wq Slot query weights [B, H, T, Slots].
 * @param decay Per-head retention decay factors [H].
 * @param s_init Optional initial state [B, H, Slots, Dh].
 * @param save_for_backward If true, caches error residuals r_t for the backward pass.
 * @return std::vector<torch::Tensor> {out, s_final, err_out}.
 */
std::vector<torch::Tensor> delta_grid_forward_binding(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& wk,
    const torch::Tensor& wq,
    const torch::Tensor& decay,
    const c10::optional<torch::Tensor>& s_init,
    bool save_for_backward)
{
    // Validate device residency
    CHECK_CUDA(q);
    CHECK_CUDA(k);
    CHECK_CUDA(v);
    CHECK_CUDA(beta);
    CHECK_CUDA(wk);
    CHECK_CUDA(wq);
    CHECK_CUDA(decay);

    // Enforce contiguous memory layouts for coalesced GPU access (q and k are unused; wq and wk are used)
    auto v_c     = v.is_contiguous() ? v : v.contiguous();
    auto beta_c  = beta.is_contiguous() ? beta : beta.contiguous();
    auto wk_c    = wk.is_contiguous() ? wk : wk.contiguous();
    auto wq_c    = wq.is_contiguous() ? wq : wq.contiguous();
    auto decay_c = decay.is_contiguous() ? decay : decay.contiguous();

    // Extract tensor dimension extents
    const int B     = v_c.size(0);
    const int H     = v_c.size(1);
    const int T     = v_c.size(2);
    const int Dh    = v_c.size(3);
    const int Slots = wk_c.size(3);

    // Handle optional initial state
    bool has_init = s_init.has_value() && s_init.value().defined();
    torch::Tensor s_init_c;
    if (has_init) {
        CHECK_CUDA(s_init.value());
        s_init_c = s_init.value().is_contiguous() ? s_init.value() : s_init.value().contiguous();
    }

    // Allocate output buffers
    auto out     = torch::empty({B, H, T, Dh}, v_c.options());
    auto s_final = torch::empty({B, H, Slots, Dh}, v_c.options());
    auto float_opts = v_c.options().dtype(torch::kFloat32);
    const bool is_fp32 = (v_c.scalar_type() == at::ScalarType::Float);
    // Conditionally allocate float32 error residual & exact state buffers when backpropagation is active
    auto err_out   = save_for_backward ? torch::empty({B, H, T, Dh}, float_opts) : torch::Tensor();
    auto s_final_f = (save_for_backward && !is_fp32) ? torch::empty({B, H, Slots, Dh}, float_opts) : s_final;

    const bool dynamic_decay = (decay_c.dim() > 1);

    // Dispatch floating-point precision to templated CUDA launch wrapper
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        v_c.scalar_type(), "delta_grid_forward_cuda", ([&] {
            launch_delta_grid_forward<scalar_t>(
                nullptr,
                nullptr,
                v_c.data_ptr<scalar_t>(),
                beta_c.data_ptr<scalar_t>(),
                wk_c.data_ptr<scalar_t>(),
                wq_c.data_ptr<scalar_t>(),
                decay_c.data_ptr<scalar_t>(),
                has_init ? s_init_c.data_ptr<scalar_t>() : nullptr,
                out.data_ptr<scalar_t>(),
                s_final.data_ptr<scalar_t>(),
                save_for_backward ? err_out.data_ptr<float>() : nullptr,
                (save_for_backward && !is_fp32) ? s_final_f.data_ptr<float>() : nullptr,
                B, H, T, Slots, Dh,
                dynamic_decay
            );
        })
    );

    return {out, s_final, err_out, s_final_f};
}

/**
 * @brief PyTorch binding for microsecond single-step autoregressive generation.
 *
 * Updates recurrent state S in constant O(1) time and constant O(1) memory.
 *
 * @param q Single-token query [B, H, Dh].
 * @param k Single-token key [B, H, Dh].
 * @param v Single-token value [B, H, Dh].
 * @param beta Single-token write gate [B, H].
 * @param wk Single-token slot key weights [B, H, Slots].
 * @param wq Single-token slot query weights [B, H, Slots].
 * @param decay Per-head retention decay alpha [H].
 * @param s_prev Preceding recurrent state [B, H, Slots, Dh].
 * @return std::vector<torch::Tensor> {out, s_next}.
 */
std::vector<torch::Tensor> delta_grid_step_binding(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& wk,
    const torch::Tensor& wq,
    const torch::Tensor& decay,
    const torch::Tensor& s_prev)
{
    // Validate device residency
    CHECK_CUDA(q);
    CHECK_CUDA(k);
    CHECK_CUDA(v);
    CHECK_CUDA(beta);
    CHECK_CUDA(wk);
    CHECK_CUDA(wq);
    CHECK_CUDA(decay);
    CHECK_CUDA(s_prev);

    // Avoid unnecessary copies: q and k are unused in kernel (wq and wk are used),
    // and v, wk, wq, beta support direct strided (b, h) indexing whenever stride(-1) == 1.
    auto v_c      = (v.stride(2) == 1) ? v : v.contiguous();
    auto wk_c     = (wk.stride(2) == 1) ? wk : wk.contiguous();
    auto wq_c     = (wq.stride(2) == 1) ? wq : wq.contiguous();
    const bool dynamic_decay = (decay.dim() > 1);
    auto decay_c  = (!dynamic_decay && decay.stride(0) != 1) ? decay.contiguous() : decay;
    auto s_prev_c = s_prev.is_contiguous() ? s_prev : s_prev.contiguous();

    // Extract dimensions (T=1 is implicit)
    const int B     = v_c.size(0);
    const int H     = v_c.size(1);
    const int Dh    = v_c.size(2);
    const int Slots = wk_c.size(2);

    const int v_s0     = v_c.stride(0);
    const int v_s1     = v_c.stride(1);
    const int beta_s0  = beta.stride(0);
    const int beta_s1  = beta.stride(1);
    const int wk_s0    = wk_c.stride(0);
    const int wk_s1    = wk_c.stride(1);
    const int wq_s0    = wq_c.stride(0);
    const int wq_s1    = wq_c.stride(1);
    const int decay_s0 = dynamic_decay ? decay_c.stride(0) : 0;
    const int decay_s1 = dynamic_decay ? decay_c.stride(1) : 1;

    // Allocate step outputs
    auto out    = torch::empty({B, H, Dh}, q.options());
    auto s_next = torch::empty({B, H, Slots, Dh}, q.options());

    // Dispatch floating-point precision
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        v_c.scalar_type(), "delta_grid_step_cuda", ([&] {
            launch_delta_grid_step<scalar_t>(
                nullptr,
                nullptr,
                v_c.data_ptr<scalar_t>(),
                beta.data_ptr<scalar_t>(),
                wk_c.data_ptr<scalar_t>(),
                wq_c.data_ptr<scalar_t>(),
                decay_c.data_ptr<scalar_t>(),
                s_prev_c.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                s_next.data_ptr<scalar_t>(),
                B, H, Slots, Dh,
                v_s0, v_s1, beta_s0, beta_s1, wk_s0, wk_s1, wq_s0, wq_s1,
                dynamic_decay, decay_s0, decay_s1
            );
        })
    );

    return {out, s_next};
}

/**
 * @brief PyTorch binding for analytical backward pass across sequence T.
 *
 * Reconstructs states backward in time inside Shared Memory and computes
 * exact analytical gradients for all inputs and model parameters.
 *
 * @param grad_out Upstream output gradients [B, H, T, Dh].
 * @param grad_s_final Optional upstream state gradient [B, H, Slots, Dh].
 * @param v Forward value tensor [B, H, T, Dh].
 * @param beta Forward write gate tensor [B, H, T].
 * @param wk Forward slot key weights [B, H, T, Slots].
 * @param wq Forward slot query weights [B, H, T, Slots].
 * @param decay Forward retention decay alpha [H].
 * @param s_final Final forward recurrent state [B, H, Slots, Dh].
 * @param err_saved Saved forward residuals r_t [B, H, T, Dh].
 * @param needs_grad_s_init True if initial state requires gradient.
 * @return std::vector<torch::Tensor> {grad_v, grad_beta, grad_wk, grad_wq, grad_decay, grad_s_init}.
 */
std::vector<torch::Tensor> delta_grid_backward_binding(
    const torch::Tensor& grad_out,
    const c10::optional<torch::Tensor>& grad_s_final,
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& wk,
    const torch::Tensor& wq,
    const torch::Tensor& decay,
    const torch::Tensor& s_final,
    const torch::Tensor& err_saved,
    bool needs_grad_s_init)
{
    CHECK_CUDA(grad_out);
    CHECK_CUDA(v);
    CHECK_CUDA(beta);
    CHECK_CUDA(wk);
    CHECK_CUDA(wq);
    CHECK_CUDA(decay);
    CHECK_CUDA(s_final);
    CHECK_CUDA(err_saved);

    auto grad_out_c  = grad_out.contiguous();
    auto v_c         = v.contiguous();
    auto beta_c      = beta.contiguous();
    auto wk_c        = wk.contiguous();
    auto wq_c        = wq.contiguous();
    auto decay_c     = decay.contiguous();
    auto s_final_c   = s_final.to(torch::kFloat32).contiguous();
    auto err_saved_c = err_saved.to(torch::kFloat32).contiguous();

    const int B     = v_c.size(0);
    const int H     = v_c.size(1);
    const int T     = v_c.size(2);
    const int Dh    = v_c.size(3);
    const int Slots = wk_c.size(3);

    // Handle optional incoming state gradient
    bool has_grad_sf = grad_s_final.has_value() && grad_s_final.value().defined();
    torch::Tensor grad_s_final_c;
    if (has_grad_sf) {
        CHECK_CUDA(grad_s_final.value());
        grad_s_final_c = grad_s_final.value().contiguous();
    }

    const bool dynamic_decay = (decay_c.dim() > 1);
    // Allocate gradient tensors matching input layouts
    auto grad_v      = torch::empty_like(v_c);
    auto grad_beta   = torch::empty_like(beta_c);
    auto grad_wk     = torch::empty_like(wk_c);
    auto grad_wq     = torch::empty_like(wq_c);
    auto grad_decay  = dynamic_decay ? torch::empty_like(decay_c) : torch::zeros_like(decay_c);
    auto grad_s_init = needs_grad_s_init ? torch::empty({B, H, Slots, Dh}, v_c.options()) : torch::Tensor();

    // When state matrix exceeds 48 KB per-block Shared Memory budget (Slots * Dh > 4096),
    // allocate float32 L2-cached state/adjoint workspaces so arbitrary Dh (e.g. 256+) works seamlessly.
    const bool use_workspace = (Slots * Dh > 4096);
    auto float_opts = v_c.options().dtype(torch::kFloat32);
    auto s_ws = use_workspace ? torch::empty({B, H, Slots, Dh}, float_opts) : torch::Tensor();
    auto g_ws = use_workspace ? torch::empty({B, H, Slots, Dh}, float_opts) : torch::Tensor();

    // Dispatch floating-point precision to templated CUDA launch wrapper
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        v_c.scalar_type(), "delta_grid_backward_cuda", ([&] {
            launch_delta_grid_backward<scalar_t>(
                grad_out_c.data_ptr<scalar_t>(),
                has_grad_sf ? grad_s_final_c.data_ptr<scalar_t>() : nullptr,
                v_c.data_ptr<scalar_t>(),
                beta_c.data_ptr<scalar_t>(),
                wk_c.data_ptr<scalar_t>(),
                wq_c.data_ptr<scalar_t>(),
                decay_c.data_ptr<scalar_t>(),
                s_final_c.data_ptr<float>(),
                err_saved_c.data_ptr<float>(),
                grad_v.data_ptr<scalar_t>(),
                grad_beta.data_ptr<scalar_t>(),
                grad_wk.data_ptr<scalar_t>(),
                grad_wq.data_ptr<scalar_t>(),
                grad_decay.data_ptr<scalar_t>(),
                needs_grad_s_init ? grad_s_init.data_ptr<scalar_t>() : nullptr,
                use_workspace ? s_ws.data_ptr<float>() : nullptr,
                use_workspace ? g_ws.data_ptr<float>() : nullptr,
                B, H, T, Slots, Dh,
                dynamic_decay
            );
        })
    );

    return {grad_v, grad_beta, grad_wk, grad_wq, grad_decay, grad_s_init};
}

/**
 * @brief PyTorch binding for Sparse Delta-PKM single-step recurrence.
 *
 * Coalesced gather of active slots, register-resident fused step, coalesced scatter write.
 */
std::vector<torch::Tensor> delta_grid_pkm_step_binding(
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& wk,
    const torch::Tensor& wq,
    const torch::Tensor& active_idx,
    const torch::Tensor& decay,
    torch::Tensor& state_1m)
{
    CHECK_CUDA(v);
    CHECK_CUDA(beta);
    CHECK_CUDA(wk);
    CHECK_CUDA(wq);
    CHECK_CUDA(active_idx);
    CHECK_CUDA(decay);
    CHECK_CUDA(state_1m);

    auto v_c          = v.is_contiguous() ? v : v.contiguous();
    auto beta_c       = beta.is_contiguous() ? beta : beta.contiguous();
    auto wk_c         = wk.is_contiguous() ? wk : wk.contiguous();
    auto wq_c         = wq.is_contiguous() ? wq : wq.contiguous();
    auto active_idx_c = active_idx.is_contiguous() ? active_idx : active_idx.contiguous();
    auto decay_c      = decay.is_contiguous() ? decay : decay.contiguous();
    auto state_c      = state_1m.is_contiguous() ? state_1m : state_1m.contiguous();

    const int B              = v_c.size(0);
    const int H              = v_c.size(1);
    const int Dh             = v_c.size(2);
    const int ActiveSlots    = wk_c.size(2);
    const int64_t TotalSlots = state_c.size(2);

    auto out = torch::empty({B, H, Dh}, v_c.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        v_c.scalar_type(), "delta_pkm_step_cuda", ([&] {
            launch_delta_grid_pkm_step<scalar_t>(
                v_c.data_ptr<scalar_t>(),
                beta_c.data_ptr<scalar_t>(),
                wk_c.data_ptr<scalar_t>(),
                wq_c.data_ptr<scalar_t>(),
                active_idx_c.data_ptr<int64_t>(),
                decay_c.data_ptr<scalar_t>(),
                state_c.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                B, H, Dh, ActiveSlots, TotalSlots
            );
        })
    );

    return {out, state_c};
}

/**
 * @brief PyTorch binding for DeltaGrid Sparse Delta-PKM forward sequence recurrence.
 */
std::vector<torch::Tensor> delta_pkm_forward_binding(
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& wk,
    const torch::Tensor& wq,
    const torch::Tensor& active_idx,
    const torch::Tensor& decay,
    torch::Tensor& state,
    bool save_for_backward)
{
    CHECK_CUDA(v);
    CHECK_CUDA(beta);
    CHECK_CUDA(wk);
    CHECK_CUDA(wq);
    CHECK_CUDA(active_idx);
    CHECK_CUDA(decay);
    CHECK_CUDA(state);

    auto v_c          = v.is_contiguous() ? v : v.contiguous();
    auto beta_c       = beta.is_contiguous() ? beta : beta.contiguous();
    auto wk_c         = wk.is_contiguous() ? wk : wk.contiguous();
    auto wq_c         = wq.is_contiguous() ? wq : wq.contiguous();
    auto active_idx_c = active_idx.is_contiguous() ? active_idx : active_idx.contiguous();
    auto decay_c      = decay.is_contiguous() ? decay : decay.contiguous();
    auto state_c      = state.is_contiguous() ? state : state.contiguous();

    const int B     = v_c.size(0);
    const int H     = v_c.size(1);
    const int T     = v_c.size(2);
    const int Dh    = v_c.size(3);
    const int ActiveSlots = wk_c.size(3);
    const int64_t TotalSlots = state_c.size(2);
    const bool dynamic_decay = (decay_c.dim() > 1);

    auto out = torch::empty({B, H, T, Dh}, v_c.options());
    auto saved_s_act = save_for_backward 
        ? torch::empty({B, H, T, ActiveSlots, Dh}, v_c.options()) 
        : torch::Tensor();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        v_c.scalar_type(), "delta_pkm_forward_cuda", ([&] {
            launch_delta_grid_pkm_forward<scalar_t>(
                v_c.data_ptr<scalar_t>(),
                beta_c.data_ptr<scalar_t>(),
                wk_c.data_ptr<scalar_t>(),
                wq_c.data_ptr<scalar_t>(),
                active_idx_c.data_ptr<int64_t>(),
                decay_c.data_ptr<scalar_t>(),
                state_c.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                save_for_backward ? saved_s_act.data_ptr<scalar_t>() : nullptr,
                B, H, T, Dh, ActiveSlots, TotalSlots,
                dynamic_decay, save_for_backward
            );
        })
    );

    return {out, state_c, saved_s_act};
}

/**
 * @brief PyTorch binding for DeltaGrid Sparse Delta-PKM analytical backward pass.
 */
std::vector<torch::Tensor> delta_pkm_backward_binding(
    const torch::Tensor& grad_out,
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& wk,
    const torch::Tensor& wq,
    const torch::Tensor& active_idx,
    const torch::Tensor& decay,
    const torch::Tensor& saved_s_act,
    int64_t total_slots,
    const c10::optional<torch::Tensor>& grad_final_state,
    bool needs_grad_decay)
{
    CHECK_CUDA(grad_out);
    CHECK_CUDA(v);
    CHECK_CUDA(beta);
    CHECK_CUDA(wk);
    CHECK_CUDA(wq);
    CHECK_CUDA(active_idx);
    CHECK_CUDA(decay);
    CHECK_CUDA(saved_s_act);

    auto grad_out_c    = grad_out.is_contiguous() ? grad_out : grad_out.contiguous();
    auto v_c           = v.is_contiguous() ? v : v.contiguous();
    auto beta_c        = beta.is_contiguous() ? beta : beta.contiguous();
    auto wk_c          = wk.is_contiguous() ? wk : wk.contiguous();
    auto wq_c          = wq.is_contiguous() ? wq : wq.contiguous();
    auto active_idx_c  = active_idx.is_contiguous() ? active_idx : active_idx.contiguous();
    auto decay_c       = decay.is_contiguous() ? decay : decay.contiguous();
    auto saved_s_act_c = saved_s_act.is_contiguous() ? saved_s_act : saved_s_act.contiguous();

    const int B     = v_c.size(0);
    const int H     = v_c.size(1);
    const int T     = v_c.size(2);
    const int Dh    = v_c.size(3);
    const int ActiveSlots = wk_c.size(3);
    const bool dynamic_decay = (decay_c.dim() > 1);

    torch::Tensor grad_state;
    if (grad_final_state.has_value() && grad_final_state.value().defined()) {
        CHECK_CUDA(grad_final_state.value());
        grad_state = grad_final_state.value().clone().contiguous();
    } else {
        grad_state = torch::zeros({B, H, total_slots, Dh}, v_c.options());
    }

    auto grad_v    = torch::empty({B, H, T, Dh}, v_c.options());
    auto grad_beta = torch::empty({B, H, T}, v_c.options());
    auto grad_wk   = torch::empty({B, H, T, ActiveSlots}, v_c.options());
    auto grad_wq   = torch::empty({B, H, T, ActiveSlots}, v_c.options());
    auto grad_decay = (dynamic_decay && needs_grad_decay) 
        ? torch::empty({B, H, T}, v_c.options()) 
        : (needs_grad_decay ? torch::empty({H}, v_c.options()) : torch::Tensor());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        v_c.scalar_type(), "delta_pkm_backward_cuda", ([&] {
            launch_delta_grid_pkm_backward<scalar_t>(
                grad_out_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                beta_c.data_ptr<scalar_t>(),
                wk_c.data_ptr<scalar_t>(),
                wq_c.data_ptr<scalar_t>(),
                active_idx_c.data_ptr<int64_t>(),
                decay_c.data_ptr<scalar_t>(),
                saved_s_act_c.data_ptr<scalar_t>(),
                grad_state.data_ptr<scalar_t>(),
                grad_v.data_ptr<scalar_t>(),
                grad_beta.data_ptr<scalar_t>(),
                grad_wk.data_ptr<scalar_t>(),
                grad_wq.data_ptr<scalar_t>(),
                needs_grad_decay ? grad_decay.data_ptr<scalar_t>() : nullptr,
                B, H, T, Dh, ActiveSlots, total_slots,
                dynamic_decay
            );
        })
    );

    return {grad_v, grad_beta, grad_wk, grad_wq, grad_decay, grad_state};
}

// =============================================================================
// PyBind11 Module Registration
// =============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "HSRU DeltaGrid Recurrent Associative Memory CUDA Extension with Shared Memory Residency";
    
    m.def("forward", &delta_grid_forward_binding, 
          "DeltaGrid Forward Sequence Recurrence (CUDA Shared Memory)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("beta"), 
          py::arg("wk"), py::arg("wq"), py::arg("decay"),
          py::arg("s_init") = py::none(), py::arg("save_for_backward") = false);

    m.def("step", &delta_grid_step_binding, 
          "DeltaGrid Microsecond Single-Step Recurrent Update (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("beta"), 
          py::arg("wk"), py::arg("wq"), py::arg("decay"),
          py::arg("s_prev"));

    m.def("pkm_step", &delta_grid_pkm_step_binding,
          "DeltaGrid Sparse Delta-PKM Microsecond Step Update (CUDA Coalesced Gather/Scatter)",
          py::arg("v"), py::arg("beta"), py::arg("wk"), py::arg("wq"),
          py::arg("active_idx"), py::arg("decay"), py::arg("state_1m"));

    m.def("pkm_forward", &delta_pkm_forward_binding,
          "DeltaGrid Sparse Delta-PKM Fused Sequence Forward Recurrence (CUDA)",
          py::arg("v"), py::arg("beta"), py::arg("wk"), py::arg("wq"),
          py::arg("active_idx"), py::arg("decay"), py::arg("state"),
          py::arg("save_for_backward") = false);

    m.def("pkm_backward", &delta_pkm_backward_binding,
          "DeltaGrid Sparse Delta-PKM Fused Sequence Backward Recurrence (CUDA)",
          py::arg("grad_out"), py::arg("v"), py::arg("beta"),
          py::arg("wk"), py::arg("wq"), py::arg("active_idx"),
          py::arg("decay"), py::arg("saved_s_act"), py::arg("total_slots"),
          py::arg("grad_final_state") = py::none(), py::arg("needs_grad_decay") = false);

    m.def("backward", &delta_grid_backward_binding, 
          "DeltaGrid Analytical Reverse-Time Backward Pass (CUDA Shared Memory)",
          py::arg("grad_out"), py::arg("grad_s_final"), py::arg("v"), py::arg("beta"), 
          py::arg("wk"), py::arg("wq"), py::arg("decay"), py::arg("s_final"), 
          py::arg("err_saved"), py::arg("needs_grad_s_init") = false);
}
