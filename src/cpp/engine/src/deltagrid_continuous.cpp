// =============================================================================
// DeltaGrid Continuous C++ Edge Engine
// Exact match to DeltaGridContinuous -> DeltaGridBlock -> DeltaGridLayer
// in src/models/continuous.py, src/modules/block.py, and src/modules/memory.py
// =============================================================================

#ifndef DELTAGRID_EXPORTS
#define DELTAGRID_EXPORTS
#endif

#include "deltagrid.h"
#include "deltagrid_internal.h"
#include "deltagrid_math.h"

extern "C" {

DELTAGRID_API void* deltagrid_init_context(const char* weights_path) {
    return static_cast<void*>(dg_load_context_from_file(weights_path, DG_MODEL_CONTINUOUS));
}

DELTAGRID_API void deltagrid_step(void* ctx_ptr, const float* input, float* output) {
    if (!ctx_ptr || !input || !output) return;
    DGRuntimeContext* ctx = static_cast<DGRuntimeContext*>(ctx_ptr);

    const int D      = ctx->header.d_model;
    const int in_dim = ctx->header.input_or_vocab;
    const int out_d  = ctx->header.output_dim;

    // 1. Continuous feature encoder: [input_dim] -> [d_model]
    dg_matmul_bias(input, ctx->input_weight, ctx->input_bias, ctx->hidden_buf, D, in_dim);

    // 2. Step through all 4-Tier DeltaGrid blocks
    dg_step_all_blocks(ctx);

    // 3. Final LayerNorm & readout head
    dg_layernorm(ctx->hidden_buf, D, ctx->final_norm.weight, ctx->final_norm.bias, ctx->final_norm_buf);
    dg_matmul_bias(ctx->final_norm_buf, ctx->head_weight, ctx->head_bias, output, out_d, D);
}

DELTAGRID_API void deltagrid_step_forecast(
    void* ctx_ptr,
    const float* input,
    float* output,
    float* forecast_out
) {
    if (!ctx_ptr || !input || !output) return;
    DGRuntimeContext* ctx = static_cast<DGRuntimeContext*>(ctx_ptr);

    const int D       = ctx->header.d_model;
    const int in_dim  = ctx->header.input_or_vocab;
    const int out_d   = ctx->header.output_dim;
    const int horizon = ctx->header.forecast_horizon;

    // 1. Continuous feature encoder
    dg_matmul_bias(input, ctx->input_weight, ctx->input_bias, ctx->hidden_buf, D, in_dim);

    // 2. Step through all 4-Tier DeltaGrid blocks
    dg_step_all_blocks(ctx);

    // 3. Final LayerNorm & step readout head
    dg_layernorm(ctx->hidden_buf, D, ctx->final_norm.weight, ctx->final_norm.bias, ctx->final_norm_buf);
    dg_matmul_bias(ctx->final_norm_buf, ctx->head_weight, ctx->head_bias, output, out_d, D);

    // 4. Multi-Horizon Direct Trajectory Forecasting Head (if requested & present)
    if (forecast_out != nullptr) {
        if (horizon > 0) {
            dg_matmul_bias(
                ctx->final_norm_buf,
                ctx->forecast_fc1_weight,
                ctx->forecast_fc1_bias,
                ctx->forecast_hid_buf,
                D,
                D
            );
            for (int i = 0; i < D; ++i) {
                ctx->forecast_hid_buf[i] = dg_silu(ctx->forecast_hid_buf[i]);
            }
            dg_matmul_bias(
                ctx->forecast_hid_buf,
                ctx->forecast_fc2_weight,
                ctx->forecast_fc2_bias,
                forecast_out,
                horizon * out_d,
                D
            );
        } else {
            for (int i = 0; i < out_d; ++i) {
                forecast_out[i] = output[i];
            }
        }
    }
}

DELTAGRID_API void deltagrid_reset_state(void* ctx_ptr) {
    dg_reset_runtime_state(static_cast<DGRuntimeContext*>(ctx_ptr));
}

DELTAGRID_API int deltagrid_save_state(void* ctx_ptr, const char* filepath) {
    return dg_save_runtime_state(static_cast<DGRuntimeContext*>(ctx_ptr), filepath);
}

DELTAGRID_API int deltagrid_load_state(void* ctx_ptr, const char* filepath) {
    return dg_load_runtime_state(static_cast<DGRuntimeContext*>(ctx_ptr), filepath);
}

DELTAGRID_API int deltagrid_get_input_dim(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->header.input_or_vocab;
}

DELTAGRID_API int deltagrid_get_output_dim(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->header.output_dim;
}

DELTAGRID_API int deltagrid_get_d_model(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->header.d_model;
}

DELTAGRID_API int deltagrid_get_num_layers(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->header.num_layers;
}

DELTAGRID_API int deltagrid_get_forecast_horizon(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->header.forecast_horizon;
}

DELTAGRID_API int deltagrid_get_step_count(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->step_count;
}

DELTAGRID_API void deltagrid_release_context(void* ctx_ptr) {
    dg_free_runtime_context(static_cast<DGRuntimeContext*>(ctx_ptr));
}

}

