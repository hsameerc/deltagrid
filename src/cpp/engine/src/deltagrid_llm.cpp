// =============================================================================
// DeltaGrid Causal LLM C++ Edge Engine
// Exact match to DeltaGridLM -> DeltaGridBlock -> DeltaGridLayer
// in src/models/language_model.py, src/modules/block.py, and src/modules/memory.py
// =============================================================================

#ifndef DELTAGRID_EXPORTS
#define DELTAGRID_EXPORTS
#endif

#include "deltagrid_llm.h"
#include "deltagrid_internal.h"
#include "deltagrid_math.h"
#include <cstring>

extern "C" {

DELTAGRID_API void* deltagrid_llm_init_context(const char* weights_path) {
    return static_cast<void*>(dg_load_context_from_file(weights_path, DG_MODEL_LM));
}

DELTAGRID_API void deltagrid_llm_step(void* ctx_ptr, int current_token, float* out_logits) {
    if (!ctx_ptr || !out_logits) return;
    DGRuntimeContext* ctx = static_cast<DGRuntimeContext*>(ctx_ptr);

    const int D          = ctx->header.d_model;
    const int vocab_size = ctx->header.input_or_vocab;

    if (current_token < 0 || current_token >= vocab_size) {
        current_token = 0;
    }

    // 1. Token Embedding Lookup: [vocab_size, d_model] -> [d_model]
    if (ctx->input_weight.q8 != nullptr) {
        const int8_t* emb_row = ctx->input_weight.q8 + (size_t)current_token * D;
        float scale = ctx->input_weight.scale;
        for (int i = 0; i < D; ++i) {
            ctx->hidden_buf[i] = (float)emb_row[i] * scale;
        }
    } else {
        const float* emb_row = ctx->input_weight.f32 + (size_t)current_token * D;
        std::memcpy(ctx->hidden_buf, emb_row, D * sizeof(float));
    }

    // 2. Step through all 4-Tier DeltaGrid blocks (Tier 0 Clock, Tier 1 Conv1D, Tier 2 Memory, Tier 3 GLU)
    dg_step_all_blocks(ctx);

    // 3. Final LayerNorm & LM Head Logits Readout
    dg_layernorm(ctx->hidden_buf, D, ctx->final_norm.weight, ctx->final_norm.bias, ctx->final_norm_buf);
    dg_matmul_bias(ctx->final_norm_buf, ctx->head_weight, nullptr, out_logits, vocab_size, D);
}

DELTAGRID_API int deltagrid_llm_get_vocab_size(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->header.input_or_vocab;
}

DELTAGRID_API int deltagrid_llm_get_d_model(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->header.d_model;
}

DELTAGRID_API int deltagrid_llm_get_num_layers(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->header.num_layers;
}

DELTAGRID_API int deltagrid_llm_get_step_count(void* ctx_ptr) {
    if (!ctx_ptr) return 0;
    return static_cast<DGRuntimeContext*>(ctx_ptr)->step_count;
}

DELTAGRID_API void deltagrid_llm_reset_state(void* ctx_ptr) {
    dg_reset_runtime_state(static_cast<DGRuntimeContext*>(ctx_ptr));
}

DELTAGRID_API int deltagrid_llm_save_state(void* ctx_ptr, const char* filepath) {
    return dg_save_runtime_state(static_cast<DGRuntimeContext*>(ctx_ptr), filepath);
}

DELTAGRID_API int deltagrid_llm_load_state(void* ctx_ptr, const char* filepath) {
    return dg_load_runtime_state(static_cast<DGRuntimeContext*>(ctx_ptr), filepath);
}

DELTAGRID_API void deltagrid_llm_free_context(void* ctx_ptr) {
    dg_free_runtime_context(static_cast<DGRuntimeContext*>(ctx_ptr));
}

// HSRU Edge SDK C-API Compatibility Aliases
DELTAGRID_API void* hsru_llm_init_context(const char* weights_path) {
    return deltagrid_llm_init_context(weights_path);
}

DELTAGRID_API void hsru_llm_step(void* ctx, int current_token, float* out_logits) {
    deltagrid_llm_step(ctx, current_token, out_logits);
}

DELTAGRID_API int hsru_llm_get_vocab_size(void* ctx) {
    return deltagrid_llm_get_vocab_size(ctx);
}

DELTAGRID_API void hsru_llm_free_context(void* ctx) {
    deltagrid_llm_free_context(ctx);
}

DELTAGRID_API int hsru_save_state(void* ctx, const char* filepath) {
    return deltagrid_llm_save_state(ctx, filepath);
}

DELTAGRID_API int hsru_load_state(void* ctx, const char* filepath) {
    return deltagrid_llm_load_state(ctx, filepath);
}

}


