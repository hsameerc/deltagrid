#ifndef DELTAGRID_LLM_H
#define DELTAGRID_LLM_H

#include <stdint.h>

#ifndef DELTAGRID_API
#if defined(_WIN32) || defined(_MSC_VER)
  #ifdef DELTAGRID_EXPORTS
    #define DELTAGRID_API __declspec(dllexport)
  #else
    #define DELTAGRID_API __declspec(dllimport)
  #endif
#else
  #define DELTAGRID_API __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes a DeltaGridLM context from an exported binary file.
 *        Loads all 4-Tier block weights, embedding tables, and allocates
 *        zero-KV associative memory states (Dense, Kronecker, or Sparse PKM).
 */
DELTAGRID_API void* deltagrid_llm_init_context(const char* weights_path);

/**
 * @brief Performs a single-token causal forward step in strictly constant O(1) RAM.
 * @param ctx Opaque pointer returned by deltagrid_llm_init_context().
 * @param current_token Integer token ID in [0, vocab_size - 1].
 * @param out_logits Output buffer of float[vocab_size] receiving next-token logits.
 */
DELTAGRID_API void deltagrid_llm_step(void* ctx, int current_token, float* out_logits);

/**
 * @brief Metadata query functions for a loaded DeltaGridLM context.
 */
DELTAGRID_API int deltagrid_llm_get_vocab_size(void* ctx);
DELTAGRID_API int deltagrid_llm_get_d_model(void* ctx);
DELTAGRID_API int deltagrid_llm_get_num_layers(void* ctx);
DELTAGRID_API int deltagrid_llm_get_step_count(void* ctx);

/**
 * @brief Resets all layer associative memory grids, Conv1D history buffers,
 *        and the harmonic clock counter to zero.
 */
DELTAGRID_API void deltagrid_llm_reset_state(void* ctx);

/**
 * @brief Saves conversational associative memory states, Conv1D buffers, and
 *        step count to a binary file for instant context resumption.
 * @return 0 on success, non-zero on error.
 */
DELTAGRID_API int deltagrid_llm_save_state(void* ctx, const char* filepath);

/**
 * @brief Restores conversational associative memory states, Conv1D buffers, and
 *        step count from a binary file.
 * @return 0 on success, non-zero on error.
 */
DELTAGRID_API int deltagrid_llm_load_state(void* ctx, const char* filepath);

/**
 * @brief Frees the DeltaGridLM context and all allocated memory.
 */
DELTAGRID_API void deltagrid_llm_free_context(void* ctx);

/**
 * @brief HSRU Edge SDK C-API Compatibility Aliases.
 */
DELTAGRID_API void* hsru_llm_init_context(const char* weights_path);
DELTAGRID_API void hsru_llm_step(void* ctx, int current_token, float* out_logits);
DELTAGRID_API int hsru_llm_get_vocab_size(void* ctx);
DELTAGRID_API void hsru_llm_free_context(void* ctx);
DELTAGRID_API int hsru_save_state(void* ctx, const char* filepath);
DELTAGRID_API int hsru_load_state(void* ctx, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // DELTAGRID_LLM_H

