#ifndef DELTAGRID_H
#define DELTAGRID_H

#include <stdint.h>
#include <stdbool.h>

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

// =============================================================================
// Hardware Authentication / DRM Unlock
// =============================================================================
DELTAGRID_API bool deltagrid_authenticate(const char* hardware_id);

// =============================================================================
// Floating-Point / INT8 Continuous Control & Telemetry Engine (DeltaGridContinuous)
// =============================================================================

/**
 * @brief Initializes a DeltaGridContinuous context from an exported binary file.
 *        Allocates all weights, recurrent states, and scratch buffers upfront.
 *        Strictly zero heap allocations occur during deltagrid_step().
 */
DELTAGRID_API void* deltagrid_init_context(const char* weights_path);

/**
 * @brief Executes a single continuous recurrent step in O(1) memory.
 * @param ctx Opaque pointer returned by deltagrid_init_context().
 * @param input Pointer to float[input_dim] sensory vector.
 * @param output Pointer to float[output_dim] output buffer.
 */
DELTAGRID_API void deltagrid_step(void* ctx, const float* input, float* output);

/**
 * @brief Executes a single continuous recurrent step and computes multi-horizon
 *        trajectory forecast if the model was exported with forecast_horizon > 0.
 * @param ctx Opaque pointer returned by deltagrid_init_context().
 * @param input Pointer to float[input_dim] sensory vector.
 * @param output Pointer to float[output_dim] step prediction buffer.
 * @param forecast_out Pointer to float[forecast_horizon * output_dim] trajectory buffer.
 */
DELTAGRID_API void deltagrid_step_forecast(
    void* ctx,
    const float* input,
    float* output,
    float* forecast_out
);

/**
 * @brief Resets all internal recurrent memory states, Conv1D history buffers,
 *        and the harmonic clock counter to zero.
 */
DELTAGRID_API void deltagrid_reset_state(void* ctx);

/**
 * @brief Saves all layer recurrent memory states, Conv1D history buffers,
 *        and the current sequence offset to a binary file.
 * @return 0 on success, non-zero on error.
 */
DELTAGRID_API int deltagrid_save_state(void* ctx, const char* filepath);

/**
 * @brief Restores all layer recurrent memory states, Conv1D history buffers,
 *        and the sequence offset from a binary file.
 * @return 0 on success, non-zero on error.
 */
DELTAGRID_API int deltagrid_load_state(void* ctx, const char* filepath);

/**
 * @brief Metadata query helpers for loaded context.
 */
DELTAGRID_API int deltagrid_get_input_dim(void* ctx);
DELTAGRID_API int deltagrid_get_output_dim(void* ctx);
DELTAGRID_API int deltagrid_get_d_model(void* ctx);
DELTAGRID_API int deltagrid_get_num_layers(void* ctx);
DELTAGRID_API int deltagrid_get_forecast_horizon(void* ctx);
DELTAGRID_API int deltagrid_get_step_count(void* ctx);

/**
 * @brief Releases all memory associated with the context.
 */
DELTAGRID_API void deltagrid_release_context(void* ctx);

// =============================================================================
// Q16.16 Fixed-Point Integer Engine for Embedded MCUs (No FPU Required)
// =============================================================================

DELTAGRID_API void* deltagrid_init_context_fxp(const char* weights_path);
DELTAGRID_API void deltagrid_step_fxp(void* ctx, const int32_t* input, int32_t* output);
DELTAGRID_API void deltagrid_step_forecast_fxp(
    void* ctx,
    const int32_t* input,
    int32_t* output,
    int32_t* forecast_out
);
DELTAGRID_API void deltagrid_reset_state_fxp(void* ctx);
DELTAGRID_API int deltagrid_save_state_fxp(void* ctx, const char* filepath);
DELTAGRID_API int deltagrid_load_state_fxp(void* ctx, const char* filepath);
DELTAGRID_API void deltagrid_release_context_fxp(void* ctx);
DELTAGRID_API float deltagrid_compute_normalized_mse_fxp(
    const float* actual,
    const float* predicted,
    int dim
);

#ifdef __cplusplus
}
#endif

#endif // DELTAGRID_H

