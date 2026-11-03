// =============================================================================
// HSRU DeltaGrid C++ FP32 vs Q16.16 Fixed-Point Divergence Verification Suite
// =============================================================================
//
// Purpose:
// --------
// Verifies that the 32-bit integer Q16.16 Fixed-Point runtime (deltagrid_fxp.cpp)
// closely tracks the IEEE-754 Float32 runtime (deltagrid_continuous.cpp) across
// multi-step recurrent trajectories without numerical overflow or drift.
//
// Modes of Operation:
// -------------------
//   1. Self-Contained Mode (no arguments, used by `make test`):
//      Synthesizes a deterministic 2-layer DeltaGridContinuous binary in both
//      FP32 and Q16.16 formats, steps both engines across 32 continuous frames,
//      and asserts that the mean absolute divergence remains within Q16.16 bounds.
//
//   2. External File Mode (6 arguments, matching HSRU verify_divergence CLI):
//      Usage: verify_divergence <weights_fp32.bin> <weights_fxp.bin> <input.bin>
//                               <out_fp32.bin> <out_fxp.bin> <seq_len>
// =============================================================================

#include "deltagrid.h"
#include "deltagrid_internal.h"
#include "deltagrid_math_fxp.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <cstdio>

static void write_f32_or_fxp(std::ofstream& f, const std::vector<float>& vals, bool is_fxp) {
    if (is_fxp) {
        std::vector<int32_t> fxp(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) {
            fxp[i] = (int32_t)std::round(vals[i] * 65536.0f);
        }
        f.write(reinterpret_cast<const char*>(fxp.data()), fxp.size() * sizeof(int32_t));
    } else {
        f.write(reinterpret_cast<const char*>(vals.data()), vals.size() * sizeof(float));
    }
}

static void generate_synthetic_weights(const char* path, bool is_fxp) {
    std::ofstream f(path, std::ios::binary);
    const int L = 2, D = 16, H = 2, K = 4, in_d = 4, out_d = 2, slots = 8;

    DeltaGridBinHeader hdr = {};
    hdr.magic            = DELTAGRID_BIN_MAGIC;
    hdr.version          = DELTAGRID_BIN_VERSION;
    hdr.model_type       = DG_MODEL_CONTINUOUS;
    hdr.precision_mode   = is_fxp ? DG_PREC_FXP : DG_PREC_FP32;
    hdr.num_layers       = L;
    hdr.d_model          = D;
    hdr.num_heads        = H;
    hdr.kernel_size      = K;
    hdr.thinking_cycles  = 0;
    hdr.input_or_vocab   = in_d;
    hdr.output_dim       = out_d;
    hdr.forecast_horizon = 0;
    hdr.num_axes         = 0;
    hdr.top_k            = 0;
    hdr.actual_slots     = slots;
    hdr.flags            = DG_FLAG_DYNAMIC_DECAY;

    f.write(reinterpret_cast<const char*>(&hdr), sizeof(DeltaGridBinHeader));

    auto make_vec = [](size_t n, float scale, int seed_offset) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = scale * std::sin(0.37f * (float)(i + seed_offset));
        }
        return v;
    };

    // Input encoder
    write_f32_or_fxp(f, make_vec(D * in_d, 0.2f, 1), is_fxp);
    write_f32_or_fxp(f, make_vec(D, 0.05f, 2), is_fxp);

    for (int l = 0; l < L; ++l) {
        int32_t has_mem = 1;
        f.write(reinterpret_cast<const char*>(&has_mem), sizeof(int32_t));

        // Tier 0: div_term & clock_scale
        std::vector<float> div_term(D / 2);
        for (int k = 0; k < D / 2; ++k) {
            div_term[k] = std::exp((float)(2 * k) * (-std::log(10000.0f) / (float)D));
        }
        write_f32_or_fxp(f, div_term, is_fxp);
        write_f32_or_fxp(f, {0.1f}, is_fxp);

        // Tier 1: Conv1D & norm_conv
        write_f32_or_fxp(f, make_vec(D * K, 0.15f, 10 + l), is_fxp);
        write_f32_or_fxp(f, make_vec(D, 0.02f, 20 + l), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(D, 1.0f), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(D, 0.0f), is_fxp);

        // Tier 2: Memory
        write_f32_or_fxp(f, {1.0f}, is_fxp);
        write_f32_or_fxp(f, make_vec((3 * D + H) * D, 0.15f, 30 + l), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(H, 1.0f), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(H, 0.0f), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(H, 0.995f), is_fxp);
        write_f32_or_fxp(f, make_vec(H * D, 0.05f, 40 + l), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(H, 2.0f), is_fxp);
        write_f32_or_fxp(f, make_vec(D * D, 0.15f, 50 + l), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(D, 1.0f), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(D, 0.0f), is_fxp);

        // Tier 3: GLU Logic
        write_f32_or_fxp(f, {1.0f}, is_fxp);
        write_f32_or_fxp(f, make_vec(2 * D * D, 0.15f, 60 + l), is_fxp);
        write_f32_or_fxp(f, make_vec(2 * D, 0.02f, 70 + l), is_fxp);
        write_f32_or_fxp(f, make_vec(D * D, 0.15f, 80 + l), is_fxp);
        write_f32_or_fxp(f, make_vec(D, 0.02f, 90 + l), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(D, 1.0f), is_fxp);
        write_f32_or_fxp(f, std::vector<float>(D, 0.0f), is_fxp);
    }

    // Final norm & head
    write_f32_or_fxp(f, std::vector<float>(D, 1.0f), is_fxp);
    write_f32_or_fxp(f, std::vector<float>(D, 0.0f), is_fxp);
    write_f32_or_fxp(f, make_vec(out_d * D, 0.2f, 100), is_fxp);
    write_f32_or_fxp(f, make_vec(out_d, 0.05f, 101), is_fxp);
}

static int run_self_contained_divergence_test() {
    std::cout << "=== Running DeltaGrid C++ FP32 vs Q16.16 FXP Divergence Verification ===\n";
    const char* fp32_path = "/tmp/dg_verify_fp32.bin";
    const char* fxp_path  = "/tmp/dg_verify_fxp.bin";

    generate_synthetic_weights(fp32_path, false);
    generate_synthetic_weights(fxp_path, true);

    void* ctx_fp32 = deltagrid_init_context(fp32_path);
    void* ctx_fxp  = deltagrid_init_context_fxp(fxp_path);
    assert(ctx_fp32 != nullptr && ctx_fxp != nullptr);

    const int in_d  = 4;
    const int out_d = 2;
    float max_diff  = 0.0f;
    float sum_diff  = 0.0f;
    const int T     = 32;

    for (int t = 0; t < T; ++t) {
        float in_fp32[in_d];
        int32_t in_fxp[in_d];
        for (int i = 0; i < in_d; ++i) {
            in_fp32[i] = 0.5f * std::sin(0.2f * (float)t + (float)i);
            in_fxp[i]  = DG_FLOAT_TO_FXP(in_fp32[i]);
        }

        float out_fp32[out_d]  = {0.0f};
        int32_t out_fxp[out_d] = {0};

        deltagrid_step(ctx_fp32, in_fp32, out_fp32);
        deltagrid_step_fxp(ctx_fxp, in_fxp, out_fxp);

        for (int i = 0; i < out_d; ++i) {
            float fxp_as_float = DG_FXP_TO_FLOAT(out_fxp[i]);
            float diff = std::fabs(out_fp32[i] - fxp_as_float);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
        }
    }

    float mean_diff = sum_diff / (float)(T * out_d);
    std::cout << std::scientific << std::setprecision(4);
    std::cout << "[INFO] 32-Step Trajectory | Max FP32-vs-FXP Diff: " << max_diff
              << " | Mean Diff: " << mean_diff << "\n";
    assert(max_diff < 5e-3f);
    std::cout << "[PASS] verify_divergence (FP32 vs Q16.16 Fixed-Point)\n";

    deltagrid_release_context(ctx_fp32);
    deltagrid_release_context_fxp(ctx_fxp);
    std::remove(fp32_path);
    std::remove(fxp_path);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        return run_self_contained_divergence_test();
    }

    const char* weights_fp32_path = argv[1];
    const char* weights_fxp_path  = argv[2];
    const char* input_path        = argv[3];
    const char* out_fp32_path     = argv[4];
    const char* out_fxp_path      = argv[5];
    int seq_len                   = std::atoi(argv[6]);

    if (!deltagrid_authenticate("MAC:00-11-22-33-44-55")) {
        std::cerr << "Authentication failed\n";
        return 1;
    }

    void* ctx_fp32 = deltagrid_init_context(weights_fp32_path);
    void* ctx_fxp  = deltagrid_init_context_fxp(weights_fxp_path);
    if (!ctx_fp32 || !ctx_fxp) {
        std::cerr << "Failed to initialize contexts\n";
        return 1;
    }

    int in_dim  = deltagrid_get_input_dim(ctx_fp32);
    int out_dim = deltagrid_get_output_dim(ctx_fp32);

    std::ifstream input_file(input_path, std::ios::binary);
    std::ofstream fp32_file(out_fp32_path, std::ios::binary);
    std::ofstream fxp_file(out_fxp_path, std::ios::binary);

    for (int t = 0; t < seq_len; ++t) {
        std::vector<float> input_fp32(in_dim);
        input_file.read(reinterpret_cast<char*>(input_fp32.data()), in_dim * sizeof(float));

        std::vector<int32_t> input_fxp(in_dim);
        for (int i = 0; i < in_dim; ++i) {
            input_fxp[i] = DG_FLOAT_TO_FXP(input_fp32[i]);
        }

        std::vector<float> output_fp32(out_dim);
        deltagrid_step(ctx_fp32, input_fp32.data(), output_fp32.data());

        std::vector<int32_t> output_fxp(out_dim);
        deltagrid_step_fxp(ctx_fxp, input_fxp.data(), output_fxp.data());

        fp32_file.write(reinterpret_cast<const char*>(output_fp32.data()), out_dim * sizeof(float));

        std::vector<float> output_fxp_float(out_dim);
        for (int i = 0; i < out_dim; ++i) {
            output_fxp_float[i] = DG_FXP_TO_FLOAT(output_fxp[i]);
        }
        fxp_file.write(reinterpret_cast<const char*>(output_fxp_float.data()), out_dim * sizeof(float));
    }

    deltagrid_release_context(ctx_fp32);
    deltagrid_release_context_fxp(ctx_fxp);
    std::cout << "Divergence verification completed successfully.\n";
    return 0;
}
