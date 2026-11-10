#include "deltagrid.h"
#include "deltagrid_math_fxp.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::cout << "====================================================\n";
    std::cout << "   HSRU DELTAGRID EDGE: Q16.16 FIXED-POINT ENGINE   \n";
    std::cout << "====================================================\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <weights_fxp.bin> [num_steps]\n";
        return 1;
    }

    const char* weights_path = argv[1];
    int num_steps = (argc > 2) ? std::atoi(argv[2]) : 100;

    void* ctx = deltagrid_init_context_fxp(weights_path);
    if (!ctx) {
        std::cerr << "[ERROR] Could not initialize Q16.16 FXP context from: " << weights_path << "\n";
        return 1;
    }

    std::vector<int32_t> input(64, 0);
    std::vector<int32_t> output(64, 0);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_steps; ++t) {
        for (int i = 0; i < 64; ++i) {
            input[i] = DG_FLOAT_TO_FXP(std::sin(0.05f * (float)t + 0.1f * (float)i));
        }
        deltagrid_step_fxp(ctx, input.data(), output.data());
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[DONE] Executed " << num_steps << " Q16.16 steps in " << (elapsed_us / 1000.0) << " ms\n";
    std::cout << "       Final Output[0]: " << DG_FXP_TO_FLOAT(output[0]) << " (raw FXP: " << output[0] << ")\n";

    deltagrid_release_context_fxp(ctx);
    return 0;
}

