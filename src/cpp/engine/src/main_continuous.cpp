#include "deltagrid.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::cout << "====================================================\n";
    std::cout << "   HSRU DELTAGRID EDGE: CONTINUOUS CONTROL ENGINE   \n";
    std::cout << "====================================================\n\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <weights.bin> [num_steps]\n";
        return 1;
    }

    const char* weights_path = argv[1];
    int num_steps = (argc > 2) ? std::atoi(argv[2]) : 100;

    if (!deltagrid_authenticate("MAC:00-11-22-33-44-55")) {
        std::cerr << "[ERROR] Hardware Authentication Failed!\n";
        return 1;
    }

    void* ctx = deltagrid_init_context(weights_path);
    if (!ctx) {
        std::cerr << "[ERROR] Could not initialize DeltaGrid context from: " << weights_path << "\n";
        return 1;
    }

    int in_dim    = deltagrid_get_input_dim(ctx);
    int out_dim   = deltagrid_get_output_dim(ctx);
    int d_model   = deltagrid_get_d_model(ctx);
    int layers    = deltagrid_get_num_layers(ctx);
    int f_horizon = deltagrid_get_forecast_horizon(ctx);

    std::cout << "[INFO] Loaded DeltaGridContinuous Model:\n";
    std::cout << "       Layers: " << layers << " | d_model: " << d_model
              << " | in_dim: " << in_dim << " | out_dim: " << out_dim
              << " | forecast_horizon: " << f_horizon << "\n\n";

    std::vector<float> input(in_dim, 0.0f);
    std::vector<float> output(out_dim, 0.0f);
    std::vector<float> forecast(std::max(1, f_horizon) * out_dim, 0.0f);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_steps; ++t) {
        for (int i = 0; i < in_dim; ++i) {
            input[i] = std::sin(0.05f * (float)t + 0.1f * (float)i);
        }
        if (f_horizon > 0) {
            deltagrid_step_forecast(ctx, input.data(), output.data(), forecast.data());
        } else {
            deltagrid_step(ctx, input.data(), output.data());
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    double us_per_step = elapsed_us / (double)num_steps;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[DONE] Executed " << num_steps << " steps in " << (elapsed_us / 1000.0) << " ms\n";
    std::cout << "       Latency: " << us_per_step << " us/step ("
              << (1e6 / std::max(us_per_step, 1e-6)) << " Hz)\n";
    std::cout << "       Final Output[0]: " << output[0] << "\n";

    deltagrid_release_context(ctx);
    return 0;
}

