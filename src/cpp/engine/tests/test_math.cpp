#include "deltagrid_math.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

static void test_layernorm() {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4] = {0.0f};

    dg_layernorm(x, 4, w, b, out);

    float sum = out[0] + out[1] + out[2] + out[3];
    assert(std::fabs(sum) < 1e-5f);
    assert(out[0] < out[1] && out[1] < out[2] && out[2] < out[3]);
    std::cout << "[PASS] test_layernorm\n";
}

static void test_l2_normalize() {
    float x[3] = {3.0f, 4.0f, 0.0f};
    float out[3] = {0.0f};
    dg_l2_normalize(x, out, 3);
    assert(std::fabs(out[0] - 0.6f) < 1e-6f);
    assert(std::fabs(out[1] - 0.8f) < 1e-6f);
    assert(std::fabs(out[2] - 0.0f) < 1e-6f);
    std::cout << "[PASS] test_l2_normalize\n";
}

static void test_swiglu() {
    float x[4] = {0.0f, 2.0f, 1.0f, 0.5f};
    // a = [0, 2], b = [1, 0.5]
    // silu(0) = 0 * sigmoid(0) = 0 -> out[0] = 0 * 1 = 0
    // silu(2) = 2 * (1 / (1 + exp(-2))) -> out[1] = silu(2) * 0.5
    float out[2] = {0.0f};
    dg_swiglu(x, out, 2);
    float expected_1 = (2.0f / (1.0f + std::exp(-2.0f))) * 0.5f;
    assert(std::fabs(out[0] - 0.0f) < 1e-6f);
    assert(std::fabs(out[1] - expected_1) < 1e-5f);
    std::cout << "[PASS] test_swiglu\n";
}

static void test_topk() {
    float scores[6] = {0.1f, 3.5f, -1.0f, 9.2f, 3.5f, 2.0f};
    float vals[3];
    int32_t idx[3];
    dg_topk_descending(scores, 6, 3, vals, idx);
    assert(idx[0] == 3 && std::fabs(vals[0] - 9.2f) < 1e-6f);
    assert(idx[1] == 1 && std::fabs(vals[1] - 3.5f) < 1e-6f);
    assert(idx[2] == 4 && std::fabs(vals[2] - 3.5f) < 1e-6f);
    std::cout << "[PASS] test_topk\n";
}

static void test_matmul_f32_and_int8() {
    float x[3] = {1.0f, 2.0f, 3.0f};
    float W_f32[6] = {
        1.0f, 0.0f, -1.0f,
        2.0f, 1.0f,  0.5f
    };
    float bias[2] = {0.5f, -0.5f};
    float out_f32[2] = {0.0f};

    DGWeightArray W;
    W.f32 = W_f32;
    W.q8  = nullptr;
    W.scale = 1.0f;

    dg_matmul_bias(x, W, bias, out_f32, 2, 3);
    // Row 0: 0.5 + 1*1 + 0*2 + (-1)*3 = -1.5
    // Row 1: -0.5 + 2*1 + 1*2 + 0.5*3 = 5.0
    assert(std::fabs(out_f32[0] - (-1.5f)) < 1e-6f);
    assert(std::fabs(out_f32[1] - 5.0f) < 1e-6f);

    // INT8 Quantized version with scale = 0.5
    int8_t W_q8[6] = {
        2, 0, -2,
        4, 2,  1
    };
    W.f32 = nullptr;
    W.q8  = W_q8;
    W.scale = 0.5f;

    float out_q8[2] = {0.0f};
    dg_matmul_bias(x, W, bias, out_q8, 2, 3);
    assert(std::fabs(out_q8[0] - (-1.5f)) < 1e-6f);
    assert(std::fabs(out_q8[1] - 5.0f) < 1e-6f);
    std::cout << "[PASS] test_matmul_f32_and_int8\n";
}

int main() {
    std::cout << "=== Running DeltaGrid C++ Math Unit Tests ===\n";
    test_layernorm();
    test_l2_normalize();
    test_swiglu();
    test_topk();
    test_matmul_f32_and_int8();
    std::cout << "=== All Math Unit Tests Passed! ===\n";
    return 0;
}

