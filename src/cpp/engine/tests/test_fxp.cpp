#include "deltagrid_math_fxp.h"
#include <iostream>
#include <cmath>
#include <cassert>

static void test_fxp_conversions_and_mul() {
    int32_t a = DG_FLOAT_TO_FXP(2.5f);
    int32_t b = DG_FLOAT_TO_FXP(-1.5f);
    int32_t c = dg_fxp_mul(a, b);
    float cf = DG_FXP_TO_FLOAT(c);
    assert(std::fabs(cf - (-3.75f)) < 1e-4f);
    std::cout << "[PASS] test_fxp_conversions_and_mul\n";
}

static void test_fxp_matmul() {
    int32_t x[2] = {DG_FLOAT_TO_FXP(1.5f), DG_FLOAT_TO_FXP(-2.0f)};
    int32_t W[4] = {
        DG_FLOAT_TO_FXP(2.0f), DG_FLOAT_TO_FXP(0.5f),
        DG_FLOAT_TO_FXP(-1.0f), DG_FLOAT_TO_FXP(1.0f)
    };
    int32_t b[2] = {DG_FLOAT_TO_FXP(0.25f), DG_FLOAT_TO_FXP(0.5f)};
    int32_t out[2] = {0};

    dg_matmul_fxp(x, W, b, out, 2, 2);
    // Row 0: 0.25 + 2.0*1.5 + 0.5*(-2.0) = 2.25
    // Row 1: 0.5 + (-1.0)*1.5 + 1.0*(-2.0) = -3.0
    assert(std::fabs(DG_FXP_TO_FLOAT(out[0]) - 2.25f) < 1e-3f);
    assert(std::fabs(DG_FXP_TO_FLOAT(out[1]) - (-3.0f)) < 1e-3f);
    std::cout << "[PASS] test_fxp_matmul\n";
}

static void test_fxp_l2_and_layernorm() {
    int32_t x[3] = {DG_FLOAT_TO_FXP(3.0f), DG_FLOAT_TO_FXP(4.0f), 0};
    int32_t out[3] = {0};
    dg_l2_normalize_fxp(x, out, 3);
    assert(std::fabs(DG_FXP_TO_FLOAT(out[0]) - 0.6f) < 1e-3f);
    assert(std::fabs(DG_FXP_TO_FLOAT(out[1]) - 0.8f) < 1e-3f);

    int32_t ln_in[4] = {
        DG_FLOAT_TO_FXP(1.0f),
        DG_FLOAT_TO_FXP(2.0f),
        DG_FLOAT_TO_FXP(3.0f),
        DG_FLOAT_TO_FXP(4.0f)
    };
    int32_t ln_w[4] = {DG_FXP_ONE, DG_FXP_ONE, DG_FXP_ONE, DG_FXP_ONE};
    int32_t ln_b[4] = {0, 0, 0, 0};
    int32_t ln_out[4] = {0};
    dg_layernorm_fxp(ln_in, 4, ln_w, ln_b, ln_out);
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += DG_FXP_TO_FLOAT(ln_out[i]);
    assert(std::fabs(sum) < 1e-2f);
    std::cout << "[PASS] test_fxp_l2_and_layernorm\n";
}

int main() {
    std::cout << "=== Running DeltaGrid Q16.16 Fixed-Point Unit Tests ===\n";
    test_fxp_conversions_and_mul();
    test_fxp_matmul();
    test_fxp_l2_and_layernorm();
    std::cout << "=== All Q16.16 Fixed-Point Unit Tests Passed! ===\n";
    return 0;
}

