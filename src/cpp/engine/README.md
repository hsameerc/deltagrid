# Unified HSRU DeltaGrid C++ Edge Engine

The `src/cpp/engine` directory contains the zero-dependency, high-performance C++ edge inference runtimes for all **DeltaGrid** 4-Tier Neuromorphic architectures:
- **Aerospace, Drones & Robotics (`libdeltagrid_continuous`)**: Sub-millisecond continuous telemetry regression, closed-loop control, and direct multi-horizon trajectory forecasting.
- **Embedded Medical & Avionics MCUs (`libdeltagrid_fxp`)**: Ultra-low-power microcontrollers running Q16.16 fixed-point integer arithmetic without a hardware FPU (IEC 62304 / DO-178C friendly).
- **Zero-KV Edge Language Models (`libdeltagrid_llm`)**: Autoregressive causal token generation in strictly constant $O(1)$ RAM with zero KV-cache growth, supporting Dense, Factorized Kronecker, and Sparse Delta-PKM ($1,048,576$ slots).

---

## Architecture & Runtime Overview

Every runtime implements the complete **4-Tier DeltaGrid Hierarchy** with **zero dynamic memory allocation (`malloc`/`new`) during `.step()`**:

1. **Tier 0 (Biological Harmonic Clock)**: Continuous multi-frequency sinusoidal phase anchoring ($\omega_k = 10000^{-2k/D}$).
2. **Tier 1 (Causal Depthwise Conv1D, $k=4$)**: High-frequency local n-gram and sensory transient binding via a circular shift buffer and SiLU activation.
3. **Tier 2 (Associative DeltaGrid Memory Layer)**:
   - Closed-loop Householder Delta error-erasing rule:
     $$\mathbf{y}_t = \mathbf{w}_{q,t}^\top \mathbf{S}_{t-1}, \quad \mathbf{r}_t = \mathbf{v}_t - \mathbf{w}_{k,t}^\top \mathbf{S}_{t-1}, \quad \mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \beta_t (\mathbf{w}_{k,t} \mathbf{r}_t^\top)$$
   - Supports **Mode 1 (Standard Dense)**, **Mode 2 (Factorized Kronecker Dense)**, and **Mode 3 (Sparse Delta-PKM Top-$K$)** across both FP32/INT8 and Q16.16 Fixed-Point runtimes.
4. **Tier 3 (Non-Linear SwiGLU Logic Reasoning Engine)**: Dual-branch Swish Gated Linear Units ($\text{SwiGLU}(x) = \text{SiLU}(x W_{\text{gate}}) \cdot (x W_{\text{up}})$) with optional cognitive working-memory recurrence (`thinking_cycles`).

---

## Verified Numerical Divergence (PyTorch vs. C++ Engine)

| Architecture & Mode | Runtime Precision | Max Abs Divergence vs. PyTorch FP32 | Status |
| :--- | :---: | :---: | :---: |
| `DeltaGridContinuous` (Standard Dense + Forecast Head) | **FP32** | $7.75 \times 10^{-7}$ | **PASS** |
| `DeltaGridContinuous` (`.forward()` vs C++ `.step()`) | **FP32** | $8.64 \times 10^{-7}$ | **PASS** |
| `DeltaGridContinuous` (Factorized Kronecker $6 \times 6$) | **FP32** | $9.54 \times 10^{-7}$ | **PASS** |
| `DeltaGridContinuous` (Sparse Delta-PKM $16 \times 16$, $K=4$) | **FP32** | $1.07 \times 10^{-6}$ | **PASS** |
| `DeltaGridLM` (Dense + `thinking_cycles=1`) | **FP32** | $2.38 \times 10^{-6}$ | **PASS** |
| `DeltaGridLM` (Sparse Delta-PKM $16 \times 16$, $K=4$) | **FP32** | $3.46 \times 10^{-6}$ | **PASS** |
| `DeltaGridContinuous` (Dense + Forecast + Thinking) | **Q16.16 FXP** | $2.47 \times 10^{-4}$ | **PASS** |
| `DeltaGridContinuous` (Factorized Kronecker $6 \times 6$) | **Q16.16 FXP** | $2.09 \times 10^{-4}$ | **PASS** |
| `DeltaGridContinuous` (Sparse Delta-PKM $12 \times 12$, $K=3$) | **Q16.16 FXP** | $7.36 \times 10^{-4}$ | **PASS** |
| `DeltaGridLM` (Symmetric `Q8` Quantization) | **INT8 (`Q8`)** | $99.96\%$ Cosine Similarity | **PASS** |

---

## Directory Layout

```text
src/cpp/engine/
├── CMakeLists.txt               # Cross-platform CMake build configuration
├── Makefile                     # Standalone GNU Makefile (macOS, Linux, Windows)
├── README.md                    # Documentation & usage guide
├── include/
│   ├── deltagrid.h              # Public C-ABI header for Continuous & FXP engines
│   ├── deltagrid_llm.h          # Public C-ABI header for Causal LLM engine
│   ├── deltagrid_internal.h     # Binary header format, state & weight structures
│   ├── deltagrid_math.h         # FP32 & INT8 (Q8) vector/matrix primitives
│   └── deltagrid_math_fxp.h     # Q16.16 fixed-point integer arithmetic primitives
├── src/
│   ├── deltagrid_auth.cpp       # Hardware ID / DRM unlock verification
│   ├── deltagrid_continuous.cpp # Continuous control & multi-horizon forecast engine
│   ├── deltagrid_fxp.cpp        # Q16.16 fixed-point integer MCU engine
│   ├── deltagrid_llm.cpp        # Zero-KV causal language model engine
│   ├── deltagrid_math.cpp       # Core 4-tier stepping, routing & state serialization
│   ├── deltagrid_math_fxp.cpp   # Q16.16 MAC array & fixed-point non-linearities
│   ├── main_continuous.cpp      # Standalone CLI benchmark for continuous control
│   ├── main_fxp.cpp             # Standalone CLI benchmark for Q16.16 fixed-point
│   └── main_llm.cpp             # Standalone CLI text generator for DeltaGridLM
├── tests/
│   ├── test_math.cpp            # Standalone C++ unit tests for FP32/INT8 math
│   ├── test_fxp.cpp             # Standalone C++ unit tests for Q16.16 fixed-point
│   └── verify_divergence.cpp    # Standalone C++ FP32 vs Q16.16 trajectory verifier
└── tools/
    └── export_deltagrid_binary.py # Exports PyTorch models to .bin (fp32, int8, fxp)
```

---

## Building & Testing the Engines

### Option A: Using GNU `make`

```bash
cd src/cpp/engine

# Build all shared libraries (.dylib / .so / .dll) and CLI binaries
make all

# Run standalone C++ unit and divergence tests (test_math, test_fxp, verify_divergence)
make test
```

### Option B: Using `CMake`

```bash
cd src/cpp/engine
cmake -B build_cmake -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build_cmake --config Release
ctest --test-dir build_cmake --output-on-failure
```

### Running Python-to-C++ Divergence Verification

```bash
python3 -m unittest tests/test_cpp_engine_parity.py -v
```

---

## Test Coverage Summary (`llvm-cov`)

- **Function Coverage**: **100.00%** (`63 / 63` functions)
- **Line Coverage**: **99.61%** (`1,552 / 1,558` executable lines)
- **Exporter Coverage (`export_deltagrid_binary.py`)**: **100.00%** (`150 / 150` statements)

| Engine Source / Header | Functions | Function Cover | Lines | Missed | Line Cover |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `include/deltagrid_math.h` | 2 / 2 | **100.00%** | 6 | 0 | **100.00%** |
| `include/deltagrid_math_fxp.h` | 1 / 1 | **100.00%** | 4 | 0 | **100.00%** |
| `src/deltagrid_auth.cpp` | 1 / 1 | **100.00%** | 6 | 0 | **100.00%** |
| `src/deltagrid_continuous.cpp` | 13 / 13 | **100.00%** | 89 | 0 | **100.00%** |
| `src/deltagrid_llm.cpp` | 10 / 10 | **100.00%** | 53 | 0 | **100.00%** |
| `src/deltagrid_math_fxp.cpp` | 9 / 9 | **100.00%** | 83 | 0 | **100.00%** |
| `src/deltagrid_math.cpp` | 16 / 16 | **100.00%** | 707 | 3 | **99.58%** |
| `src/deltagrid_fxp.cpp` | 11 / 11 | **100.00%** | 610 | 3 | **99.51%** |
| **TOTAL** | **63 / 63** | **100.00%** | **1,558** | **6** | **99.61%** |
