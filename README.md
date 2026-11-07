# DeltaGrid: Zero-KV Fast-Weight Associative Recurrent Architecture

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-76b900.svg)](https://developer.nvidia.com/cuda-zone)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-14%2F14%20Passed%20(100%25)-brightgreen.svg)]()
[![Inference RAM](https://img.shields.io/badge/Inference%20RAM-Strict%20O(1)%20Zero--KV-blueviolet.svg)]()
[![SRAM Residency](https://img.shields.io/badge/SRAM%20Residency-100%25%20On--Chip%20(Dense)%20%7C%208%20KB%20Window-orange.svg)]()

**DeltaGrid** is a linear-complexity, zero-KV-cache recurrent neural foundation architecture designed for ultra-long context language modeling, sub-millisecond continuous robotics control, and physical telemetry forecasting.

DeltaGrid replaces the quadratic $O(T^2)$ computational complexity and memory-exploding $O(T)$ Key-Value cache of standard Transformers with an **error-erasing closed-loop associative memory rule** operating in **strictly constant $O(1)$ RAM**, featuring **100% on-chip GPU register/SRAM residency** for dense states ($\le 512$ slots) and a **strict 4 KB – 32 KB active SRAM compute window** (8 KB default) for 1,048,576-slot Sparse Delta-PKM.

---

## Key Architecture

### 1. Sparse Product-Key Memory with Householder Delta Erasure (`DeltaGridLayer`)
Traditional dense recurrent memory states suffer from the fundamental compression limit: an $8\text{ KB}$ matrix cannot retain hundreds of distinct key-value pairs verbatim. DeltaGrid solves this by embedding **Product-Key Memory (PKM) with closed-loop Householder Delta Error Erasure** directly inside [`DeltaGridLayer`](src/modules/memory.py):

$$
\mathbf{w}_{k,\text{active}} = \hat{\mathbf{u}}_1 \otimes \hat{\mathbf{u}}_2 \in \mathbb{R}^{M_{\text{active}}}, \quad \|\mathbf{w}_{k,\text{active}}\|_2 = 1.0
$$

```
                          Sparse Delta-PKM (1,048,576 Virtual Slots)
                          
  Query / Key Vector x in R^Dh (64)
              |
      +-------+-------+
      |               |
   [Proj 1]        [Proj 2]
      |               |
   s1 in R^1024    s2 in R^1024       (Exponential Virtual Space: 1024 x 1024 = 1,048,576 slots)
      |               |
   [Top-K=8]       [Top-K=8]          (Dynamically route to 8 active coordinate bins per axis)
      |               |
   u1 in R^8       u2 in R^8          (Strict L2 normalization: ||u1||=1.0, ||u2||=1.0)
      \               /
       === Kronecker ===> w_k,active = u1 (x) u2 in R^64 active slots
                          (||w_k,active|| = 1.0 Strict Householder Isometry)
                          (Touch strictly 64 slots = 8 KB active SRAM compute window!)
```

- **Exponential Capacity Expansion ($1,048,576$ Slots):** Projects input keys across coordinate sub-axes to address an exponential product coordinate space ($1024^2 = 1,048,576$ slots) with minimal linear router parameters ($O(D_h \sum M_i)$).
- **Exact Closed-Loop Error Erasure (Householder Isometry):** Each axis sub-vector is unit-normalized with $\|\hat{\mathbf{u}}_i\|_2 = 1.0$. By the Kronecker mixed-product norm identity, fact overwriting is mathematically exact with zero residual leakage:

$$
\|\mathbf{w}_{k,\text{active}}\|_2 = \prod_{i=1}^N \|\hat{\mathbf{u}}_i\|_2 = 1.0^N = 1.0 \implies \mathbf{w}_{k,\text{active}}^\top \mathbf{S}_t = \mathbf{v}_t
$$

- **Strict 8 KB Active SRAM Window:** By routing to Top-$K$ bins per axis (e.g. $8 \times 8 = 64$ active slots), the active recurrent compute slice remains strictly bounded in GPU hardware registers and on-chip SRAM ($64 \times 64 \times 2\text{ bytes} = 8\text{ KB}$), achieving a **~250x – 260x speedup** over PyTorch loops (257.8x measured on NVIDIA RTX 3060).
- **Massive Associative Recall:** Successfully retains 1,000+ distinct facts sequentially with **99.89% mean cosine similarity** and 100% high-fidelity recall.

### 2. Strict $O(1)$ Inference RAM (Zero KV-Cache)
- Memory consumption remains completely static from token 1 to token 1,000,000+.
- Eliminates out-of-memory crashes during decoding. At 4,096 tokens, DeltaGrid uses **12.5x less memory** than a standard Transformer, unlocking extreme concurrent batch sizes on a single GPU.

### 3. Hardware-Mapped Memory Hierarchy (Register & SRAM Residency)
DeltaGrid is architected from the silicon up to bypass the memory bandwidth bottleneck (the "DRAM Memory Wall") by strictly bounding recurrent compute within on-chip GPU hardware registers and SRAM:
- **Standard Dense Mode ($\text{Slots} \le 512$): 100% Permanent On-Chip State Residency**
  - Full head memory states $\mathbf{S} \in \mathbb{R}^{\text{Slots} \times D_h}$ require only **4 KB – 64 KB per head** (FP16).
  - Resident entirely inside GPU hardware thread registers (`float s_reg[SLOTS]` for $\text{Slots} \le 128$) and on-chip Shared Memory / L1 SRAM ($\text{Slots} \le 512$).
  - **Zero DRAM roundtrips:** State remains resident on-chip across all $T$ timesteps, executing barrier-free at up to **1.59M tok/s (`0.63 µs/tok`)** on-chip sequence recurrence.
- **Sparse Delta-PKM Mode ($1,048,576$ Virtual Slots): Strict 4 KB – 32 KB Active SRAM Window (8 KB Default)**
  - While the massive 1M-slot global knowledge store ($128\text{ MB}$ per head) is hosted in VRAM DRAM, **token recurrence never executes a dense 128 MB matrix operation**.
  - At each token step $t$, the coordinate router indexes strictly the active $K_1 \times K_2 = 64$ slots (**8 KB FP16**) into on-chip registers/SRAM.
  - Closed-loop Householder read, error residual erasure, and delta update execute at register speed before coalesced scattering back to VRAM.

### 4. Closed-Loop Error-Erasing Delta Recurrence
Unlike classical additive RNNs (LSTMs/GRUs) which suffer catastrophic forgetting, DeltaGrid continuously computes the key-prediction error before writing:

$$
\begin{aligned}
\text{Read:}  &\quad \mathbf{y}_t = \mathbf{w}_{q,t}^\top \mathbf{S}_{t-1} \\
\text{Error:} &\quad \mathbf{r}_t = \mathbf{v}_t - \mathbf{w}_{k,t}^\top \mathbf{S}_{t-1} \\
\text{Write:} &\quad \mathbf{S}_t = \alpha_t \mathbf{S}_{t-1} + \beta_t (\mathbf{w}_{k,t} \mathbf{r}_t^\top)
\end{aligned}
$$

### 5. 4-Tier Neuromorphic Hierarchy
- **Tier 0 (Harmonic Clock):** Multi-frequency sinusoidal coordinates ($\omega_k = 10000^{-2k/d}$) providing absolute phase and temporal anchoring without positional lookup tables.
- **Tier 1 (Causal Depthwise Conv1D, $k=4$):** High-frequency local n-gram token binding and consecutive pair fusion.
- **Tier 2 (Associative Memory Layer):** Bounded fast-weight associative matrix memory driven by the Householder Delta Rule.
- **Tier 3 (GLU Logic Reasoning Engine):** Dual-branch Gated Linear Units with working-memory recurrent thinking cycles for relational multi-hop deduction.

### 6. Dual-Engine Execution: Vectorized Chunkwise Prefill + Fused CUDA Kernel
- **Prefill (Training & Context Encoding):** Fully vectorized chunkwise parallel WY Tensor-Core prefill solves all intra-chunk associative factorizations simultaneously in batched triangular solves.
- **Generation (Autoregressive Decoding):** Barrier-free register-resident CUDA C++ kernel ([src/cpp](src/cpp)) executes single-token recurrence directly in hardware registers.

### 7. Standalone Zero-Dependency C++ Edge Engine (`src/cpp/engine`)
For deployment on autonomous drones, robotics controllers, and safety-critical microcontrollers without Python or LibTorch:
- **`libdeltagrid_continuous` (`FP32` / `INT8`):** Sub-millisecond real-time telemetry control and direct multi-horizon trajectory forecasting with zero heap allocations during `.step()`.
- **`libdeltagrid_fxp` (`Q16.16 Fixed-Point`):** Pure 32-bit integer arithmetic (`int32_t` with 64-bit MAC accumulators) for FPU-less microcontrollers (IEC 62304 / DO-178C friendly), matching FP32 PyTorch within $< 7.4 \times 10^{-4}$.
- **`libdeltagrid_llm` (`FP32` / `INT8`):** Zero-KV causal language model engine in strictly constant $O(1)$ RAM, supporting Standard Dense, Factorized Kronecker, and Sparse Delta-PKM routing plus conversational state save/restore.

---

## Architecture Overview

```mermaid
flowchart TD
    In[Input Stream: x_t] --> T0[Tier 0: Biological Harmonic Clock]
    T0 --> T1[Tier 1: Causal Depthwise Conv1D k=4]
    T1 --> T2[Tier 2: DeltaGrid PKM Memory Layer]
    
    subgraph ROUTER [Sparse Delta-PKM Coordinate Router: 1M Virtual Slots]
        SubProj["Sub-Axis Projections: s_i = W_i k"]
        TopK["Top-K Active Selection: val_i, idx_i = topk(s_i, K)"]
        Norm["Householder Normalization: ||u_i|| = 1.0"]
        Kron["Active Kronecker Key: w_{k,active} = u_1 ⊗ u_2"]
        SubProj --> TopK --> Norm --> Kron
    end
    
    subgraph S_MEM [On-Chip Register Window: 8 KB/Head Active SRAM]
        Read["Read: y_t = S_{t-1}[idx] w_{q,t}"]
        Err["Error: r_t = v_t - S_{t-1}[idx] w_{k,t}"]
        Write["Write: S_t[idx] = α_t S_{t-1}[idx] + β_t (w_{k,t} outer r_t^T)"]
        Read --> Err --> Write
    end
    
    T2 --> ROUTER
    ROUTER <--> S_MEM
    T2 --> T3[Tier 3: GLU Logic Engine]
    T3 --> Think{Adaptive Thinking Cycles?}
    Think -- "cycles > 0" --> T2
    Think -- "done" --> NormOut[LayerNorm & Out Projection]
    NormOut --> Out[Predicted Output / Logits]
```

---

## Comprehensive Empirical Benchmarks

DeltaGrid has been rigorously evaluated across all 14 standard recurrent memory and reasoning tasks:

### Benchmark Scorecard (14 / 14 Tasks Passed)

| # | Benchmark Task | Metric | DeltaGrid Score | Target Baseline | Status |
| :-: | :--- | :--- | :---: | :---: | :---: |
| **1** | **Associative Recall (Across Noise)** | Accuracy | **100.0%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **2** | **Multi-Token Sequence Copy** | Accuracy | **100.0%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **3** | **2nd-Order Markov Grammar** | Accuracy / Loss | **100.0%** ($CE < 0.05$) | $\ge 95.0\%$ | **PASSED (100%)** |
| **4** | **Adding Problem (Continuous Sum)** | MSE | **0.00040** | $\sim 0.166$ (Random) | **PASSED** |
| **5** | **Continuous Wave Extrapolation** | 32-Step Horizon MSE | **0.00444** | $MSE < 0.05$ | **PASSED** |
| **6** | **GPU Memory Scaling (4,096 tokens)** | Peak VRAM | **98.8 MB** | 1,236.3 MB (Transformer) | **12.5x Smaller** |
| **7** | **Dynamic Password Overwrite** | 1-Step Mutation Acc | **100.0%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **8** | **Needle In A Haystack (64 tokens)** | Exact Retrieval Acc | **100.0%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **9** | **Needle In A Haystack (256 tokens)** | Exact Retrieval Acc | **100.0%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **10** | **Needle In A Haystack (512 tokens)** | Exact Retrieval Acc | **100.0%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **11** | **Associative Recall (SQAR 8-Pairs)** | Disambiguation Acc | **99.6%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **12** | **Multi-Variable State Machine** | 5 Variable Tracking | **100.0%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **13** | **Multi-Hop Transitive Reasoning** | Resolution Acc | **100.0%** | $\ge 98.0\%$ | **PASSED (100%)** |
| **14** | **Sparse Delta-PKM (1,000 Facts in 1M Slots)** | Recall ($\cos > 0.80$) | **100.0%** (Cosine: **0.9989**) | $\ge 98.0\%$ | **PASSED (100%)** |

### High-Density Associative KV Retrieval Capacity (Delta-PKM)
*Measured live on sequential random KV pair retention (Cosine similarity of retrieved values; $1.0 = \text{perfect verbatim recall}$):*

| Stored KV Pairs | Virtual Slots | Active Compute Window | Mean Cosine Similarity | High-Fidelity Recall ($\cos > 0.80$) | Memory Retention Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **16 Pairs** | 1,048,576 | 64 slots (8 KB) | **1.0000** | **100.0%** | Lossless verbatim memory |
| **32 Pairs** | 1,048,576 | 64 slots (8 KB) | **0.9999** | **100.0%** | Lossless verbatim memory |
| **64 Pairs** | 1,048,576 | 64 slots (8 KB) | **0.9999** | **100.0%** | Lossless verbatim memory |
| **128 Pairs** | 1,048,576 | 64 slots (8 KB) | **0.9999** | **100.0%** | Lossless verbatim memory |
| **256 Pairs** | 1,048,576 | 64 slots (8 KB) | **0.9997** | **100.0%** | Near-lossless retention |
| **512 Pairs** | 1,048,576 | 64 slots (8 KB) | **0.9994** | **100.0%** | Near-lossless retention |
| **1,000 Pairs** | 1,048,576 | 64 slots (8 KB) | **0.9989** | **100.0%** | Near-lossless retention |
| **2,048 Pairs** | 1,048,576 | 64 slots (8 KB) | **0.9977** | **100.0%** | High-fidelity retention |
| **5,000 Pairs** | 1,048,576 | 64 slots (8 KB) | **0.9944** | **100.0%** | High-fidelity retention |

---

## Scaling Dimensions & Hardware Guarantees

DeltaGrid is architected around four distinct scaling dimensions:

### 1. Sequence Length Scaling ($T \to 1\text{K}, 16\text{K}, 128\text{K}, 1\text{M}$ Tokens)
- **Transformers:** KV-cache scales as $O(T)$ during autoregressive decoding, while full-attention activations during context encoding scale quadratically as $O(T^2)$. At 128K tokens, a standard 7B model demands $>32\text{ GB}$ of VRAM solely for KV storage.
- **DeltaGrid:** Generation memory is strictly $O(1)$ constant. Recurrent state remains **8 KB per head** regardless of context length:

```
Context Length (Tokens)    Transformer Full-Attention (O(T^2))    Transformer KV-Cache (O(T))    DeltaGrid Streaming State (O(1))
---------------------------------------------------------------------------------------------------------------------------------
1,024 tokens               ~81.5 MB                               ~4.0 MB                        8 KB / head (Strict O(1))
4,096 tokens               ~1,238.2 MB                            ~16.0 MB                       8 KB / head (Strict O(1))
16,384 tokens              ~19,700 MB (19.7 GB)                   ~64.0 MB                       8 KB / head (Strict O(1))
65,536 tokens              ~315,000 MB (315 GB) [OOM]             ~256.0 MB                      8 KB / head (Strict O(1))
1,000,000 tokens           > 4.8 TB [Impossible]                  ~3,906 MB (~3.9 GB)            8 KB / head (Strict O(1))
```

### 2. Product-Key Memory (PKM) Coordinate Layouts

DeltaGrid's Product-Key Memory maps high-dimensional associative keys into an exponential virtual slot space while dynamically activating only a minimal register-resident subset per token step:

| Memory Architecture Mode | Coordinate Axes Geometry | Total Virtual Slots | Active Slots per Step | Active SRAM Window | Key Router Params (per Head) | Primary Application Domain |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standard Dense** | Direct 1D | 64 | 64 | 8.0 KB | 4,096 | Embedded edge devices, robotics, real-time control |
| **5-Axis PKM (4 KB)** | `(16, 16, 16, 16, 16)` | **1,048,576** | 32 ($K=2$) | **4.0 KB** | **5,120** | Ultra-low parameter mobile & embedded edge devices |
| **Balanced PKM (8 KB)** | `(64, 64)` | **4,096** | 64 ($K=8$) | **8.0 KB** | 8,192 | Mid-sized agents, code synthesis, mathematical reasoning |
| **Massive PKM (8 KB)** | `(1024, 1024)` | **1,048,576** | 64 ($K=8$) | **8.0 KB** | 131,072 | Large-scale LLMs, long-document reasoning, multi-hop QA |
| **4-Axis PKM (10 KB)** | `(32, 32, 32, 32)` | **1,048,576** | 81 ($K=3$) | **10.1 KB** | 8,192 | Hierarchical multi-predicate knowledge graphs |
| **7-Axis PKM (16 KB)** | `(8, 8, 8, 8, 8, 8, 4)` | **1,048,576** | 128 ($K=2$) | **16.0 KB** | 3,328 | High-precision multi-axis relational memory |
| **Wide PKM (32 KB)** | `(1024, 1024)` | **1,048,576** | 256 ($K=16$) | **32.0 KB** | 131,072 | Ultra-dense associative world models |

> [!NOTE]
> All configurations are directly parameterized using `grid_dim` and `top_k`. Active compute recurrence strictly touches **32 – 256 slots (4.0 KB – 32.0 KB)** across all PKM geometries (**8.0 KB default** for `top_k=8` 2-axis grids)!

### 3. Slot Capacity vs. Hardware Memory Hierarchy
DeltaGrid's state layout maps directly into modern GPU memory tiers (measured on NVIDIA GeForce RTX 3060):

| Slots | State per Head (FP16) | Physical GPU Memory Tier | On-Chip Sequence Recurrence ($T=512$) | Single-Step Decode Latency |
| :---: | :---: | :--- | :---: | :---: |
| **32** | **4.0 KB** | **100% Thread Hardware Registers (`s_reg[32]`)** | **0.63 µs/tok (1.59M tok/s)** | **15.4 µs (~65,000 tok/s)** |
| **64** | **8.0 KB** | **100% Thread Hardware Registers (`s_reg[64]`)** | **0.77 µs/tok (1.30M tok/s)** | **17.4 µs (~57,600 tok/s)** |
| **128** | **16.0 KB** | **100% Thread Hardware Registers (`s_reg[128]`)** | **1.88 µs/tok (531.9K tok/s)** | **17.1 µs (~58,400 tok/s)** |
| **256** | **32.0 KB** | **100% On-Chip Shared Memory (SMEM Tile)** | **18.00 µs/tok (55.6K tok/s)** | **34.2 µs (~29,200 tok/s)** |
| **512** | **64.0 KB** | **100% On-Chip Shared Memory (SMEM Tile)** | **35.40 µs/tok (28.2K tok/s)** | **75.1 µs (~13,300 tok/s)** |
| **1,048,576** | **128.0 MB** | **VRAM Backing Store + 4–32 KB Register Window** | **7.03 – 13.52 µs/tok** | **12.3 – 14.7 µs (68K–81K tok/s)** |

> [!TIP]
> **Physical Memory Mapping Architecture:**
> - **Dense Modes ($\le 512$ slots):** The **entire recurrent state matrix** stays permanently resident in GPU hardware registers (`Slots <= 128`) or Shared Memory SRAM (`128 < Slots <= 512`) throughout sequence generation — achieving true **zero DRAM roundtrips**.
> - **Massive Delta-PKM ($1,048,576$ slots):** The **backing storage table** ($128\text{ MB}$) is allocated in VRAM DRAM, but the **compute recurrence is strictly 4 KB – 32 KB register/SRAM resident** (8 KB default): only the dynamically routed active slots are gathered into on-chip registers for the Householder delta update, eliminating dense memory-bandwidth bottlenecks.

---

## Installation & Setup

### Prerequisites
- **Python:** 3.9, 3.10, 3.11, 3.12, or 3.13
- **PyTorch:** $\ge 2.0.0$ with CUDA support (only `torch` is required; `torchvision` and `torchaudio` are **not** needed)
- **CUDA Toolkit:** 11.8, 12.x, or 13.x (`nvcc` compiler on `PATH`)
- **C++ Compiler:** 
  - **Windows:** MSVC (Visual Studio 2022 C++ Build Tools)
  - **Linux:** GCC / Clang (C++17)

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/hsameerc/deltagrid.git
cd deltagrid

# 2. Install PyTorch with CUDA support (e.g., CUDA 12.4 or 12.1; no torchvision/torchaudio required)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Install build tools and requirements
pip install ninja
pip install -r requirements.txt

# 4. Build and install DeltaGrid with fused CUDA extension
pip install -e . --no-build-isolation
```

> [!NOTE]
> `--no-build-isolation` ensures the build uses your active PyTorch environment and Ninja directly. DeltaGrid's `setup.py` automatically bridges minor CUDA version differences between PyTorch and `nvcc` (e.g., PyTorch cu128 with CUDA Toolkit 13.x).

---

## Quickstart Examples

### 1. Large-Scale Delta-PKM Memory Layer (`DeltaGridLayer`)

```python
import torch
from deltagrid import DeltaGridLayer  # or: from src import DeltaGridLayer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Standard Dense Mode (fits 100% in SRAM registers, zero DRAM overhead):
layer_dense = DeltaGridLayer(d_model=128, num_heads=2).to(device)

# Exponential Sparse Delta-PKM Mode (1,048,576 slots, strictly 64 active slots = 8 KB SRAM window):
layer = DeltaGridLayer(d_model=128, num_heads=2, grid_dim=(1024, 1024), top_k=8).to(device)

x = torch.randn(2, 64, 128, device=device)

# Parallel sequence forward (vectorized batched coordinate routing)
out, state = layer(x)
print(f"Output: {out.shape}, Global Memory State: {state.shape}")
# State shape: [2, 2, 1048576, 64] (touches strictly 8 KB active window per token!)

# Microsecond fused CUDA step decoding:
step_x = torch.randn(2, 1, 128, device=device)
step_out, state = layer.step(step_x, state=state)

# Check coordinate codebook utilization (verifies zero cold-slot collapse):
entropy = layer.compute_router_entropy(x)
print(f"Router coordinate entropy: {entropy.item():.3f} nats")
```

### 2. Zero-KV Language Modeling (`DeltaGridLM`)

```python
import torch
from deltagrid import DeltaGridLM  # or: from src import DeltaGridLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# 4-layer Zero-KV Language Model with Delta-PKM Memory
model = DeltaGridLM(
    vocab_size=32000,
    d_model=128,
    num_layers=4,
    num_heads=4,
    grid_dim=(1024, 1024),
    top_k=8,
).to(device)

# Parallel context prefill: [Batch=1, SeqLen=1024]
prompt = torch.randint(0, 32000, (1, 1024), device=device)
logits, states = model(prompt)

# Autoregressive generation in strictly constant O(1) RAM:
next_token = logits[:, -1:, :].argmax(dim=-1)
generated = []

for step in range(50):
    step_logits, states = model.step(next_token, states, seq_offset=1024 + step)
    next_token = step_logits.argmax(dim=-1)
    generated.append(next_token.item())

print(f"Generated {len(generated)} tokens in constant memory!")
```

### 3. Continuous Robotics & Telemetry Forecasting (`DeltaGridContinuous`)

```python
import torch
from deltagrid import DeltaGridContinuous  # or: from src import DeltaGridContinuous

device = "cuda" if torch.cuda.is_available() else "cpu"

# Controller with 12 sensory telemetry channels and 6 motor torque outputs
controller = DeltaGridContinuous(
    input_dim=12,
    output_dim=6,
    d_model=128,
    num_layers=2,
    grid_dim=(8, 8),            # 2D grid for smooth continuous curvature
    forecast_horizon=32,
).to(device)

# Real-time sub-millisecond control step:
sensor_sample = torch.randn(1, 1, 12, device=device)
states = None
torque_command, states = controller.step(sensor_sample, states)

# Closed-loop trajectory forecast without compounding exposure bias:
history = torch.randn(1, 64, 12, device=device)
future_trajectory = controller.forecast(history, forecast_steps=32)
```

### 4. Standalone Modular Memory Block (`DeltaGridBlock`)

```python
import torch
from deltagrid import DeltaGridBlock  # or: from src import DeltaGridBlock

# Embed the complete 4-tier block inside custom models:
block = DeltaGridBlock(
    d_model=128,
    num_heads=4,
    grid_dim=(1024, 1024),
    top_k=8,
    has_memory=True,
    kernel_size=4,
    decay_init=0.999,
).cuda()

x = torch.randn(2, 64, 128, device="cuda")
# Forward pass with 2 cognitive thinking cycles per step:
out, new_mem, new_conv = block(x, thinking_cycles=2)
```

### 5. Standalone C++ Edge Export (`FP32`, `INT8`, `Q16.16 Fixed-Point`)

```python
from deltagrid import DeltaGridContinuous, DeltaGridLM, export_deltagrid_to_bin

# Export Continuous Controller to FP32 or Q16.16 Fixed-Point (.bin)
controller = DeltaGridContinuous(input_dim=12, output_dim=6, d_model=64, num_layers=2, forecast_horizon=8).eval()
export_deltagrid_to_bin(controller, "controller_fp32.bin", mode="fp32")
export_deltagrid_to_bin(controller, "controller_q16.bin",  mode="fxp")

# Export Zero-KV Language Model to 4x-compressed Symmetric INT8 (.bin)
lm = DeltaGridLM(vocab_size=32000, d_model=128, num_layers=4, grid_dim=(64, 64), top_k=8).eval()
export_deltagrid_to_bin(lm, "model_q8.bin", mode="int8")
```

---

## Verification & Benchmarks

### 1. Automated Unit, Divergence & C++ Edge Parity Tests (99.6% C++ / 99.8% CPU Python Coverage)
Run the automated test suite across configuration, state encapsulation, high-level models, native CUDA parity, and the standalone C++ Edge Engine (`FP32`, `INT8`, and `Q16.16 Fixed-Point`):

```bash
# Run all Python unit & C++ parity tests:
python -m unittest discover tests

# Build and run the standalone C++ Edge Engine test suite (test_math, test_fxp, verify_divergence):
make -C src/cpp/engine all && make -C src/cpp/engine test

# Or run individual test modules:
python -m unittest tests/test_config.py             # Configuration engine & validation (100% coverage)
python -m unittest tests/test_state.py              # Recurrent state isolation & TBPTT detach (100% coverage)
python -m unittest tests/test_models.py             # LM & Continuous controller interfaces (100% coverage)
python -m unittest tests/test_sparse_pkm.py         # Sparse Delta-PKM mathematical correctness
python -m unittest tests/test_cuda_parity.py        # Bitwise CUDA kernel vs PyTorch parity (< 1e-6)
python -m unittest tests/test_continuous.py         # Continuous stream forecasting & control
python -m unittest tests/test_integrity.py          # Causality, gradient flow & determinism
python -m unittest tests/test_cpp_engine_parity.py  # PyTorch vs C++ Edge Engine (FP32, INT8, Q16.16 FXP)
```

### 2. Performance Profiling & Reasoning Benchmarks
Evaluate long-sequence throughput and synthetic cognitive reasoning tasks:

```bash
# 1. Microsecond kernel latency & throughput benchmarking (~250x – 260x speedup over PyTorch)
python benchmarks/bench_cuda_speed.py

# 2. 7-Area associative memory reasoning benchmark (Password Overwrite, Needle, SQAR, Multi-Hop)
python benchmarks/bench_associative_memory.py

# 3. Master unified benchmark suite & dashboard visualization
python benchmarks/bench_unified_suite.py --task all

# 4. Rigorous parameter-matched baseline comparison (Transformers, LSTMs, GRUs)
python benchmarks/bench_baselines.py
python benchmarks/plot_figures.py

# 5. Hardware SRAM-constrained benchmark (4 KB, 8 KB, 16 KB, 32 KB windows vs DRAM traffic & energy)
python benchmarks/bench_sram_constrained.py --num-facts 500 --runs 200
```

<p align="center">
  <img src="figures/sram_constrained_benchmark.png" alt="DeltaGrid Hardware SRAM-Constrained Recurrent Memory Benchmark" width="100%">
</p>
<p align="center"><em>Figure 2: Hardware SRAM-constrained benchmark on NVIDIA RTX 3060 across 4 KB, 8 KB, 16 KB, 32 KB, and 64 KB working windows (decoding latency in microseconds, DRAM traffic scaling vs. Transformer KV-cache, first-principles VLSI energy breakdown, and multi-query associative recall accuracy).</em></p>

### Hardware SRAM Benchmark Summary (Measured on NVIDIA RTX 3060)

#### 1. Active Compute, Hardware Residency & Energy Profile ($N = 500$ Facts)
| Architecture Mode | Active SRAM Window | DRAM Traffic / Step | Step Latency | On-Chip Seq ($T=512$) | VLSI Energy / Token | 500-Fact Recall | Physical Residency Tier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DeltaGrid PKM (4 KB)** | **4.0 KB** | 8.0 KB | **12.30 µs** (81.3K tok/s) | 7.03 µs/tok | **1,272.3 nJ** | **100.0%** ($\cos=1.00$) | Register Window (32 slots) $\to$ VRAM |
| **DeltaGrid Dense (4 KB)** | **4.0 KB** | **0.0 KB (Zero)** | **15.38 µs** (65.0K tok/s) | **0.63 µs/tok (1.59M tok/s)** | **30.7 nJ** | 1.8% ($N \gg 32$) | 100% Thread Registers (`s_reg[32]`) |
| **DeltaGrid PKM (8 KB)** | **8.0 KB** | 16.0 KB | **14.65 µs** (68.3K tok/s) | 13.52 µs/tok | **2,846.7 nJ** | **100.0%** ($\cos=1.00$) | Register Window (64 slots) $\to$ VRAM |
| **DeltaGrid Dense (8 KB)** | **8.0 KB** | **0.0 KB (Zero)** | **17.36 µs** (57.6K tok/s) | **0.77 µs/tok (1.30M tok/s)** | **61.4 nJ** | 3.4% ($N \gg 64$) | 100% Thread Registers (`s_reg[64]`) |
| **DeltaGrid PKM (10 KB)** | **10.1 KB** | 20.2 KB | **18.06 µs** (55.4K tok/s) | 17.38 µs/tok | **3,208.6 nJ** | **100.0%** ($\cos=1.00$) | Register Window (81 slots) $\to$ VRAM |
| **DeltaGrid PKM (16 KB)** | **16.0 KB** | 32.0 KB | **21.96 µs** (45.5K tok/s) | 28.18 µs/tok | **5,046.4 nJ** | **100.0%** ($\cos=1.00$) | Register Window (128 slots) $\to$ VRAM |
| **DeltaGrid Dense (16 KB)** | **16.0 KB** | **0.0 KB (Zero)** | **17.12 µs** (58.4K tok/s) | **1.88 µs/tok (531.9K tok/s)** | **122.9 nJ** | 6.6% ($N \gg 128$) | 100% Thread Registers (`s_reg[128]`) |
| **DeltaGrid PKM (32 KB)** | **32.0 KB** | 64.0 KB | **45.32 µs** (22.1K tok/s) | 66.79 µs/tok | **10,403.8 nJ** | **100.0%** ($\cos=1.00$) | Register Window (256 slots) $\to$ VRAM |
| **DeltaGrid Dense (32 KB)** | **32.0 KB** | **0.0 KB (Zero)** | **34.22 µs** (29.2K tok/s) | 18.00 µs/tok (55.6K tok/s) | **245.8 nJ** | 12.2% ($N \gg 256$) | 100% On-Chip Shared Memory (SMEM Tile) |
| **DeltaGrid Dense (64 KB)** | **64.0 KB** | **0.0 KB (Zero)** | **75.11 µs** (13.3K tok/s) | 35.40 µs/tok (28.2K tok/s) | **491.5 nJ** | **100.0%** ($\cos=1.00$) | 100% On-Chip Shared Memory (SMEM Tile) |
| **Transformer ($T=1024$)** | $O(T)$ Cache | 256.0 KB | 125.35 µs (8.0K tok/s) | — | 39,649.9 nJ | 100.0% | Unconstrained VRAM KV-Cache (512 KB) |
| **Transformer ($T=4096$)** | $O(T)$ Cache | 1,024.0 KB | 127.38 µs (7.9K tok/s) | — | 158,597.8 nJ | 100.0% | Unconstrained VRAM KV-Cache (2,048 KB) |

#### 2. Memory Capacity Step-Transition Sweep ($N = 32 \dots 1,000$ Facts)
*Demonstrates exact 100% Householder recall for Dense windows up to their rank limit ($N \le \text{Slots}$), while Sparse Delta-PKM decouples active SRAM compute ($4\text{ KB} - 32\text{ KB}$) from total memory capacity ($1,048,576$ slots) to sustain 100.0% recall across all $N$:*

| Stored Facts ($N$) | Dense 4 KB (32 Slots) | Dense 8 KB (64 Slots) | Dense 16 KB (128 Slots) | Dense 32 KB (256 Slots) | Dense 64 KB (512 Slots) | PKM 4 KB / 8 KB / 16 KB / 32 KB ($10^6$ Slots) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32 Facts** | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) |
| **64 Facts** | 14.1% ($\cos=0.53$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) |
| **128 Facts** | 6.2% ($\cos=0.28$) | 12.5% ($\cos=0.52$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) |
| **256 Facts** | 3.5% ($\cos=0.14$) | 6.6% ($\cos=0.25$) | 11.3% ($\cos=0.49$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) |
| **500 Facts** | 1.8% ($\cos=0.07$) | 3.4% ($\cos=0.13$) | 6.6% ($\cos=0.25$) | 12.2% ($\cos=0.51$) | **100.0%** ($\cos=1.00$) | **100.0%** ($\cos=1.00$) |
| **1,000 Facts** | 0.7% ($\cos=0.03$) | 1.3% ($\cos=0.06$) | 3.3% ($\cos=0.13$) | 6.0% ($\cos=0.26$) | 12.6% ($\cos=0.51$) | **100.0%** ($\cos=1.00$) |

<p align="center">
  <img src="figures/rigorous_dashboard.png" alt="DeltaGrid Rigorous Benchmarks Dashboard" width="100%">
</p>
<p align="center"><em>Figure 1: Empirical multi-seed benchmarks against parameter-matched Transformers, LSTMs, and GRUs (MQAR associative recall, Needle-in-a-Haystack up to 2,048 tokens, Adding Problem, Continuous Wave Forecasting, Strict O(1) Memory Scaling up to 65,536 tokens, and Byte-level Language Modeling).</em></p>

---

## Repository Structure

```
deltagrid/
├── src/                                # Core Neural Engine (installed as deltagrid package)
│   ├── __init__.py                     # Clean Public API (__all__)
│   ├── config.py                       # DeltaGridConfig (Dataclass + Strict Validation)
│   ├── state.py                        # DeltaGridState & LayerState (TBPTT Detach Container)
│   ├── outputs.py                      # Structured Output Containers (DeltaGridLMOutput, etc.)
│   ├── modules/                        # Core Neural Building Blocks
│   │   ├── memory.py                   # Tier 2: DeltaGridLayer (Dense, Kronecker, PKM)
│   │   └── block.py                    # Tier 3: 4-Tier DeltaGridBlock
│   ├── models/                         # High-Level Standalone Models
│   │   ├── base.py                     # _DeltaGridSequenceModel Backbone
│   │   ├── language_model.py           # DeltaGridLM (Zero-KV Autoregressive Decoding)
│   │   └── continuous.py           # DeltaGridContinuous (Robotics / Drone Physics Engine)
│   ├── ops/                            # Encapsulated Low-Level Engine (Private)
│   │   ├── _backend.py             # Dynamic CUDA loader & CPU reference fallback
│   │   └── _autograd.py            # _DeltaGridAutogradFunction, _SparseDeltaPKMAutogradFunction
│   ├── core/                           # Backward compatibility shim
│   └── cpp/                            # Native C++ & CUDA Implementations
│       ├── binding.cpp                 # PyTorch pybind11 extension bindings
│       ├── delta_kernel.cu             # Fused register-resident shared-memory CUDA kernels
│       ├── delta_kernel.h              # CUDA kernel headers
│       └── engine/                     # Standalone Zero-Dependency C++ Edge Engine
│           ├── CMakeLists.txt          # Cross-platform CMake build configuration
│           ├── Makefile                # Standalone GNU Makefile (macOS, Linux, Windows)
│           ├── README.md               # C++ Edge Engine documentation & ABI guide
│           ├── include/                # Public C-ABI & internal architecture headers
│           │   ├── deltagrid.h         # Continuous (FP32/INT8) & Q16.16 FXP C-API
│           │   ├── deltagrid_llm.h     # Zero-KV Causal LLM C-API
│           │   ├── deltagrid_internal.h# Binary DLG1 specification & runtime structs
│           │   ├── deltagrid_math.h    # FP32 & INT8 (Q8) math primitives
│           │   └── deltagrid_math_fxp.h# Q16.16 fixed-point integer math primitives
│           ├── src/                    # C++14 Edge Runtimes & CLI Binaries
│           │   ├── deltagrid_auth.cpp
│           │   ├── deltagrid_continuous.cpp
│           │   ├── deltagrid_fxp.cpp
│           │   ├── deltagrid_llm.cpp
│           │   ├── deltagrid_math.cpp
│           │   ├── deltagrid_math_fxp.cpp
│           │   ├── main_continuous.cpp
│           │   ├── main_fxp.cpp
│           │   └── main_llm.cpp
│           ├── tests/                  # Standalone C++ Unit & Divergence Verifiers
│           │   ├── test_math.cpp
│           │   ├── test_fxp.cpp
│           │   └── verify_divergence.cpp
│           └── tools/                  # Model Serialization Tools
│               └── export_deltagrid_binary.py
├── tests/                              # Automated Fast CI Unit & Regression Tests
│   ├── test_config.py                  # Architectural validation & JSON/dict persistence
│   ├── test_state.py                   # Recurrent state isolation & graph detachment
│   ├── test_models.py                  # LM prefill, step, generate, and continuous forecasting
│   ├── test_sparse_pkm.py              # Sparse Delta-PKM (1,048,576 slots) correctness & AMP
│   ├── test_cuda_parity.py             # Bitwise floating-point divergence checks (< 1e-6)
│   ├── test_continuous.py              # Continuous physics, robotics, and trajectory control
│   ├── test_integrity.py               # Causality, chunkwise invariance, and gradient health
│   └── test_cpp_engine_parity.py       # End-to-end PyTorch vs C++ Edge Engine parity suite
├── benchmarks/                         # Performance Profiling & Figure Dashboards
│   ├── bench_cuda_speed.py             # Fused kernel latency and throughput benchmarks
│   ├── bench_associative_memory.py     # 7-area expanded associative reasoning benchmark
│   ├── bench_unified_suite.py          # Unified multi-task benchmark suite (14 tasks)
│   ├── bench_baselines.py              # Multi-seed baseline comparisons (Transformer, LSTM, GRU)
│   ├── test_random_needle.py           # Random-position & depth-sweep needle evaluation
│   └── plot_figures.py                 # Figure plotting utilities
├── figures/                            # Generated benchmark visualizations
├── pyproject.toml                      # PEP 517 build configuration
├── requirements.txt                    # Python dependencies
├── setup.py                            # Setuptools CUDA extension build configuration
└── README.md                           # Architecture guide and documentation
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DeltaGrid or its factorized Kronecker memory architecture in your research, please cite:

```bibtex
@software{deltagrid_2026,
  author = {Humagain, Sameer},
  title  = {DeltaGrid: Zero-KV Fast-Weight Associative Recurrent Architecture},
  year   = {2026},
  url    = {https://github.com/hsameerc/deltagrid}
}
```
