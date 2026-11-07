# DeltaGrid Benchmark Suite

This directory contains the performance profiling, hardware latency measurement, synthetic reasoning benchmarks, and multi-seed baseline comparisons for the **DeltaGrid** architecture.

All benchmarks are fully automated, deterministic with seed control, and runnable on both CUDA GPUs and CPUs.

---

## Benchmark Scripts Overview

| Script | Primary Focus | Evaluated Models / Modes | Outputs Generated |
| :--- | :--- | :--- | :--- |
| **[`bench_sram_constrained.py`](bench_sram_constrained.py)** | Real-world on-chip SRAM budget constraints (4 KB – 64 KB) | DeltaGrid (Dense & PKM) vs. Transformer KV | Console scorecard, `figures/sram_constrained_results.json`, `figures/sram_constrained_benchmark.png` |
| **[`bench_cuda_speed.py`](bench_cuda_speed.py)** | Microsecond hardware latency & throughput | Fused CUDA kernel vs. PyTorch recurrent loop | Console latency (ms/us), speedup factor, steps/s |
| **[`bench_unified_suite.py`](bench_unified_suite.py)** | Master 14-task multi-task benchmark suite | DeltaGrid (Dense, Factorized, Sparse PKM) | 6 task figures + `figures/unified_benchmark_dashboard.png` |
| **[`bench_associative_memory.py`](bench_associative_memory.py)** | 7 synthetic cognitive reasoning tasks | DeltaGrid LM | Console scorecard (accuracy % and execution time) |
| **[`test_random_needle.py`](test_random_needle.py)** | Needle-in-a-Haystack random position & depth sweep | DeltaGrid LM (trained $L \le 64$, tested to $L=2048$) | Random position recall & depth-wise sensitivity profiles |
| **[`bench_baselines.py`](bench_baselines.py)** | Rigorous multi-seed baseline comparisons | DeltaGrid vs. **GRU**, **LSTM**, **Causal Transformer** | `figures/rigorous_results.json` |
| **[`plot_figures.py`](plot_figures.py)** | Multi-panel dashboard visualization | Reads `figures/rigorous_results.json` | `figures/rigorous_dashboard.png` |

---

## Detailed Task Breakdown

### 1. Unified Multi-Task Suite (`bench_unified_suite.py`)
Evaluates DeltaGrid across 14 distinct tasks spanning memory retention, sequence reproduction, continuous signal forecasting, and sparse product-key associative recall:

- **Classic Suite (Tasks 1–6):**
  - **Task 1: Associative Recall across Noise:** Query a target token stored across variable distractor noise.
  - **Task 2: Sequence Copying:** Multi-token sequence memorization and verbatim reproduction.
  - **Task 3: 2nd-Order Toy Grammar:** Non-linear Markov next-token prediction ($CE < 0.05$).
  - **Task 4: The Adding Problem:** Continuous float summation over long sequences (Hochreiter & Schmidhuber, 1997).
  - **Task 5: Continuous Sine Wave Extrapolation:** 32-step history context to forecast the next 32 future steps.
  - **Task 6: Memory Scaling Profile:** Empirical peak GPU VRAM allocation from 128 to 4,096 tokens vs. naive attention.
- **Associative Reasoning Suite (Tasks 7–13):**
  - **Task 7: Dynamic Password Overwrite:** 1-step fact overwriting and key-variable mutation.
  - **Task 8: Needle-in-a-Haystack (64 tokens):** Exact retrieval across distractor background tokens.
  - **Task 9: Deep Needle (256 tokens):** Long-context needle retrieval.
  - **Task 10: Ultra-Deep Needle (512 tokens):** Ultra-long context needle retrieval.
  - **Task 11: Single-Query Associative Recall (SQAR 8-Pairs):** Disambiguation of 8 concurrent key-value pairs.
  - **Task 12: Multi-Variable State Machine:** Tracking 5 independent mutating variables simultaneously.
  - **Task 13: Multi-Hop Transitive Reasoning:** Relational inference ($A \to B, B \to C \implies A \to C$).
- **High-Capacity Sparse Delta-PKM (Task 14):**
  - **Task 14: 1,000 Facts in 1,048,576 Virtual Slots:** Evaluates multi-axis coordinate routing and Householder delta error erasure with an active 8 KB compute window.

### 2. Multi-Seed Rigorous Baselines (`bench_baselines.py`)
Compares DeltaGrid against parameter-matched standard recurrent and attention baselines:
- **Baselines:** Causal Transformer, LSTM, and GRU (automatically matched to DeltaGrid's exact parameter count).
- **Tasks Evaluated:**
  - `mqar`: Multi-Query Associative Recall (Zoology benchmark, Arora et al., 2023) across 4 to 48 key-value pairs.
  - `needle`: Random-position needle retention with distractor pairs (trained on $L=32/64$, tested up to $L=2048$).
  - `adding`: Continuous adding problem (trained on $T=50$, tested up to $T=800$).
  - `sine`: Continuous wave forecasting (in-distribution, out-of-distribution frequencies $0.25-0.35$, and noisy inputs).
  - `memory`: Physical peak Resident Set Size (RSS) deltas measured in isolated subprocesses up to 65,536 tokens.
  - `charlm`: Real-world byte-level language modeling on Python standard library source code (evaluated at context 256 and 1024).

---

## How to Run

### 1. Hardware Latency & Throughput Benchmark
Measures fused CUDA kernel execution time vs. standard PyTorch sequential recurrence:
```bash
python benchmarks/bench_cuda_speed.py
```
*Expected Output on NVIDIA RTX 3060:*
- PyTorch loop latency: `~55 ms`
- CUDA fused latency: `~0.21 ms`
- Speedup: `~250x – 260x FASTER`
- Single-step decode latency: `~14 µs` (Throughput: `~70,000 steps/sec`)

---

### 2. Unified Multi-Task Benchmark
Run all 14 benchmark tasks or target a specific sub-suite:
```bash
# Run all 14 tasks sequentially:
python benchmarks/bench_unified_suite.py --task all

# Run only the classic tasks (Tasks 1-6):
python benchmarks/bench_unified_suite.py --task classic

# Run only the associative reasoning tasks (Tasks 7-13):
python benchmarks/bench_unified_suite.py --task associative

# Run high-capacity Sparse Delta-PKM (Task 14):
python benchmarks/bench_unified_suite.py --task pkm

# Test arbitrary PKM grid geometries:
python benchmarks/bench_unified_suite.py --task pkm --grid-dim 64 64 --top-k 8
```

---

### 3. Random-Position & Depth-Wise Needle Extrapolation
Evaluates length extrapolation on needle retrieval when the needle is placed at arbitrary uniform positions or varying depths:
```bash
python benchmarks/test_random_needle.py
```
*Evaluations:*
- Uniform random position: $L = 64, 128, 256, 512, 1024, 2048$ tokens.
- Depth sweep at $L=1024$ and $L=2048$: depths from 5% to 95%.

---

### 4. Parameter-Matched Multi-Seed Baseline Comparison
Executes the comprehensive baseline suite comparing DeltaGrid against parameter-matched GRU, LSTM, and Causal Transformer:
```bash
# Full multi-seed run (MQAR, Needle, Adding, Sine, Memory, CharLM):
python benchmarks/bench_baselines.py --tasks mqar needle adding sine memory charlm --seeds 3

# Quick smoke-test run:
python benchmarks/bench_baselines.py --quick

# Render the 6-panel rigorous dashboard:
python benchmarks/plot_figures.py
```

### 5. Hardware SRAM-Constrained Recurrent Memory Benchmark (`bench_sram_constrained.py`)
Evaluates DeltaGrid across real-world on-chip GPU SRAM working window budgets:
- **4 KB, 8 KB, 16 KB, 32 KB, and 64 KB Active Working Windows**
- **Metrics Captured:**
  - Peak On-Chip SRAM / Register Usage (KB per head & layer)
  - DRAM Bus Traffic per Step (KB/step & MB/seq) comparing DeltaGrid's $0\text{ KB}$ (Dense) / $16\text{ KB}$ (PKM) against Transformer KV-cache explosion
  - Microsecond Step Latency ($\mu\text{s}$) & Token Throughput (tok/s)
  - First-Principles VLSI Energy per Token (Horowitz ISSCC Model: SRAM vs. DRAM vs. Compute)
  - Real Multi-Query Associative Recall (MQAR) Retrieval Fidelity across 32 to 1,000 distinct facts.

```bash
# Run the complete SRAM-constrained benchmark suite:
python benchmarks/bench_sram_constrained.py --num-facts 500 --runs 200
```

---

## Benchmark Outputs (`figures/`)

| File | Description |
| :--- | :--- |
| `figures/sram_constrained_benchmark.png` | 4-panel publication dashboard: Latency vs. SRAM Budget, DRAM Traffic Scaling, VLSI Energy Breakdown, and Real Fact Retention Accuracy. |
| `figures/sram_constrained_results.json` | Machine-readable metrics payload containing latency, DRAM traffic, energy, and accuracy sweeps. |
| `figures/rigorous_dashboard.png` | Master 6-panel empirical comparison against Transformers, LSTMs, and GRUs. |
| `figures/rigorous_results.json` | Structured JSON containing raw per-seed metrics and parameter counts. |
| `figures/unified_benchmark_dashboard.png` | 6-panel summary dashboard for the classic benchmark suite. |
| `figures/task1_recall_accuracy.png` | Accuracy curve for Associative Recall across noise tokens. |
| `figures/task2_copy_accuracy.png` | Accuracy curve for multi-token sequence reproduction. |
| `figures/task3_grammar_loss.png` | Loss and prediction curve for 2nd-order Markov grammar. |
| `figures/task4_adding_problem_mse.png` | Mean Squared Error curve for continuous float addition. |
| `figures/task5_sine_wave_result.png` | Trajectory forecast rollout vs. ground truth wave signal. |
| `figures/task6_memory_scaling.png` | VRAM scaling comparison (DeltaGrid $O(1)$ vs. Transformer $O(T^2)$). |
