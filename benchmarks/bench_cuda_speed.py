"""
=============================================================================
  DeltaGrid: CUDA Shared Memory Kernel Speed & Latency Benchmark
=============================================================================
Measures raw GPU execution speed and hardware latency:
  1. Forward Sequence Latency: Fused CUDA Shared Memory kernel vs. PyTorch loop.
  2. Speedup Factor: Hardware acceleration ratio on NVIDIA Tensor Cores.
  3. Single-Step Latency: O(1) step execution time in microseconds (us) for
     sub-millisecond robotics control and ultra-low-latency token decoding.
  4. Step Throughput: Generation capacity in steps/second per GPU.

Usage:
  python tests/test_benchmark_cuda_speed.py
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn.functional as F

# Safe import for CUDA kernel extension
try:
    import hsru_delta_cuda
    _HAS_CUDA_EXT = True
except ImportError:
    hsru_delta_cuda = None
    _HAS_CUDA_EXT = False


def benchmark_delta_grid():
    """
    Executes microsecond-accurate CUDA vs PyTorch speed benchmarking.
    Runs warmup passes to prime GPU clocks, followed by synchronized timing loops.
    """
    if not torch.cuda.is_available() or not _HAS_CUDA_EXT:
        print("[!] CUDA or hsru_delta_cuda extension is not available. Skipping GPU speed benchmark.")
        return

    device = "cuda"
    B = 8        # Batch size
    H = 4        # Number of associative heads
    Dh = 64      # Hidden dimension per head
    Slots = 16   # Associative memory slots
    T = 256      # Sequence context length
    num_runs = 50

    print("=" * 78)
    print("BENCHMARK: DeltaGrid CUDA Fused Kernel vs PyTorch Recurrent Loop")
    print(f"Configuration: Batch={B}, Heads={H}, Slots={Slots}, HeadDim={Dh}, SeqLen={T}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 78)

    # Initialize input tensors in CUDA DRAM
    q = torch.randn(B, H, T, Dh, device=device)
    k = F.normalize(torch.randn(B, H, T, Dh, device=device), p=2, dim=-1)
    v = torch.randn(B, H, T, Dh, device=device)
    beta = torch.sigmoid(torch.randn(B, H, T, device=device))
    wk = F.softmax(torch.randn(B, H, T, Slots, device=device), dim=-1)
    wq = F.softmax(torch.randn(B, H, T, Slots, device=device), dim=-1)
    decay = torch.full((H,), 0.999, device=device)
    state_init = torch.zeros(B, H, Slots, Dh, device=device)

    # Prime GPU clocks and warm up instruction caches
    for _ in range(5):
        _ = hsru_delta_cuda.forward(q, k, v, beta, wk, wq, decay, state_init)
    torch.cuda.synchronize()

    # -------------------------------------------------------------------------
    # 1. Benchmark Fused Shared-Memory CUDA Kernel
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    for _ in range(num_runs):
        out_cuda, s_final_cuda, _, _ = hsru_delta_cuda.forward(q, k, v, beta, wk, wq, decay, state_init)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - t0) / num_runs * 1000.0  # ms per sequence

    # -------------------------------------------------------------------------
    # 2. Benchmark PyTorch Recurrent Loop (Reference Baseline)
    # -------------------------------------------------------------------------
    decay_tensor = decay.view(1, H, 1, 1)
    # PyTorch warmup
    for _ in range(2):
        state = state_init.clone()
        for t in range(T):
            vt = v[:, :, t]
            bt = beta[:, :, t, None]
            wkt = wk[:, :, t]
            wqt = wq[:, :, t]
            yt = torch.matmul(wqt.unsqueeze(2), state).squeeze(2)
            pred_v = torch.matmul(wkt.unsqueeze(2), state).squeeze(2)
            rt = vt - pred_v
            rate = bt * wkt
            delta = rate.unsqueeze(-1) * rt.unsqueeze(2)
            state = decay_tensor * state + delta
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_runs):
        state = state_init.clone()
        for t in range(T):
            vt = v[:, :, t]
            bt = beta[:, :, t, None]
            wkt = wk[:, :, t]
            wqt = wq[:, :, t]
            yt = torch.matmul(wqt.unsqueeze(2), state).squeeze(2)
            pred_v = torch.matmul(wkt.unsqueeze(2), state).squeeze(2)
            rt = vt - pred_v
            rate = bt * wkt
            delta = rate.unsqueeze(-1) * rt.unsqueeze(2)
            state = decay_tensor * state + delta
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / num_runs * 1000.0  # ms per sequence

    speedup = pytorch_time / cuda_time
    print(f"[*] PyTorch Loop Latency:    {pytorch_time:.2f} ms")
    print(f"[*] CUDA Fused Latency:      {cuda_time:.2f} ms")
    print(f"[*] SPEEDUP:                 {speedup:.1f}x FASTER!")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 3. Benchmark Constant O(1) Single-Step Inference Latency
    # -------------------------------------------------------------------------
    qt = q[:, :, 0]
    kt = k[:, :, 0]
    vt = v[:, :, 0]
    bt = beta[:, :, 0]
    wkt = wk[:, :, 0]
    wqt = wq[:, :, 0]
    s_curr = state_init.clone()

    # Warmup single-step invocation
    for _ in range(50):
        ot, s_curr = hsru_delta_cuda.step(qt, kt, vt, bt, wkt, wqt, decay, s_curr)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    runs_step = 1000
    for _ in range(runs_step):
        ot, s_curr = hsru_delta_cuda.step(qt, kt, vt, bt, wkt, wqt, decay, s_curr)
    torch.cuda.synchronize()
    step_time_us = (time.perf_counter() - t0) / runs_step * 1e6  # microseconds

    print(f"[*] O(1) Single-Step Latency: {step_time_us:.2f} us")
    print(f"[*] Max Step Throughput:      {1e6 / step_time_us:,.0f} steps/second")
    print("=" * 78)


if __name__ == "__main__":
    benchmark_delta_grid()
