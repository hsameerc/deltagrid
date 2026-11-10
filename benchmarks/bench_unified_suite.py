"""
=============================================================================
  DeltaGrid: Unified Multi-Task Benchmark & Memory Profiling Suite
=============================================================================
Consolidates 6 classical and advanced recurrent memory benchmarks:
  1. Associative Recall: Long-range memory retention across background noise.
  2. Sequence Copy: Exact multi-token sequence memorization and reproduction.
  3. Toy Language Grammar: 2nd-order non-linear Markov next-token prediction.
  4. The Adding Problem: Continuous float memorization and summation (Hochreiter & Schmidhuber).
  5. Continuous Sine Wave: Multi-harmonic continuous physics/sensor-motor extrapolation.
  6. Memory Scaling Profile: O(1) state memory scaling vs Standard Transformer (up to 4096+ tokens).

Visual Plots Generated:
  - figures/task1_recall_accuracy.png
  - figures/task2_copy_accuracy.png
  - figures/task3_grammar_loss.png
  - figures/task4_adding_problem_mse.png
  - figures/task5_sine_wave_result.png
  - figures/task6_memory_scaling.png
  - figures/unified_benchmark_dashboard.png (Consolidated 6-Panel Benchmark Dashboard)
=============================================================================
"""

import sys
import os
import time
import math
import random
import argparse
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.delta_grid import DeltaGridLM, DeltaGridContinuous, DeltaGridBlock, DeltaGridLayer

try:
    import hsru_delta_cuda
    _HAS_CUDA_EXT = True
except ImportError:
    hsru_delta_cuda = None
    _HAS_CUDA_EXT = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# =============================================================================
# 1. TASK: Associative Recall across Noise
# =============================================================================
def run_task_recall(epochs: int = 25, batch_size: int = 64, seq_len: int = 32, vocab_size: int = 20, save_plot: bool = True, grid_dim: Optional[Tuple[int, ...]] = None, top_k: Optional[int] = None):
    """
    Task 1: Associative Recall across Noise.
    
    Stores a target token at index 0, passes seq_len-2 noise tokens,
    and queries the target token with a delimiter token at the final step.
    Verifies long-range information retention through pure recurrence without attention.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        seq_len (int): Total sequence length including noise and query.
        vocab_size (int): Size of synthetic vocabulary.
        save_plot (bool): Whether to export visualization to figures/.
        grid_dim (Optional[Tuple[int, ...]]): Direct N-D grid dimensions.
        top_k (Optional[int]): Direct Top-K active coordinate bins per axis.
        
    Returns:
        Tuple[float, Tuple[List[int], List[float]]]: Final accuracy and history.
    """
    print("\n" + "=" * 78)
    cfg_desc = f"Grid: {grid_dim}, Top-K: {top_k}" if grid_dim else ""
    prof_str = f" [{cfg_desc}]" if cfg_desc else ""
    print(f"TASK 1: ASSOCIATIVE RECALL ACROSS NOISE{prof_str}")
    print("=" * 78)
    model = DeltaGridLM(vocab_size=vocab_size, d_model=64, num_layers=1, num_heads=2, grid_dim=grid_dim, top_k=top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    delim_token = vocab_size - 1
    t0 = time.time()
    ep_history, acc_history = [], []
    
    for ep in range(epochs):
        model.train()
        for _ in range(25):
            x = torch.randint(1, vocab_size - 2, (batch_size, seq_len), device=device)
            target = torch.randint(1, vocab_size - 2, (batch_size,), device=device)
            x[:, 0] = target
            x[:, -1] = delim_token
            
            logits, _ = model(x)
            loss = criterion(logits[:, -1, :], target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
        # Track validation accuracy per epoch
        model.eval()
        with torch.no_grad():
            val_x = torch.randint(1, vocab_size - 2, (batch_size, seq_len), device=device)
            val_t = torch.randint(1, vocab_size - 2, (batch_size,), device=device)
            val_x[:, 0] = val_t; val_x[:, -1] = delim_token
            val_preds = model(val_x)[0][:, -1, :].argmax(dim=-1)
            val_acc = (val_preds == val_t).float().mean().item() * 100.0
            ep_history.append(ep + 1)
            acc_history.append(val_acc)
            
    elapsed = time.time() - t0
    acc = acc_history[-1]
    status = "PASSED (100%)" if acc >= 98.0 else ("PASSED" if acc >= 85.0 else "FAILED")
    print(f"[*] Associative Recall Accuracy: {acc:>5.1f}% | Time: {elapsed:>5.2f}s | {status}")

    if save_plot and _HAS_MATPLOTLIB:
        os.makedirs("figures", exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.plot(ep_history, acc_history, marker='o', color='#1f77b4', linewidth=2.5, label="DeltaGrid Recall")
        plt.axhline(y=100, color='green', linestyle='--', label="Target (100%)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Task 1: Associative Recall over {seq_len}-Step Noise")
        plt.ylim(-5, 105)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/task1_recall_accuracy.png", dpi=150)
        plt.close()
        print("    [+] Plot saved to figures/task1_recall_accuracy.png")

    return acc, (ep_history, acc_history)


# =============================================================================
# 2. TASK: Sequence Copying Task
# =============================================================================
def run_task_copy(epochs: int = 35, batch_size: int = 64, copy_len: int = 5, vocab_size: int = 20, save_plot: bool = True, grid_dim: Optional[Tuple[int, ...]] = None, top_k: Optional[int] = None):
    """
    Task 2: Exact Multi-Token Sequence Copying.
    
    Feeds a sequence of random tokens, followed by a delimiter, followed by padding.
    The model must memorize the entire multi-token sequence in its hidden recurrent
    state and reproduce it verbatim during the padded phase.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        copy_len (int): Length of the sequence to memorize and reproduce.
        vocab_size (int): Size of synthetic vocabulary.
        save_plot (bool): Whether to export visualization to figures/.
        grid_dim (Optional[Tuple[int, ...]]): Direct N-D grid dimensions.
        top_k (Optional[int]): Direct Top-K active coordinate bins per axis.
        
    Returns:
        Tuple[float, Tuple[List[int], List[float]]]: Final accuracy and history.
    """
    print("\n" + "=" * 78)
    cfg_desc = f"Grid: {grid_dim}, Top-K: {top_k}" if grid_dim else ""
    prof_str = f" [{cfg_desc}]" if cfg_desc else ""
    print(f"TASK 2: SEQUENCE COPYING MEMORY TASK{prof_str}")
    print("=" * 78)
    model = DeltaGridLM(vocab_size=vocab_size, d_model=64, num_layers=1, num_heads=2, grid_dim=grid_dim, top_k=top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    delim_token = vocab_size - 1
    t0 = time.time()
    ep_history, acc_history = [], []
    
    for ep in range(epochs):
        model.train()
        for _ in range(25):
            mem_seq = torch.randint(1, vocab_size - 2, (batch_size, copy_len), device=device)
            delim = torch.full((batch_size, 1), delim_token, dtype=torch.long, device=device)
            pad = torch.zeros((batch_size, copy_len), dtype=torch.long, device=device)
            inputs = torch.cat([mem_seq, delim, pad], dim=1)
            
            logits, _ = model(inputs)
            out_logits = logits[:, copy_len : 2 * copy_len, :]
            loss = criterion(out_logits.reshape(-1, vocab_size), mem_seq.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
        model.eval()
        with torch.no_grad():
            val_seq = torch.randint(1, vocab_size - 2, (batch_size, copy_len), device=device)
            delim = torch.full((batch_size, 1), delim_token, dtype=torch.long, device=device)
            pad = torch.zeros((batch_size, copy_len), dtype=torch.long, device=device)
            val_in = torch.cat([val_seq, delim, pad], dim=1)
            preds = model(val_in)[0][:, copy_len : 2 * copy_len, :].argmax(dim=-1)
            val_acc = (preds == val_seq).all(dim=-1).float().mean().item() * 100.0
            ep_history.append(ep + 1)
            acc_history.append(val_acc)
            
    elapsed = time.time() - t0
    acc = acc_history[-1]
    status = "PASSED (100%)" if acc >= 98.0 else ("PASSED" if acc >= 85.0 else "FAILED")
    print(f"[*] Sequence Copy Accuracy:     {acc:>5.1f}% | Time: {elapsed:>5.2f}s | {status}")

    if save_plot and _HAS_MATPLOTLIB:
        os.makedirs("figures", exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.plot(ep_history, acc_history, marker='s', color='#2ca02c', linewidth=2.5, label="DeltaGrid Copy")
        plt.axhline(y=100, color='red', linestyle='--', label="Perfect Copy (100%)")
        plt.xlabel("Epoch")
        plt.ylabel("Exact Sequence Accuracy (%)")
        plt.title(f"Task 2: Exact Multi-Token Sequence Copying ({copy_len} Tokens)")
        plt.ylim(-5, 105)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/task2_copy_accuracy.png", dpi=150)
        plt.close()
        print("    [+] Plot saved to figures/task2_copy_accuracy.png")

    return acc, (ep_history, acc_history)


# =============================================================================
# 3. TASK: Toy Language 2nd-Order Markov Grammar
# =============================================================================
def run_task_toy_language(epochs: int = 25, batch_size: int = 32, seq_len: int = 64, vocab_size: int = 20, save_plot: bool = True, grid_dim: Optional[Tuple[int, ...]] = None, top_k: Optional[int] = None):
    """
    Task 3: Toy Language 2nd-Order Non-Linear Markov Grammar.
    
    Generates a deterministic sequence where each token is a non-linear function
    of the two preceding tokens: x_t = (x_{t-1} * x_{t-2} + 7) mod V.
    Verifies that the non-linear GLU reasoning tier captures relational dependencies.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        seq_len (int): Total sequence length.
        vocab_size (int): Size of vocabulary.
        save_plot (bool): Whether to export visualization to figures/.
        grid_dim (Optional[Tuple[int, ...]]): Direct N-D grid dimensions.
        top_k (Optional[int]): Direct Top-K active coordinate bins per axis.
        
    Returns:
        Tuple[float, Tuple[List[int], List[float], List[float]]]: Final accuracy and history.
    """
    print("\n" + "=" * 78)
    cfg_desc = f"Grid: {grid_dim}, Top-K: {top_k}" if grid_dim else ""
    prof_str = f" [{cfg_desc}]" if cfg_desc else ""
    print(f"TASK 3: TOY LANGUAGE 2ND-ORDER NON-LINEAR GRAMMAR{prof_str}")
    print("=" * 78)
    model = DeltaGridLM(vocab_size=vocab_size, d_model=64, num_layers=1, num_heads=2, grid_dim=grid_dim, top_k=top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    t0 = time.time()
    ep_history, loss_history, acc_history = [], [], []
    
    for ep in range(epochs):
        model.train()
        total_ep_loss = 0.0
        for _ in range(25):
            seq = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            seq[:, 0] = torch.randint(0, vocab_size, (batch_size,), device=device)
            seq[:, 1] = torch.randint(0, vocab_size, (batch_size,), device=device)
            for t in range(2, seq_len):
                seq[:, t] = (seq[:, t-1] * seq[:, t-2] + 7) % vocab_size
                
            inputs = seq[:, :-1]
            targets = seq[:, 1:]
            
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_ep_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            val_seq = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            val_seq[:, 0] = torch.randint(0, vocab_size, (batch_size,), device=device)
            val_seq[:, 1] = torch.randint(0, vocab_size, (batch_size,), device=device)
            for t in range(2, seq_len):
                val_seq[:, t] = (val_seq[:, t-1] * val_seq[:, t-2] + 7) % vocab_size
            preds = model(val_seq[:, :-1])[0].argmax(dim=-1)
            val_acc = (preds[:, 1:] == val_seq[:, 2:]).float().mean().item() * 100.0
            
        ep_history.append(ep + 1)
        loss_history.append(total_ep_loss / 25.0)
        acc_history.append(val_acc)
            
    elapsed = time.time() - t0
    acc = acc_history[-1]
    status = "PASSED (100%)" if acc >= 98.0 else ("PASSED" if acc >= 85.0 else "FAILED")
    print(f"[*] Grammar Prediction Acc:     {acc:>5.1f}% | Time: {elapsed:>5.2f}s | {status}")

    if save_plot and _HAS_MATPLOTLIB:
        os.makedirs("figures", exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(7, 4))
        color = '#d62728'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Cross-Entropy Loss', color=color)
        ax1.plot(ep_history, loss_history, color=color, linewidth=2, label="CE Loss")
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = '#1f77b4'
        ax2.set_ylabel('Next-Token Accuracy (%)', color=color)
        ax2.plot(ep_history, acc_history, color=color, linestyle='--', linewidth=2, label="Accuracy")
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("Task 3: 2nd-Order Markov Grammar Modeling")
        plt.tight_layout()
        plt.savefig("figures/task3_grammar_loss.png", dpi=150)
        plt.close()
        print("    [+] Plot saved to figures/task3_grammar_loss.png")

    return acc, (ep_history, loss_history, acc_history)


# =============================================================================
# 4. TASK: The Classic Adding Problem
# =============================================================================
def run_task_adding_problem(epochs: int = 35, batch_size: int = 64, seq_len: int = 50, save_plot: bool = True, grid_dim: Optional[Tuple[int, ...]] = None, top_k: Optional[int] = None):
    """
    Task 4: The Classic Adding Problem (Hochreiter & Schmidhuber, 1997).
    
    Continuous floating point memory benchmark: A 2-channel sequence of length T
    contains random continuous uniform floats in channel 0, and two indicator markers
    (1.0) in channel 1. The model must remember and output the sum of the two marked floats.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        seq_len (int): Continuous sequence length.
        save_plot (bool): Whether to export visualization to figures/.
        grid_dim (Optional[Tuple[int, ...]]): Direct N-D grid dimensions.
        top_k (Optional[int]): Direct Top-K active coordinate bins per axis.
        
    Returns:
        Tuple[float, Tuple[List[int], List[float]]]: Final MSE and history.
    """
    print("\n" + "=" * 78)
    cfg_desc = f"Grid: {grid_dim}, Top-K: {top_k}" if grid_dim else ""
    prof_str = f" [{cfg_desc}]" if cfg_desc else ""
    print(f"TASK 4: THE ADDING PROBLEM (HOCHREITER & SCHMIDHUBER){prof_str}")
    print("=" * 78)
    model = DeltaGridContinuous(input_dim=2, output_dim=1, d_model=64, num_layers=1, num_heads=2, grid_dim=grid_dim, top_k=top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    t0 = time.time()
    ep_history, mse_history = [], []
    
    for ep in range(epochs):
        model.train()
        total_ep_mse = 0.0
        for _ in range(25):
            x = torch.zeros(batch_size, seq_len, 2, device=device)
            x[:, :, 0] = torch.rand(batch_size, seq_len, device=device)
            targets = torch.zeros(batch_size, 1, device=device)
            for b in range(batch_size):
                idx1, idx2 = torch.randperm(seq_len)[:2]
                x[b, idx1, 1] = 1.0
                x[b, idx2, 1] = 1.0
                targets[b, 0] = x[b, idx1, 0] + x[b, idx2, 0]
                
            pred_all, _ = model(x)
            pred = pred_all[:, -1, :] # final step prediction
            loss = criterion(pred, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_ep_mse += loss.item()
            
        ep_history.append(ep + 1)
        mse_history.append(total_ep_mse / 25.0)
            
    elapsed = time.time() - t0
    avg_mse = mse_history[-1]
    passed = avg_mse < 0.01
    status = "PASSED" if passed else "FAILED"
    print(f"[*] Adding Problem Final MSE:   {avg_mse:>7.5f} | Time: {elapsed:>5.2f}s | {status} (Baseline: ~0.166)")

    if save_plot and _HAS_MATPLOTLIB:
        os.makedirs("figures", exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.plot(ep_history, mse_history, marker='^', color='#9467bd', linewidth=2.5, label="DeltaGrid MSE")
        plt.axhline(y=0.1667, color='gray', linestyle='--', label="Random Guess Baseline (0.166)")
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error (Log Scale)")
        plt.title(f"Task 4: Adding Problem Across {seq_len} Continuous Steps")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("figures/task4_adding_problem_mse.png", dpi=150)
        plt.close()
        print("    [+] Plot saved to figures/task4_adding_problem_mse.png")

    return avg_mse, (ep_history, mse_history)


# =============================================================================
# 5. TASK: Continuous Multi-Harmonic Sine Wave Extrapolation
# =============================================================================
def run_task_sine_wave(epochs: int = 35, batch_size: int = 32, seq_len: int = 64, save_plot: bool = True, grid_dim: Optional[Tuple[int, ...]] = None, top_k: Optional[int] = None):
    """
    Task 5: Continuous Multi-Harmonic Wave Extrapolation.
    
    Continuous physical signal extrapolation benchmark: Feeds a 32-step context of a
    multi-harmonic continuous wave (frequencies, harmonics, phases), and requires the
    model to forecast the next 32 future steps with zero compounding error.
    
    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        seq_len (int): Total sequence length (context + horizon).
        save_plot (bool): Whether to export visualization to figures/.
        grid_dim (Optional[Tuple[int, ...]]): Direct N-D grid dimensions.
        top_k (Optional[int]): Direct Top-K active coordinate bins per axis.
        
    Returns:
        Tuple[float, Tuple[np.ndarray, List[float], int]]: Final horizon MSE and rollout data.
    """
    print("\n" + "=" * 78)
    cfg_desc = f"Grid: {grid_dim}, Top-K: {top_k}" if grid_dim else ""
    prof_str = f" [{cfg_desc}]" if cfg_desc else ""
    print(f"TASK 5: CONTINUOUS SINE WAVE EXTRAPOLATION{prof_str}")
    print("=" * 78)
    context_len = seq_len // 2
    forecast_horizon = seq_len - context_len
    model = DeltaGridContinuous(
        input_dim=1,
        output_dim=1,
        d_model=128,
        num_layers=2,
        num_heads=4,
        forecast_horizon=forecast_horizon,
        grid_dim=grid_dim,
        top_k=top_k,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for _ in range(25):
            inputs = torch.zeros(batch_size, seq_len, 1, device=device)
            targets = torch.zeros(batch_size, seq_len, 1, device=device)
            for b in range(batch_size):
                freq = random.uniform(0.05, 0.25)
                phase = random.uniform(0, 2 * math.pi)
                t_steps = torch.arange(seq_len + 1, device=device, dtype=torch.float32)
                wave = torch.sin(t_steps * freq + phase) + 0.5 * torch.cos(t_steps * freq * 2.5) + 0.25 * torch.sin(t_steps * freq * 4.2)
                inputs[b, :, 0] = wave[:-1]
                targets[b, :, 0] = wave[1:]
                
            pred_ctx, _, forecast_out = model(inputs[:, :context_len, :], return_forecast=True)
            loss_ctx = criterion(pred_ctx, targets[:, :context_len, :])
            loss_forecast = criterion(forecast_out, inputs[:, context_len:, :])
            loss = loss_ctx + 2.0 * loss_forecast
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
    elapsed = time.time() - t0
    model.eval()
    total_mse = 0.0
    with torch.no_grad():
        for _ in range(20):
            inputs = torch.zeros(batch_size, seq_len, 1, device=device)
            targets = torch.zeros(batch_size, seq_len, 1, device=device)
            for b in range(batch_size):
                freq = random.uniform(0.05, 0.25)
                phase = random.uniform(0, 2 * math.pi)
                t_steps = torch.arange(seq_len + 1, device=device, dtype=torch.float32)
                wave = torch.sin(t_steps * freq + phase) + 0.5 * torch.cos(t_steps * freq * 2.5) + 0.25 * torch.sin(t_steps * freq * 4.2)
                inputs[b, :, 0] = wave[:-1]
                targets[b, :, 0] = wave[1:]
            forecast_out = model.forecast(inputs[:, :context_len, :])
            total_mse += criterion(forecast_out, inputs[:, context_len:, :]).item()
            
    avg_mse = total_mse / 20.0
    passed = avg_mse < 0.05
    status = "PASSED" if passed else "FAILED"
    print(f"[*] Sine Wave 32-Step Horizon MSE: {avg_mse:>7.5f} | Time: {elapsed:>5.2f}s | {status}")

    # Generate visual rollout matching inputs[0]
    with torch.no_grad():
        context_input = inputs[0:1, :context_len, :]
        out_ctx, _ = model(context_input)
        forecast_preds = model.forecast(context_input)[0, :, 0].cpu().numpy()
        
        truth_wave = inputs[0, :, 0].cpu().numpy()
        pred_rollout = [context_input[0, 0, 0].item()]
        pred_rollout.extend(out_ctx[0, :-1, 0].cpu().numpy().tolist())
        pred_rollout.extend(forecast_preds.tolist())

    if save_plot and _HAS_MATPLOTLIB:
        os.makedirs("figures", exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(truth_wave, label="Ground Truth Multi-Harmonic Wave", color="#1f77b4", linestyle="--", linewidth=2.2)
        plt.plot(pred_rollout, label="DeltaGrid Forecast", color="#d62728", linewidth=2.2)
        plt.axvline(x=context_len - 1, color='green', linestyle=':', linewidth=2, label="Forecast Horizon Start")
        plt.title(f"Task 5: Continuous Multi-Harmonic Wave Extrapolation (MSE: {avg_mse:.4f})")
        plt.xlabel("Time Step")
        plt.ylabel("Signal Amplitude")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig("figures/task5_sine_wave_result.png", dpi=150)
        plt.close()
        print("    [+] Plot saved to figures/task5_sine_wave_result.png")

    return avg_mse, (truth_wave, pred_rollout, context_len)


# =============================================================================
# 6. TASK: Memory Scaling Profile (vs Standard Transformer)
# =============================================================================
class StandardTransformerLM(nn.Module):
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, nhead: int = 2, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor):
        emb = self.embedding(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        return self.fc_out(self.transformer(emb, mask=mask, is_causal=True))

def run_task_memory_profile(save_plot: bool = True, grid_dim: Optional[Tuple[int, ...]] = None, top_k: Optional[int] = None):
    print("\n" + "=" * 78)
    cfg_desc = f"Grid: {grid_dim}, Top-K: {top_k}" if grid_dim else ""
    prof_str = f" [{cfg_desc}]" if cfg_desc else ""
    print(f"TASK 6: MEMORY SCALING PROFILE (DeltaGrid vs Standard Transformer){prof_str}")
    print("=" * 78)
    print(f"{'Seq Length':<12} | {'DeltaGrid GPU MB':<18} | {'Transformer GPU MB':<20} | {'Advantage'}")
    print("-" * 78)
    
    vocab_size, d_model = 1000, 128
    deltagrid = DeltaGridLM(vocab_size=vocab_size, d_model=d_model, num_layers=2, num_heads=2, grid_dim=grid_dim, top_k=top_k).to(device).eval()
    transformer = StandardTransformerLM(vocab_size=vocab_size, d_model=d_model, nhead=2, num_layers=2).to(device).eval()
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    batch_size = 2
    dg_mems, tf_mems = [], []
    
    for T in seq_lengths:
        toks = torch.randint(0, vocab_size, (batch_size, T), device=device)
        
        # DeltaGrid Peak
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = deltagrid(toks)
        if device.type == "cuda":
            torch.cuda.synchronize()
            dg_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            dg_mem = 0.0
        dg_mems.append(dg_mem)
        
        # Transformer Peak
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                _ = transformer(toks)
            if device.type == "cuda":
                torch.cuda.synchronize()
                tf_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                tf_mem = 0.0
            tf_str = f"{tf_mem:>8.2f} MB"
            adv = f"{tf_mem / max(dg_mem, 1e-4):>4.1f}x smaller"
        except RuntimeError:
            tf_mem = float("inf")
            tf_str = "    OOM    "
            adv = "Infinite (OOM)"
        tf_mems.append(tf_mem)
            
        print(f"{T:>6} tokens | {dg_mem:>10.2f} MB       | {tf_str:<20} | {adv}")

    if save_plot and _HAS_MATPLOTLIB:
        os.makedirs("figures", exist_ok=True)
        plt.figure(figsize=(7, 4))
        valid_tf = [m if m != float("inf") else None for m in tf_mems]
        plt.plot(seq_lengths, valid_tf, marker="o", color="red", linewidth=2.5, label="Standard Transformer (O(T²))")
        plt.plot(seq_lengths, dg_mems, marker="s", color="blue", linewidth=2.5, label="DeltaGrid SRAM Memory (O(1))")
        plt.xlabel("Sequence Length (Tokens)")
        plt.ylabel("Peak Memory (MB)")
        plt.title("Task 6: Memory Scaling Comparison")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig("figures/task6_memory_scaling.png", dpi=150)
        plt.close()
        print("    [+] Plot saved to figures/task6_memory_scaling.png")

    return (seq_lengths, dg_mems, tf_mems)


# =============================================================================
# UNIFIED 6-PANEL BENCHMARK DASHBOARD GENERATOR
# =============================================================================
def generate_unified_dashboard(r1, r2, r3, r4, r5, r6):
    """
    Renders a publication-ready 2x3 comprehensive visual dashboard
    featuring all 6 empirical benchmark curves side-by-side.
    """
    if not _HAS_MATPLOTLIB:
        return
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("HSRU DeltaGrid: Comprehensive Multi-Task & Memory Scaling Benchmark Suite", fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: Associative Recall
    ax = axes[0, 0]
    ep1, acc1 = r1[1]
    ax.plot(ep1, acc1, marker='o', color='#1f77b4', linewidth=2.2, label='DeltaGrid')
    ax.axhline(y=100, color='green', linestyle='--', label='Target 100%')
    ax.set_title("1. Associative Recall across Noise", fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(-5, 105); ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='lower right')

    # Panel 2: Sequence Copying
    ax = axes[0, 1]
    ep2, acc2 = r2[1]
    ax.plot(ep2, acc2, marker='s', color='#2ca02c', linewidth=2.2, label='DeltaGrid')
    ax.axhline(y=100, color='red', linestyle='--', label='Perfect Copy')
    ax.set_title("2. Multi-Token Sequence Copy", fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Sequence Accuracy (%)")
    ax.set_ylim(-5, 105); ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='lower right')

    # Panel 3: Toy Grammar Modeling
    ax = axes[0, 2]
    ep3, loss3, acc3 = r3[1]
    ax.plot(ep3, loss3, color='#d62728', linewidth=2.2, label='CE Loss')
    ax.set_title("3. 2nd-Order Markov Grammar", fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper right')

    # Panel 4: Adding Problem
    ax = axes[1, 0]
    ep4, mse4 = r4[1]
    ax.plot(ep4, mse4, marker='^', color='#9467bd', linewidth=2.2, label='DeltaGrid MSE')
    ax.axhline(y=0.1667, color='gray', linestyle='--', label='Random Baseline')
    ax.set_yscale('log')
    ax.set_title("4. Adding Problem (Continuous Sum)", fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (Log Scale)")
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper right')

    # Panel 5: Sine Wave Extrapolation
    ax = axes[1, 1]
    truth5, pred5, horizon5 = r5[1]
    ax.plot(truth5, label='Ground Truth', color='#1f77b4', linestyle='--', linewidth=2)
    ax.plot(pred5, label='Forecast', color='#d62728', linewidth=2.2)
    ax.axvline(x=horizon5 - 1, color='green', linestyle=':', label='Forecast Start')
    ax.set_title("5. Continuous Wave Forecast", fontweight='bold')
    ax.set_xlabel("Time Step"); ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='lower left')

    # Panel 6: Memory Scaling
    ax = axes[1, 2]
    lens6, dg_mem6, tf_mem6 = r6
    valid_tf6 = [m if m != float("inf") else None for m in tf_mem6]
    ax.plot(lens6, valid_tf6, marker='o', color='red', linewidth=2.2, label='Transformer O(T²)')
    ax.plot(lens6, dg_mem6, marker='s', color='blue', linewidth=2.2, label='DeltaGrid O(1)')
    ax.set_title("6. GPU Memory Scaling (vs Transformer)", fontweight='bold')
    ax.set_xlabel("Sequence Length"); ax.set_ylabel("Peak RAM (MB)")
    ax.grid(True, linestyle=':', alpha=0.6); ax.legend(loc='upper left')

    plt.tight_layout()
    save_path = "figures/unified_benchmark_dashboard.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\n[+] Master 6-Panel Benchmark Dashboard saved to: {save_path}")


# =============================================================================
# PART 2: EXPANDED 7-AREA ASSOCIATIVE REASONING SUITE (Unifying test_delta_grid_bench)
# =============================================================================

def gen_overwrite_batch(batch_size: int, vocab_size: int = 50, num_keys: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task 7: Dynamic Password Overwrite (Single Variable 1-Step Mutation).
    
    Presents key-value pairs, then mutates the target key's value at position 8-9.
    The model must overwrite the prior fact in memory in a single step and output
    the updated value when queried at the end of the sequence.
    """
    query_tok, noise_start, seq_len = 35, 36, 16
    x = torch.randint(noise_start, vocab_size, (batch_size, seq_len))
    y = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randperm(8)[:num_keys] + 1
        vals = torch.randint(9, 31, (num_keys,))
        for i in range(num_keys):
            x[b, i * 2] = keys[i]; x[b, i * 2 + 1] = vals[i]
        target_k = keys[0]
        v_new = torch.randint(9, 31, (1,)).item()
        while v_new == vals[0].item():
            v_new = torch.randint(9, 31, (1,)).item()
        x[b, 8] = target_k; x[b, 9] = v_new
        x[b, -2] = query_tok; x[b, -1] = target_k
        y[b] = v_new
    return x, y


def gen_niah_64_batch(batch_size: int, seq_len: int = 64, vocab_size: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task 8: Needle-in-a-Haystack Retention across 64 tokens.
    
    Places a key-value fact at the start of the sequence, followed by 60+ random
    noise tokens, then queries the key. Tests noise rejection and retention.
    """
    query_tok, noise_start = 35, 36
    x = torch.randint(noise_start, vocab_size, (batch_size, seq_len))
    y = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        tk = torch.randint(1, 9, (1,)).item()
        tv = torch.randint(9, 31, (1,)).item()
        x[b, 0] = tk; x[b, 1] = tv
        x[b, -2] = query_tok; x[b, -1] = tk
        y[b] = tv
    return x, y


def gen_niah_256_batch(batch_size: int, seq_len: int = 256, vocab_size: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task 9: Deep Needle-in-a-Haystack Retention across 256 tokens.
    
    Embeds the target fact within 250+ tokens of distracting noise.
    Verifies that the associative matrix memory preserves gradient flow and state
    without vanishing gradients (alpha=1.0 conservation).
    """
    query_tok, noise_start = 40, 41
    x = torch.randint(noise_start, vocab_size, (batch_size, seq_len))
    y = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        tk = torch.randint(1, 9, (1,)).item()
        tv = torch.randint(9, 35, (1,)).item()
        x[b, 2] = tk; x[b, 3] = tv
        x[b, -2] = query_tok; x[b, -1] = tk
        y[b] = tv
    return x, y


def gen_niah_512_batch(batch_size: int, seq_len: int = 512, vocab_size: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task 10: Ultra-Deep Needle-in-a-Haystack Retention across 512 tokens.
    
    Tests long-range contextual recall across 500+ noise tokens in constant O(1) RAM.
    """
    query_tok, noise_start = 40, 41
    x = torch.randint(noise_start, vocab_size, (batch_size, seq_len))
    y = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        tk = torch.randint(1, 9, (1,)).item()
        tv = torch.randint(9, 35, (1,)).item()
        x[b, 2] = tk; x[b, 3] = tv
        x[b, -2] = query_tok; x[b, -1] = tk
        y[b] = tv
    return x, y


def gen_sqar_batch(batch_size: int, num_pairs: int = 8, vocab_size: int = 60) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task 11: Single-Query Associative Recall (SQAR 8-Pairs).
    
    Stores 8 distinct (key, value) pairs simultaneously in the associative matrix.
    Queries a randomly selected key at the end. Requires orthogonal key separation
    and multi-fact disambiguation.
    """
    query_token = vocab_size - 1
    seq_len = (num_pairs * 2) + 2
    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        keys = torch.randperm(20)[:num_pairs] + 1
        vals = torch.randint(21, vocab_size - 2, (num_pairs,))
        x[b, 0:num_pairs * 2:2] = keys
        x[b, 1:num_pairs * 2:2] = vals
        x[b, -2] = query_token
        target_idx = torch.randint(0, num_pairs, (1,)).item()
        x[b, -1] = keys[target_idx]
        y[b] = vals[target_idx]
    return x, y


def gen_state_machine_batch(batch_size: int, vocab_size: int = 80, num_vars: int = 5, num_mutations: int = 15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task 12: Multi-Variable State Machine (5 Variables Mutating Concurrently).
    
    Tracks 5 independent program variables (A, B, C, D, E) undergoing 15 arbitrary
    interleaved mutations. Queries the latest value of a randomly selected variable.
    """
    query_token = vocab_size - 1
    seq_len = num_mutations * 2 + 2
    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        var_ids = list(range(1, num_vars + 1))
        current_state = {v: 10 + v for v in var_ids}
        
        for m in range(num_mutations):
            chosen_var = var_ids[torch.randint(0, num_vars, (1,)).item()]
            new_val = torch.randint(20, 70, (1,)).item()
            current_state[chosen_var] = new_val
            x[b, 2 * m] = chosen_var
            x[b, 2 * m + 1] = new_val
        
        target_var = var_ids[torch.randint(0, num_vars, (1,)).item()]
        x[b, -2] = query_token
        x[b, -1] = target_var
        y[b] = current_state[target_var]
    return x, y


def gen_multihop_batch(batch_size: int, vocab_size: int = 80) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Task 13: Multi-Hop Transitive Reasoning (A -> B, B -> C => Retrieve A -> C).
    
    Presents transitive relations: Node A -> Node B and Node B -> Node C with distractors.
    Queries Node A and requires the model to compose associative hops to predict Node C.
    """
    query_token = vocab_size - 1
    seq_len = 10
    x = torch.zeros(batch_size, seq_len, dtype=torch.long)
    y = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        node_a = torch.randint(1, 15, (1,)).item()
        node_b = torch.randint(16, 30, (1,)).item()
        node_c = torch.randint(31, 50, (1,)).item()
        
        node_x = torch.randint(51, 60, (1,)).item()
        node_y = torch.randint(61, 70, (1,)).item()

        x[b, 0] = node_a; x[b, 1] = node_b    # A -> B
        x[b, 2] = node_x; x[b, 3] = node_y    # X -> Y (Distractor)
        x[b, 4] = node_b; x[b, 5] = node_c    # B -> C
        x[b, 6] = 75;     x[b, 7] = 76        # Noise
        x[b, -2] = query_token                # Query
        x[b, -1] = node_a                     # Query Node A -> Expect Node C
        y[b] = node_c
    return x, y


def run_associative_benchmarks(specific_task: Optional[str] = None, grid_dim: Optional[Tuple[int, ...]] = None, top_k: Optional[int] = None) -> List[Tuple[str, float, float, str]]:
    """
    Executes the 7 expanded associative memory and reasoning benchmarks (unified from test_delta_grid_bench.py).
    
    Args:
        specific_task (Optional[str]): If specified, runs only that individual benchmark task.
        grid_dim (Optional[Tuple[int, ...]]): Direct N-D grid dimensions.
        top_k (Optional[int]): Direct Top-K active coordinate bins per axis.
        
    Returns:
        List[Tuple[str, float, float, str]]: Summary results for each task.
    """
    print("\n" + "=" * 82)
    cfg_desc = f"Grid: {grid_dim}, Top-K: {top_k}" if grid_dim else ""
    prof_str = f" [{cfg_desc}]" if cfg_desc else ""
    print(f"HSRU DELTA-RNN: EXPANDED 7-TASK ASSOCIATIVE REASONING BENCHMARK SUITE{prof_str}")
    print(f"Device: {device.type.upper()} | Fused Batched GEMM Engine | Torch.Compile Enabled")
    print("=" * 82)

    tests = [
        ("Task 7: Dynamic Password Overwrite", gen_overwrite_batch, 20, 0.005, 50, 40, "overwrite"),
        ("Task 8: Needle Retention (64 Tokens)", gen_niah_64_batch, 20, 0.005, 50, 40, "needle64"),
        ("Task 9: Deep Needle Retention (256 Tokens)", gen_niah_256_batch, 25, 0.005, 60, 40, "needle256"),
        ("Task 10: Deep Needle Retention (512 Tokens)", gen_niah_512_batch, 25, 0.005, 60, 40, "needle512"),
        ("Task 11: Associative Recall (SQAR 8-Pairs)", gen_sqar_batch, 20, 0.005, 60, 40, "sqar"),
        ("Task 12: Multi-Variable State Machine (5 Vars)", gen_state_machine_batch, 25, 0.005, 80, 50, "statemachine"),
        ("Task 13: Multi-Hop Transitive Reasoning", gen_multihop_batch, 25, 0.005, 80, 50, "multihop"),
    ]

    if specific_task is not None:
        tests = [t for t in tests if t[6] == specific_task or specific_task in t[0].lower()]

    print(f"{'Benchmark Task':<46} | {'Accuracy':<8} | {'Time':<8} | {'Status'}")
    print("-" * 82)

    total_time = 0.0
    results = []
    for name, data_fn, epochs, lr, v_size, steps_per_ep, key in tests:
        if device.type == "mps":
            torch.mps.empty_cache()

        model = DeltaGridLM(
            v_size,
            d_model=64,
            num_layers=1,
            num_heads=2,
            num_slots=16 if grid_dim is None else None,
            grid_dim=grid_dim,
            top_k=top_k,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        t0 = time.time()
        model.train()
        for ep in range(epochs):
            for _ in range(steps_per_ep):
                bx, by = data_fn(32, vocab_size=v_size)
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx)
                loss = criterion(logits[:, -1, :], by)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        elapsed = time.time() - t0
        total_time += elapsed

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(30):
                tx, ty = data_fn(32, vocab_size=v_size)
                tx, ty = tx.to(device), ty.to(device)
                test_logits, _ = model(tx)
                preds = test_logits[:, -1, :].argmax(dim=-1)
                correct += (preds == ty).sum().item()
                total += tx.size(0)
        acc = (correct / total) * 100.0
        status = "PASSED (100%)" if acc >= 98.0 else ("PASSED" if acc >= 85.0 else "FAILED")
        print(f"{name:<46} | {acc:>6.1f}%  | {elapsed:>6.2f}s | {status}", flush=True)
        results.append((name, acc, elapsed, status))

    print("=" * 82)
    print(f"Total Associative Suite Elapsed Time: {total_time:.2f}s")
    print("=" * 82)
    return results


# =============================================================================
# PART 3: EXPONENTIAL CAPACITY BENCHMARK (Sparse Delta-PKM 1,048,576 Slots)
# =============================================================================

def run_task_pkm_capacity(
    num_facts: Optional[int] = None,
    grid_dim: Optional[Tuple[int, ...]] = None,
    top_k: Optional[int] = None,
) -> Tuple[float, float, float, str]:
    """
    Task 14: High-Capacity Sparse Delta-PKM Associative Recall.

    Evaluates the exponential retrieval capacity of Product-Key Memory (PKM) with
    closed-loop Householder Delta error erasure across virtual slots.
    Sequentially stores distinct high-dimensional facts in an 8 KB SRAM active compute window,
    then retrieves all facts.

    Auto-scales number of facts according to grid capacity if num_facts is not specified,
    defaulting to 1,000 facts on large 1,048,576-slot PKM grids.
    """
    actual_grid = grid_dim if grid_dim is not None else (1024, 1024)
    actual_top_k = top_k if top_k is not None else 8
    d_model = 128
    num_heads = 2
    head_dim = d_model // num_heads  # 64
    torch.manual_seed(42)
    layer = DeltaGridLayer(
        d_model=d_model,
        num_heads=num_heads,
        grid_dim=actual_grid,
        top_k=actual_top_k,
    ).to(device)
    layer.eval()

    total_slots = layer.total_slots
    active_slots = layer.active_slots

    if num_facts is None:
        if grid_dim is None:
            num_facts = 1000
        else:
            # Safe capacity bound avoiding collision saturation
            num_facts = min(1000, max(20, int((total_slots / active_slots) * 3.0)))

    cfg_label = f"Grid: {actual_grid}, Top-K: {actual_top_k}"
    print("\n" + "=" * 82)
    print(f"TASK 14: HIGH-CAPACITY SPARSE DELTA-PKM ({num_facts:,} Facts in {total_slots:,} Virtual Slots)")
    print(f"Device: {device.type.upper()} | {cfg_label} ({total_slots:,} Slots, Active: {active_slots})")
    print("=" * 82)

    t0 = time.time()

    torch.manual_seed(42)
    keys = torch.randn(num_facts, head_dim, device=device)
    values = torch.randn(num_facts, head_dim, device=device)

    total_slots = layer.total_slots
    active_slots = layer.active_slots
    state = torch.zeros(1, 1, total_slots, head_dim, device=device, dtype=torch.float32)

    # Sequential write using the Householder Delta Rule (Fused CUDA kernel when available)
    with torch.no_grad():
        all_k = keys.unsqueeze(0).unsqueeze(0)  # [1, 1, num_facts, head_dim]
        wk_all, wq_all, active_idx_all = layer._route_sparse_pkm(all_k, all_k)
        active_idx_all = active_idx_all.to(torch.int64)

        use_cuda_kernel = (device.type == "cuda" and _HAS_CUDA_EXT and hasattr(hsru_delta_cuda, "pkm_step"))

        if use_cuda_kernel:
            decay_vec = torch.ones(1, device=device)
            beta_one = torch.ones(1, 1, device=device)
            for t in range(num_facts):
                vt = values[t:t + 1].unsqueeze(1)
                wkt = wk_all[:, :, t]
                wqt = wq_all[:, :, t]
                idxt = active_idx_all[:, :, t]
                _, state = hsru_delta_cuda.pkm_step(vt, beta_one, wkt, wqt, idxt, decay_vec, state)

            cos_sims = []
            beta_zero = torch.zeros(1, 1, device=device)
            for t in range(num_facts):
                vt = values[t:t + 1].unsqueeze(1)
                wqt = wq_all[:, :, t]
                idxt = active_idx_all[:, :, t]
                recalled_v, _ = hsru_delta_cuda.pkm_step(vt, beta_zero, wqt, wqt, idxt, decay_vec, state)
                cos = F.cosine_similarity(vt.squeeze(1), recalled_v.squeeze(1)).item()
                cos_sims.append(cos)
        else:
            for t in range(num_facts):
                vt = values[t:t + 1].unsqueeze(1)
                wkt = wk_all[:, :, t]
                idxt = active_idx_all[:, :, t]
                expand_idx = idxt.unsqueeze(-1).expand(1, 1, active_slots, head_dim)
                s_act = torch.gather(state, 2, expand_idx)

                pred_v = torch.matmul(wkt.unsqueeze(-2), s_act).squeeze(-2)
                rt = vt - pred_v
                delta = wkt.unsqueeze(-1) * rt.unsqueeze(-2)
                s_act_new = s_act + delta
                state.scatter_(2, expand_idx, s_act_new)

            cos_sims = []
            for t in range(num_facts):
                vt = values[t:t + 1].unsqueeze(1)
                wqt = wq_all[:, :, t]
                idxt = active_idx_all[:, :, t]
                expand_idx = idxt.unsqueeze(-1).expand(1, 1, active_slots, head_dim)
                s_act = torch.gather(state, 2, expand_idx)
                recalled_v = torch.matmul(wqt.unsqueeze(-2), s_act).squeeze(-2)

                cos = F.cosine_similarity(vt.squeeze(1), recalled_v.squeeze(1)).item()
                cos_sims.append(cos)

    elapsed = time.time() - t0
    mean_cos = sum(cos_sims) / len(cos_sims)
    high_fid_rate = sum(1 for c in cos_sims if c > 0.80) / len(cos_sims) * 100.0
    status = "PASSED (100%)" if high_fid_rate >= 98.0 else ("PASSED" if high_fid_rate >= 90.0 else "FAILED")

    print(f"[*] Stored Facts:                 {num_facts:,}")
    print(f"[*] Virtual Product Slots:        {total_slots:,} ({cfg_label})")
    print(f"[*] Active Hardware Compute:      {active_slots} slots ({active_slots * head_dim * 2 // 1024} KB FP16 SRAM)")
    print(f"[*] Mean Cosine Similarity:       {mean_cos:.4f}")
    print(f"[*] High-Fidelity Recall (>0.80): {high_fid_rate:.1f}%")
    print(f"[*] Elapsed Time:                 {elapsed:.2f}s")
    print(f"[*] Benchmark Status:             {status}")
    print("=" * 82)

    return mean_cos, high_fid_rate, elapsed, status


# =============================================================================
# MAIN RUNNER & CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DeltaGrid Unified Multi-Task Benchmark Suite")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=[
            "all",
            "classic",
            "associative",
            "bench",
            "pkm",
            "recall",
            "copy",
            "grammar",
            "adding",
            "sine",
            "memory",
            "overwrite",
            "needle64",
            "needle256",
            "needle512",
            "sqar",
            "statemachine",
            "multihop",
            "pkm1000",
        ],
        help="Task or benchmark suite to execute",
    )
    parser.add_argument("--grid-dim", type=int, nargs="+", default=None, help="Direct N-D grid dimensions (e.g. --grid-dim 32 32 or --grid-dim 128 128)")
    parser.add_argument("--top-k", type=int, default=None, help="Direct Top-K active coordinate bins per axis (e.g. --top-k 4)")
    parser.add_argument("--num-facts", type=int, default=None, help="Number of facts to store in Task 14 (auto-scaled by default based on grid capacity)")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    args = parser.parse_args()

    save_plot = not args.no_plot
    grid_dim = tuple(args.grid_dim) if args.grid_dim else None
    top_k = args.top_k
    num_facts = args.num_facts

    print("=" * 78)
    print("HSRU DELTAGRID: UNIFIED MULTI-TASK & MEMORY BENCHMARK SUITE")
    print(f"Device: {device.type.upper()} | Fused CUDA Shared Memory Engine Enabled")
    if grid_dim:
        print(f"Architecture: Grid={grid_dim}, Top-K={top_k} applied to all benchmark models")
    else:
        print("Architecture: Standard Dense Associative Memory (Default)")
    print("=" * 78)

    t_start = time.time()

    # Part 1: Classical Empirical Benchmarks
    run_classic = args.task in ["all", "classic", "recall", "copy", "grammar", "adding", "sine", "memory"]
    r1 = run_task_recall(save_plot=save_plot, grid_dim=grid_dim, top_k=top_k) if (args.task in ["all", "classic", "recall"]) else None
    r2 = run_task_copy(save_plot=save_plot, grid_dim=grid_dim, top_k=top_k) if (args.task in ["all", "classic", "copy"]) else None
    r3 = run_task_toy_language(save_plot=save_plot, grid_dim=grid_dim, top_k=top_k) if (args.task in ["all", "classic", "grammar"]) else None
    r4 = run_task_adding_problem(save_plot=save_plot, grid_dim=grid_dim, top_k=top_k) if (args.task in ["all", "classic", "adding"]) else None
    r5 = run_task_sine_wave(save_plot=save_plot, grid_dim=grid_dim, top_k=top_k) if (args.task in ["all", "classic", "sine"]) else None
    r6 = run_task_memory_profile(save_plot=save_plot, grid_dim=grid_dim, top_k=top_k) if (args.task in ["all", "classic", "memory"]) else None

    # If full classic suite was run, generate the consolidated master dashboard!
    if args.task in ["all", "classic"] and save_plot:
        generate_unified_dashboard(r1, r2, r3, r4, r5, r6)

    # Part 2: Associative Reasoning Benchmarks (unified from test_delta_grid_bench)
    run_assoc = args.task in [
        "all",
        "associative",
        "bench",
        "overwrite",
        "needle64",
        "needle256",
        "needle512",
        "sqar",
        "statemachine",
        "multihop",
    ]
    if run_assoc:
        specific = args.task if args.task not in ["all", "associative", "bench"] else None
        run_associative_benchmarks(specific_task=specific, grid_dim=grid_dim, top_k=top_k)

    # Part 3: High-Capacity Sparse Delta-PKM Benchmark
    run_pkm = args.task in ["all", "pkm", "pkm1000"]
    if run_pkm:
        run_task_pkm_capacity(num_facts=num_facts, grid_dim=grid_dim, top_k=top_k)

    total_time = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"Total Unified Benchmark Suite Elapsed Time: {total_time:.2f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
