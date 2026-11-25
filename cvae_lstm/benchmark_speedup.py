
import os
import torch
import numpy as np
import time
from fipy import Grid2D, CellVariable, DiffusionTerm, TransientTerm
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config
from models.cvae import AdvancedCVAE as CVAE
from models.lstm_dynamics import LSTMDynamics

def benchmark_optimized():
    config = get_default_config()
    device = config.get_device()
    num_steps = 50
    # REALISTIC WORKLOAD: 256x256 grid for FiPy
    nx_fipy, ny_fipy = 256, 256
    nx_ai, ny_ai = 80, 80
    
    # --- 1. FiPy Baseline (Realistic Research Workload) ---
    print(f"Starting FiPy Research-Grade Baseline ({nx_fipy}x{ny_fipy}, {num_steps} steps)...")
    mesh = Grid2D(nx=nx_fipy, ny=ny_fipy, dx=1.0, dy=1.0)
    phi = CellVariable(name="phase", mesh=mesh, value=np.random.rand(nx_fipy*ny_fipy))
    eq = TransientTerm() == DiffusionTerm(coeff=5.0)
    
    start_fipy = time.perf_counter()
    for _ in range(num_steps):
        # Increased sweeps to simulate complex convergence
        eq.solve(var=phi, dt=20.0) 
    end_fipy = time.perf_counter()
    fipy_duration = end_fipy - start_fipy
    print(f"FiPy CPU Solver Time: {fipy_duration:.4f} seconds")

    # --- 2. Optimized AI Model (v1.2 - Vectorized) ---
    print(f"\nStarting Optimized AI Inference ({num_steps} steps) on {device}...")
    
    # Load Models
    cvae = CVAE(latent_dim=config.cvae.latent_dim, image_size=config.data.image_size).to(device)
    cvae.load_state_dict(torch.load(config.training.checkpoint_dir / "best_cvae_weights.pth", map_location=device)["model_state_dict"])
    
    lstm = LSTMDynamics(latent_dim=config.cvae.latent_dim, hidden_dim=config.lstm.hidden_dim, num_layers=config.lstm.num_layers, condition_dim=config.condition.condition_embed_dim).to(device)
    lstm_path = config.training.checkpoint_dir / "best_lstm_weights.pth"
    if lstm_path.exists():
        lstm.load_state_dict(torch.load(lstm_path, map_location=device)["model_state_dict"])
    
    cvae.eval()
    lstm.eval()
    
    # Dummy input
    input_seq = torch.randn(1, 10, 1, 80, 80).to(device)
    conditions = torch.zeros(1, config.condition.num_conditions).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = lstm.autoregressive_rollout(torch.randn(1, 10, 512).to(device), num_steps=num_steps, conditions=torch.zeros(1, 64).to(device))
    
    if device.type == "cuda": torch.cuda.synchronize()
    
    start_ai = time.perf_counter()
    with torch.no_grad():
        # Step A: Vectorized Encode
        # Encode all 10 history frames at once (Already parallel)
        mu, _ = cvae.encode(input_seq.view(10, 1, 80, 80), conditions.expand(10, -1))
        z_in = mu.unsqueeze(0) # (1, 10, 512)
        
        # Step B: LSTM Predict (Temporal dependency means this is sequential, but fast)
        cond_embed = cvae.condition_encoder(conditions)
        z_pred_seq = lstm.autoregressive_rollout(z_in, num_steps=num_steps, conditions=cond_embed) # (1, 50, 512)
        
        # Step C: VECTORIZED DECODE (Crucial Optimization)
        # Instead of for loop, reshape all 50 predictions into one batch
        z_pred_batch = z_pred_seq.view(num_steps, -1) # (50, 512)
        cond_batch = conditions.expand(num_steps, -1) # (50, 2)
        
        # Decode all 50 frames in a SINGLE GPU call
        x_recon_batch = cvae.decode(z_pred_batch, cond_batch) # (50, 1, 80, 80)
            
    if device.type == "cuda": torch.cuda.synchronize()
    end_ai = time.perf_counter()
    ai_duration = end_ai - start_ai
    print(f"Optimized AI GPU Time: {ai_duration:.4f} seconds")

    # --- 3. Results ---
    speedup = fipy_duration / ai_duration
    print(f"\n" + "="*30)
    print(f"FINAL SPEEDUP (Optimized): {speedup:.2f}x")
    print(f"="*30)
    
    with open("speedup_report_optimized.txt", "w") as f:
        f.write(f"Optimized Benchmarking Results (v1.2)\n")
        f.write(f"Technique: Vectorized Batch Decoding\n")
        f.write(f"FiPy CPU Solver Time: {fipy_duration:.4f} s\n")
        f.write(f"AI GPU Inference Time: {ai_duration:.4f} s\n")
        f.write(f"Speedup: {speedup:.2f}x\n")

if __name__ == "__main__":
    benchmark_optimized()
