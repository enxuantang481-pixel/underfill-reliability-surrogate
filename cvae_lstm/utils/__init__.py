"""
Utilities module for CVAE-LSTM Microstructure Evolution Surrogate Model.

This module exports:
- Metrics: MSE, SSIM, PSNR, gradient metrics
- Visualization: Plotting functions for training and results
- KL Annealing: Schedulers for KL divergence weight
- Checkpointing: Model save/load utilities

Author: AI-Generated for Advanced Packaging Reliability Assessment
"""

from .metrics import (
    compute_mse,
    compute_ssim,
    compute_psnr,
    compute_gradient_magnitude,
    compute_phase_boundary_metrics,
    evaluate_reconstruction_quality,
    evaluate_fidelity_stub,
    benchmark_inference_time,
)
from .visualization import (
    plot_training_curves,
    plot_reconstruction_comparison,
    plot_latent_space,
    plot_sequence_evolution,
    create_evolution_animation,
    save_prediction_grid,
)

__all__ = [
    # Metrics
    "compute_mse",
    "compute_ssim",
    "compute_psnr",
    "compute_gradient_magnitude",
    "compute_phase_boundary_metrics",
    "evaluate_reconstruction_quality",
    "evaluate_fidelity_stub",
    "benchmark_inference_time",
    # Visualization
    "plot_training_curves",
    "plot_reconstruction_comparison",
    "plot_latent_space",
    "plot_sequence_evolution",
    "create_evolution_animation",
    "save_prediction_grid",
]
