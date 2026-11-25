"""
Inference Pipeline for CVAE-LSTM Microstructure Evolution Surrogate Model.

This script provides end-to-end inference capabilities:
- Load trained CVAE and LSTM models
- Autoregressive multi-step prediction
- De-normalization to physical values
- Evaluation against proxy ground truth
- Visualization and saving results

Key Features:
- GPU-accelerated inference with proper synchronization
- Speedup benchmarking (vs FiPy placeholder)
- Conservation property checks (grain boundary integrity)
- Batch and single-sample inference

Usage:
    python inference.py --input path/to/sequence.npy --steps 50

Author: AI-Generated for Advanced Packaging Reliability Assessment
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import time
import json

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_default_config
from dataset import PhaseFieldDataset, SyntheticPhaseFieldDataset
from models.cvae import AdvancedCVAE as CVAE, fold_time_dimension, unfold_time_dimension
from models.lstm_dynamics import LSTMDynamics
from utils.metrics import (
    evaluate_reconstruction_quality,
    evaluate_fidelity_stub,
    benchmark_inference_time,
    compute_total_variation,
)
from utils.visualization import (
    plot_sequence_evolution,
    create_evolution_animation,
    save_prediction_grid,
)


class MicrostructurePredictor:
    """
    End-to-end predictor for microstructure evolution.

    Combines trained CVAE (spatial compression) and LSTM (temporal dynamics)
    for autoregressive prediction of future microstructures.
    """

    def __init__(
        self,
        cvae_checkpoint: Path,
        lstm_checkpoint: Path,
        config: Config,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize predictor with trained models.

        Args:
            cvae_checkpoint: Path to CVAE checkpoint
            lstm_checkpoint: Path to LSTM checkpoint
            config: Configuration object
            device: Device for inference
        """
        self.config = config
        self.device = device or config.get_device()

        print(f"Initializing predictor on {self.device}")

        # Load CVAE
        self.cvae = self._load_cvae(cvae_checkpoint)
        self.cvae.eval()

        # Load LSTM
        self.lstm = self._load_lstm(lstm_checkpoint)
        self.lstm.eval()

        # Normalization parameters (will be set from dataset)
        self.data_min = 0.0
        self.data_max = 1.0

        print("Predictor initialized successfully")

    def _load_cvae(self, checkpoint_path: Path) -> CVAE:
        """Load CVAE model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get architecture config from checkpoint or use default
        ckpt_config = checkpoint.get("config", {})

        model = CVAE(
            in_channels=self.config.data.num_channels,
            latent_dim=ckpt_config.get("latent_dim", self.config.cvae.latent_dim),
            encoder_channels=ckpt_config.get(
                "encoder_channels", self.config.cvae.encoder_channels
            ),
            decoder_channels=ckpt_config.get(
                "decoder_channels", self.config.cvae.decoder_channels
            ),
            image_size=ckpt_config.get("image_size", self.config.data.image_size),
            num_conditions=ckpt_config.get(
                "num_conditions", self.config.condition.num_conditions
            ),
            condition_embed_dim=ckpt_config.get(
                "condition_embed_dim", self.config.condition.condition_embed_dim
            ),
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded CVAE from {checkpoint_path}")

        return model

    def _load_lstm(self, checkpoint_path: Path) -> LSTMDynamics:
        """Load LSTM model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        ckpt_config = checkpoint.get("config", {})

        model = LSTMDynamics(
            latent_dim=ckpt_config.get("latent_dim", self.config.cvae.latent_dim),
            hidden_dim=ckpt_config.get("hidden_dim", self.config.lstm.hidden_dim),
            num_layers=ckpt_config.get("num_layers", self.config.lstm.num_layers),
            condition_dim=ckpt_config.get(
                "condition_dim", self.config.condition.condition_embed_dim
            ),
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded LSTM from {checkpoint_path}")

        return model

    def set_normalization(self, data_min: float, data_max: float) -> None:
        """Set normalization parameters for de-normalization."""
        self.data_min = data_min
        self.data_max = data_max

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to [0, 1]."""
        return (data - self.data_min) / (self.data_max - self.data_min + 1e-8)

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """De-normalize data back to original scale."""
        return data * (self.data_max - self.data_min) + self.data_min

    @torch.no_grad()
    def predict(
        self,
        input_sequence: torch.Tensor,
        conditions: torch.Tensor,
        num_steps: int,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict future microstructure evolution.

        Args:
            input_sequence: Historical frames (B, T, C, H, W) - normalized
            conditions: Condition vectors (B, num_conditions)
            num_steps: Number of future steps to predict
            return_latent: Also return latent trajectory

        Returns:
            Predicted sequence (B, num_steps, C, H, W) - normalized
            Optionally: latent trajectory (B, num_steps, latent_dim)
        """
        self.cvae.eval()
        self.lstm.eval()

        B, T, C, H, W = input_sequence.shape

        # Move to device
        input_sequence = input_sequence.to(self.device)
        conditions = conditions.to(self.device)

        # === Encode input sequence to latent space ===
        x_flat = input_sequence.view(B * T, C, H, W)
        cond_flat = conditions.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        mu, logvar = self.cvae.encode(x_flat, cond_flat)
        z_input = mu  # Use mean for deterministic inference
        z_input_seq = z_input.view(B, T, -1)  # (B, T, latent_dim)

        # === Get condition embedding for LSTM ===
        cond_embed = self.cvae.condition_encoder(conditions)  # (B, cond_embed_dim)

        # === Autoregressive rollout in latent space ===
        z_pred_seq = self.lstm.autoregressive_rollout(
            z_input_seq, num_steps=num_steps, conditions=cond_embed
        )  # (B, num_steps, latent_dim)

        # === Decode predictions back to image space ===
        predictions = []
        for t in range(num_steps):
            z_t = z_pred_seq[:, t, :]  # (B, latent_dim)
            x_t = self.cvae.decode(z_t, conditions)  # (B, C, H, W)
            predictions.append(x_t)

        pred_sequence = torch.stack(predictions, dim=1)  # (B, num_steps, C, H, W)

        if return_latent:
            return pred_sequence, z_pred_seq
        return pred_sequence

    @torch.no_grad()
    def predict_single(
        self,
        input_sequence: np.ndarray,
        conditions: np.ndarray,
        num_steps: int,
        denormalize_output: bool = True,
    ) -> np.ndarray:
        """
        Predict from numpy arrays (convenience method).

        Args:
            input_sequence: Historical frames (T, C, H, W) or (T, H, W)
            conditions: Condition vector (num_conditions,)
            num_steps: Number of future steps
            denormalize_output: Whether to denormalize output

        Returns:
            Predicted sequence (num_steps, C, H, W)
        """
        # Add batch and channel dimensions if needed
        if input_sequence.ndim == 3:
            input_sequence = input_sequence[:, np.newaxis, :, :]

        input_tensor = torch.from_numpy(input_sequence).float().unsqueeze(0)
        cond_tensor = torch.from_numpy(conditions).float().unsqueeze(0)

        # Normalize input
        input_tensor = self.normalize(input_tensor)

        # Predict
        pred_tensor = self.predict(input_tensor, cond_tensor, num_steps)

        # Denormalize if requested
        if denormalize_output:
            pred_tensor = self.denormalize(pred_tensor)

        return pred_tensor.squeeze(0).cpu().numpy()

    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 10, 1, 128, 128),
        num_steps: int = 50,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference time.

        Args:
            input_shape: Input sequence shape (B, T, C, H, W)
            num_steps: Number of prediction steps
            num_iterations: Number of benchmark iterations

        Returns:
            Dictionary of timing statistics
        """
        B, T, C, H, W = input_shape
        dummy_input = torch.randn(B, T, C, H, W, device=self.device)
        dummy_cond = torch.randn(B, self.config.condition.num_conditions, device=self.device)

        # Warmup
        for _ in range(10):
            _ = self.predict(dummy_input, dummy_cond, num_steps)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = self.predict(dummy_input, dummy_cond, num_steps)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

        times = np.array(times) * 1000  # Convert to ms

        # FiPy placeholder
        FIPY_TIME_CONST = "PENDING"

        return {
            "mean_time_ms": float(np.mean(times)),
            "std_time_ms": float(np.std(times)),
            "min_time_ms": float(np.min(times)),
            "max_time_ms": float(np.max(times)),
            "median_time_ms": float(np.median(times)),
            "throughput_steps_per_sec": num_steps * 1000 / np.mean(times),
            "prediction_steps": num_steps,
            "fipy_time_ms": FIPY_TIME_CONST,
            "speedup_vs_fipy": FIPY_TIME_CONST,
            "device": str(self.device),
        }


def sanity_check_predictions(
    predictions: torch.Tensor,
    ground_truth: Optional[torch.Tensor] = None,
) -> Dict[str, Union[float, bool, str]]:
    """
    Perform sanity checks on predictions.

    Checks:
    - Values within valid range [0, 1]
    - No NaN or Inf values
    - Grain boundary preservation (via total variation)
    - Conservation properties (mass conservation check)

    Args:
        predictions: Predicted sequence (B, T, C, H, W)
        ground_truth: Optional ground truth for comparison

    Returns:
        Dictionary of check results
    """
    results = {}

    # Check for NaN/Inf
    has_nan = torch.isnan(predictions).any().item()
    has_inf = torch.isinf(predictions).any().item()
    results["has_nan"] = has_nan
    results["has_inf"] = has_inf
    results["valid_values"] = not (has_nan or has_inf)

    if not results["valid_values"]:
        results["error"] = "Predictions contain NaN or Inf values"
        return results

    # Check value range
    pred_min = predictions.min().item()
    pred_max = predictions.max().item()
    results["pred_min"] = pred_min
    results["pred_max"] = pred_max
    results["in_range"] = (pred_min >= -0.1) and (pred_max <= 1.1)

    # Total variation (should remain roughly constant for grain coarsening)
    tv_values = []
    B, T, C, H, W = predictions.shape
    for t in range(T):
        tv = compute_total_variation(predictions[:, t, :, :, :])
        tv_values.append(tv.item())

    results["tv_start"] = tv_values[0]
    results["tv_end"] = tv_values[-1]
    results["tv_change_ratio"] = tv_values[-1] / (tv_values[0] + 1e-8)

    # For grain coarsening, TV should decrease (boundaries smooth out)
    # But shouldn't completely vanish
    results["tv_physically_plausible"] = (
        results["tv_change_ratio"] > 0.1 and results["tv_change_ratio"] < 2.0
    )

    # Mass conservation check (sum should remain roughly constant)
    mass_values = []
    for t in range(T):
        mass = predictions[:, t, :, :, :].sum().item()
        mass_values.append(mass)

    results["mass_start"] = mass_values[0]
    results["mass_end"] = mass_values[-1]
    results["mass_change_ratio"] = mass_values[-1] / (mass_values[0] + 1e-8)
    results["mass_conserved"] = abs(results["mass_change_ratio"] - 1.0) < 0.1

    # Compare with ground truth if available
    if ground_truth is not None:
        metrics = evaluate_reconstruction_quality(predictions.view(-1, C, H, W),
                                                   ground_truth.view(-1, C, H, W))
        results["vs_gt_mse"] = metrics["mse"]
        results["vs_gt_ssim"] = metrics["ssim"]

    return results


def run_inference(
    config: Config,
    cvae_checkpoint: Path,
    lstm_checkpoint: Path,
    num_steps: int = 50,
    use_synthetic: bool = True,
    save_results: bool = True,
) -> Dict:
    """
    Run full inference pipeline.

    Args:
        config: Configuration
        cvae_checkpoint: Path to CVAE checkpoint
        lstm_checkpoint: Path to LSTM checkpoint
        num_steps: Number of prediction steps
        use_synthetic: Use synthetic test data
        save_results: Save prediction results

    Returns:
        Dictionary of inference results
    """
    device = config.get_device()

    # Initialize predictor
    predictor = MicrostructurePredictor(
        cvae_checkpoint, lstm_checkpoint, config, device
    )

    # Create test dataset
    if use_synthetic:
        dataset = SyntheticPhaseFieldDataset(
            num_simulations=10,
            time_steps=60,  # Need more steps for rollout
            image_size=config.data.image_size,
            sequence_length=config.data.sequence_length,
            seed=config.training.seed + 100,  # Different seed for test
        )
    else:
        dataset = PhaseFieldDataset(
            data_path=config.data.data_root,
            sequence_length=config.data.sequence_length,
        )
        predictor.set_normalization(dataset.data_min, dataset.data_max)

    # Get a test batch
    batch = dataset[0]
    input_seq = batch["input_seq"].unsqueeze(0).to(device)  # (1, T, C, H, W)
    conditions = batch["conditions"].unsqueeze(0).to(device)  # (1, num_cond)

    print(f"\nRunning inference...")
    print(f"  Input shape: {input_seq.shape}")
    print(f"  Prediction steps: {num_steps}")

    # Predict
    start_time = time.perf_counter()
    predictions, latent_trajectory = predictor.predict(
        input_seq, conditions, num_steps, return_latent=True
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = (time.perf_counter() - start_time) * 1000

    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Latent trajectory shape: {latent_trajectory.shape}")
    print(f"  Inference time: {inference_time:.2f} ms")

    # Sanity checks
    print("\nPerforming sanity checks...")
    sanity_results = sanity_check_predictions(predictions)
    for key, value in sanity_results.items():
        print(f"  {key}: {value}")

    # Benchmark
    print("\nBenchmarking inference speed...")
    benchmark_results = predictor.benchmark(
        input_shape=(1, config.data.sequence_length, 1, *config.data.image_size),
        num_steps=num_steps,
        num_iterations=50,
    )
    print(f"  Mean inference time: {benchmark_results['mean_time_ms']:.2f} ms")
    print(f"  Throughput: {benchmark_results['throughput_steps_per_sec']:.1f} steps/sec")

    # Save results
    if save_results:
        output_dir = config.inference.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        save_prediction_grid(
            predictions.squeeze(0),
            save_dir=output_dir,
            prefix="prediction",
            format="both",
        )

        # Save evolution plot
        fig = plot_sequence_evolution(
            input_seq.squeeze(0).cpu(),
            predictions.squeeze(0).cpu(),
            num_frames=8,
            save_path=output_dir / "evolution_comparison.png",
        )

        # Save metrics
        results = {
            "inference_time_ms": inference_time,
            "prediction_steps": num_steps,
            "sanity_checks": sanity_results,
            "benchmark": benchmark_results,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_dir / "inference_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {output_dir}")

    return {
        "predictions": predictions,
        "latent_trajectory": latent_trajectory,
        "sanity_checks": sanity_results,
        "benchmark": benchmark_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run CVAE-LSTM microstructure prediction"
    )
    parser.add_argument(
        "--cvae_checkpoint", type=str, default=None,
        help="Path to CVAE checkpoint"
    )
    parser.add_argument(
        "--lstm_checkpoint", type=str, default=None,
        help="Path to LSTM checkpoint"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input sequence (npy file)"
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of prediction steps"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--use_synthetic", action="store_true",
        help="Use synthetic test data"
    )
    parser.add_argument(
        "--benchmark_only", action="store_true",
        help="Only run benchmark"
    )

    args = parser.parse_args()

    # Get configuration
    config = get_default_config()

    # Set checkpoint paths
    if args.cvae_checkpoint:
        cvae_path = Path(args.cvae_checkpoint)
    else:
        cvae_path = config.training.checkpoint_dir / "best_cvae_weights.pth"

    if args.lstm_checkpoint:
        lstm_path = Path(args.lstm_checkpoint)
    else:
        lstm_path = config.training.checkpoint_dir / "best_lstm_weights.pth"

    # Check checkpoints exist
    if not cvae_path.exists():
        print(f"Error: CVAE checkpoint not found: {cvae_path}")
        print("Please train the CVAE first using: python train_cvae.py")
        sys.exit(1)

    if not lstm_path.exists():
        print(f"Error: LSTM checkpoint not found: {lstm_path}")
        print("Please train the LSTM first using: python train_lstm.py")
        sys.exit(1)

    # Set output directory
    if args.output_dir:
        config.inference.output_dir = Path(args.output_dir)

    # Run inference
    results = run_inference(
        config,
        cvae_path,
        lstm_path,
        num_steps=args.steps,
        use_synthetic=args.use_synthetic,
        save_results=True,
    )

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
