"""
Metrics module for CVAE-LSTM Microstructure Evolution Surrogate Model.

This module implements:
- Reconstruction quality metrics (MSE, SSIM, PSNR)
- Phase boundary preservation metrics (gradient-based)
- Evaluation stubs for FiPy comparison (placeholder)
- Inference time benchmarking

Note: FiPy comparison functions are stubs that raise NotImplementedError
or return placeholder values until real FiPy data is available.

Author: AI-Generated for Advanced Packaging Reliability Assessment
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, Union
import time
import warnings


def compute_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Mean Squared Error between prediction and target.

    Args:
        pred: Predicted tensor
        target: Target tensor
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        MSE value(s)
    """
    return F.mse_loss(pred, target, reduction=reduction)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    channel_dim: int = 1,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM).

    Implements SSIM as described in:
    "Image quality assessment: from error visibility to structural similarity"
    Wang et al., IEEE TIP, 2004.

    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images (B, C, H, W)
        window_size: Gaussian window size
        sigma: Gaussian standard deviation
        data_range: Range of pixel values
        channel_dim: Channel dimension index

    Returns:
        SSIM value (scalar)
    """
    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian window
    window = _create_gaussian_window(window_size, sigma, pred.device)

    # Expand window for all channels
    num_channels = pred.size(channel_dim)
    window = window.expand(num_channels, 1, window_size, window_size)

    # Compute means
    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=num_channels)
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=num_channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    # Compute variances and covariance
    sigma_pred_sq = (
        F.conv2d(pred ** 2, window, padding=window_size // 2, groups=num_channels)
        - mu_pred_sq
    )
    sigma_target_sq = (
        F.conv2d(target ** 2, window, padding=window_size // 2, groups=num_channels)
        - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred * target, window, padding=window_size // 2, groups=num_channels)
        - mu_pred_target
    )

    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)

    return ssim_map.mean()


def _create_gaussian_window(
    window_size: int, sigma: float, device: torch.device
) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    window = g.unsqueeze(1) @ g.unsqueeze(0)
    return window.unsqueeze(0).unsqueeze(0)


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred: Predicted images
        target: Target images
        data_range: Range of pixel values

    Returns:
        PSNR value in dB
    """
    mse = compute_mse(pred, target)
    if mse < 1e-10:
        return torch.tensor(float("inf"), device=pred.device)

    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr


def compute_gradient_magnitude(
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient magnitude using Sobel operators.

    Args:
        image: Input images (B, C, H, W)

    Returns:
        Gradient magnitude (B, C, H, W)
    """
    device = image.device

    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    B, C, H, W = image.shape
    grad_mag = torch.zeros_like(image)

    for c in range(C):
        img_c = image[:, c : c + 1, :, :]
        grad_x = F.conv2d(img_c, sobel_x, padding=1)
        grad_y = F.conv2d(img_c, sobel_y, padding=1)
        grad_mag[:, c : c + 1, :, :] = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    return grad_mag


def compute_phase_boundary_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Compute metrics for phase boundary preservation.

    Evaluates how well predicted images preserve sharp phase boundaries
    compared to ground truth.

    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images (B, C, H, W)
        threshold: Gradient threshold for boundary detection

    Returns:
        Dictionary of boundary metrics
    """
    # Compute gradients
    grad_pred = compute_gradient_magnitude(pred)
    grad_target = compute_gradient_magnitude(target)

    # Gradient magnitude MSE
    grad_mse = compute_mse(grad_pred, grad_target).item()

    # Boundary detection (high gradient regions)
    boundary_pred = (grad_pred > threshold).float()
    boundary_target = (grad_target > threshold).float()

    # Boundary IoU (Intersection over Union)
    intersection = (boundary_pred * boundary_target).sum()
    union = ((boundary_pred + boundary_target) > 0).float().sum()
    boundary_iou = (intersection / (union + 1e-8)).item()

    # Boundary precision and recall
    true_positive = intersection
    pred_positive = boundary_pred.sum()
    target_positive = boundary_target.sum()

    precision = (true_positive / (pred_positive + 1e-8)).item()
    recall = (true_positive / (target_positive + 1e-8)).item()
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "gradient_mse": grad_mse,
        "boundary_iou": boundary_iou,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1_score,
    }


def evaluate_reconstruction_quality(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of reconstruction quality.

    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images (B, C, H, W)
        data_range: Range of pixel values

    Returns:
        Dictionary of all metrics
    """
    # Ensure tensors are on same device
    pred = pred.detach()
    target = target.detach()

    # Basic metrics
    mse = compute_mse(pred, target).item()
    ssim = compute_ssim(pred, target, data_range=data_range).item()
    psnr = compute_psnr(pred, target, data_range=data_range).item()

    # Phase boundary metrics
    boundary_metrics = compute_phase_boundary_metrics(pred, target)

    # Combine all metrics
    metrics = {
        "mse": mse,
        "ssim": ssim,
        "psnr": psnr,
        **boundary_metrics,
    }

    return metrics


def evaluate_fidelity_stub(
    pred_tensor: torch.Tensor,
    fipy_tensor_path: Optional[str] = None,
) -> Dict[str, Union[float, str]]:
    """
    Stub for evaluating prediction fidelity against FiPy simulation results.

    This is a placeholder function that will be implemented when real
    FiPy ground truth data becomes available.

    Args:
        pred_tensor: De-normalized predicted microstructure
        fipy_tensor_path: Path to FiPy simulation output

    Returns:
        Dictionary with placeholder metrics

    Raises:
        NotImplementedError: When fipy_tensor_path is provided
    """
    # TODO: Implement when FiPy data is ready

    if fipy_tensor_path is not None:
        raise NotImplementedError(
            "FiPy comparison not yet implemented. "
            "This function requires real FiPy simulation data."
        )

    warnings.warn(
        "evaluate_fidelity_stub called without FiPy data. "
        "Returning placeholder values."
    )

    return {
        "mse_vs_fipy": "PENDING",
        "ssim_vs_fipy": "PENDING",
        "physical_consistency": "PENDING",
        "grain_count_error": "PENDING",
        "average_grain_size_error": "PENDING",
        "note": "Implement when FiPy ground truth data is available",
    }


def benchmark_inference_time(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Benchmark model inference time.

    Args:
        model: PyTorch model to benchmark
        input_shape: Shape of input tensor
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
        device: Device for inference

    Returns:
        Dictionary with timing statistics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Synchronize if using CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed iterations
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append(end - start)

    times = np.array(times) * 1000  # Convert to milliseconds

    # FiPy comparison placeholder
    FIPY_TIME_CONST = "PENDING"

    return {
        "mean_time_ms": float(np.mean(times)),
        "std_time_ms": float(np.std(times)),
        "min_time_ms": float(np.min(times)),
        "max_time_ms": float(np.max(times)),
        "median_time_ms": float(np.median(times)),
        "throughput_samples_per_sec": 1000 / np.mean(times),
        "fipy_time_ms": FIPY_TIME_CONST,
        "speedup_vs_fipy": FIPY_TIME_CONST,
    }


def compute_total_variation(image: torch.Tensor) -> torch.Tensor:
    """
    Compute total variation of an image.

    Useful for measuring smoothness/sharpness of microstructure.

    Args:
        image: Input images (B, C, H, W)

    Returns:
        Total variation value
    """
    # Compute differences along H and W dimensions
    diff_h = image[:, :, 1:, :] - image[:, :, :-1, :]
    diff_w = image[:, :, :, 1:] - image[:, :, :, :-1]

    # Sum of absolute differences
    tv = torch.abs(diff_h).mean() + torch.abs(diff_w).mean()
    return tv


def compute_grain_statistics_stub(
    image: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, Union[float, str]]:
    """
    Stub for computing grain statistics from phase-field images.

    This would typically involve connected component analysis
    to identify individual grains and compute size distributions.

    Args:
        image: Phase-field images (B, C, H, W)
        threshold: Threshold for grain boundary detection

    Returns:
        Dictionary with grain statistics placeholders
    """
    # TODO: Implement grain segmentation and statistics
    warnings.warn(
        "compute_grain_statistics_stub is a placeholder. "
        "Implement proper grain segmentation for real analysis."
    )

    return {
        "num_grains": "PENDING",
        "mean_grain_size": "PENDING",
        "grain_size_std": "PENDING",
        "grain_size_distribution": "PENDING",
        "boundary_fraction": "PENDING",
        "note": "Implement grain segmentation for real statistics",
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test tensors
    target = torch.rand(4, 1, 128, 128, device=device)
    # Add some noise to create prediction
    pred = target + 0.1 * torch.randn_like(target)
    pred = pred.clamp(0, 1)

    # Test all metrics
    print("\n1. Basic metrics:")
    mse = compute_mse(pred, target)
    ssim = compute_ssim(pred, target)
    psnr = compute_psnr(pred, target)
    print(f"  MSE: {mse.item():.6f}")
    print(f"  SSIM: {ssim.item():.4f}")
    print(f"  PSNR: {psnr.item():.2f} dB")

    print("\n2. Phase boundary metrics:")
    boundary_metrics = compute_phase_boundary_metrics(pred, target)
    for k, v in boundary_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n3. Comprehensive evaluation:")
    all_metrics = evaluate_reconstruction_quality(pred, target)
    for k, v in all_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n4. FiPy comparison stub:")
    fipy_metrics = evaluate_fidelity_stub(pred)
    for k, v in fipy_metrics.items():
        print(f"  {k}: {v}")

    print("\n5. Total variation:")
    tv = compute_total_variation(target)
    print(f"  TV: {tv.item():.4f}")
