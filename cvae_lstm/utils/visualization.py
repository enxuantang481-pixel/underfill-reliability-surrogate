"""
Visualization module for CVAE-LSTM Microstructure Evolution Surrogate Model.

This module implements:
- Training curve plotting (loss, metrics)
- Reconstruction comparison visualization
- Latent space visualization (t-SNE, PCA)
- Sequence evolution animation
- Prediction grid saving

Author: AI-Generated for Advanced Packaging Reliability Assessment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import warnings

# Try to import optional dependencies
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Latent space visualization disabled.")


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    loss_components: Optional[Dict[str, List[float]]] = None,
    metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[Path] = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        loss_components: Optional dict of loss component histories
        metrics: Optional dict of metric histories
        save_path: Optional path to save figure
        title: Figure title

    Returns:
        Matplotlib figure
    """
    num_plots = 1
    if loss_components:
        num_plots += 1
    if metrics:
        num_plots += 1

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Main loss plot
    ax = axes[0]
    ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} - Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Loss components plot
    if loss_components and num_plots > 1:
        ax = axes[1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(loss_components)))
        for (name, values), color in zip(loss_components.items(), colors):
            if len(values) == len(epochs):
                ax.plot(epochs, values, label=name, color=color, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss Value")
        ax.set_title("Loss Components")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    # Metrics plot
    if metrics and num_plots > 2:
        ax = axes[2]
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
        for (name, values), color in zip(metrics.items(), colors):
            if len(values) == len(epochs):
                ax.plot(epochs, values, label=name, color=color, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.set_title("Evaluation Metrics")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")

    return fig


def plot_reconstruction_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    num_samples: int = 4,
    channel: int = 0,
    save_path: Optional[Path] = None,
    title: str = "Reconstruction Comparison",
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot side-by-side comparison of original and reconstructed images.

    Args:
        original: Original images (B, C, H, W)
        reconstructed: Reconstructed images (B, C, H, W)
        num_samples: Number of samples to display
        channel: Channel index to visualize
        save_path: Optional path to save figure
        title: Figure title
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()

    num_samples = min(num_samples, orig_np.shape[0])

    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))

    for i in range(num_samples):
        # Original
        ax = axes[0, i] if num_samples > 1 else axes[0]
        im = ax.imshow(orig_np[i, channel], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Original {i + 1}")
        ax.axis("off")

        # Reconstructed
        ax = axes[1, i] if num_samples > 1 else axes[1]
        ax.imshow(recon_np[i, channel], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"Reconstructed {i + 1}")
        ax.axis("off")

        # Difference
        ax = axes[2, i] if num_samples > 1 else axes[2]
        diff = np.abs(orig_np[i, channel] - recon_np[i, channel])
        im_diff = ax.imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        ax.set_title(f"Difference {i + 1}")
        ax.axis("off")

    # Add colorbar for difference
    plt.colorbar(im_diff, ax=axes[2, :] if num_samples > 1 else axes[2], shrink=0.8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Reconstruction comparison saved to {save_path}")

    return fig


def plot_latent_space(
    latent_vectors: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    method: str = "tsne",
    save_path: Optional[Path] = None,
    title: str = "Latent Space Visualization",
) -> Optional[plt.Figure]:
    """
    Visualize latent space using dimensionality reduction.

    Args:
        latent_vectors: Latent vectors (N, latent_dim)
        labels: Optional labels for coloring (N,)
        method: Reduction method ('tsne' or 'pca')
        save_path: Optional path to save figure
        title: Figure title

    Returns:
        Matplotlib figure or None if sklearn unavailable
    """
    if not SKLEARN_AVAILABLE:
        warnings.warn("scikit-learn required for latent space visualization")
        return None

    # Convert to numpy
    z = latent_vectors.detach().cpu().numpy()

    # Apply dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        z_2d = reducer.fit_transform(z)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2)
        z_2d = reducer.fit_transform(z)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    if labels is not None:
        labels_np = labels.detach().cpu().numpy()
        scatter = ax.scatter(
            z_2d[:, 0], z_2d[:, 1], c=labels_np, cmap="viridis", alpha=0.7, s=20
        )
        plt.colorbar(scatter, ax=ax, label="Label")
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.7, s=20)

    ax.set_xlabel(f"{method.upper()} Dimension 1")
    ax.set_ylabel(f"{method.upper()} Dimension 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Latent space plot saved to {save_path}")

    return fig


def plot_sequence_evolution(
    sequence: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    num_frames: int = 10,
    channel: int = 0,
    save_path: Optional[Path] = None,
    title: str = "Microstructure Evolution",
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot sequence of microstructure evolution frames.

    Args:
        sequence: Input/ground truth sequence (T, C, H, W)
        predictions: Optional predicted sequence (T_pred, C, H, W)
        num_frames: Number of frames to display
        channel: Channel index to visualize
        save_path: Optional path to save figure
        title: Figure title
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    seq_np = sequence.detach().cpu().numpy()
    T = seq_np.shape[0]

    num_rows = 1 if predictions is None else 2
    frame_indices = np.linspace(0, T - 1, min(num_frames, T), dtype=int)

    fig, axes = plt.subplots(
        num_rows, len(frame_indices), figsize=(2 * len(frame_indices), 2.5 * num_rows)
    )

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot ground truth
    for j, t in enumerate(frame_indices):
        ax = axes[0, j]
        ax.imshow(seq_np[t, channel], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"t={t}")
        ax.axis("off")

    axes[0, 0].set_ylabel("Ground Truth", fontsize=10)

    # Plot predictions if provided
    if predictions is not None:
        pred_np = predictions.detach().cpu().numpy()
        T_pred = pred_np.shape[0]
        pred_indices = np.linspace(0, T_pred - 1, min(num_frames, T_pred), dtype=int)

        for j, t in enumerate(pred_indices):
            if j < len(axes[1]):
                ax = axes[1, j]
                ax.imshow(pred_np[t, channel], cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f"t={t}")
                ax.axis("off")

        axes[1, 0].set_ylabel("Predicted", fontsize=10)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sequence evolution saved to {save_path}")

    return fig


def create_evolution_animation(
    sequence: torch.Tensor,
    channel: int = 0,
    save_path: Optional[Path] = None,
    fps: int = 5,
    cmap: str = "viridis",
    title: str = "Microstructure Evolution",
) -> animation.FuncAnimation:
    """
    Create animated visualization of microstructure evolution.

    Args:
        sequence: Image sequence (T, C, H, W)
        channel: Channel index to visualize
        save_path: Optional path to save animation (mp4 or gif)
        fps: Frames per second
        cmap: Colormap
        title: Animation title

    Returns:
        Matplotlib animation object
    """
    seq_np = sequence.detach().cpu().numpy()
    T = seq_np.shape[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(seq_np[0, channel], cmap=cmap, vmin=0, vmax=1)
    ax.axis("off")
    time_text = ax.set_title(f"{title} - t=0")
    plt.colorbar(im, ax=ax, shrink=0.8)

    def update(frame):
        im.set_data(seq_np[frame, channel])
        time_text.set_text(f"{title} - t={frame}")
        return [im, time_text]

    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 // fps, blit=True
    )

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == ".gif":
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)
        print(f"Animation saved to {save_path}")

    return anim


def save_prediction_grid(
    predictions: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    save_dir: Path = Path("./outputs"),
    prefix: str = "pred",
    channel: int = 0,
    format: str = "png",
    cmap: str = "viridis",
) -> None:
    """
    Save prediction images as a grid or individual files.

    Args:
        predictions: Predicted images (B, C, H, W) or (T, C, H, W)
        targets: Optional target images for comparison
        save_dir: Directory to save images
        prefix: Filename prefix
        channel: Channel index to visualize
        format: Image format ('png', 'npy', 'both')
        cmap: Colormap for png
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pred_np = predictions.detach().cpu().numpy()
    N = pred_np.shape[0]

    for i in range(N):
        img = pred_np[i, channel]

        # Save as numpy
        if format in ["npy", "both"]:
            np.save(save_dir / f"{prefix}_{i:04d}.npy", img)

        # Save as image
        if format in ["png", "both"]:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            fig.savefig(
                save_dir / f"{prefix}_{i:04d}.png",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)

    # Save comparison grid if targets provided
    if targets is not None:
        target_np = targets.detach().cpu().numpy()

        num_show = min(8, N)
        fig, axes = plt.subplots(2, num_show, figsize=(2 * num_show, 4))

        for i in range(num_show):
            axes[0, i].imshow(target_np[i, channel], cmap=cmap, vmin=0, vmax=1)
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Target {i}")

            axes[1, i].imshow(pred_np[i, channel], cmap=cmap, vmin=0, vmax=1)
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Pred {i}")

        fig.suptitle("Prediction vs Target Comparison", fontsize=12)
        plt.tight_layout()
        fig.savefig(save_dir / f"{prefix}_comparison_grid.png", dpi=150)
        plt.close(fig)

    print(f"Saved {N} predictions to {save_dir}")


def plot_kl_annealing_schedule(
    epochs: int,
    anneal_type: str = "cyclical",
    cycles: int = 4,
    warmup_epochs: int = 10,
    beta_min: float = 0.0,
    beta_max: float = 1.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Visualize KL annealing schedule.

    Args:
        epochs: Total number of epochs
        anneal_type: Annealing type ('linear', 'sigmoid', 'cyclical')
        cycles: Number of cycles for cyclical annealing
        warmup_epochs: Warmup epochs per cycle
        beta_min: Minimum beta value
        beta_max: Maximum beta value
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    betas = []

    for epoch in range(epochs):
        if anneal_type == "linear":
            beta = min(beta_max, beta_min + (beta_max - beta_min) * epoch / warmup_epochs)

        elif anneal_type == "sigmoid":
            x = 10 * (epoch / warmup_epochs - 0.5)
            beta = beta_min + (beta_max - beta_min) / (1 + np.exp(-x))

        elif anneal_type == "cyclical":
            cycle_length = epochs // cycles
            cycle_pos = epoch % cycle_length
            cycle_warmup = min(warmup_epochs, cycle_length // 2)
            if cycle_pos < cycle_warmup:
                beta = beta_min + (beta_max - beta_min) * cycle_pos / cycle_warmup
            else:
                beta = beta_max

        betas.append(beta)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(epochs), betas, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Beta (KL Weight)")
    ax.set_title(f"KL Annealing Schedule ({anneal_type})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, beta_max + 0.1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"KL annealing schedule saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Test visualization functions
    print("Testing visualization functions...")

    # Create dummy data
    batch_size = 8
    T = 20
    C, H, W = 1, 64, 64

    # Test training curves
    print("\n1. Testing training curves...")
    train_losses = [1.0 * np.exp(-i / 20) for i in range(100)]
    val_losses = [1.1 * np.exp(-i / 25) for i in range(100)]
    loss_components = {
        "recon_loss": [0.8 * np.exp(-i / 20) for i in range(100)],
        "kl_loss": [0.2 * np.exp(-i / 30) for i in range(100)],
    }
    fig = plot_training_curves(
        train_losses, val_losses, loss_components, save_path=None
    )
    plt.close(fig)
    print("  Training curves: OK")

    # Test reconstruction comparison
    print("\n2. Testing reconstruction comparison...")
    original = torch.rand(batch_size, C, H, W)
    reconstructed = original + 0.1 * torch.randn_like(original)
    reconstructed = reconstructed.clamp(0, 1)
    fig = plot_reconstruction_comparison(original, reconstructed, num_samples=4)
    plt.close(fig)
    print("  Reconstruction comparison: OK")

    # Test sequence evolution
    print("\n3. Testing sequence evolution...")
    sequence = torch.rand(T, C, H, W)
    predictions = torch.rand(10, C, H, W)
    fig = plot_sequence_evolution(sequence, predictions, num_frames=8)
    plt.close(fig)
    print("  Sequence evolution: OK")

    # Test latent space visualization
    print("\n4. Testing latent space visualization...")
    if SKLEARN_AVAILABLE:
        latent_vectors = torch.randn(100, 128)
        labels = torch.randint(0, 5, (100,))
        fig = plot_latent_space(latent_vectors, labels, method="pca")
        plt.close(fig)
        print("  Latent space: OK")
    else:
        print("  Latent space: SKIPPED (sklearn not available)")

    # Test KL annealing schedule
    print("\n5. Testing KL annealing schedule...")
    fig = plot_kl_annealing_schedule(100, anneal_type="cyclical", cycles=4)
    plt.close(fig)
    print("  KL annealing schedule: OK")

    print("\nAll visualization tests passed!")
