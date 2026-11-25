
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config
from models.cvae import AdvancedCVAE as CVAE
from models.lstm_dynamics import LSTMDynamics

from dataset import PhaseFieldDataset

def plot_evolution_timeline_v12(output_path, num_steps=1):
    config = get_default_config()
    device = config.get_device()
    data_dir = config.data.data_root
    
    # Load dataset to get conditions and normalization
    dataset = PhaseFieldDataset(data_path=data_dir, sequence_length=config.data.sequence_length)
    
    # Load models
    cvae = CVAE(
        latent_dim=config.cvae.latent_dim,
        image_size=config.data.image_size,
        num_conditions=config.condition.num_conditions,
        condition_embed_dim=config.condition.condition_embed_dim,
        encoder_channels=config.cvae.encoder_channels,
        decoder_channels=config.cvae.decoder_channels,
    ).to(device)
    cvae.load_state_dict(torch.load(config.training.checkpoint_dir / "best_cvae_weights.pth", map_location=device)["model_state_dict"])
    cvae.eval()
    
    lstm = LSTMDynamics(latent_dim=config.cvae.latent_dim, hidden_dim=config.lstm.hidden_dim, num_layers=config.lstm.num_layers, condition_dim=config.condition.condition_embed_dim).to(device)
    lstm.load_state_dict(torch.load(config.training.checkpoint_dir / "best_lstm_weights.pth", map_location=device)["model_state_dict"])
    lstm.eval()
    
    # Select a clean grain-structure sample.
    # Exclusion rules: scale bar patches have rows with mean≈0 (dark bar) or ≈1 (white bg).
    # Preference: low mean (dark = grain boundaries visible) + high std (spatial variation).
    best_idx, best_score = 0, -1.0
    for candidate in range(len(dataset)):
        s = dataset[candidate]
        frame = s["input_seq"][-1, 0]  # (H, W)
        row_means = frame.mean(dim=1)
        col_means = frame.mean(dim=0)
        # Annotation artifacts: any row/column mean >0.55, or any pixel >0.7 in corners
        if (row_means > 0.55).any() or (col_means > 0.55).any():
            continue
        # Also reject patches with any very bright pixel cluster (color legend corners)
        if frame.max().item() > 0.70:
            continue
        frame_mean = frame.mean().item()
        frame_std = frame.std().item()
        score = frame_std - 0.3 * frame_mean  # prefer low mean (dark grains) + high std
        if score > best_score:
            best_score, best_idx = score, int(candidate)
    idx = best_idx
    sample = dataset[idx]
    print(f"Selected sample {idx} (mean={dataset[idx]['input_seq'][-1,0].mean():.3f}, std={dataset[idx]['input_seq'][-1,0].std():.3f})")
    
    input_seq = sample["input_seq"].unsqueeze(0).to(device) # (1, T, C, H, W)
    target_gt = sample["target"].unsqueeze(0).to(device)    # (1, 1, C, H, W) or (1, C, H, W)
    conditions = sample["conditions"].unsqueeze(0).to(device)
    
    if target_gt.dim() == 4:
        target_gt = target_gt.unsqueeze(1) # (1, 1, C, H, W)

    # Encode, reconstruct and predict
    with torch.no_grad():
        B, T, C, H, W = input_seq.shape
        cond_flat = conditions.unsqueeze(1).expand(-1, T, -1).reshape(B*T, -1)
        mu, _ = cvae.encode(input_seq.view(B*T, C, H, W), cond_flat)
        z_input_seq = mu.view(B, T, -1)

        # CVAE reconstruction of last history frame (diagnostic: proves CVAE quality)
        cvae_recon_t = cvae.decode(z_input_seq[:, -1, :], conditions).squeeze()
        last_frame_t = input_seq[0, -1, 0]
        cvae_ssim = 1.0 - cvae.ssim_loss(cvae_recon_t.unsqueeze(0).unsqueeze(0), last_frame_t.unsqueeze(0).unsqueeze(0)).item()
        cvae_recon = cvae_recon_t.cpu().numpy()

        # LSTM rollout
        z_pred_seq = lstm.autoregressive_rollout(
            z_input_seq, num_steps=num_steps,
            conditions=cvae.condition_encoder(conditions)
        )
        pred_frames = [
            cvae.decode(z_pred_seq[:, t, :], conditions).squeeze().cpu().numpy()
            for t in range(num_steps)
        ]

    history_frame = input_seq[0, -1, 0].cpu().numpy()
    gt_frames = [target_gt[0, t, 0].cpu().numpy() for t in range(min(num_steps, target_gt.shape[1]))]

    # Layout: History | CVAE Recon | GT t+1 | Pred t+1
    n_cols = 2 + num_steps * 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(history_frame, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("History (last frame)", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(cvae_recon, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f"CVAE Recon\n(SSIM={cvae_ssim:.4f})", fontsize=10)
    axes[1].axis('off')

    for t in range(num_steps):
        gt_col = 2 + t * 2
        pred_col = gt_col + 1
        if t < len(gt_frames):
            axes[gt_col].imshow(gt_frames[t], cmap='viridis', vmin=0, vmax=1)
            axes[gt_col].set_title(f"GT t+{t+1}", fontsize=10)
        else:
            axes[gt_col].text(0.5, 0.5, "No GT", ha='center', va='center',
                              transform=axes[gt_col].transAxes)
        axes[gt_col].axis('off')

        axes[pred_col].imshow(pred_frames[t], cmap='viridis', vmin=0, vmax=1)
        axes[pred_col].set_title(f"LSTM Pred t+{t+1}", fontsize=10)
        axes[pred_col].axis('off')

    fig.suptitle("Microstructure Evolution: Ground Truth vs CVAE-LSTM Prediction", fontsize=12, y=1.02)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"v1.2 Timeline comparison saved to: {output_path}")

if __name__ == "__main__":
    plot_evolution_timeline_v12(
        output_path="visualizations/v12_evolution_timeline.png"
    )
