
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config
from dataset import create_dataloaders
from models.cvae import AdvancedCVAE as CVAE, fold_time_dimension
from models.lstm_dynamics import LSTMDynamics, ScheduledSampler
from utils.metrics import compute_mse, compute_ssim

def train_lstm_v12():
    config = get_default_config()
    device = config.get_device()
    print(f"Training LSTM (v1.2) on {device}")

    # Load CVAE
    cvae_checkpoint = config.training.checkpoint_dir / "best_cvae_weights.pth"
    cvae = CVAE(
        latent_dim=config.cvae.latent_dim, 
        image_size=config.data.image_size,
        encoder_channels=config.cvae.encoder_channels,
        decoder_channels=config.cvae.decoder_channels
    ).to(device)
    cvae.load_state_dict(torch.load(cvae_checkpoint, map_location=device)["model_state_dict"])
    cvae.eval()
    for param in cvae.parameters(): param.requires_grad = False

    # Data — use same sequence_length as inference (config.data.sequence_length=4)
    # sequence_length=10 would exclude all T=6 zenodo sequences, leaving only
    # the 2 synthetic T=50 sequences → LSTM learns nothing from real data.
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=config.data.data_root,
        batch_size=2,
        sequence_length=config.data.sequence_length,
        prediction_horizon=1,
        use_synthetic=False
    )

    # LSTM Model
    lstm = LSTMDynamics(
        latent_dim=config.cvae.latent_dim,
        hidden_dim=config.lstm.hidden_dim,
        num_layers=config.lstm.num_layers,
        condition_dim=config.condition.condition_embed_dim
    ).to(device)

    optimizer = optim.AdamW(lstm.parameters(), lr=config.training.lstm_lr)
    
    # Scheduled Sampling
    ss_scheduler = ScheduledSampler(initial_prob=1.0, final_prob=0.0, total_steps=len(train_loader)*100)

    best_loss = float("inf")

    for epoch in range(100):
        lstm.train()
        total_loss = 0.0
        teacher_prob = ss_scheduler.get_prob()
        
        for batch in train_loader:
            input_seq = batch["input_seq"].to(device) # (B, T, C, H, W)
            target = batch["target"].to(device)       # (B, C, H, W)
            cond = batch["conditions"].to(device)
            
            B, T, C, H, W = input_seq.shape
            
            with torch.no_grad():
                # Encode history
                x_flat = input_seq.view(B * T, C, H, W)
                c_flat = cond.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                mu, _ = cvae.encode(x_flat, c_flat)
                z_in = mu.view(B, T, -1)
                
                # Encode target
                z_target, _ = cvae.encode(target.unsqueeze(1).view(B, C, H, W), cond)
                z_target = z_target.unsqueeze(1) # (B, 1, latent_dim)

            # Predict
            cond_embed = cvae.condition_encoder(cond)
            z_pred = lstm.forward_with_scheduled_sampling(z_in, z_target, teacher_prob, cond_embed)

            # Latent-space MSE (z_target detached — no grad needed for target)
            latent_loss = compute_mse(z_pred, z_target.detach())

            # Pixel-space reconstruction loss: decode z_pred through frozen CVAE.
            # CVAE params have no grad, but z_pred does → gradient flows to LSTM.
            recon_pred = cvae.decode(z_pred.squeeze(1), cond)
            pixel_loss = F.mse_loss(recon_pred, target)

            loss = latent_loss + 0.1 * pixel_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm.parameters(), config.training.grad_clip_norm)
            optimizer.step()
            ss_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | TF: {teacher_prob:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": lstm.state_dict(),
                "config": {"latent_dim": config.cvae.latent_dim, "hidden_dim": config.lstm.hidden_dim, "num_layers": config.lstm.num_layers},
                "epoch": epoch
            }, config.training.checkpoint_dir / "best_lstm_weights.pth")
            print(f"  Saved best LSTM (loss={best_loss:.6f})")

if __name__ == "__main__":
    train_lstm_v12()
