
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_default_config
from dataset import create_dataloaders
from models.cvae import AdvancedCVAE, fold_time_dimension, unfold_time_dimension
from utils.metrics import compute_ssim

def train_v12():
    config = get_default_config()
    device = config.get_device()
    print(f"Training Advanced CVAE (v1.2) on {device}")

    # Data — sequence_length must match config so that T=6 zenodo sequences are included.
    # Default sequence_length=10 would exclude all zenodo data (T=6 < 10+1), leaving only
    # the 2 synthetic T=50 sequences and causing the CVAE to learn only Gaussian-blur textures.
    train_loader, val_loader, _ = create_dataloaders(
        data_path=config.data.data_root,
        batch_size=config.data.batch_size,
        sequence_length=config.data.sequence_length,
        use_synthetic=False
    )

    # Model
    model = AdvancedCVAE(
        latent_dim=config.cvae.latent_dim,
        image_size=config.data.image_size,
        num_conditions=config.condition.num_conditions,
        condition_embed_dim=config.condition.condition_embed_dim,
        encoder_channels=config.cvae.encoder_channels,
        decoder_channels=config.cvae.decoder_channels
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.training.cvae_lr)
    
    best_val_ssim = 0.0
    
    for epoch in range(config.training.cvae_epochs):
        model.train()
        train_loss = 0.0
        
        # Simple Beta scheduler for cyclical annealing
        beta = min(1.0, epoch / 20.0) 
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x = batch["input_seq"].to(device)
            cond = batch["conditions"].to(device)
            
            x_flat, B, T = fold_time_dimension(x)
            cond_flat = cond.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            
            recon, mu, logvar, _ = model(x_flat, cond_flat)
            loss, loss_dict = model.compute_loss(x_flat, recon, mu, logvar, beta=beta, weights=config.training.cvae_loss_weights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "ssim": 1 - loss_dict["ssim_loss"]})
            
        # Validation
        model.eval()
        val_ssim = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_seq"].to(device)
                cond = batch["conditions"].to(device)
                x_flat, B, T = fold_time_dimension(x)
                cond_flat = cond.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                
                recon, _, _, _ = model(x_flat, cond_flat)
                val_ssim += compute_ssim(recon, x_flat).item()
        
        val_ssim /= len(val_loader)
        print(f"Validation SSIM: {val_ssim:.4f}")
        
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "latent_dim": config.cvae.latent_dim, 
                    "image_size": config.data.image_size,
                    "encoder_channels": config.cvae.encoder_channels,
                    "decoder_channels": config.cvae.decoder_channels,
                    "num_conditions": config.condition.num_conditions,
                    "condition_embed_dim": config.condition.condition_embed_dim
                },
                "epoch": epoch,
                "loss": train_loss/len(train_loader)
            }, config.training.checkpoint_dir / "best_cvae_weights.pth")
            print(f"Saved new best model with SSIM: {val_ssim:.4f}")

if __name__ == "__main__":
    train_v12()
