
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class LaplacianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, pred, target):
        grad_pred = F.conv2d(pred, self.kernel, padding=1)
        grad_target = F.conv2d(target, self.kernel, padding=1)
        return F.mse_loss(grad_pred, grad_target)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._create_window(window_size))

    def _create_window(self, size):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * 1.5**2))
        g /= g.sum()
        g2d = g.view(1, -1) * g.view(-1, 1)
        return g2d.view(1, 1, size, size)

    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=1)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=1)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=1) - mu1_mu2
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class ConditionEncoder(nn.Module):
    def __init__(self, num_conditions=2, hidden_dims=[32, 64], output_dim=64):
        super().__init__()
        layers = []
        in_dim = num_conditions
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mlp = nn.Sequential(*layers)
    def forward(self, conditions):
        return self.mlp(conditions)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, hidden_channels=[32, 64, 128, 256], image_size=(80, 80), condition_embed_dim=64):
        super().__init__()
        layers = []
        curr_ch = in_channels
        for ch in hidden_channels:
            layers.append(ResidualBlock(curr_ch, ch, stride=2))
            curr_ch = ch
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *image_size)
            flat_size = self.net(dummy).view(1, -1).size(1)
        self.fc_mu = nn.Linear(flat_size + condition_embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_size + condition_embed_dim, latent_dim)

    def forward(self, x, cond_embed=None):
        h = self.net(x).view(x.size(0), -1)
        if cond_embed is not None:
            h = torch.cat([h, cond_embed], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=512, hidden_channels=[256, 128, 64, 32], image_size=(80, 80), condition_embed_dim=64):
        super().__init__()
        self.init_h, self.init_w = image_size[0] // 16, image_size[1] // 16
        self.fc = nn.Linear(latent_dim + condition_embed_dim, hidden_channels[0] * self.init_h * self.init_w)
        layers = []
        curr_ch = hidden_channels[0]
        for ch in hidden_channels[1:]:
            # Bilinear upsampling + Conv avoids the checkerboard artifacts
            # produced by ConvTranspose2d with stride=2.
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(curr_ch, ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(ResidualBlock(ch, ch))
            curr_ch = ch
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(curr_ch, out_channels, kernel_size=3, padding=1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z, cond_embed=None):
        if cond_embed is not None:
            z = torch.cat([z, cond_embed], dim=1)
        h = self.fc(z).view(-1, 256, self.init_h, self.init_w)
        return self.net(h)

class AdvancedCVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512, num_conditions=2, condition_embed_dim=64, image_size=(80, 80), encoder_channels=[32, 64, 128, 256], decoder_channels=[256, 128, 64, 32], **kwargs):
        super().__init__()
        self.condition_encoder = ConditionEncoder(num_conditions, output_dim=condition_embed_dim)
        self.encoder = Encoder(in_channels, latent_dim, hidden_channels=encoder_channels, image_size=image_size, condition_embed_dim=condition_embed_dim)
        self.decoder = Decoder(in_channels, latent_dim, hidden_channels=decoder_channels, image_size=image_size, condition_embed_dim=condition_embed_dim)
        self.ssim_loss = SSIMLoss()
        self.laplace_loss = LaplacianLoss()

    def encode(self, x, conditions=None):
        ce = self.condition_encoder(conditions) if conditions is not None else None
        return self.encoder(x, ce)

    def decode(self, z, conditions=None):
        ce = self.condition_encoder(conditions) if conditions is not None else None
        return self.decoder(z, ce)

    def forward(self, x, conditions=None):
        mu, logvar = self.encode(x, conditions)
        z = reparameterize(mu, logvar)
        return self.decode(z, conditions), mu, logvar, z

    def compute_loss(self, x, x_recon, mu, logvar, beta=1.0, weights=[1.0, 1.0, 1.0, 1.0]):
        mse_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        ssim_loss = self.ssim_loss(x_recon, x)
        lap_loss = self.laplace_loss(x_recon, x)
        total_loss = weights[0]*mse_loss + beta*weights[1]*kl_loss + weights[2]*ssim_loss + weights[3]*lap_loss
        return total_loss, {"total_loss": total_loss.item(), "mse_loss": mse_loss.item(), "kl_loss": kl_loss.item(), "ssim_loss": ssim_loss.item(), "laplace_loss": lap_loss.item()}

CVAE = AdvancedCVAE

def fold_time_dimension(x: torch.Tensor):
    B, T, C, H, W = x.shape
    return x.view(B * T, C, H, W), B, T

def unfold_time_dimension(x: torch.Tensor, B: int, T: int):
    if x.dim() == 2: return x.view(B, T, -1)
    return x.view(B, T, *x.shape[1:])
