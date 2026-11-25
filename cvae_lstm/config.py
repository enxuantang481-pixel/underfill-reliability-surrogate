"""
Configuration module for v1.2 (Advanced Residual CVAE-LSTM).
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from pathlib import Path
import torch

@dataclass
class DataConfig:
    data_root: Path = Path(__file__).parent / "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    image_size: Tuple[int, int] = (80, 80)
    num_channels: int = 1
    sequence_length: int = 4
    prediction_horizon: int = 1
    sliding_stride: int = 1
    normalize: bool = True
    norm_min: float = 0.0
    norm_max: float = 1.0
    batch_size: int = 16
    num_workers: int = 2
    pin_memory: bool = True
    shuffle_train: bool = True

@dataclass
class ConditionConfig:
    num_conditions: int = 2
    condition_names: List[str] = field(default_factory=lambda: ["temperature", "stress"])
    condition_embed_dim: int = 64
    condition_mlp_hidden: List[int] = field(default_factory=lambda: [32, 64])

@dataclass
class CVAEConfig:
    latent_dim: int = 128 # Increased for higher detail retention
    use_residual: bool = True
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])

@dataclass
class LSTMConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    bidirectional: bool = False
    dropout_rate: float = 0.2
    ss_initial_prob: float = 1.0
    ss_final_prob: float = 0.0
    ss_decay_type: str = "inverse_sigmoid"
    ss_decay_k: float = 5.0

@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    cvae_epochs: int = 100
    cvae_lr: float = 5e-5
    kl_anneal_type: str = "cyclical"
    kl_anneal_cycles: int = 8 # Doubled cycles for better convergence
    kl_warmup_epochs: int = 10
    
    # Loss weights for AdvancedCVAE [MSE, KL, SSIM, Laplace]
    cvae_loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.001, 5.0, 2.0])
    
    lstm_epochs: int = 100
    lstm_lr: float = 1e-3
    grad_clip_norm: float = 1.0
    early_stopping: bool = True
    patience: int = 15
    checkpoint_dir: Path = Path(__file__).parent / "checkpoints"
    log_dir: Path = Path(__file__).parent / "logs"

@dataclass
class InferenceConfig:
    rollout_steps: int = 50
    output_dir: Path = Path(__file__).parent / "outputs"
    save_format: str = "png"

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    condition: ConditionConfig = field(default_factory=ConditionConfig)
    cvae: CVAEConfig = field(default_factory=CVAEConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        if self.training.device == "auto":
            self.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training.log_dir.mkdir(parents=True, exist_ok=True)
        self.inference.output_dir.mkdir(parents=True, exist_ok=True)

    def get_device(self) -> torch.device:
        return torch.device(self.training.device)

def get_default_config() -> Config:
    return Config()
