"""
LSTM Dynamics Network for Latent Space Temporal Modeling.

This module implements:
- LSTMDynamics: Multi-layer LSTM for latent vector sequence prediction
- ScheduledSampler: Scheduled sampling for training robustness
- Autoregressive rollout for multi-step prediction

Key Features:
- Scheduled sampling to prevent error accumulation
- Support for multi-step prediction horizon
- Condition injection at LSTM input
- Device-agnostic implementation

Tensor Flow:
Latent sequence (B, T, Latent_Dim) -> LSTM -> Predicted future (B, Latent_Dim)

Author: AI-Generated for Advanced Packaging Reliability Assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class ScheduledSampler:
    """
    Scheduled sampling scheduler for training robustness.

    Controls the probability of using teacher forcing vs model predictions
    during training. Starts with high teacher forcing probability and
    gradually decreases to use more model predictions.

    Decay types:
    - linear: Linear decay from initial to final probability
    - exponential: Exponential decay
    - inverse_sigmoid: Inverse sigmoid decay (smoother transition)
    """

    def __init__(
        self,
        initial_prob: float = 1.0,
        final_prob: float = 0.0,
        total_steps: int = 10000,
        decay_type: str = "inverse_sigmoid",
        k: float = 5.0,
    ):
        """
        Initialize scheduled sampler.

        Args:
            initial_prob: Initial teacher forcing probability
            final_prob: Final teacher forcing probability
            total_steps: Total number of training steps
            decay_type: Type of decay ('linear', 'exponential', 'inverse_sigmoid')
            k: Decay rate parameter (for inverse_sigmoid)
        """
        self.initial_prob = initial_prob
        self.final_prob = final_prob
        self.total_steps = total_steps
        self.decay_type = decay_type
        self.k = k
        self.current_step = 0

    def get_prob(self, step: Optional[int] = None) -> float:
        """
        Get current teacher forcing probability.

        Args:
            step: Optional step number (uses internal counter if None)

        Returns:
            Current teacher forcing probability
        """
        if step is None:
            step = self.current_step

        progress = min(step / self.total_steps, 1.0)

        if self.decay_type == "linear":
            prob = self.initial_prob - progress * (
                self.initial_prob - self.final_prob
            )

        elif self.decay_type == "exponential":
            prob = self.final_prob + (self.initial_prob - self.final_prob) * math.exp(
                -self.k * progress
            )

        elif self.decay_type == "inverse_sigmoid":
            # Inverse sigmoid provides smoother transition
            x = self.k * (2 * progress - 1)
            sigmoid = 1 / (1 + math.exp(-x))
            prob = self.final_prob + (self.initial_prob - self.final_prob) * (
                1 - sigmoid
            )

        else:
            prob = self.initial_prob

        return max(min(prob, 1.0), 0.0)

    def step(self):
        """Increment internal step counter."""
        self.current_step += 1

    def reset(self):
        """Reset step counter."""
        self.current_step = 0

    def should_use_teacher(self) -> bool:
        """
        Stochastically decide whether to use teacher forcing.

        Returns:
            True if should use ground truth, False if should use prediction
        """
        prob = self.get_prob()
        return torch.rand(1).item() < prob


class LSTMDynamics(nn.Module):
    """
    LSTM network for predicting latent space dynamics.

    Takes a sequence of latent vectors and predicts the next latent vector(s).
    Supports scheduled sampling for robust training.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        condition_dim: int = 0,
        use_layer_norm: bool = True,
    ):
        """
        Initialize LSTM dynamics network.

        Args:
            latent_dim: Input/output latent dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
            condition_dim: Dimension of condition embedding (0 = no conditions)
            use_layer_norm: Use layer normalization
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.condition_dim = condition_dim
        self.num_directions = 2 if bidirectional else 1

        # Input projection (latent + condition)
        input_dim = latent_dim + condition_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(hidden_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output projection
        lstm_output_dim = hidden_dim * self.num_directions
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights using orthogonal initialization."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        z_seq: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.

        Args:
            z_seq: Latent sequence (B, T, latent_dim)
            conditions: Condition vectors (B, condition_dim) - broadcast across time
            hidden: Optional initial hidden state

        Returns:
            Tuple of (output_sequence, (h_n, c_n))
            output_sequence: (B, T, latent_dim)
        """
        B, T, _ = z_seq.shape

        # Expand conditions across time if provided
        if conditions is not None and self.condition_dim > 0:
            conditions_expanded = conditions.unsqueeze(1).expand(-1, T, -1)
            lstm_input = torch.cat([z_seq, conditions_expanded], dim=-1)
        else:
            lstm_input = z_seq

        # Input projection
        lstm_input = self.input_projection(lstm_input)
        if self.use_layer_norm:
            lstm_input = self.input_norm(lstm_input)

        # LSTM forward
        lstm_out, hidden_out = self.lstm(lstm_input, hidden)

        # Output projection
        output = self.output_projection(lstm_out)

        return output, hidden_out

    def predict_next(
        self,
        z_seq: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict the next latent vector given a sequence.

        Args:
            z_seq: Latent sequence (B, T, latent_dim)
            conditions: Condition vectors (B, condition_dim)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (next_z, hidden_state)
            next_z: Predicted next latent (B, latent_dim)
        """
        output, hidden_out = self.forward(z_seq, conditions, hidden)
        # Take the last time step as prediction
        next_z = output[:, -1, :]  # (B, latent_dim)
        return next_z, hidden_out

    def predict_single_step(
        self,
        z_current: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single step prediction for autoregressive rollout.

        Args:
            z_current: Current latent vector (B, latent_dim)
            conditions: Condition vectors (B, condition_dim)
            hidden: Hidden state from previous step

        Returns:
            Tuple of (next_z, hidden_state)
        """
        # Add time dimension
        z_seq = z_current.unsqueeze(1)  # (B, 1, latent_dim)
        return self.predict_next(z_seq, conditions, hidden)

    def autoregressive_rollout(
        self,
        z_initial: torch.Tensor,
        num_steps: int,
        conditions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive multi-step prediction.

        Args:
            z_initial: Initial latent sequence (B, T_init, latent_dim)
            num_steps: Number of future steps to predict
            conditions: Condition vectors (B, condition_dim)

        Returns:
            Predicted future latent vectors (B, num_steps, latent_dim)
        """
        B = z_initial.size(0)
        device = z_initial.device

        # Get initial hidden state from input sequence
        _, hidden = self.forward(z_initial, conditions)

        # Autoregressive rollout
        predictions = []
        z_current = z_initial[:, -1, :]  # Start from last input

        for _ in range(num_steps):
            z_next, hidden = self.predict_single_step(z_current, conditions, hidden)
            predictions.append(z_next)
            z_current = z_next

        return torch.stack(predictions, dim=1)  # (B, num_steps, latent_dim)

    def forward_with_scheduled_sampling(
        self,
        z_seq: torch.Tensor,
        z_target: torch.Tensor,
        teacher_forcing_prob: float,
        conditions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with scheduled sampling for training.

        Mixes teacher forcing (using ground truth) with model predictions
        based on the teacher_forcing_prob.

        Args:
            z_seq: Input latent sequence (B, T, latent_dim)
            z_target: Target latent sequence for teacher forcing (B, T_target, latent_dim)
            teacher_forcing_prob: Probability of using teacher forcing
            conditions: Condition vectors (B, condition_dim)

        Returns:
            Predicted sequence (B, T_target, latent_dim)
        """
        B, T_in, _ = z_seq.shape
        T_target = z_target.size(1)
        device = z_seq.device

        # Initial forward pass to get hidden state
        _, hidden = self.forward(z_seq, conditions)

        predictions = []
        z_current = z_seq[:, -1, :]  # Start from last input

        for t in range(T_target):
            # Predict next
            z_pred, hidden = self.predict_single_step(z_current, conditions, hidden)
            predictions.append(z_pred)

            # Scheduled sampling: decide whether to use prediction or ground truth
            if t < T_target - 1:  # Not the last step
                use_teacher = torch.rand(1).item() < teacher_forcing_prob
                if use_teacher and self.training:
                    z_current = z_target[:, t, :]  # Use ground truth
                else:
                    z_current = z_pred  # Use prediction

        return torch.stack(predictions, dim=1)

    def get_hidden_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size
            device: Device for tensors

        Returns:
            Tuple of (h_0, c_0)
        """
        num_layers = self.num_layers * self.num_directions
        h_0 = torch.zeros(num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(num_layers, batch_size, self.hidden_dim, device=device)
        return h_0, c_0


class LSTMDynamicsWithAttention(nn.Module):
    """
    LSTM with temporal attention for improved long-range dependencies.

    Uses attention mechanism to weight historical latent vectors
    when making predictions.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        condition_dim: int = 0,
    ):
        """
        Initialize LSTM with attention.

        Args:
            latent_dim: Latent dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            condition_dim: Condition embedding dimension
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # Input projection
        input_dim = latent_dim + condition_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with attention.

        Args:
            z_seq: Latent sequence (B, T, latent_dim)
            conditions: Condition vectors (B, condition_dim)
            hidden: Optional initial hidden state

        Returns:
            Tuple of (output, hidden_state)
        """
        B, T, _ = z_seq.shape

        # Expand conditions
        if conditions is not None and self.condition_dim > 0:
            conditions_expanded = conditions.unsqueeze(1).expand(-1, T, -1)
            lstm_input = torch.cat([z_seq, conditions_expanded], dim=-1)
        else:
            lstm_input = z_seq

        # Input projection and normalization
        lstm_input = self.input_projection(lstm_input)
        lstm_input = self.input_norm(lstm_input)

        # LSTM
        lstm_out, hidden_out = self.lstm(lstm_input, hidden)

        # Self-attention (query from last time step, keys/values from all)
        query = lstm_out[:, -1:, :]  # (B, 1, hidden_dim)
        attn_out, _ = self.attention(query, lstm_out, lstm_out)
        attn_out = self.attention_norm(attn_out + query)

        # Combine LSTM and attention outputs
        combined = torch.cat([lstm_out[:, -1:, :], attn_out], dim=-1)
        output = self.output_projection(combined.squeeze(1))

        return output, hidden_out


if __name__ == "__main__":
    # Test LSTM dynamics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing LSTMDynamics on {device}")

    # Create model
    lstm = LSTMDynamics(
        latent_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        condition_dim=64,
    ).to(device)

    # Test inputs
    batch_size = 4
    seq_len = 10
    z_seq = torch.randn(batch_size, seq_len, 128, device=device)
    conditions = torch.randn(batch_size, 64, device=device)

    # Test forward pass
    output, hidden = lstm(z_seq, conditions)
    print(f"Input shape: {z_seq.shape}")
    print(f"Output shape: {output.shape}")

    # Test predict_next
    next_z, _ = lstm.predict_next(z_seq, conditions)
    print(f"Next prediction shape: {next_z.shape}")

    # Test autoregressive rollout
    future = lstm.autoregressive_rollout(z_seq, num_steps=5, conditions=conditions)
    print(f"Rollout shape (5 steps): {future.shape}")

    # Test scheduled sampling
    print("\nTesting scheduled sampler:")
    sampler = ScheduledSampler(
        initial_prob=1.0,
        final_prob=0.0,
        total_steps=1000,
        decay_type="inverse_sigmoid",
    )

    for step in [0, 250, 500, 750, 1000]:
        prob = sampler.get_prob(step)
        print(f"  Step {step}: teacher_forcing_prob = {prob:.4f}")

    # Test forward with scheduled sampling
    z_target = torch.randn(batch_size, 5, 128, device=device)
    output_ss = lstm.forward_with_scheduled_sampling(
        z_seq, z_target, teacher_forcing_prob=0.5, conditions=conditions
    )
    print(f"\nScheduled sampling output shape: {output_ss.shape}")
