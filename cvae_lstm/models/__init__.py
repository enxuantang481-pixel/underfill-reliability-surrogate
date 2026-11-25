"""
Models module for CVAE-LSTM Microstructure Evolution Surrogate Model.

This module exports:
- CVAE: Conditional Variational Autoencoder for spatial compression
- Encoder: CVAE encoder network
- Decoder: CVAE decoder network
- LSTMDynamics: LSTM network for latent space temporal dynamics
- ConditionEncoder: MLP for encoding external conditions

Author: AI-Generated for Advanced Packaging Reliability Assessment
"""

from .cvae import (
    CVAE,
    Encoder,
    Decoder,
    ConditionEncoder,
    reparameterize,
)
from .lstm_dynamics import (
    LSTMDynamics,
    ScheduledSampler,
)

__all__ = [
    "CVAE",
    "Encoder",
    "Decoder",
    "ConditionEncoder",
    "reparameterize",
    "LSTMDynamics",
    "ScheduledSampler",
]
