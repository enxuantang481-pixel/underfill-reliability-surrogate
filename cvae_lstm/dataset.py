"""
Dataset module for CVAE-LSTM Microstructure Evolution Surrogate Model.

This module handles:
- Loading phase-field data from .npy/.h5 files
- Min-Max normalization to [0, 1] range
- Sliding window slicing for time series
- Condition tensor generation
- Train/Val/Test splitting

Data format: (B, T, C, H, W) - 5D tensors
- B: Batch size
- T: Time steps (sequence length)
- C: Channels (phase-field variables)
- H, W: Spatial dimensions

Author: AI-Generated for Advanced Packaging Reliability Assessment
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Union
import h5py
import scipy.io
import warnings


class PhaseFieldDataset(Dataset):
    """
    PyTorch Dataset for phase-field microstructure time series data.

    Handles loading from .npy, .h5, or .mat files with sliding window extraction
    for autoregressive sequence modeling.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        stride: int = 1,
        normalize: bool = True,
        norm_range: Tuple[float, float] = (0.0, 1.0),
        image_size: Optional[Tuple[int, int]] = None,
        conditions: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize the PhaseFieldDataset.
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.normalize = normalize
        self.norm_range = norm_range
        self.image_size = image_size
        self.transform = transform

        # Load data
        self.raw_data, self.conditions = self._load_data(conditions)

        # Compute normalization statistics
        if self.normalize:
            if isinstance(self.raw_data, list):
                self.data_min = min(float(d.min()) for d in self.raw_data)
                self.data_max = max(float(d.max()) for d in self.raw_data)
            else:
                self.data_min = float(self.raw_data.min())
                self.data_max = float(self.raw_data.max())
            
            if self.data_max - self.data_min < 1e-8:
                warnings.warn("Data range is too small, normalization may be unstable")
                self.data_max = self.data_min + 1.0
        else:
            self.data_min = 0.0
            self.data_max = 1.0

        # Create sliding window indices
        self.sequences = self._create_sequences()

        print(f"Dataset initialized:")
        if isinstance(self.raw_data, list):
            print(f"  - Raw data: {len(self.raw_data)} sequences with varying lengths")
        else:
            print(f"  - Raw data shape: {self.raw_data.shape}")
        print(f"  - Number of sequences: {len(self.sequences)}")
        print(f"  - Data range: [{self.data_min:.4f}, {self.data_max:.4f}]")

    def _load_data(
        self, conditions: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.data_path.is_file():
            return self._load_single_file(self.data_path, conditions)
        elif self.data_path.is_dir():
            return self._load_directory(conditions)
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def _load_single_file(
        self, filepath: Path, conditions: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        suffix = filepath.suffix.lower()

        if suffix == ".npy":
            data = np.load(filepath)
        elif suffix in [".h5", ".hdf5"]:
            with h5py.File(filepath, "r") as f:
                for key in ["data", "phase_field", "microstructure", "phi"]:
                    if key in f:
                        data = f[key][:]
                        break
                else:
                    first_key = list(f.keys())[0]
                    data = f[first_key][:]
                if conditions is None:
                    for cond_key in ["conditions", "parameters", "cond"]:
                        if cond_key in f:
                            conditions = f[cond_key][:]
                            break
        elif suffix == ".mat":
            mat_data = scipy.io.loadmat(filepath)
            # For Zenodo data, 'phase' contains the microstructure
            if "phase" in mat_data:
                data = mat_data["phase"]
            elif "phi" in mat_data:
                data = mat_data["phi"]
            else:
                # Fallback to the first array that isn't metadata
                for k, v in mat_data.items():
                    if not k.startswith("__") and isinstance(v, np.ndarray):
                        data = v
                        break
                else:
                    raise ValueError(f"No valid data array found in {filepath}")
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        data = self._ensure_dimensions(data)
        return data, conditions

    def _load_directory(
        self, conditions: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        all_data = []
        all_conditions = []

        # Find all data files recursively
        data_files = []
        for ext in ["*.npy", "*.h5", "*.hdf5", "*.mat"]:
            data_files.extend(list(self.data_path.rglob(ext)))
        data_files = sorted(data_files)

        if not data_files:
            raise FileNotFoundError(f"No data files found in {self.data_path}")

        for i, filepath in enumerate(data_files):
            cond = conditions[i] if conditions is not None else None
            data, file_cond = self._load_single_file(filepath, cond)
            if data.ndim == 4:  # (T, C, H, W)
                data = data[np.newaxis, ...]
            all_data.append(data)
            if file_cond is not None:
                all_conditions.append(file_cond)

        combined_conditions = np.stack(all_conditions, axis=0) if all_conditions else None
        return all_data, combined_conditions

    def _ensure_dimensions(self, data: np.ndarray) -> np.ndarray:
        """Ensure data has correct dimensionality (N, T, C, H, W) or (T, C, H, W)."""
        if data.ndim == 2:  # (H, W) - single frame, single channel
            data = data[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
        elif data.ndim == 3:  # (T, H, W) - single channel
            data = data[:, np.newaxis, :, :]  # (T, 1, H, W)
        elif data.ndim == 4:  # (T, C, H, W) or (N, T, H, W)
            # Assume (T, C, H, W) if second dim is small
            if data.shape[1] > 10:  # Likely (N, T, H, W)
                data = data[:, :, np.newaxis, :, :]  # (N, T, 1, H, W)
        elif data.ndim == 5:  # Already (N, T, C, H, W)
            pass
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Resize if needed
        if self.image_size is not None:
            data = self._resize_data(data)

        return data

    def _resize_data(self, data: np.ndarray) -> np.ndarray:
        """Resize spatial dimensions using interpolation."""
        from scipy.ndimage import zoom

        target_h, target_w = self.image_size

        if data.ndim == 4:  # (T, C, H, W)
            current_h, current_w = data.shape[-2:]
            if (current_h, current_w) != (target_h, target_w):
                zoom_h = target_h / current_h
                zoom_w = target_w / current_w
                data = zoom(data, (1, 1, zoom_h, zoom_w), order=1)
        elif data.ndim == 5:  # (N, T, C, H, W)
            current_h, current_w = data.shape[-2:]
            if (current_h, current_w) != (target_h, target_w):
                zoom_h = target_h / current_h
                zoom_w = target_w / current_w
                data = zoom(data, (1, 1, 1, zoom_h, zoom_w), order=1)

        return data

    def _create_sequences(self) -> List[Tuple[int, int, int]]:
        """
        Create sliding window sequence indices.

        Returns:
            List of (simulation_idx, start_frame, end_frame) tuples
        """
        sequences = []
        min_frames = self.sequence_length + self.prediction_horizon

        if isinstance(self.raw_data, list):
            for sim_idx, sim_data in enumerate(self.raw_data):
                # sim_data shape is (1, T, C, H, W) because of np.newaxis in _load_directory
                # or (T, C, H, W) depending on where it came from.
                # _load_directory adds newaxis if it was (T, C, H, W)
                if sim_data.ndim == 5:
                    time_dim = sim_data.shape[1]
                else:
                    time_dim = sim_data.shape[0]

                if time_dim >= min_frames:
                    for start in range(0, time_dim - min_frames + 1, self.stride):
                        sequences.append((sim_idx, start, start + self.sequence_length))
        
        elif self.raw_data.ndim == 4:  # Single simulation (T, C, H, W)
            num_simulations = 1
            time_dim = self.raw_data.shape[0]

            if time_dim >= min_frames:
                for start in range(0, time_dim - min_frames + 1, self.stride):
                    sequences.append((0, start, start + self.sequence_length))
        else:  # Multiple simulations (N, T, C, H, W)
            num_simulations = self.raw_data.shape[0]

            for sim_idx in range(num_simulations):
                time_dim = self.raw_data.shape[1]
                if time_dim >= min_frames:
                    for start in range(0, time_dim - min_frames + 1, self.stride):
                        sequences.append((sim_idx, start, start + self.sequence_length))

        return sequences

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Apply Min-Max normalization."""
        normalized = (data - self.data_min) / (self.data_max - self.data_min)
        # Scale to target range
        range_min, range_max = self.norm_range
        return normalized * (range_max - range_min) + range_min

    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Reverse Min-Max normalization to original scale."""
        range_min, range_max = self.norm_range
        normalized = (data - range_min) / (range_max - range_min)
        return normalized * (self.data_max - self.data_min) + self.data_min

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Returns:
            Dictionary containing:
                - 'input_seq': Input sequence (T, C, H, W)
                - 'target': Target frame(s) (prediction_horizon, C, H, W)
                - 'conditions': Condition vector (num_conditions,) if available
        """
        sim_idx, start, end = self.sequences[idx]

        # Extract sequence
        if isinstance(self.raw_data, list):
            sim_data = self.raw_data[sim_idx]
            if sim_data.ndim == 5:  # (1, T, C, H, W)
                input_seq = sim_data[0, start:end].copy()
                target = sim_data[
                    0, end : end + self.prediction_horizon
                ].copy()
            else:  # (T, C, H, W)
                input_seq = sim_data[start:end].copy()
                target = sim_data[
                    end : end + self.prediction_horizon
                ].copy()
        elif self.raw_data.ndim == 4:
            input_seq = self.raw_data[start:end].copy()
            target = self.raw_data[
                end : end + self.prediction_horizon
            ].copy()
        else:
            input_seq = self.raw_data[sim_idx, start:end].copy()
            target = self.raw_data[
                sim_idx, end : end + self.prediction_horizon
            ].copy()

        # Normalize
        if self.normalize:
            input_seq = self.normalize_data(input_seq)
            target = self.normalize_data(target)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_seq).float()
        target_tensor = torch.from_numpy(target).float()

        # Squeeze prediction horizon if only predicting 1 frame
        if self.prediction_horizon == 1:
            target_tensor = target_tensor.squeeze(0)  # (C, H, W)

        # Apply transforms
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        result = {
            "input_seq": input_tensor,
            "target": target_tensor,
        }

        # Add conditions if available
        if self.conditions is not None:
            if self.raw_data.ndim == 4:
                cond = self.conditions[0] if self.conditions.ndim > 1 else self.conditions
            else:
                cond = self.conditions[sim_idx]
            result["conditions"] = torch.from_numpy(np.asarray(cond)).float()
        else:
            # Generate dummy conditions for testing
            result["conditions"] = torch.zeros(2, dtype=torch.float32)

        return result


class SyntheticPhaseFieldDataset(Dataset):
    """
    Synthetic dataset for testing when real data is unavailable.

    Generates grain coarsening-like patterns using simple dynamics.
    """

    def __init__(
        self,
        num_simulations: int = 100,
        time_steps: int = 50,
        image_size: Tuple[int, int] = (128, 128),
        num_channels: int = 1,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        seed: int = 42,
    ):
        """
        Initialize synthetic dataset.

        Args:
            num_simulations: Number of synthetic simulations to generate
            time_steps: Total time steps per simulation
            image_size: Spatial dimensions (H, W)
            num_channels: Number of phase-field channels
            sequence_length: Input sequence length
            prediction_horizon: Number of future frames to predict
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.num_simulations = num_simulations
        self.time_steps = time_steps
        self.image_size = image_size
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        np.random.seed(seed)
        self.data, self.conditions = self._generate_synthetic_data()
        self.sequences = self._create_sequences()

        print(f"Synthetic dataset created:")
        print(f"  - Shape: {self.data.shape}")
        print(f"  - Sequences: {len(self.sequences)}")

    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic grain coarsening data.

        Uses a simplified diffusion-like process to simulate grain boundary motion.
        """
        H, W = self.image_size
        data = np.zeros(
            (self.num_simulations, self.time_steps, self.num_channels, H, W),
            dtype=np.float32,
        )
        conditions = np.random.uniform(0, 1, size=(self.num_simulations, 2)).astype(
            np.float32
        )

        for sim in range(self.num_simulations):
            # Initialize with random grain structure (Voronoi-like)
            num_grains = np.random.randint(10, 30)
            centers = np.random.rand(num_grains, 2) * np.array([H, W])

            # Create initial grain structure
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            coords = np.stack([yy, xx], axis=-1)

            distances = np.zeros((H, W, num_grains))
            for g, center in enumerate(centers):
                distances[:, :, g] = np.sqrt(
                    (coords[:, :, 0] - center[0]) ** 2
                    + (coords[:, :, 1] - center[1]) ** 2
                )

            grain_ids = np.argmin(distances, axis=-1)
            initial = (grain_ids / num_grains).astype(np.float32)

            # Apply condition-dependent evolution rate
            rate = 0.02 * (1 + conditions[sim, 0])  # Temperature effect

            # Evolve over time (simplified grain coarsening)
            current = initial.copy()
            for t in range(self.time_steps):
                # Smooth (simulate grain boundary motion)
                from scipy.ndimage import gaussian_filter

                smoothed = gaussian_filter(current, sigma=rate * (t + 1) * 0.5)
                # Re-normalize to maintain phase field properties
                smoothed = (smoothed - smoothed.min()) / (
                    smoothed.max() - smoothed.min() + 1e-8
                )
                data[sim, t, 0] = smoothed
                current = smoothed

        return data, conditions

    def _create_sequences(self) -> List[Tuple[int, int]]:
        """Create sequence indices."""
        sequences = []
        min_frames = self.sequence_length + self.prediction_horizon

        for sim_idx in range(self.num_simulations):
            for start in range(self.time_steps - min_frames + 1):
                sequences.append((sim_idx, start))

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sim_idx, start = self.sequences[idx]
        end = start + self.sequence_length

        input_seq = torch.from_numpy(self.data[sim_idx, start:end])
        target = torch.from_numpy(
            self.data[sim_idx, end : end + self.prediction_horizon]
        )

        if self.prediction_horizon == 1:
            target = target.squeeze(0)

        return {
            "input_seq": input_seq,
            "target": target,
            "conditions": torch.from_numpy(self.conditions[sim_idx]),
        }


def create_dataloaders(
    data_path: Union[str, Path],
    batch_size: int = 16,
    sequence_length: int = 10,
    prediction_horizon: int = 1,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    use_synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders.

    Args:
        data_path: Path to data
        batch_size: Batch size
        sequence_length: Input sequence length
        prediction_horizon: Prediction horizon
        train_split: Training data fraction
        val_split: Validation data fraction
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        seed: Random seed
        use_synthetic: Use synthetic data for testing

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    if use_synthetic:
        dataset = SyntheticPhaseFieldDataset(
            num_simulations=100,
            time_steps=50,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            seed=seed,
        )
    else:
        dataset = PhaseFieldDataset(
            data_path=data_path,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            normalize=True,
        )

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"\nDataLoaders created:")
    print(f"  - Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  - Val: {len(val_dataset)} samples ({len(val_loader)} batches)")
    print(f"  - Test: {len(test_dataset)} samples ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing synthetic dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_path=".",
        batch_size=4,
        sequence_length=5,
        prediction_horizon=1,
        use_synthetic=True,
    )

    # Verify batch shape
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  - input_seq: {batch['input_seq'].shape}")  # (B, T, C, H, W)
    print(f"  - target: {batch['target'].shape}")  # (B, C, H, W)
    print(f"  - conditions: {batch['conditions'].shape}")  # (B, num_conditions)
