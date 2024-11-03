import pathlib

import numpy as np
import torch as T
import torch.utils.data as torch_data

from neural_lagrangian_modeling import datamodels


def custom_collate(batch):
    """Custom collate function to properly batch the different-sized tensors."""

    # masses = T.stack([item[0] for item in batch])
    # positions = T.stack([item[1] for item in batch])
    # velocities = T.stack([item[2] for item in batch])
    # accelerations = T.stack([item[3] for item in batch])
    return batch[0], batch[1], batch[2], batch[3]


class ThreeBodyDataset(torch_data.Dataset):
    def __init__(
        self,
        masses: T.Tensor,      # Shape: (N, 3)
        positions: T.Tensor,   # Shape: (N, 6) for 2D or (N, 9) for 3D
        velocities: T.Tensor,  # Shape: (N, 6) for 2D or (N, 9) for 3D
        accelerations: T.Tensor # Shape: (N, 6) for 2D or (N, 9) for 3D
    ):
        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations

    def __len__(self):
        return len(self.masses)

    def __getitem__(self, idx):
        return (
            self.masses[idx],
            self.positions[idx],
            self.velocities[idx],
            self.accelerations[idx]
        )


def create_torch_dataset(data_dir: pathlib.Path) -> ThreeBodyDataset:
    """Create custom dataset from saved trajectories.

    Args:
        data_dir: Directory containing trajectory files

    Returns:
        ThreeBodyDataset containing:
        - masses: tensor of shape (total_timesteps, 3)
        - positions: tensor of shape (total_timesteps, 6) for 2D or (total_timesteps, 9) for 3D
        - velocities: tensor of shape (total_timesteps, 6) for 2D or (total_timesteps, 9) for 3D
        - accelerations: tensor of shape (total_timesteps, 6) for 2D or (total_timesteps, 9) for 3D
    """
    trajectories_list = datamodels.load_saved_trajectories(data_dir)

    # Process each trajectory file
    masses_list = []
    positions_list = []
    velocities_list = []
    accelerations_list = []

    for trajectories in trajectories_list:
        # Get inputs
        masses, positions, velocities = datamodels.serialize_state_to_inputs(trajectories)
        masses_list.append(masses)
        positions_list.append(positions)
        velocities_list.append(velocities)

        # Get outputs
        accelerations_list.append(datamodels.get_accelerations_from_state(trajectories))

    # Stack all trajectories along timestep dimension
    masses_data = np.vstack(masses_list)
    positions_data = np.vstack(positions_list)
    velocities_data = np.vstack(velocities_list)
    accelerations_data = np.vstack(accelerations_list)

    # Convert to torch tensors
    masses_tensor = T.tensor(masses_data, dtype=T.float64)
    positions_tensor = T.tensor(positions_data, dtype=T.float64)
    velocities_tensor = T.tensor(velocities_data, dtype=T.float64)
    accelerations_tensor = T.tensor(accelerations_data, dtype=T.float64)

    return ThreeBodyDataset(
        masses=masses_tensor,
        positions=positions_tensor,
        velocities=velocities_tensor,
        accelerations=accelerations_tensor
    )
