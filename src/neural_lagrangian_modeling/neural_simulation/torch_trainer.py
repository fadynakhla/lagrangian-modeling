from typing import List, Tuple
import torch.nn as nn
import torch.utils.data as torch_data
import torch as T
import numpy as np
import os
import pathlib
from neural_lagrangian_modeling import datamodels


def train(
    model: nn.Module,
    dataset: torch_data.Dataset,
    optimizer: T.optim.Optimizer,
    batch_size: int = 32,
):
    train_loader = torch_data.DataLoader(dataset, batch_size, shuffle=True)

    criterion = nn.MSELoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            # Move data to the appropriate device
            inputs, targets = inputs.to(model.device), targets.to(model.device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute gradients w.r.t inputs
            loss.backward(retain_graph=True)  # Calculate gradients
            input_grads = inputs.grad  # Access input gradients

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}"
            )


def load_saved_trajectories(
    dir_path: pathlib.Path,
) -> List[Tuple[datamodels.Trajectory, ...]]:

    def get_npz_files(dir_path: pathlib.Path):
        return [f for f in os.listdir(dir_path) if f.endswith(".npz")]

    files: List[str] = get_npz_files(dir_path)

    trajectories = []
    for f in files:
        trajectory = datamodels.load_trajectories(f"{dir_path}/{f}")
        trajectories.append(trajectory)
    return trajectories


def create_torch_dataset(data_dir: pathlib.Path) -> torch_data.Dataset:
    trajectories_list = load_saved_trajectories(data_dir)
    input_list = [
        datamodels.serialize_state_to_inputs(trajectories)
        for trajectories in trajectories_list
    ]
    accelerations_list = [
        datamodels.get_accelerations_from_state(trajectories)
        for trajectories in trajectories_list
    ]
    input_data = np.hstack(input_list, dtype=np.float64)
    output_data = np.hstack(accelerations_list, dtype=np.float64)
    input_tensor = T.tensor(input_data, dtype=T.float64)
    output_tensor = T.tensor(output_data, dtype=T.float64)
    dataset = torch_data.TensorDataset(input_tensor, output_tensor)
    return dataset


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent.parent.parent.parent / "data"
    dataset = create_torch_dataset(data_dir=data_dir)
