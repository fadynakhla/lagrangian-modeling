import os
import pathlib
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from torch.utils.tensorboard import SummaryWriter

from neural_lagrangian_modeling import datamodels
from neural_lagrangian_modeling.neural_simulation import three_body_dataset, vanilla_neural_network, accel_calculator


def train(
    model: nn.Module,
    dataset: torch_data.Dataset,
    optimizer: T.optim.Optimizer,
    batch_size: int = 32,
    num_epochs: int = 10,
    log_dir: pathlib.Path = "./iter.log",
):
    train_loader = torch_data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=three_body_dataset.custom_collate)

    criterion = nn.MSELoss()

    # Create a SummaryWriter to log metrics
    writer = SummaryWriter(log_dir)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for masses, positions, velocities, targets in train_loader:
            # Move data to the appropriate device
            masses = masses.to(model.device)
            positions = positions.to(model.device)
            velocities = velocities.to(model.device)
            targets = targets.to(model.device)

            # Forward pass
            outputs = accel_calculator.calculate_acceleration_vectorized(
                model=model, masses=masses, q=positions, q_dot=velocities
            )
            loss = criterion(outputs, targets)

            # Compute gradients w.r.t inputs
            # loss.backward(retain_graph=True)  # Calculate gradients
            # input_grads = inputs.grad  # Access input gradients

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

        # Log average loss for the epoch
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}"
        )
    writer.close()




# def create_torch_dataset(data_dir: pathlib.Path) -> torch_data.Dataset:
#     trajectories_list = load_saved_trajectories(data_dir)
#     input_list = [
#         datamodels.serialize_state_to_inputs(trajectories)
#         for trajectories in trajectories_list
#     ]
#     accelerations_list = [
#         datamodels.get_accelerations_from_state(trajectories)
#         for trajectories in trajectories_list
#     ]
#     input_data = np.hstack(input_list, dtype=np.float64)
#     output_data = np.hstack(accelerations_list, dtype=np.float64)
#     input_tensor = T.tensor(input_data, dtype=T.float64)
#     output_tensor = T.tensor(output_data, dtype=T.float64)
#     return torch_data.TensorDataset(input_tensor, output_tensor)


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent.parent.parent.parent / "data"
    dataset = three_body_dataset.create_torch_dataset(data_dir=data_dir)[:1]
    model_config = vanilla_neural_network.ModelConfig(input_dim=15, output_dim=1, hidden_dim=128, num_layers=4, activation="softplus")
    # device = "cuda:0" if T.cuda.is_available() else "cpu"
    model = vanilla_neural_network.Model(model_config)
    model.to(model.device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    log_dir = model_dir = pathlib.Path(__file__).parent.parent.parent / "logs"

    train(model=model, dataset=dataset, optimizer=optimizer, num_epochs=10, log_dir=log_dir)

    model_dir = pathlib.Path(__file__).parent.parent.parent / "model"

    # Define a path where you want to save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f'{timestamp}.pth'  # You can change the name as needed

    # Save the model state dictionary
    T.save(model.state_dict(), model_path)
