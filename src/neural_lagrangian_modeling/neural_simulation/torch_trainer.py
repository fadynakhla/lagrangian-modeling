
import torch.nn as nn
import torch.utils.data as torch_data
import torch as T
import numpy as np

from neural_lagrangian_modeling import datamodels

def calculate_accelerations(model: nn.Module, inputs: T.Tensor) -> T.Tensor:
    positions = []
    velocities = []
    L = model(inputs)
    # dl_dq_dot = 

def train(model: nn.Module, dataset: torch_data.Dataset, optimizer: T.optim.Optimizer, batch_size: int = 32):
    train_loader = torch_data.DataLoader(dataset, batch_size, shuffle=True)

    criterion = nn.MSELoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs,targets in train_loader:
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

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}")


def create_dummy_massive_body(num_bodies: int):
    bodies = []
    for _ in range(num_bodies):
        mass = np.random.uniform(1.0, 10.0)  # Mass between 1 and 10 units
        position = np.random.uniform(-10.0, 10.0, size=3).astype(np.float128)  # Random 3D position
        velocity = np.random.uniform(-5.0, 5.0, size=3).astype(np.float128)  # Random 3D velocity
        body = datamodels.MassiveBody(mass=mass, position=position, velocity=velocity)
        bodies.append(body)
    return bodies

if __name__ == "__main__":
    bodies = create_dummy_massive_body(3)
    print(bodies)
