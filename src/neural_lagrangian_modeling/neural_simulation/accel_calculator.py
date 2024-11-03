import torch as T
from torch import Tensor
import torch.nn as nn


def calculate_acceleration_vectorized(
    model: nn.Module,
    masses: Tensor,  # Shape: (batch_size, n_objects)
    q: Tensor,  # Shape: (batch_size, n_objects * n_coordinates)
    q_dot: Tensor,  # Shape: (batch_size, n_objects * n_coordinates)
) -> Tensor:  # Shape: (batch_size, n_objects * n_coordinates)
    """
    Vectorized calculation of acceleration using Euler-Lagrange equation.

    Args:
        model: Neural network that predicts scalar Lagrangian value
        q: Position coordinates tensor
            Shape: (batch_size, n_objects * n_coordinates)
            Example for 3-body in 3D: (batch_size, 9) where each 9 is [x1,y1,z1, x2,y2,z2, x3,y3,z3]
        q_dot: Velocity coordinates tensor
            Shape: (batch_size, n_objects * n_coordinates)
        masses: Masses of the objects
            Shape: (batch_size, n_objects)
            Example for 3-body: (batch_size, 3) where each 3 is [m1, m2, m3]

    Returns:
        q_ddot: Acceleration coordinates tensor
            Shape: (batch_size, n_objects * n_coordinates)
    """
    batch_size = q.shape[0]
    n_dims = q.shape[1]  # n_objects * n_coordinates

    q = q.requires_grad_(True)
    q_dot = q_dot.requires_grad_(True)

    # Combine into state vector and compute Lagrangian
    state: Tensor = T.cat([q, q_dot, masses], dim=1)  # Shape: (batch_size, 2 * n_dims + n_objects)
    L: Tensor = model(state)  # Shape: (batch_size, 1)

    # Rest of the function remains the same
    dL_dq: Tensor = T.stack([
        T.autograd.grad(L[i], q, create_graph=True)[0][i]
        for i in range(batch_size)
    ])  # Shape: (batch_size, n_dims)

    dL_dq_dot: Tensor = T.stack([
        T.autograd.grad(L[i], q_dot, create_graph=True)[0][i]
        for i in range(batch_size)
    ])  # Shape: (batch_size, n_dims)

    basis: Tensor = T.eye(n_dims)  # Shape: (n_dims, n_dims)

    d2L_dq_dot2: Tensor = T.stack([
        T.stack([
            T.autograd.grad(dL_dq_dot[b], q_dot, basis[i], create_graph=True)[0][b]
            for i in range(n_dims)
        ])
        for b in range(batch_size)
    ])  # Shape: (batch_size, n_dims, n_dims)

    d2L_dqdq_dot: Tensor = T.stack([
        T.stack([
            T.autograd.grad(dL_dq[b], q_dot, basis[i], create_graph=True)[0][b]
            for i in range(n_dims)
        ])
        for b in range(batch_size)
    ])  # Shape: (batch_size, n_dims, n_dims)

    q_ddot: Tensor = T.stack([
        T.linalg.solve(
            d2L_dq_dot2[b],  # Shape: (n_dims, n_dims)
            dL_dq[b] - T.matmul(d2L_dqdq_dot[b], q_dot[b])  # Shape: (n_dims,)
        )
        for b in range(batch_size)
    ])  # Shape: (batch_size, n_dims)

    return q_ddot
