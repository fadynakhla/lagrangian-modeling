from typing import List, Literal

import torch as T
import torch.nn as nn
import pydantic


class ModelConfig(pydantic.BaseModel):
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    activation: Literal["relu", "leaky_relu", "softmax", "softplus"]


activation_fn_map = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "softmax": nn.Softmax,
    "softplus": nn.Softplus,
}


class Model(nn.Module):

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.input_dim = model_config.input_dim
        self.num_hidden_layers = model_config.num_layers
        self.hidden_dim = model_config.hidden_dim
        self.output_dim = model_config.output_dim

        dim_vector: List[int] = (
            [self.input_dim]
            + [self.hidden_dim for i in range(self.num_hidden_layers)]
            + [self.output_dim]
        )
        self.neural_layers: List[nn.Linear] = [
            nn.Linear(dim_vector[i - 1], dim_vector[i])
            for i in range(1, len(dim_vector))
        ]

        self.activation = activation_fn_map[model_config.activation]

    def forward(self, x: T.Tensor) -> T.Tensor:
        h = x
        for layer in self.neural_layers:
            h = layer(h)
            h = self.activation(h)

        return h


if __name__ == "__main__":
    model_config = ModelConfig(
        input_dim=5, output_dim=1, hidden_dim=2, num_layers=3, activation="softplus"
    )
    model = Model(model_config)
    print((5, 1, 2, 3))
    print(model.neural_layers)
