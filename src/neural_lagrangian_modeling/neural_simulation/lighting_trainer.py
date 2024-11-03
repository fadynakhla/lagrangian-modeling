import pydantic
import lightning as L
from neural_lagrangian_modeling.neural_simulation import vanilla_neural_network
import torch.nn as nn
import torch as T
from torch.utils.data import Dataset, DataLoader

# class LightningConfig(pydantic.BaseModel):
#     num_iter: NotImplementedError

# class TrainingConfig(pydantic.BaseModel):

#     lightning_config: LightningConfig
#     model_config: vanilla_neural_network.ModelConfig

# Example dataset
class YourDataset(Dataset):
    def __init__(self):
        # initialize data here
        self.data = T.randn(100, 10)  # Example input
        self.targets = T.randn(100, 1)  # Example target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class LitModel(L.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        # self.model = model
        self.model = nn.Linear(10, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x:T.Tensor)-> T.Tensor:
        return self.model(x)
    
    def training_step(self, batch: T.Tensor, batch_idx: int) -> T.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return T.optim.Adam(self.parameters(), lr=1e-3)

def train():

    # model_config = vanilla_neural_network.ModelConfig(input_dim=1, output_dim=1, hidden_dim=1, num_layers=1, activation="softplus")
    # model = vanilla_neural_network.Model(model_config=model_config)
    
    # lightning_config = LightningConfig
    # trainer = L.Trainer()
    dataset = YourDataset()

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = LitModel()
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == "__main__":
    # trainer_config = TrainerConfig(num_iter=1)
    train()
