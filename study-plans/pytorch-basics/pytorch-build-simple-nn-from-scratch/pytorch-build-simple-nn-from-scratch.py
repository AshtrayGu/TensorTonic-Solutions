import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        self.hidden = nn.Linear(in_features,hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size,out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        h = self.hidden(x)
        h = self.activation(h)
        return self.output(h)
