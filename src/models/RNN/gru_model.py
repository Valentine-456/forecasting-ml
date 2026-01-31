import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            hidden_size: int = 64, 
            num_layers: int = 2, 
            dropout: float = 0.1
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        h_last = out[:, -1, :]
        y = self.head(h_last)
        return y
