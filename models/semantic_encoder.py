import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 256, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)