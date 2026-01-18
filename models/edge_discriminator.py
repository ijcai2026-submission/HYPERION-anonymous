from typing import Optional
import torch
import torch.nn as nn


class EdgeDiscriminator(nn.Module):
    """
    Adaptive edge discriminator that automatically adjusts its internal MLP
    to match the true embedding dimensionality coming from the encoder.
    Prevents shape mismatches such as (B,256) x (512,256).
    """
    def __init__(self, emb_dim: Optional[int] = None, hidden_dim: Optional[int] = None):
        super().__init__()
        self._hint_emb_dim = emb_dim
        self._hint_hidden = hidden_dim
        self.fc = None  # built lazily

    def _build(self, node_emb_dim: int, device: torch.device):
        input_dim = 2 * node_emb_dim  # concatenated src+dst
        hidden_dim = self._hint_hidden or max(64, input_dim // 2)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_src, h_dst], dim=-1)  # shape [B, 2*D]
        if self.fc is None:
            self._build(h_src.size(-1), h_src.device)
        else:
            # check dynamically if size changed, rebuild if mismatch
            current_in = self.fc[0].in_features
            expected_in = x.size(-1)
            if current_in != expected_in:
                print(f"[WARN] Rebuilding discriminator (in_features {current_in}→{expected_in})")
                self._build(h_src.size(-1), h_src.device)
        return self.fc(x).squeeze(-1)
