import torch
import torch.nn as nn


class SpectralMLP(nn.Module):
    """
    Input:  (B, 8, 16, 16)
    Output: (B, 121, 16, 16)
    MLP per-pixel: R^8 -> R^121 applicato su ogni (h,w).
    """
    def __init__(self, hidden_dim=256, num_layers=3, out_activation=None):
        super().__init__()
        act = nn.ReLU()

        layers = []
        in_dim = 8
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(in_dim, hidden_dim), act]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 121))
        if out_activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif out_activation == "softplus":
            layers.append(nn.Softplus(beta=1.0, threshold=20.0))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 8, 16, 16)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()   # (B,16,16,8)
        x = x.view(B * H * W, 8)                 # (BHW, 8)
        y = self.mlp(x)                          # (BHW, 121)
        y = y.view(B, H, W, 121).permute(0, 3, 1, 2).contiguous()  # (B,121,16,16)
        return y