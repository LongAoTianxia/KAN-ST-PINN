import torch
import torch.nn as nn


class TDSinePINN(nn.Module):
    """Task-decomposed PINN sine MLP from Zou et al. TD-PINN code."""

    def __init__(self, layers=None, lb=(0.0, 0.0, 0.0), ub=(1.0, 1.0, 1.0)):
        super().__init__()
        if layers is None:
            layers = [3] + 5 * [64] + 3 * [32] + [1]
        self.layers = [int(v) for v in layers]
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32).view(1, -1))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32).view(1, -1))

        modules = []
        for in_dim, out_dim in zip(self.layers[:-1], self.layers[1:]):
            layer = nn.Linear(in_dim, out_dim)
            nn.init.normal_(layer.weight, mean=0.0, std=(2.0 / (in_dim + out_dim)) ** 0.5)
            nn.init.zeros_(layer.bias)
            modules.append(layer)
        self.linears = nn.ModuleList(modules)

    def get_config(self):
        return {
            "layers": list(self.layers),
            "lb": self.lb.detach().cpu().reshape(-1).tolist(),
            "ub": self.ub.detach().cpu().reshape(-1).tolist(),
        }

    def forward(self, r):
        h = 2.0 * (r - self.lb) / (self.ub - self.lb + 1e-12) - 1.0
        for layer in self.linears[:-1]:
            h = torch.sin(layer(h))
        return self.linears[-1](h)
