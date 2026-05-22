import math

import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GaussianFourierFeatures(nn.Module):
    """Fixed Gaussian Fourier feature mapping used by FF-PINNs."""

    def __init__(
        self,
        input_dim=3,
        n_frequencies=128,
        sigma=1.0,
        include_input=True,
        feature_order="sin_cos",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.include_input = include_input
        self.feature_order = feature_order
        B = torch.randn(input_dim, n_frequencies) * sigma
        self.register_buffer("B", B)

    @property
    def out_dim(self):
        dim = 2 * self.n_frequencies
        if self.include_input:
            dim += self.input_dim
        return dim

    def forward(self, x):
        proj = 2.0 * math.pi * (x @ self.B)
        if self.feature_order == "cos_sin":
            features = [torch.cos(proj), torch.sin(proj)]
        else:
            features = [torch.sin(proj), torch.cos(proj)]
        if self.include_input:
            features.append(x)
        return torch.cat(features, dim=-1)


class FFPINN(nn.Module):
    """Gaussian Fourier feature PINN with Swish MLP."""

    def __init__(
        self,
        input_dim=3,
        output_dim=1,
        n_frequencies=128,
        sigma=1.0,
        hidden_dim=128,
        n_hidden_layers=5,
        include_input=True,
        feature_order="sin_cos",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.include_input = include_input
        self.feature_order = feature_order

        self.ff = GaussianFourierFeatures(
            input_dim, n_frequencies, sigma, include_input, feature_order
        )
        layers = []
        in_dim = self.ff.out_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(Swish())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "n_frequencies": self.n_frequencies,
            "sigma": self.sigma,
            "hidden_dim": self.hidden_dim,
            "n_hidden_layers": self.n_hidden_layers,
            "include_input": self.include_input,
            "feature_order": self.feature_order,
        }

    def forward(self, r):
        return self.net(self.ff(r))


class HardInitialFFPINN(nn.Module):
    """FF-PINN with hard embedded initial conditions."""

    INIT_PARAMS = {
        "threelayer": (0.5, 0.5, 0.05),
        "marmousi": (0.5, 0.15, 0.04),
        "overthrust": (0.5, 0.12, 0.035),
    }

    def __init__(self, dataset="threelayer", p_max=1.0, initial_mode="initial_pressure", **ff_config):
        super().__init__()
        dataset = dataset.lower()
        if dataset not in self.INIT_PARAMS:
            raise ValueError(f"Unknown dataset={dataset!r}")
        if initial_mode not in ("initial_pressure", "zero"):
            raise ValueError(f"Unknown initial_mode={initial_mode!r}")
        self.dataset = dataset
        self.p_max = float(p_max) if p_max is not None and float(p_max) > 1e-12 else 1.0
        self.initial_mode = initial_mode
        self.base = FFPINN(**ff_config)

    def get_config(self):
        config = self.base.get_config()
        config.update({
            "dataset": self.dataset,
            "p_max": self.p_max,
            "initial_mode": self.initial_mode,
        })
        return config

    def initial_pressure(self, r):
        x0, y0, std = self.INIT_PARAMS[self.dataset]
        x = r[:, 0:1]
        y = r[:, 1:2]
        p0 = torch.exp(-0.5 * (((x - x0) / std) ** 2 + ((y - y0) / std) ** 2))
        return p0 / self.p_max

    def forward(self, r):
        t = r[:, 2:3]
        if self.initial_mode == "zero":
            return (t ** 2) * self.base(r)
        return self.initial_pressure(r) + (t ** 2) * self.base(r)
