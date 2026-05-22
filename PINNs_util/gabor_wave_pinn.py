import math
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


def _as_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


class FourierEmbedding(nn.Module):
    """NeRF-style Fourier features for normalized (x, y, t) coordinates."""

    def __init__(self, input_dim: int = 3, multires: int = 4, include_input: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.multires = int(multires)
        self.include_input = bool(include_input)
        if self.multires > 0:
            freq_bands = 2.0 ** torch.arange(self.multires, dtype=torch.float32)
        else:
            freq_bands = torch.empty(0, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.input_dim * self.multires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x] if self.include_input else []
        sin_parts = []
        cos_parts = []
        for freq in self.freq_bands:
            z = x * freq
            sin_parts.append(torch.sin(z))
            cos_parts.append(torch.cos(z))
        parts.extend(sin_parts)
        parts.extend(cos_parts)
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


class ActivationBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str = "sin"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "gelu":
            return torch.nn.functional.gelu(x)
        return torch.sin(x)


class LegacyFourierEmbedding(nn.Module):
    """Backward-compatible embedding for older local checkpoints."""

    def __init__(self, input_dim: int = 3, multires: int = 4, include_input: bool = True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.multires = int(multires)
        self.include_input = bool(include_input)
        if self.multires > 0:
            freq_bands = 2.0 ** torch.arange(self.multires, dtype=torch.float32)
        else:
            freq_bands = torch.empty(0, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    @property
    def out_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.input_dim * self.multires

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.include_input:
            parts.append(x)
        for freq in self.freq_bands:
            z = x * freq
            parts.append(torch.sin(z))
            parts.append(torch.cos(z))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


class SineBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x))


class SpatialGaborLayer(nn.Module):
    """Spatial Gabor filters adapted from PINNbasedgabor for scalar wavefields."""

    def __init__(
        self,
        out_features: int,
        freq: float = 6.0,
        alpha: float = 6.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.out_features = int(out_features)
        self.freq = float(freq)
        self.delta = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((self.out_features,))
        )
        self.gamma = nn.Parameter(
            torch.distributions.Uniform(0.5, 1.0).sample((self.out_features,))
        )
        self.phi = nn.Parameter(
            torch.distributions.Uniform(-math.pi, math.pi).sample((self.out_features,))
        )
        self.theta = nn.Parameter(
            torch.distributions.Uniform(-math.pi, math.pi).sample((self.out_features,))
        )

    def _rotate_x(self, xy: torch.Tensor) -> torch.Tensor:
        return xy[..., 0:1] * torch.cos(self.theta) + xy[..., 1:2] * torch.sin(self.theta)

    def _rotate_y(self, xy: torch.Tensor) -> torch.Tensor:
        return -xy[..., 0:1] * torch.sin(self.theta) + xy[..., 1:2] * torch.cos(self.theta)

    def forward(self, xy: torch.Tensor, centers: torch.Tensor, velocity_scale=1.0) -> torch.Tensor:
        shifted = xy - centers
        xr = self._rotate_x(shifted)
        yr = self._rotate_y(shifted)
        envelope = torch.exp(-0.5 * (xr ** 2 + self.gamma ** 2 * yr ** 2) * self.delta[None, :] ** 2)
        phase = self.freq * velocity_scale * xr + torch.tanh(self.phi)[None, :] * math.pi
        return torch.cos(phase) * envelope


class GaborWavePINN(nn.Module):
    """Gabor last-layer PINN adapted to p(x, y, t) -> pressure."""

    def __init__(
        self,
        hidden_layers: Sequence[int] = (128, 128, 128, 128, 64),
        gabor_units: Optional[int] = None,
        multires: int = 4,
        freq: float = 6.0,
        alpha: float = 6.0,
        beta: float = 1.0,
        center_hidden: int = 32,
    ):
        super().__init__()
        hidden = _as_list(hidden_layers)
        if not hidden:
            raise ValueError("hidden_layers must contain at least one layer size.")
        self.hidden_layers = hidden
        self.gabor_units = int(gabor_units if gabor_units is not None else hidden[-1])
        self.multires = int(multires)
        self.freq = float(freq)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.center_hidden = int(center_hidden)

        self.embedding = FourierEmbedding(input_dim=3, multires=self.multires, include_input=True)
        layers = []
        in_dim = self.embedding.out_dim
        for width in hidden:
            layers.append(SineBlock(in_dim, width))
            in_dim = width
        self.trunk = nn.Sequential(*layers)
        self.coeff = nn.Linear(hidden[-1], self.gabor_units)
        self.center_net = nn.Sequential(
            nn.Linear(2, self.center_hidden),
            nn.GELU(),
            nn.Linear(self.center_hidden, 2),
            nn.Sigmoid(),
        )
        self.gabor = SpatialGaborLayer(
            out_features=self.gabor_units,
            freq=self.freq,
            alpha=self.alpha,
            beta=self.beta,
        )
        self.bias_head = nn.Linear(hidden[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def get_config(self) -> dict:
        return {
            "hidden_layers": list(self.hidden_layers),
            "gabor_units": self.gabor_units,
            "multires": self.multires,
            "freq": self.freq,
            "alpha": self.alpha,
            "beta": self.beta,
            "center_hidden": self.center_hidden,
        }

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        xy = r[:, :2]
        h = self.trunk(self.embedding(r))
        coeff = self.coeff(h)
        centers = 2.0 * self.center_net(xy) - 1.0
        gabor = self.gabor(xy, centers)
        return torch.sum(coeff * gabor, dim=1, keepdim=True) + self.bias_head(h)


class AbediGaborFunctionLayer(nn.Module):
    """PyTorch adaptation of GaborFunctionLayer from Gabor-Enhanced-PINN."""

    def __init__(self, neurons: int, v0: float = 1.0, omega: float = 6.0, delta: float = 10.0):
        super().__init__()
        self.neurons = int(neurons)
        self.v0 = float(v0)
        self.omega = float(omega)
        self.delta = float(delta)
        self.theta = nn.Parameter(torch.ones(self.neurons) * (-math.pi / 4.0))
        self.trainable_v = nn.Parameter(torch.ones(self.neurons) * self.v0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dx, dz = torch.chunk(inputs, 2, dim=-1)
        v = torch.clamp(self.trainable_v, min=1e-4)
        x_theta = dx * torch.cos(self.theta) + dz * torch.sin(self.theta)
        z_theta = -dx * torch.sin(self.theta) + dz * torch.cos(self.theta)
        distance = x_theta ** 2 + z_theta ** 2
        envelope = torch.exp(-0.5 * distance * (self.delta ** 2))
        phase = self.omega / v * x_theta
        g_real = torch.cos(phase) * envelope
        g_imag = torch.sin(phase) * envelope
        return torch.stack([g_real, g_imag], dim=0)


class CustomScalarLayer3D(nn.Module):
    """Scalar-output analogue of Abedi et al.'s CustomLayer3D."""

    def __init__(self, neurons: int):
        super().__init__()
        self.kernel = nn.Parameter(torch.empty(2, int(neurons), 1))
        nn.init.xavier_normal_(self.kernel, gain=1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("jbn,jnk->bk", inputs, self.kernel)


class GaborEnhancedWavePINN(nn.Module):
    """Gabor-Enhanced-PINN architecture adapted from TensorFlow Helmholtz to p(x,y,t)."""

    def __init__(
        self,
        neurons: int = 128,
        neurons_final: int = 64,
        embed_k: int = 4,
        omega: float = 6.0,
        v0: float = 1.0,
        delta: float = 10.0,
        penultimate_activation: str = "sigmoid",
        legacy_embedding_order: bool = False,
    ):
        super().__init__()
        if neurons_final % 2 != 0:
            raise ValueError("neurons_final must be even because the Gabor layer splits it in half.")
        self.neurons = int(neurons)
        self.neurons_final = int(neurons_final)
        self.embed_k = int(embed_k)
        self.omega = float(omega)
        self.v0 = float(v0)
        self.delta = float(delta)
        self.penultimate_activation = str(penultimate_activation)
        self.legacy_embedding_order = bool(legacy_embedding_order)
        self.gabor_units = self.neurons_final // 2

        embedding_cls = LegacyFourierEmbedding if self.legacy_embedding_order else FourierEmbedding
        self.embedding = embedding_cls(input_dim=3, multires=self.embed_k, include_input=True)
        self.layer_1 = ActivationBlock(self.embedding.out_dim, self.neurons, "sin")
        self.layer_2 = ActivationBlock(self.neurons, self.neurons, "sin")
        self.penultimate = ActivationBlock(
            self.neurons, self.neurons_final, self.penultimate_activation
        )
        self.gabor = AbediGaborFunctionLayer(
            neurons=self.gabor_units,
            v0=self.v0,
            omega=self.omega,
            delta=self.delta,
        )
        self.output = CustomScalarLayer3D(self.gabor_units)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def get_config(self) -> dict:
        return {
            "neurons": self.neurons,
            "neurons_final": self.neurons_final,
            "embed_k": self.embed_k,
            "omega": self.omega,
            "v0": self.v0,
            "delta": self.delta,
            "penultimate_activation": self.penultimate_activation,
            "legacy_embedding_order": self.legacy_embedding_order,
        }

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        h = self.layer_1(self.embedding(r))
        h = self.layer_2(h)
        h = self.penultimate(h)
        return self.output(self.gabor(h))
