import copy

import torch
import torch.nn as nn


class WaveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        return x + self.ff(x2)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        return x + self.ff(x2)


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, heads):
        super().__init__()
        self.n_layers = int(n_layers)
        self.layers = get_clones(EncoderLayer(d_model, heads), self.n_layers)
        self.act = WaveAct()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, heads):
        super().__init__()
        self.n_layers = int(n_layers)
        self.layers = get_clones(DecoderLayer(d_model, heads), self.n_layers)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for layer in self.layers:
            x = layer(x, e_outputs)
        return self.act(x)


class PINNsFormerWave2D(nn.Module):
    """PINNsFormer adapted from official demos to coordinates (x, y, t)."""

    def __init__(
        self,
        d_out=1,
        d_model=32,
        d_hidden=512,
        n_layers=1,
        heads=2,
        num_step=5,
        step_size=1e-4,
    ):
        super().__init__()
        self.d_out = int(d_out)
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.n_layers = int(n_layers)
        self.heads = int(heads)
        self.num_step = int(num_step)
        self.step_size = float(step_size)

        self.linear_emb = nn.Linear(3, self.d_model)
        self.encoder = Encoder(self.d_model, self.n_layers, self.heads)
        self.decoder = Decoder(self.d_model, self.n_layers, self.heads)
        self.linear_out = nn.Sequential(
            nn.Linear(self.d_model, self.d_hidden),
            WaveAct(),
            nn.Linear(self.d_hidden, self.d_hidden),
            WaveAct(),
            nn.Linear(self.d_hidden, self.d_out),
        )

    def get_config(self):
        return {
            "d_out": self.d_out,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "n_layers": self.n_layers,
            "heads": self.heads,
            "num_step": self.num_step,
            "step_size": self.step_size,
        }

    def make_sequence(self, r):
        seq = r.unsqueeze(1).repeat(1, self.num_step, 1)
        offsets = torch.arange(self.num_step, device=r.device, dtype=r.dtype).view(1, -1, 1)
        seq[:, :, 2:3] = seq[:, :, 2:3] + offsets * self.step_size
        return seq

    def forward_sequence(self, r_seq):
        src = self.linear_emb(r_seq)
        e_outputs = self.encoder(src)
        d_output = self.decoder(src, e_outputs)
        return self.linear_out(d_output)

    def forward(self, r):
        return self.forward_sequence(self.make_sequence(r))[:, 0, :]
