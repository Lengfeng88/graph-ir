"""
model.py -- the sample input for mini-compiler.

Convention (the interface the frontend currently expects):
    the module must define a get_model() function returning
    (model, example_inputs). example_inputs isn't used for an actual
    forward pass right now, but the interface is kept -- it'll matter
    once we switch to a trace backend that needs a real forward pass
    (e.g. torch.export).
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        score = q @ k.transpose(-2, -1)
        score = torch.softmax(score, dim=-1)
        out = score @ v
        return out


def get_model():
    d_model = 512
    model = Attention(d_model)
    example_inputs = (torch.randn(1, 16, d_model),)
    return model, example_inputs
