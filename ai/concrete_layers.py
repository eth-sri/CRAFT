import torch
import torch.nn as nn

class Bias(nn.Module):
    def __init__(self, in_dim=None, bias=None):
        super().__init__()
        assert in_dim is not None or bias is not None
        in_dim = list(bias.shape) if in_dim is None else in_dim
        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]
        if bias is not None:
            self.bias = bias
        else:
            self.bias = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        return x + self.bias