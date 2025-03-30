import torch.nn as nn


class LloraModule(nn.Module):
    def __init__(self, initial_dim: int, rank: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downscale = nn.Linear(initial_dim, rank)
        self.upscale = nn.Linear(rank, initial_dim)

    def forward(self, x):
        x = self.downscale(x)
        return self.upscale(x)


class AdapterModule(nn.Module):
    def __init__(self, initial_dim: int, rank: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.downscale = nn.Linear(initial_dim, rank)
        self.upscale = nn.Linear(rank, initial_dim)

    def forward(self, x):
        x = self.downscale(x)
        x = nn.LeakyReLU(x)
        return self.upscale(x)


class PrefixModule(nn.Module):
    def __init__(self, initial_dim: int, reduction_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = nn.Linear(initial_dim, initial_dim % reduction_size)

    def forward(self, x):
        return self.embeddings(x)
