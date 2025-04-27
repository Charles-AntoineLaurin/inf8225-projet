import torch.nn.functional as F
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
        x = F.leaky_relu(x)
        return self.upscale(x)


class PrefixModule(nn.Module):
    def __init__(self, initial_dim: int, reduction_size: int, num_heads: int, head_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction_size = reduction_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embeddings = nn.Linear(initial_dim, reduction_size * num_heads * head_dim //8)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, 768)
        prefix = self.embeddings(x)  # [B, prefix_len * num_heads * head_dim]
        
        batch_size = original_shape[0]
        
        if batch_size == 196 : # hidden states shape shape = [196, 8, 768]
            # Reshape to [B, num_heads, prefix_len, head_dim]
            prefix = prefix.view(batch_size, self.num_heads, self.reduction_size, self.head_dim)
        else : # hidden states shape = [8, 197, 768]
            prefix = prefix.view(self.reduction_size, self.num_heads, original_shape[1] , self.head_dim)

        return prefix

