import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor):
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        self.time_embedding = TimeEmbedding(dim_out)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.fc(x)
        alpha = self.time_embedding(t).view(-1, self.dim_out)

        return alpha * x
'''''

class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int
    ):
        super().__init__()
        """
        (TODO) Build a noise estimating network.

        Args:
            dim_in: dimension of input
            dim_out: dimension of output
            dim_hids: dimensions of hidden features
            num_timesteps: number of timesteps
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        self.num_timesteps = num_timesteps

        layers = []
        in_dim = dim_in + 1  
        for h in dim_hids:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, dim_out))
        self.net = nn.Sequential(*layers)
        ######################
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
        """
        
        ######## TODO ########
        # DO NOT change the code outside this part.
        if t.ndim == 0:
            t = torch.full((x.shape[0], 1), t, device=x.device)
        elif t.ndim == 1:
            t = t[:, None]
        t_norm = t.float() / self.num_timesteps
        z = torch.cat([x, t_norm], dim=1)
        return self.net(z)
        ######################
        '''
class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int,
        dim_hids: List[int], num_timesteps: int
    ):
        super().__init__()
        """
        Noise-prediction network modulated by time (ε_θ).

        Args
        ----
        dim_in        : dimension of input x  (2 for the toy data)
        dim_out       : dimension of predicted noise  (2)
        dim_hids      : hidden sizes, e.g. [256, 256, 256]
        num_timesteps : total diffusion steps T
        """

        ######## TODO ########
        # Build a stack of TimeLinear → SiLU layers
        self.layers = nn.ModuleList()
        prev_dim = dim_in
        for h in dim_hids:
            self.layers.append(TimeLinear(prev_dim, h, num_timesteps))
            prev_dim = h

        # Final plain Linear, no time modulation
        self.out = nn.Linear(prev_dim, dim_out)
        ######################

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Predict ε_θ(x_t , t).

        Args
        ----
        x : noisy sample, shape (B, 2)
        t : timestep,      shape (B,) or scalar
        """
        ######## TODO ########
        # Ensure t has shape (B,) expected by TimeLinear / TimeEmbedding
        if t.ndim == 0:
            t = t.repeat(x.size(0))
        elif t.ndim == 1:
            t = t.view(-1)

        h = x
        for tl in self.layers:
            h = tl(h, t)          # Time-modulated linear
            h = F.silu(h)         # smoother than ReLU

        return self.out(h)
        ######################

