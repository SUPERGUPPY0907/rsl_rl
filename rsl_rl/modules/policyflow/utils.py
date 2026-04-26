from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn


def at_least_ndim(
    x: Union[np.ndarray, torch.Tensor, int, float],
    ndim: int,
    pad: int = 0,
) -> Union[np.ndarray, torch.Tensor, int, float]:
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        return x
    if isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        return x
    if isinstance(x, (int, float)):
        return x
    raise ValueError(f"Unsupported type {type(x)}.")


def uniform_sampling_step_schedule_continuous(
    trange: list[float] | tuple[float, float] | None = None,
    sampling_steps: int = 10,
) -> torch.Tensor:
    if trange is None:
        trange = [1e-3, 1.0]
    return torch.linspace(trange[0], trange[1], sampling_steps + 1, dtype=torch.float32)


def quad_sampling_step_schedule_continuous(
    trange: list[float] | tuple[float, float] | None = None,
    sampling_steps: int = 10,
    n: float = 1.5,
) -> torch.Tensor:
    if trange is None:
        trange = [1e-3, 1.0]
    schedule = (trange[1] - trange[0]) * (
        torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32) ** n
    ) + trange[0]
    return schedule


def cat_cos_sampling_step_schedule_continuous(
    trange: list[float] | tuple[float, float] | None = None,
    sampling_steps: int = 10,
    n: float = 2.0,
) -> torch.Tensor:
    if trange is None:
        trange = [1e-3, 1.0]
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = 0.5 * (2 * (idx > 0.5) - 1) * torch.sin(np.pi * torch.abs(idx - 0.5)) ** (1 / n) + 0.5
    return (trange[1] - trange[0]) * idx + trange[0]


def quad_cos_sampling_step_schedule_continuous(
    trange: list[float] | tuple[float, float] | None = None,
    sampling_steps: int = 10,
    n: float = 2.0,
) -> torch.Tensor:
    if trange is None:
        trange = [1e-3, 1.0]
    idx = torch.linspace(0, 1, sampling_steps + 1, dtype=torch.float32)
    idx = ((torch.sin(np.pi * (idx - 0.5)) + 1) / 2) ** n
    return (trange[1] - trange[0]) * idx + trange[0]


SAMPLING_STEP_SCHEDULE = {
    "uniform_continuous": uniform_sampling_step_schedule_continuous,
    "quad_continuous": quad_sampling_step_schedule_continuous,
    "cat_cos_continuous": cat_cos_sampling_step_schedule_continuous,
    "quad_cos_continuous": quad_cos_sampling_step_schedule_continuous,
}


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        emb = torch.einsum("i,j->ij", x, freqs.to(x.dtype))
        return torch.nn.functional.pad(
            torch.cat([emb.cos(), emb.sin()], dim=1),
            pad=[0, self.dim - freqs.shape[-1] * 2],
        )


class UntrainablePositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000, endpoint: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(start=0, end=self.dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.dim // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        emb = torch.einsum("...i,j->...ij", x, freqs.to(x.dtype))
        return torch.cat([emb.cos(), emb.sin()], dim=1)


class FourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 8) * scale)
        self.mlp = nn.Sequential(
            nn.Linear(2 * (dim // 8), dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("...i,j->...ij", x, (2 * np.pi * self.freqs).to(x.dtype))
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
        return self.mlp(emb)


class UntrainableFourierEmbedding(nn.Module):
    def __init__(self, dim: int, scale: float = 16.0) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(dim // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("...i,j->...ij", x, (2 * np.pi * self.freqs).to(x.dtype))
        return torch.cat([emb.cos(), emb.sin()], dim=-1)


TIMESTEP_EMBEDDING = {
    "positional": PositionalEmbedding,
    "fourier": FourierEmbedding,
    "untrainable_fourier": UntrainableFourierEmbedding,
    "untrainable_positional": UntrainablePositionalEmbedding,
}
