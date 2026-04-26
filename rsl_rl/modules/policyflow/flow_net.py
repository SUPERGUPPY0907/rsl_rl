from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation

from .utils import TIMESTEP_EMBEDDING


def resolve_policyflow_activation(name: str) -> nn.Module:
    if name.lower() == "linear":
        return nn.Identity()
    return resolve_nn_activation(name)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activations: list[str],
        hidden_dims: list[int],
    ) -> None:
        super().__init__()

        dims = [input_size, *hidden_dims, output_size]
        if len(activations) != len(dims) - 1:
            raise ValueError(
                "Expected one activation entry per linear layer, "
                f"got {len(activations)} activations for {len(dims) - 1} layers."
            )

        layers: list[nn.Module] = []
        for layer_idx, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(resolve_policyflow_activation(activations[layer_idx]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FlowNetBase(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if timestep_emb_type not in TIMESTEP_EMBEDDING:
            raise ValueError(f"Unsupported timestep embedding type: {timestep_emb_type}.")
        self.map_noise = TIMESTEP_EMBEDDING[timestep_emb_type](emb_dim, **(timestep_emb_params or {}))

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class FlowMlp(FlowNetBase):
    def __init__(
        self,
        x_dim: int,
        emb_dim: int = 16,
        activations: list[str] = ["relu", "relu", "relu", "linear"],
        hidden_dims: list[int] = [256, 256, 256],
        timestep_emb_type: str = "positional",
        timestep_emb_params: Optional[dict] = None,
    ) -> None:
        super().__init__(emb_dim, timestep_emb_type, timestep_emb_params)
        self.mlp = FeedForwardNetwork(
            input_size=x_dim + emb_dim,
            output_size=x_dim,
            activations=activations,
            hidden_dims=hidden_dims,
        )

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = self.map_noise(noise)
        if condition is not None:
            t = t + condition
        return self.mlp(torch.cat([x, t], dim=-1))


class ConditionNetBase(nn.Module):
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityCondition(ConditionNetBase):
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return condition


class ConditionLinearLayer(ConditionNetBase):
    def __init__(self, cond_dim: int, emb_dim: int = 16) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, emb_dim)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.linear(condition)


class ConditionMlp(ConditionNetBase):
    def __init__(
        self,
        cond_dim: int,
        emb_dim: int = 16,
        activations: list[str] = ["elu", "elu", "elu", "linear"],
        hidden_dims: list[int] = [256, 256, 256],
    ) -> None:
        super().__init__()
        self.mlp = FeedForwardNetwork(cond_dim, emb_dim, activations=activations, hidden_dims=hidden_dims)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.mlp(condition)


class LearnableVariance(nn.Module):
    def __init__(
        self,
        dims: int,
        log_std_max: float = 4.0,
        log_std_min: float = -20.0,
        std_init: float = 1.0,
    ) -> None:
        super().__init__()
        self._log_std_max = log_std_max
        self._log_std_min = log_std_min
        self._log_std = nn.Parameter(torch.ones(dims) * np.log(std_init))

    @property
    def std(self) -> torch.Tensor:
        return self._log_std.clamp(self._log_std_min, self._log_std_max).exp()
