# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class MLP(nn.Module):
    """MLP used for the learned vector field."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: tuple[int] | list[int],
        output_dim: int,
        t_dim: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        if len(hidden_dim) == 0:
            raise ValueError("hidden_dim must not be empty.")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim + t_dim, hidden_dim[0], bias=True))
        layers.append(activation)

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=True))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim[-1], output_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x_input: torch.Tensor, observations: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x_input, observations, t], dim=1)
        return self.net(inputs)


class MLP_L(nn.Module):
    """MLP with zero-initialized output layer used by leapfrog acceleration field."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: tuple[int] | list[int],
        output_dim: int,
        t_dim: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        if len(hidden_dim) == 0:
            raise ValueError("hidden_dim must not be empty.")

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim + t_dim, hidden_dim[0], bias=True))
        layers.append(activation)

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=True))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim[-1], output_dim, bias=True))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_input: torch.Tensor, observations: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x_input, observations, t], dim=1)
        return self.net(inputs)


class Flow(nn.Module):
    """Heun-like coupling flow over augmented action state [z, y]."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        a_dim: int,
        actor_hidden_dim: tuple[int] | list[int],
        time_dim: int,
        time_hidden_dim: tuple[int] | list[int],
        activation: nn.Module,
        n_steps: int,
        mix_coeff: float,
        device: str | torch.device,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a_dim = a_dim
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self.mix_coeff = mix_coeff
        self.device = device

        self.vec_field = MLP(
            input_dim=self.input_dim,
            hidden_dim=actor_hidden_dim,
            output_dim=self.output_dim,
            t_dim=time_dim,
            activation=activation,
        )

        time_layers: list[nn.Module] = [SinusoidalPosEmb(time_dim)]
        in_dim = time_dim
        for hidden_dim in time_hidden_dim:
            time_layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            time_layers.append(activation)
            in_dim = hidden_dim
        time_layers.append(nn.Linear(in_dim, time_dim))
        self.time_mlp = nn.Sequential(*time_layers)

    def _standard_gaussian_log_prob(self, latent: torch.Tensor) -> torch.Tensor:
        latent_sq_norm = latent.square().sum(dim=-1)
        return -0.5 * latent_sq_norm - self.a_dim * math.log(2.0 * math.pi)

    def sample_with_latent(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = observations.shape[0]
        latent = torch.randn(num_envs, self.a_dim * 2, device=self.device)
        action_aug = self._heun_method(observations, latent)
        return action_aug, latent

    def _heun_method(self, observations: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        z = x[..., : self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., : self.a_dim]
        y = x[..., self.a_dim :].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim :]

        num_envs = observations.shape[0]
        for i in range(self.n_steps):
            t = torch.full((num_envs,), i / self.n_steps, device=self.device)
            t_emb = self.time_mlp(t)

            z_transformed = self.vec_field(y, observations, t_emb)
            z_in = z + z_transformed * self.dt

            y_transformed = self.vec_field(z_in, observations, t_emb)
            y_in = y + y_transformed * self.dt

            z = self.mix_coeff * z_in + (1.0 - self.mix_coeff) * y_in
            y = self.mix_coeff * y_in + (1.0 - self.mix_coeff) * z

        out = torch.cat([z, y], dim=-1)
        return out.squeeze(0)

    def _heun_method_inverse(self, observations: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        z = x[..., : self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., : self.a_dim]
        y = x[..., self.a_dim :].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim :]

        num_envs = observations.shape[0]
        for i in reversed(range(self.n_steps)):
            t = torch.full((num_envs,), i / self.n_steps, device=self.device)
            t_emb = self.time_mlp(t)

            y_in = (y - (1.0 - self.mix_coeff) * z) / self.mix_coeff
            z_in = (z - (1.0 - self.mix_coeff) * y_in) / self.mix_coeff

            y_transformed = self.vec_field(z_in, observations, t_emb)
            y = y_in - y_transformed * self.dt

            z_transformed = self.vec_field(y, observations, t_emb)
            z = z_in - z_transformed * self.dt

        out = torch.cat([z, y], dim=-1)
        return out.squeeze(0)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action_aug, action_aug_0 = self.sample_with_latent(observations)
        log_probs = self._standard_gaussian_log_prob(action_aug_0)
        return action_aug, log_probs

    def inference(self, observations: torch.Tensor) -> torch.Tensor:
        num_envs = observations.shape[0]
        action_aug_0 = torch.randn(num_envs, self.a_dim * 2, device=self.device)
        # action_aug_0 = torch.zeros(num_envs, self.a_dim * 2, device=self.device)
        return self._heun_method(observations, action_aug_0)

    def inverse(self, observations: torch.Tensor, action_aug: torch.Tensor) -> torch.Tensor:
        action_aug_0 = self._heun_method_inverse(observations, action_aug)
        log_probs = self._standard_gaussian_log_prob(action_aug_0)
        return log_probs

    def inverse_latent(self, observations: torch.Tensor, action_aug: torch.Tensor) -> torch.Tensor:
        return self._heun_method_inverse(observations, action_aug)
