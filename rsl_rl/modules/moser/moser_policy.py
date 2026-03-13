# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Moser flow policy for reinforcement learning."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import distributions
from typing import Callable


def parse_activation(activation_name: str) -> nn.Module:
    activations = {
        "tanh": nn.Tanh(),
        "softplus": nn.Softplus(),
        "softplus100": nn.Softplus(100),
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "mish": nn.Mish(),
    }
    return activations.get(activation_name, nn.ELU())


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int] | list[int],
    output_dim: int,
    activation: nn.Module,
    last_activation: nn.Module | None = None,
    add_batchnorm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), activation]

    if add_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(activation)
        if add_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if last_activation is not None:
        layers.append(last_activation)

    return nn.Sequential(*layers)


class GaussianPrior(distributions.MultivariateNormal):
    def __init__(self, dim: int, device: str | torch.device, std: float) -> None:
        super().__init__(torch.zeros(dim, device=device), std * torch.eye(dim, device=device))


class MoserFlowPolicy(nn.Module):
    """Moser flow based policy over Euclidean action space."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "mish",
        envelope_scale: float = 1.0,
        ode: str = "rk4",
        flow_num_steps: int = 10,
        device: str | torch.device = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.device = device
        self.eps = 1e-7
        self.envelope_scale = envelope_scale
        self.ode = ode
        self.flow_num_steps = flow_num_steps

        self.prior = GaussianPrior(num_actions, device, std=1.0)

        self.v = build_mlp(
            input_dim=num_actions + num_obs,
            hidden_dims=hidden_dims,
            output_dim=num_actions,
            activation=parse_activation(activation),
            last_activation=None,
            add_batchnorm=False,
        )

        self.initialize_weights()
        self.to(device)

    def initialize_weights(self) -> None:
        def init_orthogonal_hidden(layer: nn.Module) -> None:
            if isinstance(layer, nn.Linear):
                gain = nn.init.calculate_gain("relu")
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.v.apply(init_orthogonal_hidden)

        last_linear_layer = None
        for layer in reversed(self.v):
            if isinstance(layer, nn.Linear):
                last_linear_layer = layer
                break

        if last_linear_layer is not None:
            with torch.no_grad():
                nn.init.orthogonal_(last_linear_layer.weight, gain=0.0)
                if last_linear_layer.bias is not None:
                    nn.init.zeros_(last_linear_layer.bias)

    def envelope(self, actions: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.linalg.norm(actions, ord=2, dim=-1, keepdim=True)
        return torch.exp(-self.envelope_scale * l2_norm)

    def u_theta(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([actions, obs], dim=1)
        return self.v(combined)

    def u(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.envelope(actions) * self.u_theta(actions, obs)

    def nu(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.prior.log_prob(actions)).unsqueeze(-1)

    def divergence_u(self, actions: torch.Tensor, obs: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        if not actions.requires_grad:
            actions = actions.requires_grad_(True)

        u_val = self.u(actions, obs)
        div = torch.zeros(actions.shape[0], device=actions.device)

        for i in range(self.num_actions):
            grad_i = torch.autograd.grad(
                u_val[:, i].sum(),
                actions,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            div += grad_i[:, i]

        return div

    def signed_mu(self, actions: torch.Tensor, obs: torch.Tensor, create_graph: bool) -> torch.Tensor:
        nu_val = self.nu(actions)
        div_u = self.divergence_u(actions, obs, create_graph)
        return (nu_val.squeeze(-1) - div_u).unsqueeze(-1)

    def mu_plus(self, actions: torch.Tensor, obs: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        signed = self.signed_mu(actions, obs, create_graph)
        return torch.relu(signed) + self.eps

    def density(self, actions: torch.Tensor, obs: torch.Tensor, create_graph: bool) -> torch.Tensor:
        return self.signed_mu(actions, obs, create_graph)

    def ode_func(self, t: torch.Tensor, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        denominator = (1.0 - t) * self.nu(x).view(-1, 1) + t * self.mu_plus(x, obs)
        return self.u(x, obs) / (denominator + self.eps)

    def odeint(
        self,
        func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t: torch.Tensor,
        method: str = "rk4",
    ) -> torch.Tensor:
        dt = 1.0 / max(len(t), 1)
        x = x0
        for i in range(self.flow_num_steps):
            t_curr = t[i]
            if method == "euler":
                x = x + dt * func(t_curr, x)
            elif method == "rk4":
                k1 = func(t_curr, x)
                k2 = func(t_curr + dt / 2, x + dt * k1 / 2)
                k3 = func(t_curr + dt / 2, x + dt * k2 / 2)
                k4 = func(t_curr + dt, x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown ODE method: {method}")
        return x

    def transport(self, x: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, steps=self.flow_num_steps + 1, device=x.device)[:-1]
        return self.odeint(lambda tt, xx: self.ode_func(tt, xx, obs), x, t, self.ode)

    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        num_samples = obs.shape[0]
        random_samples = self.prior.sample((num_samples,)).to(obs.device)
        random_samples.requires_grad = True
        return self.transport(random_samples, obs)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return -self.log_density(actions, obs)

    def log_density(self, actions: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        density = self.mu_plus(actions, obs, create_graph=False)
        return torch.log(density + self.eps).squeeze(-1)

    def get_prob(self, actions: torch.Tensor, obs: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
        if not actions.requires_grad:
            actions = actions.detach().requires_grad_(True)
        return self.density(actions, obs, create_graph)
