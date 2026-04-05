# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any, NoReturn

from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.utils import resolve_nn_activation

from .genpo.flow import BELMFlow


class ActorCriticBELMGenPO(nn.Module):
    """Actor-critic module with a BELM-style actor over augmented actions [x0, x1]."""

    is_recurrent: bool = False
    diagnostic_names = (
        "a_term_norm",
        "b_term_norm",
        "eps_term_norm",
        "x_prev_norm",
        "latent_norm",
        "dummy_gap_norm",
    )

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        flow_num_steps: int = 5,
        a_coeff: float | None = None,
        b_coeff: float | None = None,
        eps_coeff: float | None = None,
        lag_coeff: float | None = None,
        mix_para: float | None = None,
        std: float = 1.0,
        time_dim: int = 32,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        time_hidden_dims: tuple[int] | list[int] = [256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        device: str | torch.device = torch.device("cpu"),
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticBELMGenPO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticBELMGenPO module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticBELMGenPO module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.action_dim = num_actions
        self.full_action_dim = num_actions * 2
        self.device = device

        activation_mod = resolve_nn_activation(activation)
        resolved_a_coeff, resolved_b_coeff, resolved_eps_coeff = BELMFlow.resolve_coefficients(
            a_coeff=a_coeff,
            b_coeff=b_coeff,
            eps_coeff=eps_coeff,
            lag_coeff=lag_coeff,
            mix_para=mix_para,
        )
        self._a_coeff = resolved_a_coeff
        self._b_coeff = resolved_b_coeff
        self._eps_coeff = resolved_eps_coeff

        self.actor = BELMFlow(
            input_dim=num_actor_obs + self.action_dim,
            output_dim=self.action_dim,
            a_dim=self.action_dim,
            actor_hidden_dim=actor_hidden_dims,
            time_dim=time_dim,
            time_hidden_dim=time_hidden_dims,
            activation=activation_mod,
            n_steps=flow_num_steps,
            a_coeff=resolved_a_coeff,
            b_coeff=resolved_b_coeff,
            eps_coeff=resolved_eps_coeff,
            device=self.device,
        )

        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Time Net: {self.actor.time_mlp}")
        print(f"Actor MLP: {self.actor.vec_field}")
        print(f"Critic MLP: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(self.full_action_dim))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.full_action_dim)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.ip_std = std
        self.last_latents: torch.Tensor | None = None
        self._track_inverse_diagnostics = False
        self._diagnostic_updates = 0
        self._diagnostic_totals = {name: 0.0 for name in self.diagnostic_names}

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_std(self) -> NoReturn:
        raise NotImplementedError

    @property
    def entropy(self) -> NoReturn:
        raise NotImplementedError

    @property
    def a_coeff(self) -> float:
        return self._a_coeff

    @property
    def b_coeff(self) -> float:
        return self._b_coeff

    @property
    def eps_coeff(self) -> float:
        return self._eps_coeff

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actions, latents = self.actor.sample_with_latent(actor_obs)
        self.last_latents = latents
        return actions

    def inverse(self, actions: torch.Tensor, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        log_probs = self.actor.inverse(actor_obs, actions)
        return log_probs

    def inverse_latent(self, actions: torch.Tensor, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        latent = self.actor.inverse_latent(
            actor_obs,
            actions,
            track_diagnostics=self._track_inverse_diagnostics,
        )
        if self._track_inverse_diagnostics and self.actor.last_inverse_diagnostics is not None:
            for name, value in self.actor.last_inverse_diagnostics.items():
                self._diagnostic_totals[name] += float(value.item())
            self._diagnostic_updates += 1
        return latent

    def forward_latent(self, latent: torch.Tensor, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor.forward_from_latent(actor_obs, latent)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.last_latents is None:
            raise ValueError("No latents stored. Call act() first.")
        return self.actor._standard_gaussian_log_prob(self.last_latents)

    def get_actions_latent(self, actions: torch.Tensor) -> torch.Tensor:
        if self.last_latents is None:
            raise ValueError("No latents stored. Call act() first.")
        return self.last_latents

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor.inference(actor_obs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def begin_diagnostic_accumulation(self) -> None:
        self._track_inverse_diagnostics = True
        self._diagnostic_updates = 0
        self._diagnostic_totals = {name: 0.0 for name in self.diagnostic_names}

    def end_diagnostic_accumulation(self) -> dict[str, float]:
        self._track_inverse_diagnostics = False
        num_updates = max(1, self._diagnostic_updates)
        diagnostics = {
            name: self._diagnostic_totals[name] / num_updates
            for name in self.diagnostic_names
        }
        self._diagnostic_updates = 0
        self._diagnostic_totals = {name: 0.0 for name in self.diagnostic_names}
        return diagnostics

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        self._a_coeff = self.actor.a_coeff
        self._b_coeff = self.actor.b_coeff
        self._eps_coeff = self.actor.eps_coeff
        return True
