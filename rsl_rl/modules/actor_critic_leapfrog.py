# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.utils import resolve_nn_activation

from .genpo.flow import MLP_L, SinusoidalPosEmb


class LeapfrogFlow(nn.Module):
    """Volume-preserving leapfrog flow over augmented action state [x, v]."""

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
        device: str | torch.device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a_dim = a_dim
        self.n_steps = n_steps
        self.dt = 1.0 / n_steps
        self.device = device

        self.acc_field = MLP_L(
            input_dim=self.input_dim,
            hidden_dim=actor_hidden_dim,
            output_dim=self.output_dim,
            t_dim=time_dim,
            activation=activation,
        )
        self.dist = Normal(
            torch.zeros(self.a_dim * 2, device=device),
            torch.ones(self.a_dim * 2, device=device),
        )

        layers: list[nn.Module] = [SinusoidalPosEmb(time_dim)]
        in_dim = time_dim
        for hidden_dim in time_hidden_dim:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, time_dim))
        self.time_mlp = nn.Sequential(*layers)

    def _leapfrog(self, observations: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)

        num_envs = observations.shape[0]
        pos = state[..., : self.a_dim]
        vel = state[..., self.a_dim :]

        for i in range(self.n_steps):
            t_mid = torch.full((num_envs,), (i + 0.5) / self.n_steps, device=observations.device)
            t_emb = self.time_mlp(t_mid)

            pos_half = pos + 0.5 * self.dt * vel
            accel = self.acc_field(pos_half, observations, t_emb)
            vel = vel + self.dt * accel
            pos = pos_half + 0.5 * self.dt * vel

        out = torch.cat([pos, vel], dim=-1)
        return out.squeeze(0) if single else out

    def _leapfrog_inverse(self, observations: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)

        num_envs = observations.shape[0]
        pos = state[..., : self.a_dim]
        vel = state[..., self.a_dim :]

        for i in reversed(range(self.n_steps)):
            t_mid = torch.full((num_envs,), (i + 0.5) / self.n_steps, device=observations.device)
            t_emb = self.time_mlp(t_mid)

            pos_half = pos - 0.5 * self.dt * vel
            accel = self.acc_field(pos_half, observations, t_emb)
            vel = vel - self.dt * accel
            pos = pos_half - 0.5 * self.dt * vel

        out = torch.cat([pos, vel], dim=-1)
        return out.squeeze(0) if single else out

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = observations.shape[0]
        state_0 = torch.randn(num_envs, self.a_dim * 2, device=observations.device)
        log_probs = self.dist.log_prob(state_0).sum(dim=-1)
        state = self._leapfrog(observations, state_0)
        return state, log_probs

    def inference(self, observations: torch.Tensor) -> torch.Tensor:
        num_envs = observations.shape[0]
        state_0 = torch.randn(num_envs, self.a_dim * 2, device=observations.device)
        return self._leapfrog(observations, state_0)

    def inverse(self, observations: torch.Tensor, state: torch.Tensor, jac: bool = False) -> torch.Tensor:
        state_0 = self._leapfrog_inverse(observations, state)
        # Leapfrog is volume preserving; jacobian determinant is 1.
        return self.dist.log_prob(state_0).sum(dim=-1)


class ActorCriticLeapfrog(nn.Module):
    """Actor-critic with leapfrog flow actor over augmented actions [a, v]."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        flow_num_steps: int = 5,
        std: float = 1.0,
        time_dim: int = 32,
        flow_interations: int = 5,
        flow_distill_batch_size: int = 256,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        time_hidden_dims: tuple[int] | list[int] = [256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticLeapfrog.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticLeapfrog module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticLeapfrog module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.action_dim = num_actions
        self.full_action_dim = num_actions * 2
        self.distillation_ites = flow_interations
        self.flow_distill_batch_size = flow_distill_batch_size

        activation_mod = resolve_nn_activation(activation)

        self.actor = LeapfrogFlow(
            input_dim=num_actor_obs + self.action_dim,
            output_dim=self.action_dim,
            a_dim=self.action_dim,
            time_dim=time_dim,
            time_hidden_dim=time_hidden_dims,
            actor_hidden_dim=actor_hidden_dims,
            activation=activation_mod,
            n_steps=flow_num_steps,
            device=obs.device,
        )

        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Time Net: {self.actor.time_mlp}")
        print(f"Actor MLP: {self.actor.acc_field}")
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
        self.last_log_probs: torch.Tensor | None = None
        self.last_actions: torch.Tensor | None = None

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        if self.last_actions is None:
            raise RuntimeError("No actions sampled yet. Call act() before querying action_mean.")
        return self.last_actions

    @property
    def action_std(self) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std
        return torch.exp(self.log_std)

    @property
    def entropy(self) -> torch.Tensor:
        if self.last_log_probs is None:
            raise RuntimeError("No log probabilities available. Call act() before querying entropy.")
        return -self.last_log_probs

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actions, log_probs = self.actor(actor_obs)
        self.last_actions = actions
        self.last_log_probs = log_probs
        return actions

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.last_log_probs is None:
            raise ValueError("No log_probs stored. Call act() first.")
        return self.last_log_probs

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

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
