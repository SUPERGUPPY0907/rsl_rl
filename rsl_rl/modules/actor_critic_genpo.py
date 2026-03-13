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

from .genpo.flow import Flow


class ActorCriticGenPO(nn.Module):
    """Actor-critic module with a flow-based actor over augmented actions [a, v]."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        flow_num_steps: int = 5,
        mix_para: float = 0.95,
        std: float = 1.0,
        time_dim: int = 32,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        time_hidden_dims: tuple[int] | list[int] = [256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        device = torch.device("cpu"),
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticFlow.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticFlow module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticFlow module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.action_dim = num_actions
        self.full_action_dim = num_actions * 2
        self. device= device

        activation_mod = resolve_nn_activation(activation)

        self.actor = Flow(
            input_dim=num_actor_obs + self.action_dim,
            output_dim=self.action_dim,
            a_dim=self.action_dim,
            actor_hidden_dim=actor_hidden_dims,
            time_dim=time_dim,
            time_hidden_dim=time_hidden_dims,
            activation=activation_mod,
            n_steps=flow_num_steps,
            mix_coeff=mix_para,
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
        self.last_log_probs: torch.Tensor | None = None

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

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actions, log_probs = self.actor(actor_obs, jac=False)
        self.last_log_probs = log_probs
        return actions
    
    def inverse(self, actions: torch.tensor, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        log_probs = self.actor.inverse(actor_obs, actions, jac=False)
        return log_probs

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
