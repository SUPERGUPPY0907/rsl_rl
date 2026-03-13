# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.genpo.flow import MLP_L, SinusoidalPosEmb


class LeapfrogFlow(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        a_dim,
        actor_hidden_dim,
        time_dim,
        time_hidden_dim,
        activation,
        N,
        device,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a_dim = a_dim
        self.N = N
        self.dt = 1 / N
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

        layers = [SinusoidalPosEmb(time_dim)]
        in_dim = time_dim
        for hidden_dim in time_hidden_dim:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, time_dim))
        self.time_mlp = nn.Sequential(*layers)

    def _leapfrog(self, observations, state):
        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)

        num_envs = observations.shape[0]
        pos = state[..., : self.a_dim]
        vel = state[..., self.a_dim :]

        for i in range(self.N):
            t_mid = torch.full((num_envs,), (i + 0.5) / self.N, device=self.device)
            t_em = self.time_mlp(t_mid)

            pos_half = pos + 0.5 * self.dt * vel
            accel = self.acc_field(pos_half, observations, t_em)
            vel = vel + self.dt * accel
            pos = pos_half + 0.5 * self.dt * vel


        # for i in range(self.N): 
        #     t = torch.full((num_envs,), i / self.N, device=self.device) 
        #     t_em = self.time_mlp(t) 

        #     accel = self.acc_field(pos, observations, t_em) 
        #     vel_half = vel + 0.5 * self.dt * accel 
        #     pos = pos + self.dt * vel_half 
        #     t_next = torch.full((num_envs,), (i + 1) / self.N, device=self.device) 
        #     t_next_em = self.time_mlp(t_next) 
        #     accel_next = self.acc_field(pos, observations, t_next_em) 
        #     vel = vel_half + 0.5 * self.dt * accel_next

        out = torch.cat([pos, vel], dim=-1)
        return out.squeeze(0) if single else out

    def _leapfrog_inverse(self, observations, state):
        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)

        num_envs = observations.shape[0]
        pos = state[..., : self.a_dim]
        vel = state[..., self.a_dim :]

        for i in reversed(range(self.N)):
            t_mid = torch.full((num_envs,), (i + 0.5) / self.N, device=self.device)
            t_em = self.time_mlp(t_mid)

            pos_half = pos - 0.5 * self.dt * vel
            accel = self.acc_field(pos_half, observations, t_em)
            vel = vel - self.dt * accel
            pos = pos_half - 0.5 * self.dt * vel

        # for i in reversed(range(self.N)):
        #     t_next = torch.full((num_envs,), (i + 1) / self.N, device=self.device)
        #     t_next_em = self.time_mlp(t_next)

    
        #     accel_next = self.acc_field(pos, observations, t_next_em)
        #     vel_half = vel - 0.5 * self.dt * accel_next
        #     pos = pos - self.dt * vel_half
        #     t = torch.full((num_envs,), i / self.N, device=self.device)
        #     t_em = self.time_mlp(t)
        #     accel = self.acc_field(pos, observations, t_em)
        #     vel = vel_half - 0.5 * self.dt * accel

        out = torch.cat([pos, vel], dim=-1)
        return out.squeeze(0) if single else out

    def forward(self, observations):
        num_envs = observations.shape[0]

        state_0 = torch.randn(num_envs, self.a_dim * 2, device=self.device)
        # x_0 = torch.randn(num_envs, self.a_dim, device=self.device)
        # v_0 = torch.randn(num_envs, self.a_dim, device=self.device) 
        # state_0 = torch.cat([x_0, v_0], dim=-1)

        log_probs = self.dist.log_prob(state_0).sum(dim=-1)
        state = self._leapfrog(observations, state_0)

        return state, log_probs

    def inference(self, observations):
        num_envs = observations.shape[0]
        state_0 = torch.randn(num_envs, self.a_dim * 2, device=self.device)
        # x_0 = torch.randn(num_envs, self.a_dim, device=self.device)
        # v_0 = torch.randn(num_envs, self.a_dim, device=self.device)
        # state_0 = torch.cat([x_0, v_0], dim=-1)

        state = self._leapfrog(observations, state_0)

        return state

    def inverse(self, observations, state, jac=False):
        state_0 = self._leapfrog_inverse(observations, state)
        log_probs = self.dist.log_prob(state_0).sum(dim=-1)
        # Leapfrog is volume-preserving, so log_probs remain unchanged even if jac=True.
        return log_probs


class ActorCriticLeapfrog(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        flow_num_steps: int = 5,
        mix_para: float = 0.95,
        std: float = 1.0,
        time_dim: int = 32,
        device = torch.device("cpu"),
        flow_interations: int = 5,
        flow_distill_batch_size: int = 256,
        actor_hidden_dims: list = [256, 256, 256],
        critic_hidden_dims:list = [256, 256, 256],
        time_hidden_dims:list = [256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticLeapfrog.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)

        # self.N = flow_num_steps
        self.distillation_ites = flow_interations
        self.flow_distill_batch_size = flow_distill_batch_size

        self.a_o_dim = num_actor_obs
        self.c_o_dim = num_critic_obs
        self.a_dim = num_actions

        self.device = device
        self.last_log_probs = None
        #### actor input
        mlp_input_dim_a = self.a_o_dim + self.a_dim
        #### critic input
        mlp_input_dim_c = self.c_o_dim

        # Policy
        self.actor = LeapfrogFlow(
            input_dim=mlp_input_dim_a,
            output_dim=self.a_dim,
            a_dim=self.a_dim,
            time_dim=time_dim,
            time_hidden_dim=time_hidden_dims,
            actor_hidden_dim=actor_hidden_dims,
            activation=activation,
            N=flow_num_steps,
            device=device,
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Time Net: {self.actor.time_mlp}")
        print(f"Actor MLP: {self.actor.acc_field}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # entropy
        self.ip_std = std
        self.dist = Normal(torch.zeros(self.a_dim * 2, device=self.device),
                           self.ip_std * torch.ones(self.a_dim * 2, device=self.device))

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        raise NotImplementedError

    @property
    def action_std(self):
        # return self.distribution.stddev
        return torch.tensor(0.1, device=self.device)


    def entropy(self, obs):
        num_envs = obs.shape[0]
        x_sample = torch.randn(num_envs, self.a_dim * 2, device=self.device) * self.ip_std
        probs_Q = self.dist.log_prob(x_sample).sum(dim=-1).exp()
        log_probs_P = self.actor.inverse(obs, x_sample, jac=True)
        probs_P = log_probs_P.exp()
        weights = probs_P / (probs_Q + 1e-8)
        entropy_estimate = -(weights * log_probs_P).sum(dim=0)
        return entropy_estimate

    def update_distribution(self, observations):
        raise NotImplementedError

    def act(self, observations, **kwargs):
        action, log_probs = self.actor(observations)
        self.last_log_probs = log_probs
        return action

    def get_actions_log_prob(self, actions):
        if self.last_log_probs is None:
            raise ValueError("No log_probs stored. Call act() first.")
        return self.last_log_probs

    def act_inference(self, observations):
        actions_mean = self.actor.inference(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True
