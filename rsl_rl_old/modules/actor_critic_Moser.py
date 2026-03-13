# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.moser.moser_policy import MoserFlowPolicy


class ActorCriticMoser(nn.Module):
    """Actor-Critic with Moser Flow policy instead of Gaussian."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        envelope_scale=1.0,
        ode = "rk4",
        flow_num_steps: int = 10,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticMoser.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_fn = resolve_nn_activation(activation)

        mlp_input_dim_c = num_critic_obs

        # Moser Flow Policy
        self.actor = MoserFlowPolicy(
            num_obs=num_actor_obs,
            num_actions=num_actions,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            envelope_scale=envelope_scale,
            ode = ode,
            flow_num_steps = flow_num_steps,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Value function (standard MLP)
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor: Moser Flow Policy: {self.actor.v})")
        print(f"Critic MLP: {self.critic}")

    @staticmethod
    def init_weights(sequential, scales):
        """Initialize weights for sequential model."""
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """Reset hidden states (not used for non-recurrent)."""
        pass

    def forward(self):
        raise NotImplementedError

    def act(self, observations, **kwargs):
        """Sample actions from Moser Flow policy."""
        actions = self.actor.sample(observations)
        return actions
    
    def get_actions_prob(self, actions, observations, create_graph):
        # Compute log probability
        with torch.enable_grad():
            actions_copy = actions.detach().requires_grad_(True)
            obs_copy = observations.detach().requires_grad_(False)
            probs = self.actor.get_prob(actions_copy, obs_copy, create_graph)

        return probs

    def act_inference(self, observations):
        """Get deterministic actions for inference."""
        return self.actor.sample(observations)

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value function."""
        value = self.critic(critic_observations)
        return value

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training.
        """
        super().load_state_dict(state_dict, strict=strict)
        return True
