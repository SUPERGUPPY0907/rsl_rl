# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from tensordict import TensorDict

from rsl_rl.algorithms import LeapfrogPPO, PPO
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticLeapfrog,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class OnPolicyLeapfrogRunner(OnPolicyRunner):
    """On-policy runner with leapfrog-flow aware algorithm/policy construction."""

    def _construct_algorithm(self, obs: TensorDict) -> PPO | LeapfrogPPO:
        # Resolve RND config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)

        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: ActorCritic | ActorCriticLeapfrog | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO | LeapfrogPPO = alg_class(
            actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Initialize the storage
        if hasattr(alg, "get_storage_action_shape"):
            actions_shape = alg.get_storage_action_shape(self.env.num_actions)
        else:
            actions_shape = [self.env.num_actions]

        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            actions_shape,
        )

        return alg

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        if hasattr(self.alg, "act_inference"):
            return self.alg.act_inference
        return self.alg.policy.act_inference
