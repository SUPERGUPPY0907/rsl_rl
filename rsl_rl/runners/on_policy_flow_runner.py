# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import statistics
import time
import torch
import warnings
from tensordict import TensorDict

from rsl_rl.algorithms import BELMGenPO, FPO, GenPO, GenPOPFClip, GenPOPlusPlus, GenPOU0Clip, PPO, SGenPO
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticBELMGenPO,
    ActorCriticFPO,
    ActorCriticGenPO,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class OnPolicyFlowRunner(OnPolicyRunner):
    """On-policy runner with flow-aware algorithm/policy construction."""

    _alg_registry = {
        "BELMGenPO": BELMGenPO,
        "FPO": FPO,
        "GenPO": GenPO,
        "GenPOPFClip": GenPOPFClip,
        "SGenPO": SGenPO,
        "GenPOPlusPlus": GenPOPlusPlus,
        "GenPOU0Clip": GenPOU0Clip,
        "GenPO++": GenPOPlusPlus,
        "genpo++": GenPOPlusPlus,
        "FPO++": FPO,
        "fpo++": FPO,
    }

    def _construct_algorithm(
        self, obs: TensorDict
    ) -> BELMGenPO | FPO | GenPO | GenPOPFClip | GenPOU0Clip | SGenPO | GenPOPlusPlus:
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
        actor_critic: ActorCritic | ActorCriticBELMGenPO | ActorCriticFPO | ActorCriticGenPO | ActorCriticRecurrent = (
            actor_critic_class(
                obs,
                self.cfg["obs_groups"],
                self.env.num_actions,
                device=self.device,
                **self.policy_cfg,
            ).to(self.device)
        )

        # Initialize the algorithm
        alg_name = self.alg_cfg.pop("class_name")
        alg_class = self._alg_registry.get(alg_name)
        if alg_class is None:
            alg_class = eval(alg_name)

        alg: BELMGenPO | FPO | GenPO | GenPOPFClip | GenPOU0Clip | SGenPO | GenPOPlusPlus = alg_class(
            actor_critic, device=self.device, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        # Initialize the storage
        if hasattr(alg, "get_storage_action_shape"):
            actions_shape = alg.get_storage_action_shape(self.env.num_actions)
        else:
            actions_shape = [self.env.num_actions]

        if hasattr(alg, "get_storage_training_type"):
            training_type = alg.get_storage_training_type()
        else:
            training_type = "rl"

        alg.init_storage(
            training_type,
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

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Compute the collection size
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        # Update total time-steps and time
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # Log episode information
        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f"Mean episode {key}:":>{pad}} {value:.4f}\n"""

        # GenPO policies may not expose Gaussian std
        mean_std = None
        try:
            policy_std = self.alg.policy.action_std
            mean_std = policy_std.mean() if isinstance(policy_std, torch.Tensor) else torch.as_tensor(policy_std).mean()
        except Exception:
            mean_std = None

        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

        # Log losses
        for key, value in locs["loss_dict"].items():
            if isinstance(value, dict):
                for metric_name, metric_value in value.items():
                    self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, locs["it"])
            else:
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        # Log noise std only if available
        if mean_std is not None:
            self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

        # Log performance
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # Log training
        if len(locs["rewbuffer"]) > 0:
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
                self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{title.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
            )
            if mean_std is not None:
                log_string += f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            for key, value in locs["loss_dict"].items():
                if isinstance(value, dict):
                    for metric_name, metric_value in value.items():
                        log_string += f"""{f"{metric_name}:":>{pad}} {metric_value:.4f}\n"""
                else:
                    log_string += f"""{f"Mean {key} loss:":>{pad}} {value:.4f}\n"""
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                log_string += (
                    f"""{"Mean extrinsic reward:":>{pad}} {statistics.mean(locs["erewbuffer"]):.2f}\n"""
                    f"""{"Mean intrinsic reward:":>{pad}} {statistics.mean(locs["irewbuffer"]):.2f}\n"""
                )
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(locs["rewbuffer"]):.2f}\n"""
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(locs["lenbuffer"]):.2f}\n"""
        else:
            log_string = (
                f"""{"#" * width}\n"""
                f"""{title.center(width, " ")}\n\n"""
                f"""{"Computation:":>{pad}} {fps:.0f} steps/s (collection: {locs["collection_time"]:.3f}s, learning {
                    locs["learn_time"]:.3f}s)\n"""
            )
            if mean_std is not None:
                log_string += f"""{"Mean action noise std:":>{pad}} {mean_std.item():.2f}\n"""
            for key, value in locs["loss_dict"].items():
                if isinstance(value, dict):
                    for metric_name, metric_value in value.items():
                        log_string += f"""{f"{metric_name}:":>{pad}} {metric_value:.4f}\n"""
                else:
                    log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

        log_string += ep_string
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{"ETA:":>{pad}} {
                time.strftime(
                    "%H:%M:%S",
                    time.gmtime(
                        self.tot_time
                        / (locs["it"] - locs["start_iter"] + 1)
                        * (locs["start_iter"] + locs["num_learning_iterations"] - locs["it"])
                    ),
                )
            }\n"""
        )
        print(log_string)
