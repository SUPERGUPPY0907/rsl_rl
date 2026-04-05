# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.genpo import GenPO
from rsl_rl.modules import ActorCriticGenPO


class _GenPOPushforwardClipBase(GenPO):
    """Shared update path for pushforward-measure clip variants."""

    policy: ActorCriticGenPO
    kl_metric_name = "trust_region_kl"

    def __init__(
        self,
        policy: ActorCriticGenPO,
        log_clip_delta: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(policy, **kwargs)

        self.log_clip_delta = self.clip_param if log_clip_delta is None else log_clip_delta
        if self.log_clip_delta <= 0.0:
            raise ValueError(f"'log_clip_delta' must be positive, got {self.log_clip_delta}.")

    @staticmethod
    def _repeat_observations(obs: TensorDict, repeats: int) -> TensorDict:
        if repeats < 1:
            raise ValueError(f"'repeats' must be positive, got {repeats}.")

        repeated_obs = {key: value.repeat_interleave(repeats, dim=0) for key, value in obs.items()}
        return TensorDict(repeated_obs, batch_size=[obs.batch_size[0] * repeats], device=obs.device)

    def _create_reference_policy(self) -> ActorCriticGenPO:
        reference_policy = copy.deepcopy(self.policy).eval()
        reference_policy.to(self.device)
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)
        return reference_policy

    def _split_dummy_action(self, actions_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = actions_batch[..., : self.policy.action_dim]
        right = actions_batch[..., self.policy.action_dim :]
        return left, right

    def _real_action(self, actions_batch: torch.Tensor) -> torch.Tensor:
        left, right = self._split_dummy_action(actions_batch)
        return 0.5 * (left + right)

    def _section_actions(self, actions_batch: torch.Tensor) -> torch.Tensor:
        real_action = self._real_action(actions_batch)
        return torch.cat((real_action, real_action), dim=-1)

    def _log_clip(self, log_ratio_batch: torch.Tensor) -> torch.Tensor:
        return log_ratio_batch.clamp(-self.log_clip_delta, +self.log_clip_delta)
    def _ratio_clip(self, ratio: torch.Tensor) -> torch.Tensor:
        return ratio.clamp(1-self.clip_param, 1+self.clip_param)

    def _surrogate_loss_from_log_ratio(
        self,
        log_ratio_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(log_ratio_batch)
        clipped_ratio = self._ratio_clip(ratio)
        # clipped_ratio = torch.exp(self._log_clip(log_ratio_batch))
        
        advantages = torch.squeeze(advantages_batch)

        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * clipped_ratio
        return torch.max(surrogate, surrogate_clipped).mean()
    

    def _estimated_kl_from_log_ratio(self, log_ratio_batch: torch.Tensor) -> torch.Tensor:
        return (-log_ratio_batch).mean()

    def _adapt_learning_rate_from_log_ratio(self, log_ratio_batch: torch.Tensor) -> float:
        with torch.inference_mode():
            # kl_mean = self._estimated_kl_from_log_ratio(log_ratio_batch)
            kl_mean = (-log_ratio_batch).mean()

            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size

            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        return kl_mean.item()

    def _estimate_log_ratio(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        latent_batch: torch.Tensor | None,
        reference_policy: ActorCriticGenPO,
    ) -> torch.Tensor:
        raise NotImplementedError

    def update(self) -> dict[str, float]:  # noqa: C901
        reference_policy = self._create_reference_policy()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_compress_loss = 0.0
        mean_estimated_kl = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_action_latent_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)
                old_action_latent_batch = old_action_latent_batch.repeat(num_aug, 1)

            log_ratio_batch = self._estimate_log_ratio(obs_batch, actions_batch, old_action_latent_batch, reference_policy)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])

            if self.desired_kl is not None and self.schedule == "adaptive":
                estimated_kl_value = self._adapt_learning_rate_from_log_ratio(log_ratio_batch)
            else:
                estimated_kl_value = self._estimated_kl_from_log_ratio(log_ratio_batch).item()

            # ratio = torch.exp(log_ratio_batch)
            # surrogate = -torch.squeeze(advantages_batch) * ratio
            # surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            #     ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            # )
            # surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            surrogate_loss = self._surrogate_loss_from_log_ratio(log_ratio_batch, advantages_batch)

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if self.use_compress:
                actions = self.policy.act_inference(obs_batch)
                actions_left = actions[:, : self.policy.actor.a_dim]
                actions_right = actions[:, self.policy.actor.a_dim :]
                compress_loss = torch.norm(actions_left - actions_right, dim=1).mean()
            else:
                compress_loss = surrogate_loss.new_zeros(())

            loss = surrogate_loss + self.value_loss_coef * value_loss + self.compress_coef * compress_loss

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.batch_size[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:],
                    actions_mean_symm_batch.detach()[original_batch_size:],
                )

                if self.symmetry["use_mirror_loss"]:
                    loss = loss + self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()

            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += 0.0
            mean_compress_loss += compress_loss.item()
            mean_estimated_kl += estimated_kl_value

            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_compress_loss /= num_updates
        mean_estimated_kl /= num_updates

        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "compress": mean_compress_loss,
            self.kl_metric_name: mean_estimated_kl,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict


class GenPOU0Clip(_GenPOPushforwardClipBase):
    """GenPO variant using the diagonal dummy section (u = 0)."""

    kl_metric_name = "section_kl"

    def _estimate_log_ratio(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        latent_batch: torch.Tensor | None,
        reference_policy: ActorCriticGenPO,
    ) -> torch.Tensor:
        section_actions = self._section_actions(actions_batch)
        new_action_latent_batch = self.policy.inverse_latent(section_actions, obs_batch)
        with torch.no_grad():
            old_action_latent_batch = reference_policy.inverse_latent(section_actions, obs_batch)
        return self._log_ratio_from_latents(new_action_latent_batch, latent_batch)


class GenPOPFClip(_GenPOPushforwardClipBase):
    """Proposal-based pushforward clip approximation with antithetic fiber samples."""

    kl_metric_name = "pushforward_kl"

    def __init__(
        self,
        policy: ActorCriticGenPO,
        pf_num_samples: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(policy, **kwargs)

        if pf_num_samples < 1:
            raise ValueError(f"'pf_num_samples' must be at least 1, got {pf_num_samples}.")
        self.pf_num_samples = pf_num_samples

    def _estimate_log_ratio(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        reference_policy: ActorCriticGenPO,
    ) -> torch.Tensor:
        real_action = self._real_action(actions_batch)
        actions_left, actions_right = self._split_dummy_action(actions_batch)
        fiber = 0.5 * (actions_left - actions_right)
        fiber_scale = fiber.square().mean(dim=0).sqrt()

        proposal_noise = torch.randn(
            actions_batch.shape[0],
            self.pf_num_samples,
            self.policy.action_dim,
            device=actions_batch.device,
            dtype=actions_batch.dtype,
        )
        fiber_proposals = proposal_noise * fiber_scale.view(1, 1, -1)

        plus_actions = torch.cat(
            (real_action.unsqueeze(1) + fiber_proposals, real_action.unsqueeze(1) - fiber_proposals),
            dim=-1,
        )
        minus_actions = torch.cat(
            (real_action.unsqueeze(1) - fiber_proposals, real_action.unsqueeze(1) + fiber_proposals),
            dim=-1,
        )
        proposal_actions = torch.cat((plus_actions, minus_actions), dim=1)

        repeated_obs = self._repeat_observations(obs_batch, proposal_actions.shape[1])
        proposal_actions_flat = proposal_actions.flatten(0, 1)
        new_action_latent_batch = self.policy.inverse_latent(proposal_actions_flat, repeated_obs)
        with torch.no_grad():
            old_action_latent_batch = reference_policy.inverse_latent(proposal_actions_flat, repeated_obs)

        sample_log_ratio = self._log_ratio_from_latents(new_action_latent_batch, old_action_latent_batch)
        sample_log_ratio = sample_log_ratio.view(actions_batch.shape[0], proposal_actions.shape[1])
        return torch.logsumexp(sample_log_ratio, dim=1) - math.log(proposal_actions.shape[1])
