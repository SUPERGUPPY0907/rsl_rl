# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.genpo import GenPO


class GenPOPlusPlus(GenPO):
    """GenPO variant with latent directional diversity regularization."""

    def __init__(
        self,
        policy,
        lambda_dir: float = 0.0,
        lambda_mirror: float = 0.0,
        directional_num_samples: int = 4,
        directional_advantage_quantile: float = 0.75,
        directional_max_groups: int = 32,
        directional_eps: float = 1.0e-8,
        **kwargs,
    ) -> None:
        super().__init__(policy, **kwargs)

        if lambda_dir < 0.0:
            raise ValueError(f"'lambda_dir' must be non-negative, got {lambda_dir}.")
        if lambda_mirror < 0.0:
            raise ValueError(f"'lambda_mirror' must be non-negative, got {lambda_mirror}.")
        if directional_num_samples < 2:
            raise ValueError(
                f"'directional_num_samples' must be at least 2, got {directional_num_samples}."
            )
        if not 0.0 <= directional_advantage_quantile <= 1.0:
            raise ValueError(
                "'directional_advantage_quantile' must be in [0, 1], "
                f"got {directional_advantage_quantile}."
            )
        if directional_max_groups < 1:
            raise ValueError(f"'directional_max_groups' must be positive, got {directional_max_groups}.")
        if directional_eps <= 0.0:
            raise ValueError(f"'directional_eps' must be positive, got {directional_eps}.")

        self.lambda_dir = lambda_dir
        self.lambda_mirror = lambda_mirror
        self.directional_num_samples = directional_num_samples
        self.directional_advantage_quantile = directional_advantage_quantile
        self.directional_max_groups = directional_max_groups
        self.directional_eps = directional_eps

        self.gipo_sigma = 1.0
        self.rho_min = 1e-4
        self.rho_max = 1e4

        self.tau_pos = 1.0
        self.tau_neg = 1.05

    def _repeat_observations(self, obs: TensorDict, repeats: int) -> TensorDict:
        if repeats < 1:
            raise ValueError(f"'repeats' must be positive, got {repeats}.")

        repeated_obs = {key: value.repeat_interleave(repeats, dim=0) for key, value in obs.items()}
        return TensorDict(repeated_obs, batch_size=[obs.batch_size[0] * repeats], device=obs.device)

    def _select_directional_anchor_indices(self, advantages: torch.Tensor) -> torch.Tensor:
        advantages = advantages.detach().flatten()
        positive_mask = advantages > 0.0
        if not positive_mask.any():
            return torch.empty(0, dtype=torch.long, device=advantages.device)

        positive_advantages = advantages[positive_mask]
        threshold = torch.quantile(positive_advantages, self.directional_advantage_quantile)
        anchor_mask = positive_mask & (advantages >= threshold)
        anchor_indices = anchor_mask.nonzero(as_tuple=False).squeeze(-1)

        if anchor_indices.numel() > self.directional_max_groups:
            topk = torch.topk(advantages[anchor_indices], k=self.directional_max_groups).indices
            anchor_indices = anchor_indices[topk]

        return anchor_indices

    def _compute_directional_diversity_loss(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
    ) -> torch.Tensor:
        zero = self.new_method(actions_batch)
        if self.lambda_dir <= 0.0:
            return zero

        anchor_indices = self._select_directional_anchor_indices(torch.squeeze(advantages_batch, -1))
        if anchor_indices.numel() == 0:
            return zero

        anchor_obs = obs_batch[anchor_indices]
        anchor_actions = actions_batch[anchor_indices]

        action_samples = [anchor_actions]
        with torch.no_grad():
            for _ in range(self.directional_num_samples - 1):
                # The extra actions are treated as fixed samples so the inverse-latent term still
                # produces a gradient on the current flow parameters.
                action_samples.append(self.policy.act_inference(anchor_obs).detach())

        sampled_actions = torch.stack(action_samples, dim=1)
        repeated_obs = self._repeat_observations(anchor_obs, self.directional_num_samples)
        latent_batch = self.policy.inverse_latent(sampled_actions.flatten(0, 1), repeated_obs)
        latent_batch = latent_batch.view(anchor_actions.shape[0], self.directional_num_samples, -1)

        directions = latent_batch / (latent_batch.norm(dim=-1, keepdim=True) + self.directional_eps)
        cosine_sq = torch.matmul(directions, directions.transpose(-1, -2)).square()
        diag_mask = torch.eye(
            self.directional_num_samples, device=cosine_sq.device, dtype=torch.bool
        ).unsqueeze(0)
        cosine_sq = cosine_sq.masked_fill(diag_mask, 0.0)

        num_pairs = self.directional_num_samples * (self.directional_num_samples - 1)
        return (cosine_sq.sum(dim=(-1, -2)) / num_pairs).mean()

    def _compute_mirrored_action_loss(
        self,
        obs_batch: TensorDict,
        actions_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
    ) -> torch.Tensor:
        zero = self.new_method(actions_batch)
        if self.lambda_mirror <= 0.0:
            return zero

        advantages = torch.squeeze(advantages_batch, -1).detach()
        anchor_indices = self._select_directional_anchor_indices(advantages)
        if anchor_indices.numel() == 0:
            return zero

        anchor_obs = obs_batch[anchor_indices]
        anchor_actions = actions_batch[anchor_indices]
        anchor_advantages = advantages[anchor_indices]

        anchor_latents = self.policy.inverse_latent(anchor_actions, anchor_obs)
        mirrored_actions = self.policy.forward_latent(-anchor_latents, anchor_obs)
        per_sample_loss = (mirrored_actions - anchor_actions.detach()).pow(2).mean(dim=-1)
        weights = anchor_advantages / (anchor_advantages.mean() + self.directional_eps)

        return (weights * per_sample_loss).mean()

    def new_method(self, actions_batch):
        zero = actions_batch.new_zeros(())
        return zero

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_compress_loss = 0.0
        mean_directional_loss = 0.0
        mean_mirrored_loss = 0.0
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
                old_action_latent_batch = old_action_latent_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            new_action_latent_batch = self.policy.inverse_latent(actions_batch, obs_batch)
            log_ratio_batch = self._log_ratio_from_latents(new_action_latent_batch, old_action_latent_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = -log_ratio_batch
                    kl_mean = torch.mean(kl)

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

            ratio = torch.exp(log_ratio_batch)
            adv = torch.squeeze(advantages_batch, -1)
            tau = torch.where(
                adv > 0,
                torch.full_like(ratio, self.tau_pos),
                torch.full_like(ratio, self.tau_neg),
            )
            # Use the original ratio where PPO would not clip, and SAPO soft clipping where it would.
            soft_factor = (4.0 / tau) * torch.sigmoid(tau * (ratio - 1.0))
            hard_factor = torch.zeros_like(ratio)
            keep_factor = torch.ones_like(ratio)
            use_clip = ((adv > 0) & (ratio > 1.0 + self.clip_param)) | (
                (adv < 0) & (ratio < 1.0 - self.clip_param)
            )
            effective_factor = torch.where(use_clip, hard_factor, ratio)
            surrogate_loss = -(adv * effective_factor).mean()
            # weight = torch.exp(-0.5 * (torch.log(ratio_for_weight) / self.gipo_sigma) ** 2)
            # effective_multiplier = weight * ratio
            # surrogate_loss = -(effective_multiplier * adv).mean()

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

            directional_loss = self._compute_directional_diversity_loss(obs_batch, actions_batch, advantages_batch)
            mirrored_loss = self._compute_mirrored_action_loss(obs_batch, actions_batch, advantages_batch)
            action_log_prob_batch = self._log_prob_from_latent(new_action_latent_batch)
            entropy = -((action_log_prob_batch).detach() * action_log_prob_batch).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.compress_coef * compress_loss
                + self.lambda_dir * directional_loss
                + self.lambda_mirror * mirrored_loss
            )

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
            mean_entropy += entropy.item()
            mean_compress_loss += compress_loss.item()
            mean_directional_loss += directional_loss.item()
            mean_mirrored_loss += mirrored_loss.item()

            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_compress_loss /= num_updates
        mean_directional_loss /= num_updates
        mean_mirrored_loss /= num_updates

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
            "directional": mean_directional_loss,
            "mirrored": mean_mirrored_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
