from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticFPO
from rsl_rl.modules.ema import ExponentialMovingAverage
from rsl_rl.storage import RolloutStorage


def clamp_ste(x: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
    clamped = x.clamp(min=min, max=max)
    return x + (clamped - x).detach()


class FPO:
    """Flow Policy Optimization."""

    policy: ActorCriticFPO

    def __init__(
        self,
        policy: ActorCriticFPO,
        num_learning_epochs: int = 16,
        num_mini_batches: int = 4,
        clip_param: float = 0.05,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = False,
        schedule: str = "fixed",
        desired_kl: float = 1e-4,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        normalize_advantage: bool = True,
        n_samples_per_action: int = 16,
        cfm_diff_clamp_max: float = 10.0,
        cfm_loss_clamp: float = 20.0,
        cfm_loss_clamp_negative_advantages: bool = True,
        cfm_loss_clamp_negative_advantages_max: float = 20.0,
        storage_action_noise_std: float = 0.0,
        trust_region_mode: str = "aspo",
        advantage_clamp: tuple[float, float] = (100.0, 100.0),
        knn_entropy_coef: float = 0.0,
        knn_entropy_k: int = 1,
        ema_decay: float = 0.95,
        ema_warmup_steps: int = 500,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print("FPO.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs]))

        if rnd_cfg is not None:
            raise ValueError("FPO does not support RND yet.")
        if symmetry_cfg is not None:
            raise ValueError("FPO does not support symmetry augmentation yet.")
        if trust_region_mode not in {"ppo", "spo", "aspo"}:
            raise ValueError(f"Unsupported trust-region mode: {trust_region_mode}.")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy
        self.policy.to(self.device)
        if weight_decay > 0.0:
            self.optimizer = optim.AdamW(
                self.policy.parameters(),
                lr=learning_rate,
                betas=adam_betas,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.policy.parameters(),
                lr=learning_rate,
                betas=adam_betas,
            )

        self.ema_decay = ema_decay
        self.ema_warmup_steps = ema_warmup_steps
        self.tot_timesteps = 0
        if ema_decay > 0.0:
            self.ema = ExponentialMovingAverage(self.policy.actor, decay=ema_decay, device=self.device)
        else:
            self.ema = None

        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.knn_entropy_coef = knn_entropy_coef
        self.knn_entropy_k = knn_entropy_k
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.normalize_advantage = normalize_advantage
        self.n_samples_per_action = n_samples_per_action
        self.cfm_diff_clamp_max = cfm_diff_clamp_max
        self.cfm_loss_clamp = cfm_loss_clamp
        self.cfm_loss_clamp_negative_advantages = cfm_loss_clamp_negative_advantages
        self.cfm_loss_clamp_negative_advantages_max = cfm_loss_clamp_negative_advantages_max
        self.storage_action_noise_std = storage_action_noise_std
        self.trust_region_mode = trust_region_mode
        self.advantage_clamp = advantage_clamp
        self.update_counter = 0

        self.rnd = None
        self.rnd_optimizer = None
        self.symmetry = None

    def get_storage_training_type(self) -> str:
        return "rl_fpo"

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
            n_samples_per_action=self.n_samples_per_action,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        if self.storage_action_noise_std > 0:
            self.transition.actions = self.transition.actions + self.storage_action_noise_std * torch.randn_like(
                self.transition.actions
            )

        cfm_loss_eps = torch.randn(
            self.storage.num_envs,
            self.n_samples_per_action,
            self.policy.num_actions,
            device=self.device,
        )
        uniform_t = torch.rand(
            self.storage.num_envs,
            self.n_samples_per_action,
            1,
            device=self.device,
        )
        beta = self.policy.cfm_loss_t_inverse_cdf_beta
        cfm_loss_t = 0.005 + 0.99 * (1.0 - (1.0 - uniform_t) ** (1.0 / beta))
        initial_cfm_loss, x1_pred, _ = self.policy.get_cfm_loss(obs, self.transition.actions, cfm_loss_eps, cfm_loss_t)

        self.transition.initial_cfm_loss = initial_cfm_loss.detach()
        self.transition.x1_pred = x1_pred.detach()
        self.transition.cfm_loss_eps = cfm_loss_eps
        self.transition.cfm_loss_t = cfm_loss_t
        self.transition.observations = obs
        return self.transition.actions

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.policy.act_inference(obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=self.normalize_advantage and not self.normalize_advantage_per_mini_batch,
        )

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        all_grad_norms_before: list[float] = []
        all_grad_norms_after: list[float] = []

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_x1_pred_batch,
            old_cfm_loss_batch,
            old_cfm_eps_batch,
            old_cfm_t_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            del hid_states_batch, masks_batch
            batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            with torch.no_grad():
                pos_clamp, neg_clamp = self.advantage_clamp
                advantages_batch = advantages_batch.clamp(-neg_clamp, pos_clamp)

            cfm_loss_batch, x1_pred_batch, x0_pred_batch = self.policy.get_cfm_loss(
                obs_batch,
                actions_batch,
                old_cfm_eps_batch,
                old_cfm_t_batch,
            )
            value_batch = self.policy.evaluate(obs_batch)

            entropy_bonus = None
            if self.knn_entropy_coef > 0:
                entropy_bonus = self._compute_knn_entropy(x0_pred_batch, k=self.knn_entropy_k)

            if self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = (x1_pred_batch.detach() - old_x1_pred_batch).pow(2).mean()
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
                    mean_kl += kl_mean.item()

            if self.cfm_loss_clamp > 0:
                old_cfm_loss_batch = torch.clamp(old_cfm_loss_batch, max=self.cfm_loss_clamp)
                cfm_loss_batch = torch.clamp(cfm_loss_batch, max=self.cfm_loss_clamp)

            if self.cfm_loss_clamp_negative_advantages:
                cfm_loss_batch = torch.where(
                    advantages_batch < 0,
                    cfm_loss_batch.clamp(max=self.cfm_loss_clamp_negative_advantages_max),
                    cfm_loss_batch,
                )

            log_ratio = old_cfm_loss_batch - cfm_loss_batch
            log_ratio = clamp_ste(log_ratio, max=self.cfm_diff_clamp_max)
            ratio = torch.exp(log_ratio)
            advantages = advantages_batch.squeeze(-1)

            if self.trust_region_mode == "ppo":
                surrogate = -advantages.unsqueeze(-1) * ratio
                surrogate_clipped = -advantages.unsqueeze(-1) * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            elif self.trust_region_mode == "spo":
                surrogate_loss = -torch.mean(
                    ratio * advantages.unsqueeze(-1)
                    - torch.abs(advantages).unsqueeze(-1) / (2.0 * self.clip_param) * (ratio - 1.0).pow(2)
                )
            else:
                surrogate = -advantages.unsqueeze(-1) * ratio
                surrogate_clipped = -advantages.unsqueeze(-1) * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
                ppo_loss = torch.max(surrogate, surrogate_clipped)
                spo_loss = -(
                    ratio * advantages.unsqueeze(-1)
                    - torch.abs(advantages).unsqueeze(-1) / (2.0 * self.clip_param) * (ratio - 1.0).pow(2)
                )
                surrogate_loss = torch.where(advantages.unsqueeze(-1) > 0, ppo_loss, spo_loss).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param,
                    self.clip_param,
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss
            if entropy_bonus is not None:
                loss -= self.knn_entropy_coef * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            grad_norm_before = 0.0
            for param in self.policy.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    grad_norm_before += param_norm.item() ** 2
            grad_norm_before = grad_norm_before**0.5
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            grad_norm_after = 0.0
            for param in self.policy.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    grad_norm_after += param_norm.item() ** 2
            grad_norm_after = grad_norm_after**0.5

            self.optimizer.step()

            all_grad_norms_before.append(grad_norm_before)
            all_grad_norms_after.append(grad_norm_after)
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_bonus.item() if entropy_bonus is not None else 0.0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if self.schedule == "adaptive":
            mean_kl /= num_updates

        self.storage.clear()
        self.update_counter += 1
        self.tot_timesteps += 1

        loss_dict: dict[str, Any] = {
            "surrogate_loss": mean_surrogate_loss,
            "value_loss": mean_value_loss,
        }
        if self.knn_entropy_coef > 0:
            loss_dict["entropy_loss"] = mean_entropy

        metrics_dict = {"clip_param": self.clip_param}
        if self.schedule == "adaptive":
            metrics_dict["kl"] = mean_kl
        if all_grad_norms_before:
            metrics_dict["mean_grad_norm_before_clip"] = float(np.mean(all_grad_norms_before))
            metrics_dict["mean_grad_norm_after_clip"] = float(np.mean(all_grad_norms_after))
        loss_dict["metrics"] = metrics_dict
        return loss_dict

    def post_update(self) -> None:
        if self.ema is None:
            return
        if self.tot_timesteps == self.ema_warmup_steps:
            self.ema.reset_to_current()
        elif self.tot_timesteps > self.ema_warmup_steps:
            self.ema.update()

    def get_checkpoint_policy_state_dict(self) -> dict[str, torch.Tensor]:
        model_state_dict = {name: tensor.detach().clone() for name, tensor in self.policy.state_dict().items()}
        if self.ema is not None and self.tot_timesteps > self.ema_warmup_steps:
            for name, ema_param in self.ema.shadow_params.items():
                full_name = f"actor.{name}"
                if full_name in model_state_dict:
                    model_state_dict[full_name] = ema_param.detach().clone().to(model_state_dict[full_name].device)
        return model_state_dict

    def get_additional_checkpoint_state(self) -> dict[str, object]:
        if self.ema is None:
            return {}
        return {"ema_state_dict": self.ema.state_dict()}

    def load_additional_checkpoint_state(self, loaded_dict: dict[str, object]) -> None:
        if self.ema is not None and "ema_state_dict" in loaded_dict:
            self.ema.load_state_dict(loaded_dict["ema_state_dict"])

    def _compute_knn_entropy(self, x0_pred: torch.Tensor, k: int) -> torch.Tensor:
        batch_size, n_samples, action_dim = x0_pred.shape
        if not (1 <= k < n_samples):
            raise ValueError(f"k must be in [1, {n_samples}), got {k}.")

        dists = torch.cdist(x0_pred, x0_pred, p=2)
        eye_mask = torch.eye(n_samples, device=self.device).unsqueeze(0).expand(batch_size, -1, -1)
        dists = dists + eye_mask * 1e10
        kth_dists, _ = torch.topk(dists, k=k, dim=2, largest=False, sorted=True)
        rho_k = kth_dists[:, :, -1].clamp(min=1e-6, max=1e9)

        psi_n = torch.digamma(torch.tensor(float(n_samples), device=self.device))
        psi_k = torch.digamma(torch.tensor(float(k), device=self.device))
        log_cd = (action_dim / 2) * math.log(math.pi) - float(torch.lgamma(torch.tensor(action_dim / 2 + 1)))
        log_rho_mean = torch.log(rho_k).mean(dim=1)
        entropy_per_batch = psi_n - psi_k + log_cd + action_dim * log_rho_mean
        return entropy_per_batch.mean()

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
