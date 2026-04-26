from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticPolicyFlow
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class AdaptiveKLScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        kl_threshold: float = 0.008,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        kl_factor: float = 2.0,
        lr_factor: float = 1.5,
    ) -> None:
        self.optimizer = optimizer
        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.kl_factor = kl_factor
        self.lr_factor = lr_factor

    def step(self, kl: float | torch.Tensor | None = None) -> None:
        if kl is None:
            return
        if isinstance(kl, torch.Tensor):
            kl = float(kl.item())

        for group in self.optimizer.param_groups:
            if kl > self.kl_threshold * self.kl_factor:
                group["lr"] = max(group["lr"] / self.lr_factor, self.min_lr)
            elif kl < self.kl_threshold / self.kl_factor:
                group["lr"] = min(group["lr"] * self.lr_factor, self.max_lr)

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class PolicyFlow:
    policy: ActorCriticPolicyFlow

    def __init__(
        self,
        policy: ActorCriticPolicyFlow,
        desired_kl: float = 0.01,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        gamma: float | None = None,
        lam: float = 0.95,
        time_limit_bootstrap: bool = True,
        mini_batches: int | None = None,
        num_mini_batches: int | None = None,
        learning_epochs: int | None = None,
        num_learning_epochs: int | None = None,
        gaussian_entropy_loss_scale: float = 0.01,
        brownian_reg_loss_scale: float = 0.01,
        ratio_clip: float | None = None,
        clip_param: float | None = None,
        clip_predicted_values: bool = True,
        value_clip: float = 0.2,
        value_loss_scale: float | None = None,
        value_loss_coef: float | None = None,
        grad_norm_clip: float | None = None,
        max_grad_norm: float | None = None,
        degenerate2gaussian: bool = False,
        schedule: str = "adaptive",
        learning_rate_scheduler_kwargs: dict | None = None,
        optimizer: str = "adamw",
        optimizer_kwargs: dict | None = None,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,
    ) -> None:
        if kwargs:
            print("PolicyFlow.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs]))

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

        self.discount_factor = gamma if gamma is not None else discount_factor
        self.gamma = self.discount_factor
        self.lam = lam
        self.desired_kl = desired_kl
        self.time_limit_bootstrap = time_limit_bootstrap
        self.mini_batches = mini_batches if mini_batches is not None else (num_mini_batches or 4)
        self.learning_epochs = learning_epochs if learning_epochs is not None else (num_learning_epochs or 5)
        self.gaussian_entropy_loss_scale = gaussian_entropy_loss_scale
        self.brownian_reg_loss_scale = brownian_reg_loss_scale
        self.ratio_clip = ratio_clip if ratio_clip is not None else (clip_param if clip_param is not None else 0.2)
        self.clip_predicted_values = clip_predicted_values
        self.value_clip = value_clip
        self.value_loss_scale = (
            value_loss_scale if value_loss_scale is not None else (value_loss_coef if value_loss_coef is not None else 1.0)
        )
        self.grad_norm_clip = grad_norm_clip if grad_norm_clip is not None else (max_grad_norm or 1.0)
        self.degenerate2gaussian = degenerate2gaussian
        self.schedule = schedule
        self.learning_rate = learning_rate

        optimizer_kwargs = {"weight_decay": 1e-5} if optimizer_kwargs is None else dict(optimizer_kwargs)
        optimizer_cls = resolve_optimizer(optimizer)
        self.optimizer = optimizer_cls(self._trainable_parameters(), lr=self.learning_rate, **optimizer_kwargs)

        scheduler_kwargs = {"kl_threshold": self.desired_kl}
        if learning_rate_scheduler_kwargs is not None:
            scheduler_kwargs.update(learning_rate_scheduler_kwargs)
        self.lr_scheduler = AdaptiveKLScheduler(self.optimizer, **scheduler_kwargs)

        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()
        self.rnd = None
        self.rnd_optimizer = None
        self.symmetry = None

    def get_storage_training_type(self) -> str:
        return "rl_policyflow"

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
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            raise ValueError("PolicyFlow recurrent policies are not supported in rsl_rl.")

        actions, action_info = self.policy.sample_actions(obs, degenerate2gaussian=self.degenerate2gaussian)
        self.transition.actions = actions
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_prior = action_info["actions_prior"]
        self.transition.flow_x0 = action_info["flow_x0"]
        self.transition.delta_actions = action_info["delta_actions"]
        self.transition.delta_actions_std = action_info["delta_actions_std"]
        self.transition.delta_actions_log_prob = action_info["delta_actions_log_prob"]
        self.transition.observations = obs
        return self.transition.actions

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.policy.act_inference(obs, degenerate2gaussian=self.degenerate2gaussian)

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

        if self.time_limit_bootstrap and "time_outs" in extras:
            self.transition.rewards += self.discount_factor * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(last_values, self.discount_factor, self.lam)

    def update(self) -> dict[str, float | dict[str, float]]:  # noqa: C901
        if self.policy.is_recurrent:
            raise ValueError("PolicyFlow recurrent policies are not supported in rsl_rl.")

        mean_policy_loss = 0.0
        mean_entropy_loss = 0.0
        mean_value_loss = 0.0
        mean_brownian_reg_loss = 0.0
        delta_vel_max = float("-inf")
        delta_vel_min = float("inf")
        last_kl = 0.0
        last_noise_std = 0.0

        for _ in range(self.learning_epochs):
            epoch_kls: list[torch.Tensor] = []
            generator = self.storage.mini_batch_generator(self.mini_batches, num_epochs=1)
            for (
                obs_batch,
                _actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_prior_batch,
                old_flow_x0_batch,
                old_delta_actions_batch,
                old_delta_actions_std_batch,
                old_delta_actions_log_prob_batch,
                hidden_states_batch,
                masks_batch,
            ) in generator:
                del hidden_states_batch, masks_batch

                if self.brownian_reg_loss_scale > 0.0 and not self.degenerate2gaussian:
                    delta_vel, delta_std_new, brownian_reg_loss = self.policy.compute_flow_variation(
                        obs_batch,
                        old_actions_prior_batch,
                        old_flow_x0_batch,
                        compute_brownian_reg_loss=True,
                    )
                    brownian_reg_loss = self.brownian_reg_loss_scale * brownian_reg_loss
                else:
                    delta_vel, delta_std_new = self.policy.compute_flow_variation(
                        obs_batch,
                        old_actions_prior_batch,
                        old_flow_x0_batch,
                        compute_brownian_reg_loss=False,
                    )
                    brownian_reg_loss = torch.zeros((), device=self.device)

                action_distribution_new = torch.distributions.Normal(delta_vel, delta_std_new)
                actions_log_prob_new = action_distribution_new.log_prob(old_delta_actions_batch).sum(-1)
                kl = self._compute_kl_divergence(
                    delta_vel.detach(),
                    delta_std_new.detach(),
                    torch.zeros_like(delta_vel),
                    old_delta_actions_std_batch,
                )
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                    kl /= self.gpu_world_size
                epoch_kls.append(kl)

                if self.gaussian_entropy_loss_scale > 0.0:
                    gaussian_entropy_loss = (
                        -self.gaussian_entropy_loss_scale * action_distribution_new.entropy().sum(dim=-1).mean()
                    )
                else:
                    gaussian_entropy_loss = torch.zeros((), device=self.device)

                old_log_prob = old_delta_actions_log_prob_batch.squeeze(-1)
                advantages = advantages_batch.squeeze(-1)
                ratio = torch.exp(actions_log_prob_new - old_log_prob)
                surrogate = advantages * ratio
                surrogate_clipped = advantages * torch.clamp(
                    ratio,
                    1.0 - self.ratio_clip,
                    1.0 + self.ratio_clip,
                )
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                predicted_values = self.policy.evaluate(obs_batch)
                if self.clip_predicted_values:
                    predicted_values_clipped = target_values_batch + torch.clamp(
                        predicted_values - target_values_batch,
                        min=-self.value_clip,
                        max=self.value_clip,
                    )
                    value_loss = self.value_loss_scale * torch.max(
                        (predicted_values - returns_batch).pow(2),
                        (predicted_values_clipped - returns_batch).pow(2),
                    ).mean()
                else:
                    value_loss = self.value_loss_scale * torch.nn.functional.mse_loss(returns_batch, predicted_values)

                self.optimizer.zero_grad()
                (policy_loss + gaussian_entropy_loss + brownian_reg_loss + value_loss).backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self._trainable_parameters(), self.grad_norm_clip)
                self.optimizer.step()

                if self.policy.actor.using_ema:
                    self.policy.actor.ema_update()

                mean_policy_loss += policy_loss.item()
                mean_entropy_loss += gaussian_entropy_loss.item()
                mean_value_loss += value_loss.item()
                mean_brownian_reg_loss += brownian_reg_loss.item()
                delta_vel_max = max(delta_vel_max, float(delta_vel.max().item()))
                delta_vel_min = min(delta_vel_min, float(delta_vel.min().item()))
                last_noise_std = float(delta_std_new.mean().item())

            if epoch_kls:
                kl_value = torch.stack(epoch_kls).mean()
                last_kl = float(kl_value.item())
                if self.schedule == "adaptive":
                    self.lr_scheduler.step(kl_value)

        self.policy.sync_reference_model()
        self.storage.clear()
        self.learning_rate = self.lr_scheduler.get_last_lr()[0]

        num_updates = self.learning_epochs * self.mini_batches
        loss_dict: dict[str, float | dict[str, float]] = {
            "policy_loss": mean_policy_loss / num_updates,
            "gaussian_entropy_loss": mean_entropy_loss / num_updates,
            "value_loss": mean_value_loss / num_updates,
            "metrics": {
                "mean_noise_std": last_noise_std,
                "delta_vel_max": delta_vel_max,
                "delta_vel_min": delta_vel_min,
                "kl": last_kl,
            },
        }
        if self.brownian_reg_loss_scale > 0.0:
            loss_dict["brownian_reg_loss"] = mean_brownian_reg_loss / num_updates
        return loss_dict

    def _compute_kl_divergence(
        self,
        actions_mean: torch.Tensor,
        actions_std: torch.Tensor,
        last_action_mean: torch.Tensor,
        last_action_std: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            std_ratio = actions_std / last_action_std + 1.0e-5
            std_drifted = torch.square(last_action_std) + torch.square(last_action_mean - actions_mean)
            kl = torch.sum(
                torch.log(std_ratio) + std_drifted / (2.0 * torch.square(actions_std)) - 0.5,
                dim=-1,
            )
        return kl.mean().detach()

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [param for param in self.policy.parameters() if param.requires_grad]

    def reduce_parameters(self) -> None:
        trainable_params = self._trainable_parameters()
        grads = [param.grad.view(-1) for param in trainable_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in trainable_params:
            if param.grad is None:
                continue
            numel = param.numel()
            param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
            offset += numel
