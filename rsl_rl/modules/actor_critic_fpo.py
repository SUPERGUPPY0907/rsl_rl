from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.utils import resolve_nn_activation


class ActorCriticFPO(nn.Module):
    """Flow-matching actor-critic used by FPO++."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        timestep_embed_dim: int = 8,
        sampling_steps: int = 64,
        training_sampling_steps: int | None = None,
        actor_scale: float = 1.0,
        actor_mlp_output_scale: float = 1.0,
        actor_final_layer_weight_scale: float | None = None,
        cfm_loss_t_inverse_cdf_beta: float = 1.0,
        cfm_loss_reduction: str = "sqrt",
        action_perturb_std: float = 0.02,
        device: str | torch.device = "cpu",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticFPO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        if timestep_embed_dim < 0:
            raise ValueError("'timestep_embed_dim' must be non-negative.")
        if timestep_embed_dim % 2 != 0:
            raise ValueError("'timestep_embed_dim' must be even.")
        if sampling_steps < 1:
            raise ValueError("'sampling_steps' must be positive.")
        if cfm_loss_reduction not in {"mean", "sum", "sqrt"}:
            raise ValueError(f"Unsupported CFM loss reduction: {cfm_loss_reduction}.")

        self.obs_groups = obs_groups
        self.device = torch.device(device)

        self.num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticFPO module only supports 1D observations."
            self.num_actor_obs += obs[obs_group].shape[-1]

        self.num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticFPO module only supports 1D observations."
            self.num_critic_obs += obs[obs_group].shape[-1]

        self.num_actions = num_actions
        self.timestep_embed_dim = timestep_embed_dim
        self.sampling_steps = sampling_steps
        self.training_sampling_steps = training_sampling_steps or sampling_steps
        self.actor_scale = actor_scale
        self.actor_mlp_output_scale = actor_mlp_output_scale
        self.cfm_loss_t_inverse_cdf_beta = cfm_loss_t_inverse_cdf_beta
        self.cfm_loss_reduction = cfm_loss_reduction
        self.action_perturb_std = action_perturb_std

        activation_mod = resolve_nn_activation(activation)
        actor_input_dim = self.num_actor_obs + self.timestep_embed_dim + self.num_actions

        actor_layers: list[nn.Module] = [nn.Linear(actor_input_dim, actor_hidden_dims[0]), activation_mod]
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], self.num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(resolve_nn_activation(activation))
        self.actor = nn.Sequential(*actor_layers)

        if actor_final_layer_weight_scale is not None and actor_final_layer_weight_scale != 1.0:
            final_layer = self.actor[-1]
            if not isinstance(final_layer, nn.Linear):
                raise TypeError("Expected the actor's final layer to be a linear projection.")
            with torch.no_grad():
                final_layer.weight.data.mul_(actor_final_layer_weight_scale)
                if final_layer.bias is not None:
                    final_layer.bias.data.mul_(actor_final_layer_weight_scale)

        self.critic = MLP(self.num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(self.num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self._integrate_flow_impl = self._integrate_flow
        if (
            hasattr(torch, "compile")
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            try:
                self._integrate_flow_impl = torch.compile(self._integrate_flow, mode="reduce-overhead")
            except Exception:
                self._integrate_flow_impl = self._integrate_flow

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

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
        del kwargs
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        actions = self._act_from_actor_obs(actor_obs, training=self.training)
        if self.training and self.action_perturb_std > 0:
            actions = actions + self.action_perturb_std * torch.randn_like(actions)
        return actions

    def act_inference(
        self,
        obs: TensorDict,
        eval_mode: str = "zero",
        eval_fixed_seed: int = 12345,
    ) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.act_inference_from_actor_obs(
            actor_obs,
            eval_mode=eval_mode,
            eval_fixed_seed=eval_fixed_seed,
        )

    def act_inference_from_actor_obs(
        self,
        actor_obs: torch.Tensor,
        eval_mode: str = "zero",
        eval_fixed_seed: int = 12345,
    ) -> torch.Tensor:
        return self._act_from_actor_obs(
            actor_obs,
            training=False,
            eval_mode=eval_mode,
            eval_fixed_seed=eval_fixed_seed,
        )

    def get_cfm_loss(
        self,
        observations: TensorDict,
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_obs = self.get_actor_obs(observations)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.get_cfm_loss_from_actor_obs(actor_obs, actions, eps, t, actor=actor)

    def get_cfm_loss_from_actor_obs(
        self,
        actor_obs: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
        actor: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actor is None:
            actor = self.actor

        batch_size, action_dim = actions.shape
        num_samples = eps.shape[1]
        if actor_obs.shape != (batch_size, self.num_actor_obs):
            raise ValueError(f"Expected normalized actor observations of shape {(batch_size, self.num_actor_obs)}.")
        if eps.shape != (batch_size, num_samples, action_dim):
            raise ValueError("Unexpected CFM epsilon shape.")
        if t.shape != (batch_size, num_samples, 1):
            raise ValueError("Unexpected CFM timestep shape.")

        scaled_actions = actions / self.actor_scale
        embedded_t = self._embed_timestep(t)
        x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
        actor_obs_expanded = actor_obs[:, None, :].expand(batch_size, num_samples, -1)
        actor_input = torch.cat([actor_obs_expanded, embedded_t, x_t], dim=-1)
        mlp_output = actor(actor_input)
        mlp_output = self.actor_mlp_output_scale * mlp_output

        velocity_pred = mlp_output
        x0_pred = x_t - t * velocity_pred
        x1_pred = x0_pred + velocity_pred
        target_velocity = eps - scaled_actions[:, None, :]
        loss = self._compute_squared_error(velocity_pred, target_velocity)
        return loss, x1_pred, x0_pred

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        del kwargs
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

    def _act_from_actor_obs(
        self,
        actor_obs: torch.Tensor,
        *,
        training: bool,
        eval_mode: str = "zero",
        eval_fixed_seed: int = 12345,
    ) -> torch.Tensor:
        batch_size = actor_obs.shape[0]
        device = actor_obs.device
        dtype = actor_obs.dtype

        if training:
            x_t = torch.randn(batch_size, self.num_actions, device=device, dtype=dtype)
        elif eval_mode == "zero":
            x_t = torch.zeros(batch_size, self.num_actions, device=device, dtype=dtype)
        elif eval_mode == "fixed_seed":
            generator = torch.Generator(device=device)
            generator.manual_seed(eval_fixed_seed)
            x_t = torch.randn(batch_size, self.num_actions, device=device, dtype=dtype, generator=generator)
        elif eval_mode == "random":
            x_t = torch.randn(batch_size, self.num_actions, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown evaluation mode: {eval_mode}.")

        flow_steps = self.training_sampling_steps if training else self.sampling_steps
        t_path = torch.linspace(1.0, 0.0, flow_steps + 1, device=device, dtype=dtype)
        t_current = t_path[:-1]
        dt = t_path[1:] - t_current
        x_t = self._integrate_flow_impl(actor_obs, x_t, t_current, dt, flow_steps)
        return self.actor_scale * x_t

    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        if self.timestep_embed_dim == 0:
            return t.new_zeros(*t.shape[:-1], 0)
        half_dim = self.timestep_embed_dim // 2
        freqs = 2 ** torch.arange(half_dim, device=t.device, dtype=t.dtype)
        scaled_t = t * freqs
        return torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)

    def _integrate_flow(
        self,
        actor_obs: torch.Tensor,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        dt: torch.Tensor,
        flow_steps: int,
    ) -> torch.Tensor:
        batch_size = actor_obs.shape[0]
        half_dim = self.timestep_embed_dim // 2
        freqs = (
            2 ** torch.arange(half_dim, device=actor_obs.device, dtype=actor_obs.dtype)
            if self.timestep_embed_dim > 0
            else actor_obs.new_zeros(0)
        )

        for step in range(flow_steps):
            if self.timestep_embed_dim > 0:
                t_val = t_current[step].reshape(1, 1)
                scaled_t = t_val * freqs
                embedded_t = torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1).expand(batch_size, -1)
                actor_input = torch.cat([actor_obs, embedded_t, x_t], dim=-1)
            else:
                actor_input = torch.cat([actor_obs, x_t], dim=-1)

            mlp_output = self.actor(actor_input)
            mlp_output = self.actor_mlp_output_scale * mlp_output
            x_t = x_t + mlp_output * dt[step]

        return x_t

    def _compute_squared_error(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        squared_errors = (predictions - targets) ** 2
        if self.cfm_loss_reduction == "mean":
            return torch.mean(squared_errors, dim=-1)
        if self.cfm_loss_reduction == "sum":
            return torch.sum(squared_errors, dim=-1)
        return torch.sum(squared_errors, dim=-1) / (squared_errors.shape[-1] ** 0.5)
