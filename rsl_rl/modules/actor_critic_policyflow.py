from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization

from .policyflow import ConditionLinearLayer, ConditionMlp, ContinuousNormalizingFlow, FeedForwardNetwork, FlowMlp


class ActorCriticPolicyFlow(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        actor_activations: list[str] | None = None,
        critic_activations: list[str] | None = None,
        flow_condition_hidden_dims: tuple[int] | list[int] | None = None,
        flow_condition_activations: list[str] | None = None,
        activation: str = "elu",
        flow_embedding_dim: int = 64,
        emb_dim: int | None = None,
        flow_sample_steps: int = 10,
        flow_sample_step_schedule: str = "uniform_continuous",
        flow_interpolation_type: str = "rectified_flow",
        flow_timestep_embedding_type: str = "fourier",
        timestep_emb_type: str | None = None,
        flow_conditioning: str = "linear",
        flow_use_ema: bool = False,
        flow_ema_rate: float = 0.995,
        variance_log_std_max: float = 4.0,
        variance_log_std_min: float = -20.0,
        variance_std_init: float = 1.0,
        device: str | torch.device = "cpu",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticPolicyFlow.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.num_actions = num_actions

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            if len(obs[obs_group].shape) != 2:
                raise ValueError("The ActorCriticPolicyFlow module only supports 1D observations.")
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            if len(obs[obs_group].shape) != 2:
                raise ValueError("The ActorCriticPolicyFlow module only supports 1D observations.")
            num_critic_obs += obs[obs_group].shape[-1]

        flow_embedding_dim = emb_dim if emb_dim is not None else flow_embedding_dim
        flow_timestep_embedding_type = (
            timestep_emb_type if timestep_emb_type is not None else flow_timestep_embedding_type
        )
        actor_hidden_dims = list(actor_hidden_dims)
        critic_hidden_dims = list(critic_hidden_dims)
        flow_condition_hidden_dims = list(actor_hidden_dims if flow_condition_hidden_dims is None else flow_condition_hidden_dims)

        if actor_activations is None:
            actor_activations = [activation] * len(actor_hidden_dims) + ["linear"]
        if critic_activations is None:
            critic_activations = [activation] * len(critic_hidden_dims) + ["linear"]
        if flow_condition_activations is None:
            flow_condition_activations = [activation] * len(flow_condition_hidden_dims) + ["linear"]

        flow_network = FlowMlp(
            x_dim=num_actions,
            emb_dim=flow_embedding_dim,
            activations=actor_activations,
            hidden_dims=actor_hidden_dims,
            timestep_emb_type=flow_timestep_embedding_type,
        )

        conditioning_mode = flow_conditioning.lower()
        if conditioning_mode == "linear":
            condition_network = ConditionLinearLayer(cond_dim=num_actor_obs, emb_dim=flow_embedding_dim)
        elif conditioning_mode == "mlp":
            condition_network = ConditionMlp(
                cond_dim=num_actor_obs,
                emb_dim=flow_embedding_dim,
                activations=flow_condition_activations,
                hidden_dims=flow_condition_hidden_dims,
            )
        else:
            raise ValueError(f"Unsupported PolicyFlow conditioning mode: {flow_conditioning}.")

        self.actor = ContinuousNormalizingFlow(
            x_dims=num_actions,
            nn_flow=flow_network,
            nn_condition=condition_network,
            ema_rate=flow_ema_rate,
            using_ema=flow_use_ema,
            sample_steps=flow_sample_steps,
            sample_step_schedule=flow_sample_step_schedule,
            interpolation_type=flow_interpolation_type,
            log_std_max=variance_log_std_max,
            log_std_min=variance_log_std_min,
            std_init=variance_std_init,
            device=device,
        )
        self.critic = FeedForwardNetwork(
            input_size=num_critic_obs,
            output_size=1,
            activations=critic_activations,
            hidden_dims=critic_hidden_dims,
        )

        print(f"PolicyFlow actor: {self.actor.model}")
        print(f"PolicyFlow critic: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else torch.nn.Identity()
        )

        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else torch.nn.Identity()
        )

        self._distribution: Normal | None = None

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        if self._distribution is None:
            raise ValueError("PolicyFlow distribution is not populated. Call 'sample_actions' or 'act_inference' first.")
        return self._distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self._distribution is None:
            raise ValueError("PolicyFlow distribution is not populated. Call 'sample_actions' or 'act_inference' first.")
        return self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self._distribution is None:
            raise ValueError("PolicyFlow distribution is not populated. Call 'sample_actions' or 'act_inference' first.")
        return self._distribution.entropy().sum(dim=-1)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actions, _ = self.sample_actions(obs, degenerate2gaussian=kwargs.get("degenerate2gaussian", False))
        return actions

    def sample_actions(
        self,
        obs: TensorDict,
        degenerate2gaussian: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        x0 = torch.randn((actor_obs.shape[0], self.num_actions), device=actor_obs.device, dtype=actor_obs.dtype)
        if degenerate2gaussian:
            x0 = torch.zeros_like(x0)

        actions_prior, std = self.actor.sample(
            x0=x0,
            condition=actor_obs,
            n_samples=actor_obs.shape[0],
        )
        delta_distribution = Normal(torch.zeros_like(actions_prior), std)
        delta_actions = delta_distribution.sample().detach()
        actions = actions_prior.detach() + delta_actions
        delta_actions_log_prob = delta_distribution.log_prob(delta_actions).sum(-1).detach()
        self._distribution = Normal(actions_prior.detach(), std.detach())

        info = {
            "actions_prior": actions_prior.detach(),
            "flow_x0": x0.detach(),
            "delta_actions": delta_actions,
            "delta_actions_std": std.detach(),
            "delta_actions_log_prob": delta_actions_log_prob,
        }
        return actions.detach(), info

    def act_inference(self, obs: TensorDict, degenerate2gaussian: bool = False) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        x0 = torch.randn((actor_obs.shape[0], self.num_actions), device=actor_obs.device, dtype=actor_obs.dtype)
        if degenerate2gaussian:
            x0 = torch.zeros_like(x0)

        action_mean, action_std = self.actor.sample(
            x0=x0,
            condition=actor_obs,
            n_samples=actor_obs.shape[0],
        )
        self._distribution = Normal(action_mean.detach(), action_std.detach())
        return self._distribution.sample().detach()

    def compute_flow_variation(
        self,
        obs: TensorDict,
        actions_prior: torch.Tensor,
        flow_x0: torch.Tensor,
        compute_brownian_reg_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self.actor.compute_flow_variation(
            x1=actions_prior,
            condition=actor_obs,
            x0=flow_x0,
            compute_brownian_reg_loss=compute_brownian_reg_loss,
        )

    def sync_reference_model(self) -> None:
        self.actor.sync_reference_model()

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        del kwargs
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.critic(critic_obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
