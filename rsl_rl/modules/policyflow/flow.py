from __future__ import annotations

import copy
import math
from typing import Callable

import torch
import torch.nn as nn

from .flow_net import ConditionNetBase, FlowNetBase, IdentityCondition, LearnableVariance
from .utils import SAMPLING_STEP_SCHEDULE, at_least_ndim


class ContinuousNormalizingFlow(nn.Module):
    def __init__(
        self,
        x_dims: int,
        nn_flow: FlowNetBase,
        nn_condition: ConditionNetBase | None = None,
        ema_rate: float = 0.995,
        using_ema: bool = False,
        sample_steps: int = 10,
        sample_step_schedule: str | Callable = "uniform_continuous",
        interpolation_type: str = "rectified_flow",
        log_std_max: float = 4.0,
        log_std_min: float = -20.0,
        std_init: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        if sample_steps < 1:
            raise ValueError("'sample_steps' must be positive.")

        self.ema_rate = ema_rate
        self.using_ema = using_ema
        self.sample_steps = sample_steps
        self.interpolation_type = interpolation_type

        if interpolation_type in {"stochastic_interpolant", "rectified_flow"}:
            final_t = 1.0
        elif interpolation_type == "trigflow":
            final_t = math.pi / 2.0
        else:
            raise ValueError(f"Interpolation type {interpolation_type} is not supported.")

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule not in SAMPLING_STEP_SCHEDULE:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
            schedule = SAMPLING_STEP_SCHEDULE[sample_step_schedule]([0.0, final_t], self.sample_steps)
        elif callable(sample_step_schedule):
            schedule = sample_step_schedule([0.0, final_t], self.sample_steps)
        else:
            raise ValueError("'sample_step_schedule' must be a callable or a string.")

        schedule = torch.as_tensor(schedule, dtype=torch.float32, device=device)
        if schedule.shape[0] != sample_steps + 1:
            raise ValueError("Sampling schedule must contain 'sample_steps + 1' values.")
        self.register_buffer("sample_step_schedule_tensor", schedule)

        time_steps = []
        for idx in range(self.sample_steps):
            t = schedule[idx]
            time_steps.append(t)
            delta_t = schedule[idx + 1] - schedule[idx]
            time_steps.append(t + delta_t / 2)
        time_steps.append(schedule[self.sample_steps])
        self.register_buffer("time_steps_tensor", torch.stack(time_steps))

        if nn_condition is None:
            nn_condition = IdentityCondition()

        self.model = nn.ModuleDict(
            {
                "flow": nn_flow.to(device),
                "condition": nn_condition.to(device),
                "variance": LearnableVariance(
                    dims=x_dims,
                    log_std_max=log_std_max,
                    log_std_min=log_std_min,
                    std_init=std_init,
                ).to(device),
            }
        )
        self.model_ema = copy.deepcopy(self.model).requires_grad_(False)
        self.model_last = copy.deepcopy(self.model).requires_grad_(False)
        self.model_ema.eval()
        self.model_last.eval()

    def train(self, mode: bool = True) -> ContinuousNormalizingFlow:
        super().train(mode)
        self.model_ema.eval()
        self.model_last.eval()
        return self

    def ema_update(self) -> None:
        with torch.no_grad():
            for param, param_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                param_ema.data.mul_(self.ema_rate).add_(param.data, alpha=1.0 - self.ema_rate)

    def compute_flow_variation(
        self,
        x1: torch.Tensor,
        condition: torch.Tensor,
        x0: torch.Tensor | None = None,
        compute_brownian_reg_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x0 is None:
            x0 = torch.randn_like(x1)
        elif x0.shape != x1.shape:
            raise ValueError("'x0' and 'x1' must have the same shape.")

        idx = torch.randint(
            low=0,
            high=self.time_steps_tensor.shape[0],
            size=(x1.shape[0],),
            device=x1.device,
        )
        t = self.time_steps_tensor[idx]
        alpha = at_least_ndim(t, x1.dim())

        if self.interpolation_type == "rectified_flow":
            xt = (1.0 - alpha) * x0 + alpha * x1
        elif self.interpolation_type == "stochastic_interpolant":
            xt = (
                (1.0 - alpha) * x0
                + alpha * x1
                + torch.sqrt(2.0 * alpha * (1.0 - alpha).clamp(min=1e-6)) * torch.randn_like(x1)
            )
        elif self.interpolation_type == "trigflow":
            xt = torch.cos(alpha) * x0 + torch.sin(alpha) * x1
        else:
            raise ValueError(f"Interpolation type {self.interpolation_type} is not supported.")

        with torch.no_grad():
            condition_embedded_last = self.model_last["condition"](condition)
            vel_field_last = self.model_last["flow"](xt, t, condition_embedded_last).detach()

        condition_embedded = self.model["condition"](condition)
        vel_field = self.model["flow"](xt, t, condition_embedded)
        delta_vel = vel_field - vel_field_last
        std = torch.ones_like(x1) * self.model["variance"].std

        if not compute_brownian_reg_loss:
            return delta_vel, std

        beta = 1.0
        if self.interpolation_type == "rectified_flow":
            brownian_reg_loss = torch.nn.functional.mse_loss(
                (1 - alpha) * vel_field,
                beta * (xt - alpha * vel_field_last),
            )
        elif self.interpolation_type == "stochastic_interpolant":
            brownian_reg_loss = torch.nn.functional.mse_loss(
                (2.0 * ((alpha - 0.5) ** 2) + 0.5) * vel_field,
                beta * (xt - alpha * vel_field_last),
            )
        else:
            brownian_reg_loss = torch.nn.functional.mse_loss(
                torch.cos(alpha) * vel_field,
                beta * (torch.cos(alpha) * xt - torch.sin(alpha) * vel_field_last),
            )
        return delta_vel, std, brownian_reg_loss

    def sync_reference_model(self) -> None:
        source_model = self.model_ema if self.using_ema else self.model
        source_state = source_model.state_dict()
        self.model_last.load_state_dict(source_state)

    def sample(
        self,
        x0: torch.Tensor,
        condition: torch.Tensor,
        n_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xt = x0
        model = self.model_ema if self.using_ema else self.model
        condition_embedded = model["condition"](condition)

        for idx in range(self.sample_steps):
            t = torch.full(
                (n_samples,),
                float(self.sample_step_schedule_tensor[idx].item()),
                dtype=xt.dtype,
                device=xt.device,
            )
            delta_t = self.sample_step_schedule_tensor[idx + 1] - self.sample_step_schedule_tensor[idx]
            vel_t = model["flow"](xt, t, condition_embedded)
            xt_middle = xt + vel_t * delta_t / 2
            vel_t = model["flow"](xt_middle, t + delta_t / 2, condition_embedded)
            xt = xt + delta_t * vel_t

        std = torch.ones_like(xt) * model["variance"].std
        return xt.detach(), std.detach()
