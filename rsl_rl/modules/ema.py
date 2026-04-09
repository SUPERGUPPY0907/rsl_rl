from __future__ import annotations

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Track an exponential moving average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.95,
        device: torch.device | str | None = None,
    ) -> None:
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.shadow_params: dict[str, torch.Tensor] = {}
        self.model_params: dict[str, torch.nn.Parameter] = {}
        self.backup_params: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.model_params[name] = param
                self.shadow_params[name] = param.data.detach().clone().to(self.device)

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model_params.items():
            if param.requires_grad:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data.to(self.device),
                    alpha=1.0 - self.decay,
                )

    @torch.no_grad()
    def reset_to_current(self) -> None:
        for name, param in self.model_params.items():
            if param.requires_grad:
                self.shadow_params[name].copy_(param.data.to(self.device))

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "shadow_params": {name: value.clone() for name, value in self.shadow_params.items()},
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        shadow_params = state_dict["shadow_params"]
        self.shadow_params = {name: value.clone().to(self.device) for name, value in shadow_params.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].to(param.device))

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        self.backup_params = {}
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                self.backup_params[name] = param.data.detach().clone()

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name].to(param.device))
        self.backup_params = {}
