# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.genpo import GenPO
from rsl_rl.modules import ActorCriticBELMGenPO


class BELMGenPO(GenPO):
    """GenPO variant using a BELM-style reversible backbone and x0 environment actions."""

    policy: ActorCriticBELMGenPO

    def __init__(self, policy: ActorCriticBELMGenPO, **kwargs) -> None:
        kwargs.setdefault("use_compress", False)
        super().__init__(policy, **kwargs)

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        actions_full = self.policy.act(obs).detach()
        self.transition.actions = actions_full
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.action_latent = self.policy.get_actions_latent(actions_full).detach()
        self.transition.observations = obs

        return actions_full[..., : self.policy.action_dim]

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actions_full = self.policy.act_inference(obs)
        return actions_full[..., : self.policy.action_dim]

    def update(self) -> dict[str, float]:
        self.policy.begin_diagnostic_accumulation()
        diagnostics: dict[str, float] = {}
        try:
            loss_dict = super().update()
        finally:
            diagnostics = self.policy.end_diagnostic_accumulation()

        loss_dict.update(diagnostics)
        return loss_dict
