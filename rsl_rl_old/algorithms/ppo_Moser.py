# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain

from rsl_rl.modules import ActorCriticMoser
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable


class MoserPPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCriticMoser
    """The actor critic module."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        lambda_minus=1.0,  # Moser Flow: weight for negative density penalty
        sigma=1e-3,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        # Create rollout storage
        self.storage: RolloutStorage = None  # type: ignore
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lambda_minus = lambda_minus  # Moser Flow parameter
        self.sigma = sigma
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape
    ):
        rnd_state_shape = None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            rnd_state_shape,
            self.device,
        )

    def act(self, obs, critic_obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.privileged_observations = critic_obs

        self.transition.values = self.policy.evaluate(critic_obs).detach()
        ## prob in storage here
        self.transition.actions_prob = self.policy.get_actions_prob(self.transition.actions, obs, create_graph=False).detach()
    
        
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_positivity_loss = 0  # Track Moser Flow positivity loss


        generator = self.storage.moser_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_prob_batch,    
        ) in generator:

            print('actions_batch:', torch.norm(actions_batch, p=2, dim=1).mean())
            
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=None, hidden_states=None)
            actions_prob_batch = self.policy.get_actions_prob(actions_batch, obs_batch, create_graph=True)
            # -- critic
            value_batch = self.policy.evaluate(critic_obs_batch, masks=None, hidden_states=None)
            # -- entropy (approximate)
            # entropy_batch = self.policy.entropy

            # prob_plus and prob_minus
            old_actions_log_prob_batch_plus = torch.log(torch.nn.functional.relu(old_actions_prob_batch - self.sigma) + self.sigma)
            actions_log_prob_barch_plus = torch.log(torch.nn.functional.relu(actions_prob_batch - self.sigma) + self.sigma)

            # KL
            # if self.desired_kl is not None and self.schedule == "adaptive":
            #     with torch.inference_mode():
            #         kl = old_actions_log_prob_batch_plus - actions_log_prob_barch_plus
            #         kl_mean = torch.mean(kl)

            #         # Reduce the KL divergence across all GPUs
            #         if self.is_multi_gpu:
            #             torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            #             kl_mean /= self.gpu_world_size

            #         # Update the learning rate
            #         # Perform this adaptation only on the main process
            #         # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
            #         #       then the learning rate should be the same across all GPUs.
            #         if self.gpu_global_rank == 0:
            #             if kl_mean > self.desired_kl * 2.0:
            #                 self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            #             elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            #                 self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            #         # Update the learning rate for all GPUs
            #         if self.is_multi_gpu:
            #             lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            #             torch.distributed.broadcast(lr_tensor, src=0)
            #             self.learning_rate = lr_tensor.item()
            #         # Update the learning rate for all parameter groups
            #         for param_group in self.optimizer.param_groups:
            #             param_group["lr"] = self.learning_rate
            # Surrogate loss
            log_diff = old_actions_log_prob_batch_plus - actions_log_prob_barch_plus
            ratio = torch.exp(log_diff)
            ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
            # ratio = torch.exp(old_actions_log_prob_batch_plus - actions_log_prob_barch_plus)
            # print('ratio:', ratio)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            # print('surrogate_loss', surrogate_loss)
            
            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()


            # Moser Flow: Compute positivity loss (penalty for negative density)
            if self.lambda_minus > 0:
                # positivity_loss = torch.mean(actions_prob_barch_minus)
                # positivity_loss = torch.mean(torch.nn.functional.softplus(actions_prob_batch))
                positivity_loss = 0.0
            else:
                positivity_loss = 0.0

            loss = surrogate_loss + self.value_loss_coef * value_loss  + self.lambda_minus * positivity_loss

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            
            loss.backward()

            # for name, param in self.policy.actor.named_parameters():
            #     if param.grad is not None:
            #         print(f"参数: {name}")
            #         print(f"  梯度形状: {param.grad.shape}")
            #         print(f"  梯度值（部分）: {param.grad.view(-1)[:5]}") 
            #     else:
            #         print(f"参数: {name} 梯度为 None (可能requires_grad=False)")

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            
            if isinstance(positivity_loss, torch.Tensor):
                mean_positivity_loss += positivity_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_positivity_loss /= num_updates

        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "positivity": mean_positivity_loss,  # Add Moser Flow positivity loss
        }

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[1])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
