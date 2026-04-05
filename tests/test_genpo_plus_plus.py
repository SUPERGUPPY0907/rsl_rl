import unittest

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.genpo_plus_plus import GenPOPlusPlus
from rsl_rl.modules.actor_critic_genpo import ActorCriticGenPO


def make_obs(batch_size: int, offset: float = 0.0) -> TensorDict:
    base = torch.linspace(0.0 + offset, 1.0 + offset, batch_size)
    policy_obs = torch.stack((base, base + 0.1, base + 0.2), dim=-1)
    privileged_obs = torch.stack((base + 0.3, base + 0.4), dim=-1)
    return TensorDict(
        {
            "policy": policy_obs,
            "privileged": privileged_obs,
        },
        batch_size=[batch_size],
    )


def make_algorithm(
    *,
    lambda_dir: float = 0.0,
    lambda_mirror: float = 0.0,
    directional_advantage_quantile: float = 0.5,
    directional_max_groups: int = 2,
) -> tuple[ActorCriticGenPO, GenPOPlusPlus]:
    obs = make_obs(batch_size=5)
    policy = ActorCriticGenPO(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["policy", "privileged"]},
        num_actions=2,
        flow_num_steps=2,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        time_hidden_dims=[16],
        time_dim=8,
        device="cpu",
    )
    algorithm = GenPOPlusPlus(
        policy,
        lambda_dir=lambda_dir,
        lambda_mirror=lambda_mirror,
        directional_advantage_quantile=directional_advantage_quantile,
        directional_max_groups=directional_max_groups,
        num_learning_epochs=1,
        num_mini_batches=1,
        use_compress=False,
        device="cpu",
    )
    return policy, algorithm


def populate_storage(algorithm: GenPOPlusPlus, num_envs: int = 2, num_steps: int = 2) -> TensorDict:
    initial_obs = make_obs(num_envs)
    algorithm.init_storage(
        algorithm.get_storage_training_type(),
        num_envs,
        num_steps,
        initial_obs,
        algorithm.get_storage_action_shape(algorithm.policy.action_dim),
    )

    current_obs = initial_obs
    for step in range(num_steps):
        algorithm.act(current_obs)
        next_obs = make_obs(num_envs, offset=float(step + 1))
        rewards = torch.full((num_envs,), float(step + 1))
        dones = torch.zeros(num_envs, dtype=torch.uint8)
        algorithm.process_env_step(next_obs, rewards, dones, {})
        current_obs = next_obs

    algorithm.compute_returns(current_obs)
    return current_obs


class TestGenPOPlusPlusMirroredLoss(unittest.TestCase):
    def test_select_directional_anchor_indices_respects_quantile_and_cap(self) -> None:
        _, algorithm = make_algorithm(lambda_mirror=1.0, directional_advantage_quantile=0.5, directional_max_groups=2)
        advantages = torch.tensor([-0.5, 0.1, 0.8, 0.9, 1.0])

        indices = algorithm._select_directional_anchor_indices(advantages)

        self.assertEqual(indices.numel(), 2)
        self.assertEqual(set(indices.tolist()), {3, 4})

    def test_mirrored_action_loss_returns_zero_when_disabled_or_no_anchor(self) -> None:
        torch.manual_seed(0)
        policy, algorithm = make_algorithm(lambda_mirror=0.0)
        obs_batch = make_obs(batch_size=5)
        actions_batch = policy.act(obs_batch).detach()
        advantages_batch = torch.ones(5, 1)

        disabled_loss = algorithm._compute_mirrored_action_loss(obs_batch, actions_batch, advantages_batch)

        self.assertEqual(disabled_loss.item(), 0.0)

        _, enabled_algorithm = make_algorithm(lambda_mirror=1.0)
        no_anchor_loss = enabled_algorithm._compute_mirrored_action_loss(
            obs_batch,
            actions_batch,
            -torch.ones(5, 1),
        )

        self.assertEqual(no_anchor_loss.item(), 0.0)

    def test_forward_latent_roundtrip_has_expected_shape_and_grad(self) -> None:
        torch.manual_seed(1)
        policy, _ = make_algorithm(lambda_mirror=1.0)
        obs_batch = make_obs(batch_size=5)
        actions_batch = policy.act(obs_batch).detach()

        policy.zero_grad()
        latent_batch = policy.inverse_latent(actions_batch, obs_batch)
        mirrored_actions = policy.forward_latent(-latent_batch, obs_batch)
        loss = mirrored_actions.square().mean()
        loss.backward()

        self.assertEqual(mirrored_actions.shape, actions_batch.shape)
        self.assertTrue(any(param.grad is not None and torch.any(param.grad != 0) for param in policy.parameters()))

    def test_mirrored_action_loss_detaches_advantages(self) -> None:
        torch.manual_seed(2)
        policy, algorithm = make_algorithm(lambda_mirror=1.0)
        obs_batch = make_obs(batch_size=5)
        actions_batch = policy.act(obs_batch).detach()
        base_advantages = torch.tensor([-0.5, 0.2, 0.8, 1.0, 1.2], requires_grad=True)

        loss = algorithm._compute_mirrored_action_loss(obs_batch, actions_batch, base_advantages.unsqueeze(-1))
        loss.backward()

        self.assertIsNone(base_advantages.grad)

    def test_update_reports_mirrored_loss_key_and_keeps_it_zero_when_disabled(self) -> None:
        torch.manual_seed(3)
        _, algorithm = make_algorithm(lambda_dir=0.0, lambda_mirror=0.0)
        populate_storage(algorithm, num_envs=2, num_steps=2)

        loss_dict = algorithm.update()

        self.assertIn("directional", loss_dict)
        self.assertIn("mirrored", loss_dict)
        self.assertEqual(loss_dict["mirrored"], 0.0)


if __name__ == "__main__":
    unittest.main()
