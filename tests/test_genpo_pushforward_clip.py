import copy
import math
import unittest

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.genpo_pushforward_clip import GenPOPFClip, GenPOU0Clip
from rsl_rl.modules.actor_critic_genpo import ActorCriticGenPO
from rsl_rl.modules.genpo.flow import Flow


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


def make_policy() -> ActorCriticGenPO:
    obs = make_obs(batch_size=4)
    return ActorCriticGenPO(
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


def perturb_policy(policy: ActorCriticGenPO, scale: float) -> None:
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(scale)


class TestGenPOPushforwardClip(unittest.TestCase):
    def test_flow_jacobian_determinant_matches_constant_scale(self) -> None:
        torch.manual_seed(0)
        mix_coeff = 0.9
        n_steps = 2
        action_dim = 1
        flow = Flow(
            input_dim=3,
            output_dim=action_dim,
            a_dim=action_dim,
            actor_hidden_dim=[8],
            time_dim=4,
            time_hidden_dim=[8],
            activation=torch.nn.Tanh(),
            n_steps=n_steps,
            mix_coeff=mix_coeff,
            device="cpu",
        )
        observation = torch.tensor([[0.25, -0.50]], dtype=torch.float32)
        expected_det = mix_coeff ** (2 * action_dim * n_steps)

        for latent in (
            torch.tensor([0.2, -0.1], dtype=torch.float32, requires_grad=True),
            torch.tensor([0.7, 0.3], dtype=torch.float32, requires_grad=True),
        ):
            jac = torch.autograd.functional.jacobian(
                lambda value: flow.forward_from_latent(observation, value),
                latent,
            )
            determinant = torch.linalg.det(jac)
            self.assertAlmostEqual(determinant.item(), expected_det, places=5)

    def test_u0_log_ratio_matches_dummy_log_ratio_on_diagonal_actions(self) -> None:
        torch.manual_seed(1)
        policy = make_policy()
        algorithm = GenPOU0Clip(policy, num_learning_epochs=1, num_mini_batches=1, use_compress=False, device="cpu")
        reference_policy = copy.deepcopy(policy)
        perturb_policy(policy, scale=0.05)

        obs_batch = make_obs(batch_size=4, offset=0.2)
        real_action = torch.tensor(
            [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2], [0.0, 0.6]],
            dtype=torch.float32,
        )
        diagonal_actions = torch.cat((real_action, real_action), dim=-1)

        section_log_ratio = algorithm._estimate_log_ratio(obs_batch, diagonal_actions, reference_policy)
        current_latent = policy.inverse_latent(diagonal_actions, obs_batch)
        old_latent = reference_policy.inverse_latent(diagonal_actions, obs_batch)
        dummy_log_ratio = algorithm._log_ratio_from_latents(current_latent, old_latent)

        self.assertTrue(torch.allclose(section_log_ratio, dummy_log_ratio, atol=1e-5, rtol=1e-5))

    def test_pf_log_ratio_degenerates_to_u0_when_fiber_variance_is_zero(self) -> None:
        torch.manual_seed(2)
        policy = make_policy()
        pf_algorithm = GenPOPFClip(
            policy,
            pf_num_samples=2,
            num_learning_epochs=1,
            num_mini_batches=1,
            use_compress=False,
            device="cpu",
        )
        u0_algorithm = GenPOU0Clip(policy, num_learning_epochs=1, num_mini_batches=1, use_compress=False, device="cpu")
        reference_policy = copy.deepcopy(policy)
        perturb_policy(policy, scale=0.02)

        obs_batch = make_obs(batch_size=4, offset=0.4)
        real_action = torch.tensor(
            [[0.2, -0.1], [0.5, 0.0], [-0.2, 0.3], [0.1, 0.1]],
            dtype=torch.float32,
        )
        diagonal_actions = torch.cat((real_action, real_action), dim=-1)

        pf_log_ratio = pf_algorithm._estimate_log_ratio(obs_batch, diagonal_actions, reference_policy)
        u0_log_ratio = u0_algorithm._estimate_log_ratio(obs_batch, diagonal_actions, reference_policy)

        self.assertTrue(torch.allclose(pf_log_ratio, u0_log_ratio, atol=1e-5, rtol=1e-5))

    def test_section_kl_uses_section_estimator_not_dummy_sample_ratio(self) -> None:
        torch.manual_seed(3)
        policy = make_policy()
        algorithm = GenPOU0Clip(policy, num_learning_epochs=1, num_mini_batches=1, use_compress=False, device="cpu")
        reference_policy = copy.deepcopy(policy)
        perturb_policy(policy, scale=0.03)

        obs_batch = make_obs(batch_size=4, offset=0.6)
        dummy_actions = torch.tensor(
            [
                [0.6, -0.2, -0.1, 0.3],
                [0.4, 0.5, 0.1, -0.2],
                [-0.3, 0.7, 0.2, -0.5],
                [0.1, -0.4, -0.6, 0.8],
            ],
            dtype=torch.float32,
        )

        section_log_ratio = algorithm._estimate_log_ratio(obs_batch, dummy_actions, reference_policy)
        dummy_current_latent = policy.inverse_latent(dummy_actions, obs_batch)
        dummy_old_latent = reference_policy.inverse_latent(dummy_actions, obs_batch)
        dummy_log_ratio = algorithm._log_ratio_from_latents(dummy_current_latent, dummy_old_latent)

        self.assertFalse(torch.allclose(section_log_ratio, dummy_log_ratio, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(
                algorithm._estimated_kl_from_log_ratio(section_log_ratio),
                (-section_log_ratio).mean(),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_pushforward_kl_uses_pushforward_estimator_not_dummy_sample_ratio(self) -> None:
        torch.manual_seed(4)
        policy = make_policy()
        algorithm = GenPOPFClip(
            policy,
            pf_num_samples=2,
            num_learning_epochs=1,
            num_mini_batches=1,
            use_compress=False,
            device="cpu",
        )
        reference_policy = copy.deepcopy(policy)
        perturb_policy(policy, scale=0.03)

        obs_batch = make_obs(batch_size=4, offset=0.8)
        dummy_actions = torch.tensor(
            [
                [0.7, -0.1, -0.2, 0.4],
                [0.2, 0.6, 0.0, -0.3],
                [-0.4, 0.5, 0.3, -0.6],
                [0.0, -0.5, -0.7, 0.9],
            ],
            dtype=torch.float32,
        )

        pf_log_ratio = algorithm._estimate_log_ratio(obs_batch, dummy_actions, reference_policy)
        dummy_current_latent = policy.inverse_latent(dummy_actions, obs_batch)
        dummy_old_latent = reference_policy.inverse_latent(dummy_actions, obs_batch)
        dummy_log_ratio = algorithm._log_ratio_from_latents(dummy_current_latent, dummy_old_latent)

        self.assertFalse(torch.allclose(pf_log_ratio, dummy_log_ratio, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(
                algorithm._estimated_kl_from_log_ratio(pf_log_ratio),
                (-pf_log_ratio).mean(),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_log_domain_clipping_is_symmetric_in_log_space(self) -> None:
        policy = make_policy()
        algorithm = GenPOU0Clip(
            policy,
            log_clip_delta=0.3,
            num_learning_epochs=1,
            num_mini_batches=1,
            use_compress=False,
            device="cpu",
        )

        log_ratio = torch.tensor([-0.8, -0.1, 0.2, 0.9], dtype=torch.float32)
        clipped_ratio = torch.exp(algorithm._log_clip(log_ratio))

        self.assertTrue(torch.allclose(algorithm._log_clip(log_ratio), torch.tensor([-0.3, -0.1, 0.2, 0.3])))
        self.assertAlmostEqual(clipped_ratio[0].item(), math.exp(-0.3), places=6)
        self.assertAlmostEqual(clipped_ratio[-1].item(), math.exp(0.3), places=6)


if __name__ == "__main__":
    unittest.main()
