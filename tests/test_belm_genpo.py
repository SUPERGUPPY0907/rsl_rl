import copy
import unittest

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.belm_genpo import BELMGenPO
from rsl_rl.modules.actor_critic_belm_genpo import ActorCriticBELMGenPO
from rsl_rl.modules.genpo.flow import BELMFlow
from rsl_rl.runners.on_policy_flow_runner import OnPolicyFlowRunner


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


def make_policy(
    *,
    a_coeff: float | None = None,
    b_coeff: float | None = None,
    eps_coeff: float | None = None,
    lag_coeff: float | None = 0.95,
    mix_para: float | None = None,
    flow_num_steps: int = 2,
) -> ActorCriticBELMGenPO:
    obs = make_obs(batch_size=4)
    return ActorCriticBELMGenPO(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["policy", "privileged"]},
        num_actions=2,
        flow_num_steps=flow_num_steps,
        a_coeff=a_coeff,
        b_coeff=b_coeff,
        eps_coeff=eps_coeff,
        lag_coeff=lag_coeff,
        mix_para=mix_para,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        time_hidden_dims=[16],
        time_dim=8,
        device="cpu",
    )


def perturb_policy(policy: ActorCriticBELMGenPO, scale: float) -> None:
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(scale)


def copy_flow_network_weights(target: BELMFlow, source: BELMFlow) -> None:
    target.vec_field.load_state_dict(source.vec_field.state_dict())
    target.time_mlp.load_state_dict(source.time_mlp.state_dict())


def populate_storage(algorithm: BELMGenPO, num_envs: int = 2, num_steps: int = 2) -> TensorDict:
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


class DummyFlowEnv:
    def __init__(self, num_envs: int = 4, num_actions: int = 2) -> None:
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = "cpu"
        self.cfg = {}
        self.unwrapped = self
        self.step_dt = 0.05

    def get_observations(self) -> TensorDict:
        return make_obs(self.num_envs)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError


class TestBELMFlow(unittest.TestCase):
    def test_explicit_coefficients_match_legacy_lag_configuration(self) -> None:
        torch.manual_seed(0)
        flow_legacy = BELMFlow(
            input_dim=5,
            output_dim=2,
            a_dim=2,
            actor_hidden_dim=[8],
            time_dim=4,
            time_hidden_dim=[8],
            activation=torch.nn.Tanh(),
            n_steps=3,
            lag_coeff=0.9,
            device="cpu",
        )
        flow_explicit = BELMFlow(
            input_dim=5,
            output_dim=2,
            a_dim=2,
            actor_hidden_dim=[8],
            time_dim=4,
            time_hidden_dim=[8],
            activation=torch.nn.Tanh(),
            n_steps=3,
            a_coeff=0.1,
            b_coeff=0.9,
            eps_coeff=1.0,
            device="cpu",
        )
        copy_flow_network_weights(flow_explicit, flow_legacy)

        observations = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32)
        latent = torch.tensor([[0.5, -0.2, 0.1, 0.7]], dtype=torch.float32)

        legacy_actions = flow_legacy.forward_from_latent(observations, latent)
        explicit_actions = flow_explicit.forward_from_latent(observations, latent)

        self.assertTrue(torch.allclose(legacy_actions, explicit_actions, atol=1e-6, rtol=1e-6))

    def test_forward_inverse_roundtrip_with_explicit_coefficients(self) -> None:
        torch.manual_seed(1)
        flow = BELMFlow(
            input_dim=5,
            output_dim=2,
            a_dim=2,
            actor_hidden_dim=[8],
            time_dim=4,
            time_hidden_dim=[8],
            activation=torch.nn.Tanh(),
            n_steps=3,
            a_coeff=0.25,
            b_coeff=0.9,
            eps_coeff=2.0,
            device="cpu",
        )
        observations = torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32)
        latent = torch.tensor([[0.5, -0.2, 0.1, 0.7]], dtype=torch.float32)

        actions = flow.forward_from_latent(observations, latent)
        recovered = flow.inverse_latent(observations, actions)

        self.assertTrue(torch.allclose(recovered, latent, atol=1e-5, rtol=1e-5))

    def test_jacobian_determinant_depends_only_on_b_coeff(self) -> None:
        torch.manual_seed(2)
        b_coeff = 0.9
        n_steps = 3
        action_dim = 2

        flow_reference = BELMFlow(
            input_dim=4,
            output_dim=action_dim,
            a_dim=action_dim,
            actor_hidden_dim=[8],
            time_dim=4,
            time_hidden_dim=[8],
            activation=torch.nn.Tanh(),
            n_steps=n_steps,
            a_coeff=0.05,
            b_coeff=b_coeff,
            eps_coeff=1.0,
            device="cpu",
        )
        flow_shifted = BELMFlow(
            input_dim=4,
            output_dim=action_dim,
            a_dim=action_dim,
            actor_hidden_dim=[8],
            time_dim=4,
            time_hidden_dim=[8],
            activation=torch.nn.Tanh(),
            n_steps=n_steps,
            a_coeff=0.5,
            b_coeff=b_coeff,
            eps_coeff=2.0,
            device="cpu",
        )
        copy_flow_network_weights(flow_shifted, flow_reference)

        observation = torch.tensor([[0.25, -0.50]], dtype=torch.float32)
        latent = torch.tensor([0.2, -0.1, 0.3, 0.4], dtype=torch.float32, requires_grad=True)

        jac_reference = torch.autograd.functional.jacobian(
            lambda value: flow_reference.forward_from_latent(observation, value),
            latent,
        )
        jac_shifted = torch.autograd.functional.jacobian(
            lambda value: flow_shifted.forward_from_latent(observation, value),
            latent,
        )

        det_reference = torch.linalg.det(jac_reference)
        det_shifted = torch.linalg.det(jac_shifted)
        expected_det = abs(b_coeff) ** (action_dim * n_steps)

        self.assertAlmostEqual(abs(det_reference.item()), expected_det, places=5)
        self.assertAlmostEqual(abs(det_shifted.item()), expected_det, places=5)


class TestBELMGenPO(unittest.TestCase):
    def test_policy_maps_legacy_and_explicit_coefficients(self) -> None:
        legacy_policy = make_policy(lag_coeff=0.8)
        mix_policy = make_policy(lag_coeff=None, mix_para=0.7)
        explicit_policy = make_policy(a_coeff=0.3, b_coeff=0.8, eps_coeff=0.5, mix_para=0.95)

        self.assertAlmostEqual(legacy_policy.a_coeff, 0.2, places=6)
        self.assertAlmostEqual(legacy_policy.b_coeff, 0.8, places=6)
        self.assertAlmostEqual(legacy_policy.eps_coeff, 1.0, places=6)
        self.assertAlmostEqual(mix_policy.a_coeff, 0.3, places=6)
        self.assertAlmostEqual(mix_policy.b_coeff, 0.7, places=6)
        self.assertAlmostEqual(explicit_policy.a_coeff, 0.3, places=6)
        self.assertAlmostEqual(explicit_policy.b_coeff, 0.8, places=6)
        self.assertAlmostEqual(explicit_policy.eps_coeff, 0.5, places=6)

    def test_act_returns_x0_and_storage_keeps_full_dummy_action(self) -> None:
        torch.manual_seed(3)
        policy = make_policy()
        algorithm = BELMGenPO(policy, num_learning_epochs=1, num_mini_batches=1, device="cpu")
        initial_obs = make_obs(batch_size=3)
        algorithm.init_storage(
            algorithm.get_storage_training_type(),
            3,
            2,
            initial_obs,
            algorithm.get_storage_action_shape(algorithm.policy.action_dim),
        )

        env_actions = algorithm.act(initial_obs)
        dummy_actions = algorithm.transition.actions.clone()
        next_obs = make_obs(batch_size=3, offset=0.5)
        rewards = torch.ones(3)
        dones = torch.zeros(3, dtype=torch.uint8)
        algorithm.process_env_step(next_obs, rewards, dones, {})

        self.assertEqual(env_actions.shape, (3, policy.action_dim))
        self.assertTrue(torch.allclose(env_actions, dummy_actions[..., : policy.action_dim]))
        self.assertEqual(algorithm.storage.actions.shape[-1], policy.full_action_dim)
        self.assertTrue(torch.allclose(algorithm.storage.actions[0], dummy_actions))

    def test_log_ratio_matches_latent_energy_difference(self) -> None:
        torch.manual_seed(4)
        policy = make_policy(a_coeff=0.2, b_coeff=0.75, eps_coeff=1.5)
        algorithm = BELMGenPO(policy, num_learning_epochs=1, num_mini_batches=1, device="cpu")
        reference_policy = copy.deepcopy(policy)
        perturb_policy(policy, scale=0.03)

        obs_batch = make_obs(batch_size=4, offset=0.2)
        dummy_actions = torch.tensor(
            [
                [0.2, -0.1, 0.3, 0.5],
                [0.1, 0.4, -0.2, 0.0],
                [-0.3, 0.6, 0.2, -0.5],
                [0.7, -0.2, -0.1, 0.8],
            ],
            dtype=torch.float32,
        )

        current_latent = policy.inverse_latent(dummy_actions, obs_batch)
        old_latent = reference_policy.inverse_latent(dummy_actions, obs_batch)
        expected_log_ratio = 0.5 * (old_latent.square().sum(dim=-1) - current_latent.square().sum(dim=-1))
        observed_log_ratio = algorithm._log_ratio_from_latents(current_latent, old_latent)

        self.assertTrue(torch.allclose(observed_log_ratio, expected_log_ratio, atol=1e-5, rtol=1e-5))

    def test_update_reports_belm_diagnostics(self) -> None:
        torch.manual_seed(5)
        policy = make_policy(a_coeff=0.25, b_coeff=0.95, eps_coeff=0.5)
        algorithm = BELMGenPO(
            policy,
            num_learning_epochs=1,
            num_mini_batches=1,
            use_compress=False,
            device="cpu",
        )
        populate_storage(algorithm, num_envs=2, num_steps=2)

        loss_dict = algorithm.update()

        for key in ActorCriticBELMGenPO.diagnostic_names:
            self.assertIn(key, loss_dict)
            self.assertGreaterEqual(loss_dict[key], 0.0)

    def test_flow_runner_constructs_belm_algorithm_with_doubled_storage(self) -> None:
        env = DummyFlowEnv()
        train_cfg = {
            "num_steps_per_env": 2,
            "save_interval": 10,
            "obs_groups": {"policy": ["policy"], "critic": ["policy", "privileged"]},
            "policy": {
                "class_name": "ActorCriticBELMGenPO",
                "a_coeff": 0.05,
                "b_coeff": 0.9,
                "eps_coeff": 1.0,
                "flow_num_steps": 2,
                "actor_hidden_dims": [8, 8],
                "critic_hidden_dims": [8, 8],
                "time_hidden_dims": [8],
                "time_dim": 4,
            },
            "algorithm": {
                "class_name": "BELMGenPO",
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
            },
        }

        runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(train_cfg), device="cpu")

        self.assertIsInstance(runner.alg, BELMGenPO)
        self.assertIsInstance(runner.alg.policy, ActorCriticBELMGenPO)
        self.assertEqual(runner.alg.storage.actions.shape[-1], env.num_actions * 2)
        self.assertAlmostEqual(runner.alg.policy.a_coeff, 0.05, places=6)
        self.assertAlmostEqual(runner.alg.policy.b_coeff, 0.9, places=6)
        self.assertAlmostEqual(runner.alg.policy.eps_coeff, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
