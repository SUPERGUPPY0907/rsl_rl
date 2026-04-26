import copy
import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.policyflow import PolicyFlow
from rsl_rl.modules.actor_critic_policyflow import ActorCriticPolicyFlow
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


def make_policy() -> ActorCriticPolicyFlow:
    obs = make_obs(batch_size=4)
    return ActorCriticPolicyFlow(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["policy", "privileged"]},
        num_actions=2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        actor_activations=["elu", "elu", "linear"],
        critic_activations=["elu", "elu", "linear"],
        flow_embedding_dim=16,
        flow_sample_steps=4,
        flow_timestep_embedding_type="fourier",
        flow_conditioning="linear",
        variance_std_init=1.0,
        device="cpu",
    )


def make_algorithm() -> tuple[ActorCriticPolicyFlow, PolicyFlow]:
    policy = make_policy()
    algorithm = PolicyFlow(
        policy,
        desired_kl=0.01,
        learning_rate=1e-3,
        discount_factor=0.99,
        lam=0.95,
        mini_batches=1,
        learning_epochs=1,
        gaussian_entropy_loss_scale=0.01,
        brownian_reg_loss_scale=0.01,
        ratio_clip=0.2,
        clip_predicted_values=True,
        value_clip=0.2,
        value_loss_scale=1.0,
        grad_norm_clip=1.0,
        device="cpu",
    )
    return policy, algorithm


def populate_storage(algorithm: PolicyFlow, num_envs: int = 2, num_steps: int = 2) -> TensorDict:
    initial_obs = make_obs(num_envs)
    algorithm.init_storage(
        algorithm.get_storage_training_type(),
        num_envs,
        num_steps,
        initial_obs,
        [algorithm.policy.num_actions],
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
        del actions
        raise NotImplementedError


class TestActorCriticPolicyFlow(unittest.TestCase):
    def test_policyflow_actor_sampling_shapes(self) -> None:
        torch.manual_seed(0)
        policy = make_policy()
        obs = make_obs(batch_size=3)

        actions, info = policy.sample_actions(obs)

        self.assertEqual(actions.shape, (3, policy.num_actions))
        self.assertEqual(info["actions_prior"].shape, (3, policy.num_actions))
        self.assertEqual(info["flow_x0"].shape, (3, policy.num_actions))
        self.assertEqual(info["delta_actions"].shape, (3, policy.num_actions))
        self.assertEqual(info["delta_actions_std"].shape, (3, policy.num_actions))
        self.assertEqual(info["delta_actions_log_prob"].shape, (3,))


class TestPolicyFlowAlgorithm(unittest.TestCase):
    def test_update_reports_metrics_dictionary(self) -> None:
        torch.manual_seed(1)
        _, algorithm = make_algorithm()
        populate_storage(algorithm, num_envs=2, num_steps=2)

        loss_dict = algorithm.update()

        self.assertIn("policy_loss", loss_dict)
        self.assertIn("gaussian_entropy_loss", loss_dict)
        self.assertIn("value_loss", loss_dict)
        self.assertIn("metrics", loss_dict)
        self.assertIn("kl", loss_dict["metrics"])

    def test_flow_runner_constructs_policyflow_algorithm_and_storage(self) -> None:
        env = DummyFlowEnv()
        train_cfg = {
            "num_steps_per_env": 2,
            "save_interval": 10,
            "obs_groups": {"policy": ["policy"], "critic": ["policy", "privileged"]},
            "policy": {
                "class_name": "ActorCriticPolicyFlow",
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
                "actor_hidden_dims": [16, 16],
                "critic_hidden_dims": [16, 16],
                "actor_activations": ["elu", "elu", "linear"],
                "critic_activations": ["elu", "elu", "linear"],
                "flow_embedding_dim": 16,
                "flow_sample_steps": 4,
                "flow_timestep_embedding_type": "fourier",
                "flow_conditioning": "linear",
            },
            "algorithm": {
                "class_name": "PolicyFlow",
                "mini_batches": 1,
                "learning_epochs": 1,
                "gaussian_entropy_loss_scale": 0.01,
                "brownian_reg_loss_scale": 0.01,
            },
        }

        runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(train_cfg), device="cpu")

        self.assertIsInstance(runner.alg, PolicyFlow)
        self.assertIsInstance(runner.alg.policy, ActorCriticPolicyFlow)
        self.assertEqual(runner.alg.storage.actions.shape[-1], env.num_actions)
        self.assertEqual(runner.alg.storage.actions_prior.shape[-1], env.num_actions)
        self.assertEqual(runner.alg.storage.delta_actions_std.shape[-1], env.num_actions)

    def test_checkpoint_save_and_load_supports_policyflow(self) -> None:
        env = DummyFlowEnv()
        train_cfg = {
            "num_steps_per_env": 2,
            "save_interval": 10,
            "obs_groups": {"policy": ["policy"], "critic": ["policy", "privileged"]},
            "policy": {
                "class_name": "ActorCriticPolicyFlow",
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
                "actor_hidden_dims": [16, 16],
                "critic_hidden_dims": [16, 16],
                "actor_activations": ["elu", "elu", "linear"],
                "critic_activations": ["elu", "elu", "linear"],
                "flow_embedding_dim": 16,
                "flow_sample_steps": 4,
                "flow_timestep_embedding_type": "fourier",
                "flow_conditioning": "linear",
            },
            "algorithm": {
                "class_name": "PolicyFlow",
                "mini_batches": 1,
                "learning_epochs": 1,
            },
        }

        runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(train_cfg), device="cpu")
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "policyflow.pt"
            runner.save(str(checkpoint_path))
            self.assertTrue(checkpoint_path.exists())

            reloaded_runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(train_cfg), device="cpu")
            reloaded_runner.load(str(checkpoint_path), load_optimizer=True)


if __name__ == "__main__":
    unittest.main()
