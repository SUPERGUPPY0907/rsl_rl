import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.belm_genpo import BELMGenPO
from rsl_rl.algorithms.fpo import FPO
from rsl_rl.modules.actor_critic_belm_genpo import ActorCriticBELMGenPO
from rsl_rl.modules.actor_critic_fpo import ActorCriticFPO
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


def make_policy() -> ActorCriticFPO:
    obs = make_obs(batch_size=4)
    return ActorCriticFPO(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["policy", "privileged"]},
        num_actions=2,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[16, 16],
        critic_hidden_dims=[16, 16],
        activation="elu",
        timestep_embed_dim=8,
        sampling_steps=4,
        training_sampling_steps=4,
        actor_scale=1.0,
        actor_mlp_output_scale=1.0,
        cfm_loss_reduction="sqrt",
        action_perturb_std=0.0,
        device="cpu",
    )


def make_algorithm() -> tuple[ActorCriticFPO, FPO]:
    policy = make_policy()
    algorithm = FPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.05,
        learning_rate=1e-3,
        schedule="fixed",
        n_samples_per_action=4,
        knn_entropy_coef=0.0,
        ema_decay=0.95,
        ema_warmup_steps=1,
        device="cpu",
    )
    return policy, algorithm


def populate_storage(algorithm: FPO, num_envs: int = 2, num_steps: int = 2) -> TensorDict:
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


class TestActorCriticFPO(unittest.TestCase):
    def test_actor_outputs_and_cfm_loss_shapes(self) -> None:
        torch.manual_seed(0)
        policy = make_policy()
        obs = make_obs(batch_size=3)
        actions = policy.act(obs)
        eps = torch.randn(3, 5, policy.num_actions)
        t = torch.rand(3, 5, 1)
        loss, x1_pred, x0_pred = policy.get_cfm_loss(obs, actions, eps, t)

        self.assertEqual(actions.shape, (3, policy.num_actions))
        self.assertEqual(loss.shape, (3, 5))
        self.assertEqual(x1_pred.shape, (3, 5, policy.num_actions))
        self.assertEqual(x0_pred.shape, (3, 5, policy.num_actions))

    def test_actor_obs_inference_path_matches_policy_path(self) -> None:
        torch.manual_seed(1)
        policy = make_policy()
        obs = make_obs(batch_size=3)
        actor_obs = policy.get_actor_obs(obs)

        actions_from_obs = policy.act_inference(obs)
        actions_from_actor_obs = policy.act_inference_from_actor_obs(actor_obs)

        self.assertTrue(torch.allclose(actions_from_obs, actions_from_actor_obs, atol=1e-6, rtol=1e-6))


class TestFPOAlgorithm(unittest.TestCase):
    def test_update_reports_metrics_dictionary(self) -> None:
        torch.manual_seed(2)
        _, algorithm = make_algorithm()
        populate_storage(algorithm, num_envs=2, num_steps=2)

        loss_dict = algorithm.update()

        self.assertIn("surrogate_loss", loss_dict)
        self.assertIn("value_loss", loss_dict)
        self.assertIn("metrics", loss_dict)
        self.assertIn("clip_param", loss_dict["metrics"])

    def test_flow_runner_constructs_fpo_algorithm_and_storage(self) -> None:
        env = DummyFlowEnv()
        train_cfg = {
            "num_steps_per_env": 2,
            "save_interval": 10,
            "obs_groups": {"policy": ["policy"], "critic": ["policy", "privileged"]},
            "policy": {
                "class_name": "ActorCriticFPO",
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
                "actor_hidden_dims": [16, 16],
                "critic_hidden_dims": [16, 16],
                "activation": "elu",
                "timestep_embed_dim": 8,
                "sampling_steps": 4,
                "training_sampling_steps": 4,
                "action_perturb_std": 0.0,
            },
            "algorithm": {
                "class_name": "FPO",
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
                "n_samples_per_action": 4,
                "ema_decay": 0.95,
                "ema_warmup_steps": 1,
            },
        }

        runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(train_cfg), device="cpu")

        self.assertIsInstance(runner.alg, FPO)
        self.assertIsInstance(runner.alg.policy, ActorCriticFPO)
        self.assertEqual(runner.alg.storage.actions.shape[-1], env.num_actions)
        self.assertEqual(runner.alg.storage.initial_cfm_loss.shape[-1], 4)

    def test_checkpoint_save_load_supports_fpo_ema_and_existing_flow_algorithms(self) -> None:
        env = DummyFlowEnv()
        fpo_train_cfg = {
            "num_steps_per_env": 2,
            "save_interval": 10,
            "obs_groups": {"policy": ["policy"], "critic": ["policy", "privileged"]},
            "policy": {
                "class_name": "ActorCriticFPO",
                "actor_obs_normalization": False,
                "critic_obs_normalization": False,
                "actor_hidden_dims": [16, 16],
                "critic_hidden_dims": [16, 16],
                "activation": "elu",
                "timestep_embed_dim": 8,
                "sampling_steps": 4,
                "training_sampling_steps": 4,
                "action_perturb_std": 0.0,
            },
            "algorithm": {
                "class_name": "FPO",
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
                "n_samples_per_action": 4,
                "ema_decay": 0.95,
                "ema_warmup_steps": 1,
            },
        }
        belm_train_cfg = {
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            fpo_path = str(Path(tmp_dir) / "fpo.pt")
            belm_path = str(Path(tmp_dir) / "belm.pt")

            fpo_runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(fpo_train_cfg), device="cpu")
            fpo_runner.current_learning_iteration = 7
            fpo_runner.alg.tot_timesteps = 2
            for tensor in fpo_runner.alg.ema.shadow_params.values():
                tensor.fill_(0.123)
            fpo_runner.save(fpo_path)

            saved_fpo = torch.load(fpo_path, weights_only=False)
            self.assertIn("ema_state_dict", saved_fpo)

            reloaded_fpo_runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(fpo_train_cfg), device="cpu")
            reloaded_fpo_runner.load(fpo_path)
            self.assertEqual(reloaded_fpo_runner.current_learning_iteration, 7)
            self.assertIsNotNone(reloaded_fpo_runner.alg.ema)

            belm_runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(belm_train_cfg), device="cpu")
            self.assertIsInstance(belm_runner.alg, BELMGenPO)
            self.assertIsInstance(belm_runner.alg.policy, ActorCriticBELMGenPO)
            belm_runner.current_learning_iteration = 5
            belm_runner.save(belm_path)

            saved_belm = torch.load(belm_path, weights_only=False)
            self.assertNotIn("ema_state_dict", saved_belm)

            reloaded_belm_runner = OnPolicyFlowRunner(env=env, train_cfg=copy.deepcopy(belm_train_cfg), device="cpu")
            reloaded_belm_runner.load(belm_path)
            self.assertEqual(reloaded_belm_runner.current_learning_iteration, 5)

    def test_exporter_prefers_policy_inference_path(self) -> None:
        exporter_path = Path("/home/superguppy/Isaaclab_pr/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py")
        spec = importlib.util.spec_from_file_location("isaaclab_rsl_exporter", exporter_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        policy = make_policy()
        actor_obs = policy.get_actor_obs(make_obs(batch_size=2))
        exporter = module._TorchPolicyExporter(policy, normalizer=None)

        exported_actions = exporter(actor_obs)
        reference_actions = policy.act_inference_from_actor_obs(actor_obs)

        self.assertTrue(torch.allclose(exported_actions, reference_actions, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
