# Moser Flow Policy for Reinforcement Learning

This implementation integrates **Moser Flow** (divergence-based generative modeling on manifolds) as a policy network for PPO in Isaac Lab.

## Overview

Moser Flow replaces the standard Gaussian policy with a learned probability distribution that directly models the target action density. Instead of using a fixed distribution type (like Gaussian), Moser Flow learns the optimal distribution shape through a PDE-based approach.

## Key Differences from Gaussian Policy

| Aspect | Gaussian Policy | Moser Flow Policy |
|--------|----------------|-------------------|
| Distribution | Fixed Gaussian | Learned density via PDE |
| Parameters | Mean μ and std σ | Potential function v(a\|s) |
| Expressiveness | Limited to unimodal | Can be multimodal |
| Action Space | Unbounded (with clipping) | Naturally bounded to [-1, 1] |

## Architecture

The Moser Flow policy consists of:
1. **Fourier Features**: Encode actions using sin/cos features
2. **Potential Network**: MLP that outputs potential function u(a|s)
3. **Density Computation**: ρ(a|s) = ν(a) - ∇·u(a|s), where ν is uniform prior
4. **Positivity Enforcement**: μ₊(a|s) = ReLU(ρ(a|s) - ε) + ε

## Usage

### 1. Import the necessary modules

```python
from rsl_rl.algorithms import MoserPPO
from rsl_rl.modules import ActorCriticMoser
```

### 2. Create the policy

```python
policy = ActorCriticMoser(
    num_actor_obs=env.num_obs,
    num_critic_obs=env.num_privileged_obs,
    num_actions=env.num_actions,
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    activation="elu",
    n_fourier_features=4,  # Number of Fourier features for action encoding
)
```

### 3. Create the MoserPPO algorithm

```python
alg = MoserPPO(
    policy=policy,
    num_learning_epochs=5,
    num_mini_batches=4,
    clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    value_loss_coef=2.0,
    entropy_coef=0.01,
    learning_rate=3e-4,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
    schedule="adaptive",
    desired_kl=0.01,
    device="cuda",
)
```

### 4. Configuration in IsaacLab Tasks

Modify your agent configuration file (e.g., `rsl_rl_ppo_cfg.py`):

```python
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from rsl_rl.modules import ActorCriticMoser
from rsl_rl.algorithms import MoserPPO

# Policy configuration
class MoserPolicyNetworkCfg:
    """Configuration for Moser Flow policy network."""
    class_name = ActorCriticMoser
    actor_hidden_dims = [256, 256, 256]
    critic_hidden_dims = [256, 256, 256]
    activation = "elu"
    n_fourier_features = 4

# Algorithm configuration
class MoserPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for MoserPPO algorithm."""
    class_name = MoserPPO
    entropy_coef = 0.01

# Runner configuration
class MoserRlCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for training with Moser Flow."""
    policy = MoserPolicyNetworkCfg()
    algorithm = MoserPpoAlgorithmCfg()
```

## Hyperparameter Tuning

### Key Hyperparameters

1. **n_fourier_features** (default: 4)
   - Controls the expressiveness of action encoding
   - Higher values → more complex distributions
   - Recommended range: 2-8

2. **entropy_coef** (default: 0.01)
   - Controls exploration
   - Moser Flow naturally has variable entropy, so may need different values than Gaussian
   - Recommended range: 0.001-0.05

3. **actor_hidden_dims** (default: [256, 256, 256])
   - Network capacity for learning density
   - Larger networks → more expressive policies
   - Trade-off with training speed

### Suggested Configurations

#### For Continuous Control (e.g., Locomotion)
```python
n_fourier_features = 4
entropy_coef = 0.01
actor_hidden_dims = [256, 256, 256]
learning_rate = 3e-4
```

#### For Manipulation Tasks
```python
n_fourier_features = 6
entropy_coef = 0.02
actor_hidden_dims = [512, 512, 512]
learning_rate = 1e-4
```

## Advantages

1. **Multimodal Policies**: Can naturally represent multiple action modes
2. **Bounded Actions**: Actions are naturally in [-1, 1] without explicit clipping
3. **Adaptive Exploration**: Entropy adapts to task complexity automatically
4. **Theoretical Foundation**: Based on optimal transport and PDEs

## Limitations

1. **Computational Cost**: Requires gradient computation for divergence (2-3x slower than Gaussian)
2. **Memory**: Slightly higher memory usage due to gradient tracking
3. **Sampling**: Approximate sampling using gradient-based adjustment

## Troubleshooting

### Issue: Training is slow
- **Solution**: Reduce `n_fourier_features` or use smaller networks
- Use batch gradient computation sparingly

### Issue: Policy doesn't converge
- **Solution**:
  - Increase `entropy_coef` for more exploration
  - Reduce `learning_rate`
  - Try fewer Fourier features first (e.g., 2-3)

### Issue: Actions are too deterministic
- **Solution**: Increase `entropy_coef` or reduce `n_fourier_features`

### Issue: NaN in training
- **Solution**:
  - Check that `eps` parameter is not too small (default: 1e-5)
  - Reduce learning rate
  - Enable gradient clipping (already enabled by default)

## References

1. Moser Flow Paper: "Moser Flow: Divergence-based Generative Modeling on Manifolds" ([arXiv:2108.08052](https://arxiv.org/abs/2108.08052))
2. Original Implementation: https://github.com/noamroze/moserflow

## Example: Complete Training Script

```python
# Example for IsaacLab locomotion task
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.algorithms import MoserPPO
from rsl_rl.modules import ActorCriticMoser

# Create environment
env = gym.make("Isaac-Velocity-Rough-Anymal-D-v0")

# Create policy
policy = ActorCriticMoser(
    num_actor_obs=env.num_obs,
    num_critic_obs=env.num_privileged_obs,
    num_actions=env.num_actions,
    n_fourier_features=4,
)

# Create algorithm
ppo = MoserPPO(policy=policy, device=env.device)

# Initialize storage
ppo.init_storage(
    "on_policy",
    env.num_envs,
    num_transitions_per_env=24,
    actor_obs_shape=[env.num_obs],
    critic_obs_shape=[env.num_privileged_obs],
    actions_shape=[env.num_actions],
)

# Training loop
for epoch in range(1000):
    # Collect rollouts
    obs = env.get_observations()
    for step in range(24):
        actions = ppo.act(obs["policy"], obs["critic"])
        obs, rewards, dones, infos = env.step(actions)
        ppo.process_env_step(rewards, dones, infos)

    # Update policy
    ppo.compute_returns(obs["critic"])
    loss_dict = ppo.update()

    print(f"Epoch {epoch}: {loss_dict}")
```

## Notes

- Moser Flow is experimental and may require more tuning than standard Gaussian policies
- Best suited for tasks where multimodal or complex action distributions are beneficial
- Monitor entropy during training to ensure sufficient exploration
