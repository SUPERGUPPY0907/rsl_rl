"""
Moser Flow Policy for RL
Adapted for reinforcement learning from the original Moser Flow paper.
This version is self-contained and designed to replace Gaussian policies in PPO.
"""

import torch
import torch.nn as nn

import numpy as np
from torch import distributions


def parse_activation(activation_name):
    """Parse activation function name to PyTorch module."""
    activations = {
        "tanh": nn.Tanh(),
        "softplus": nn.Softplus(),
        "softplus100": nn.Softplus(100),
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "mish": nn.Mish(),
    }
    return activations.get(activation_name, nn.ELU())


def build_mlp(input_dim, hidden_dims, output_dim, activation, last_activation=None, add_batchnorm=False):
    """Build MLP network."""
    layers = [nn.Linear(input_dim, hidden_dims[0]), activation]

    if add_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(activation)
        if add_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if last_activation is not None:
        layers.append(last_activation)

    return nn.Sequential(*layers)


# class GaussianPrior:
#     """Standard Gaussian prior N(0, I) for unbounded action space."""
#     def __init__(self, dim, device):
#         self.dim = dim
#         self.device = device
#         self.normalizer = (2 * np.pi) ** (-dim / 2)

#     def log_prob(self, x):
#         # log N(x; 0, I) = -0.5 * ||x||^2 - 0.5 * dim * log(2π)
#         squared_norm = (x ** 2).sum(dim=-1)
#         return -0.5 * squared_norm - 0.5 * self.dim * np.log(2 * np.pi)

#     def sample(self, shape):
#         n, = shape
#         return torch.randn(n, self.dim, device=self.device)

class GaussianPrior(distributions.MultivariateNormal):
    def __init__(self, dim, device, std):
        super(GaussianPrior, self).__init__(torch.zeros(dim, device=device), std * torch.eye(dim, device=device))

class MoserFlowPolicy(nn.Module):
    """
    Moser Flow Policy for RL in Euclidean space R^n.

    Uses envelope function exp(-||a||^2) to ensure divergence integral = 0.
    Implements Jordan decomposition: μ = μ+ - μ- for handling negative densities.
    """

    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims=[256, 256, 256],
        activation="mish",
        envelope_scale=1.0,
        ode = "rk4",
        flow_num_steps = 10,
        device="cpu",
        **kwargs,
    ):
        super().__init__()

        self.num_obs = num_obs
        self.num_actions = num_actions
        self.device = device
        self.eps = 1e-7
        self.envelope_scale = envelope_scale  # Scale factor for envelope decay
        self.ode = ode
        self.flow_num_steps = flow_num_steps


        # Prior distribution: Standard Gaussian N(0, I)
        self.prior = GaussianPrior(num_actions, device, std=1.0)
        # self.monte_carlo_prior = GaussianPrior(num_actions, device)

        # Potential function network u_θ(a|s)
        # Input: actions + observations
        self.v = build_mlp(
            input_dim=num_actions + num_obs,
            hidden_dims=hidden_dims,
            output_dim=num_actions,
            activation=parse_activation(activation),
            last_activation=None,
            add_batchnorm=False
        )

        self.initialize_weights()
        self.to(device)

        # Cache for current observation
        self.current_obs = None

    # def initialize_weights(self):
    #     """Initialize network weights."""
    #     def init_xavier(layer):
    #         if isinstance(layer, nn.Linear):
    #             nn.init.xavier_normal_(layer.weight)
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)
    #     def init_orthogonal(layer):
    #         if isinstance(layer, nn.Linear):
    #             # 使用正交初始化，并设置极小的 gain
    #             # 0.01 可以让初始输出非常接近 0
    #             nn.init.orthogonal_(layer.weight, gain=1.0)
    #             if layer.bias is not None:
    #                 nn.init.zeros_(layer.bias)

    #     self.v.apply(init_orthogonal)
    
    def initialize_weights(self):
        """Initialize network weights."""
        
        # 1. 定义隐藏层的初始化逻辑
        # 注意：gain 的值取决于你在 build_mlp 中使用的激活函数
        # - 如果是 ReLU，使用 nn.init.calculate_gain('relu') (约 1.414)
        # - 如果是 Tanh，使用 nn.init.calculate_gain('tanh') (约 1.67)
        def init_orthogonal_hidden(layer):
            if isinstance(layer, nn.Linear):
                # 这里假设你使用的是 ReLU，如果是 Tanh 请自行修改字符串
                gain = nn.init.calculate_gain('relu') 
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # 2. 先对整个网络应用隐藏层的初始化
        self.v.apply(init_orthogonal_hidden)

        # 3. 单独重置最后一层 (Output Layer) 的 gain 为 0.01
        # 我们需要找到网络中最后一个 Linear 层（防止 self.v[-1] 是激活函数的情况）
        last_linear_layer = None
        # 倒序遍历寻找最后一个 Linear 层
        for layer in reversed(self.v):
            if isinstance(layer, nn.Linear):
                last_linear_layer = layer
                break
        
        # 对最后一层应用 gain=0.01
        if last_linear_layer is not None:
            with torch.no_grad():
                nn.init.orthogonal_(last_linear_layer.weight, gain=0)
                if last_linear_layer.bias is not None:
                    nn.init.zeros_(last_linear_layer.bias)

        # self.v.apply(init_xavier)
        # Initialize last layer to zero for initial near-uniform policy
        # with torch.no_grad():
        #     if isinstance(self.v[-1], nn.Linear):
        #         nn.init.zeros_(self.v[-1].weight)
        #         if self.v[-1].bias is not None:
        #             nn.init.zeros_(self.v[-1].bias)

    def envelope(self, actions):
        """
        Compute envelope function exp(-scale * ||a||^2).
        This ensures u(a) → 0 as ||a|| → ∞, guaranteeing ∫ ∇·u da = 0.
        """
        # squared_norm = (actions).sum(dim=-1, keepdim=True)
        squared_norm = torch.linalg.norm(actions, ord=2, dim=-1, keepdim=True)
        return torch.exp(-self.envelope_scale * squared_norm)

    def u_theta(self, actions, obs):
        """Compute raw potential function u_θ(a|s) from network."""
        combined = torch.cat([actions, obs], dim=1)
        return self.v(combined)

    def u(self, actions, obs):
        """
        Compute enveloped potential function: u(a|s) = exp(-||a||^2) * u_θ(a|s).
        The envelope ensures proper decay at infinity for divergence theorem.
        """
        envelope = self.envelope(actions)
        u_theta = self.u_theta(actions, obs)
        # print('envelope:', self.envelope(actions))
        # print('u_theta:', self.u_theta(actions, obs))
        return envelope * u_theta

    def nu(self, actions):
        """Prior density."""
        return torch.exp(self.prior.log_prob(actions)).unsqueeze(-1)

    def divergence_u(self, actions, obs, create_graph=False):
        """Compute divergence of u with respect to actions."""
        if not actions.requires_grad:
            actions.requires_grad = True

        u_val = self.u(actions, obs)
        div = torch.zeros(actions.shape[0], device=self.device)

        for i in range(self.num_actions):
            grad_i = torch.autograd.grad(
                u_val[:, i].sum(),
                actions,
                create_graph=create_graph,
                retain_graph=True
            )[0]
            div += grad_i[:, i]

        return div

    def signed_mu(self, actions, obs, create_graph):
        """
        Compute signed density: μ(a|s) = ν(a) - ∇·u(a|s).
        This can be positive or negative.
        """
        nu_val = self.nu(actions)
        div_u = self.divergence_u(actions, obs, create_graph)
        return (nu_val.squeeze(-1) - div_u).unsqueeze(-1)

    def mu_plus(self, actions, obs, create_graph=False):
        """
        Positive part of Jordan decomposition: μ+(a|s) = max(μ(a|s), 0) + ε.
        This is the actual density used for sampling and likelihood.
        """
        signed = self.signed_mu(actions, obs, create_graph)
        return torch.relu(signed) + self.eps

    def mu_minus(self, actions, obs, create_graph):
        """
        Negative part of Jordan decomposition: μ-(a|s) = max(-μ(a|s), 0).
        This represents the "violation" of positivity constraint.
        """
        signed = self.signed_mu(actions, obs, create_graph)
        return torch.relu(-signed)

    def density(self, actions, obs, create_graph):
        """Compute action density given observations."""
        return self.signed_mu(actions, obs, create_graph)
    
    def ode_func(self, t, x, obs):
        
        out = (self.u(x, obs) 
                / ((1 - t) * self.nu(x).view(-1, 1)
                + t * self.mu_plus(x, obs)))
        print('u:', self.u(x, obs))
        print('INTER', ((1 - t) * self.nu(x).view(-1, 1)
                + t * self.mu_plus(x, obs)))
        print('v_theta:', out)
        if torch.isnan(out).any():
            raise ValueError("nans in v_t")
        return out
    

    def odeint(self, func, x0, t, method='rk4'):
        """
        Solves a system of ODEs given a defined time tensor.
    
        Args:
            func: The ODE function f(t, x) that returns velocity.
            x0: Initial state tensor.
            t: A 1D tensor representing the time steps (e.g., [0, 0.1, ..., 1]).
            method: 'euler' or 'rk4'.
        
        Returns:
            Tensor of shape (len(t), *x0.shape) containing the trajectory.
        """
        # n_steps = t.shape[0]
        # trajectory = [x0]
        dt = 1/len(t)
        x = x0
        for i in range(self.flow_num_steps):
            t_curr = t[i]
            if method == 'euler':
                # print('input', x)
                # print('t_curr', t_curr)
                x = x + dt * func(t_curr, x)
                # print('output', x)
                
            elif method == 'rk4':
                k1 = func(t_curr, x)
                k2 = func(t_curr + dt / 2, x + dt * k1 / 2)
                k3 = func(t_curr + dt / 2, x + dt * k2 / 2)
                k4 = func(t_curr + dt, x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown ODE method: {method}")
        # breakpoint()
        return x
    
    def transport(self, x, obs):
        t = torch.linspace(0.0, 1.0, steps=self.flow_num_steps+1, device=self.device)[:-1]
        return self.odeint(lambda t, x: self.ode_func(t, x, obs), x, t, self.ode)

    def sample(self, obs):
        num_samples = obs.shape[0]
        random_samples = self.prior.sample((num_samples,))
        random_samples.requires_grad = True
        samples = self.transport(random_samples, obs)
        return samples.detach()
    
    def forward(self, obs, actions):
        """Compute negative log probability (for PPO loss)."""
        return -self.log_density(actions, obs)
    
    def get_prob(self, actions, obs, create_graph=True):
        """Get probability of actions given observations."""
        return self.density(actions, obs, create_graph)