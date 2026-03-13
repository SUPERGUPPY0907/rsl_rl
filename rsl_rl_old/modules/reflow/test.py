import torch
from torch.linalg import slogdet
import torch
import math
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap
from torch.distributions import Normal

class MLP(nn.Module):
    """Multi-layer Perceptron with Jacobian computation capability.

    Args:
        input_dim (int): Dimension of input features. Default: 2
        hidden_num (int): Number of hidden units in each layer. Default: 256
        output_dim (int): Dimension of output features. Default: 2
    """

    def __init__(self, input_dim: int = 2, hidden_dim: list = [256, 256, 256], output_dim: int = 2, t_dim = 0, activation: torch.nn.Module = torch.nn.Tanh(),N=5) :
        super().__init__()
        assert len(hidden_dim) > 0, "hidden_dim list must not be empty"
        self.output_dim = output_dim
        self.p=0.5
        layers = []
        layers.append(nn.Linear(input_dim + t_dim, hidden_dim[0], bias=True))
        layers.append(activation)

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=True))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim[-1], output_dim, bias=True))
        self.net = nn.Sequential(*layers)
        self.N = N
        self.dt = 1/N





    def forward(self,
                x_input: torch.Tensor,
                observations: torch.Tensor,
               ) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            observations (torch.Tensor): Observation tensor of shape (batch_size, 1)
            t (torch.Tensor): Time tensor of shape (batch_size, 1)


        Returns:
            torch.Tensor:
                - Output tensor of shape (batch_size, output_dim)
        """
        # Combine inputs
        return self._Heun_method_inverse(observations, x_input)

    def _Heun_method_inverse(self, observations,  x):

        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations

        z = x[..., :self.output_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.output_dim]
        y = x[..., self.output_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.output_dim:]
        for i in reversed(range(self.N)):

            y_in = (y - (1 - self.p) * z)/self.p
            z_in = (z - (1 - self.p) * y_in)/self.p
            k = torch.cat([z_in, observations], dim=-1)
            y_transformed = self.net(k)
            y = y_in - y_transformed * self.dt
            k = torch.cat([y, observations], dim=-1)
            z_transformed = self.net(k)
            z = z_in - z_transformed * self.dt

        out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)

class Flow(nn.Module):
    def __init__(self, input_dim, output_dim, a_dim, actor_hidden_dim, activation, N, p, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a_dim = a_dim
        self.N = N
        self.dt = 1/N
        self.p = p
        self.vec_field = MLP(input_dim=self.input_dim, hidden_dim=actor_hidden_dim, output_dim=self.output_dim, t_dim=0, activation=activation)
        self.device = device
        self.dist  = Normal(torch.zeros(self.a_dim *2, device=device), torch.ones(self.a_dim * 2, device=device))
        self.saved = False
        self.cache = []
        # t = torch.arange(self.N, device=self.device)/self.N
        # self.t_em = self.time_mlp(t)

    def _Heun_method(self, observations,  x):
        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        num_envs = observations.shape[0]

        z = x[..., :self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.a_dim]
        y = x[..., self.a_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim:]


        for i in range(self.N):

            t = torch.ones((num_envs, ), device=observations.device) * i / self.N
            t_em = self.time_mlp(t)


            z_transformed = self.vec_field(y, observations, t_em)
            z_in = z + z_transformed * self.dt

            y_transformed = self.vec_field(z_in, observations, t_em)
            y_in = y + y_transformed * self.dt
            z = self.p * z_in + (1-self.p) * y_in
            y = self.p * y_in + (1-self.p) * z

        out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)

    def _Heun_method_2(self, observations,  x):


        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        num_envs = observations.shape[0]

        z = x[..., :self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.a_dim]
        y = x[..., self.a_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim:]

        for i in range(self.N):
            if self.saved:
                t_em = self.cache[i].expand(num_envs, -1).to(observations.device)

            else:
                t = torch.ones((num_envs,), device=observations.device) * i / self.N
                t_em = self.time_mlp(t)
            z_transformed = self.vec_field(y, observations, t_em)
            z_in = z + z_transformed * self.dt

            y_transformed = self.vec_field(z_in, observations, t_em)
            y_in = y + y_transformed * self.dt
            z = self.p * z_in + (1-self.p) * y_in
            y = self.p * y_in + (1-self.p) * z

        out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)

    def _Heun_method_inverse(self, observations,  x):

        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        num_envs = observations.shape[0]

        z = x[..., :self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.a_dim]
        y = x[..., self.a_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim:]
        for i in reversed(range(self.N)):

            t = torch.ones((num_envs, ), device=observations.device) * i / self.N
            t_em = self.time_mlp(t)

            y_in = (y - (1 - self.p) * z)/self.p
            z_in = (z - (1 - self.p) * y_in)/self.p
            y_transformed = self.vec_field(z_in, observations, t_em)
            y = y_in - y_transformed * self.dt
            z_transformed = self.vec_field(y, observations, t_em)
            z = z_in - z_transformed * self.dt

        out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)

    def forward(self, observations, jac):
        batch = 2048*2
        # print(observations.shape)
        num_envs = observations.shape[0]

        action_aug_0 = torch.randn(num_envs, self.a_dim * 2, device=observations.device)
        log_probs = self.dist.log_prob(action_aug_0).sum(dim=-1)
        action_aug = torch.empty_like(action_aug_0)

        n = num_envs // batch
        n += int(num_envs % batch != 0)

        for i in range(n):
            action_aug[i*batch:min((i+1)*batch, num_envs)] = self._Heun_method(observations[i*batch:min((i+1)*batch, num_envs)],  action_aug_0[i*batch:min((i+1)*batch, num_envs)])
            # breakpoint()
            if jac:
                # assert action_aug_0.requires_grad, "input requires grad"
                jacobian_fn = jacrev(self._Heun_method, argnums = 1)
                batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0,  0))
                J = batched_jacobian_fn(observations[i*batch:min((i+1)*batch, num_envs)], action_aug_0[i*batch:min((i+1)*batch, num_envs)]).detach()
                # print('J dims:', J.shape)

                # log_probs[i*batch:min((i+1)*batch, num_envs)] = log_probs[i*batch:min((i+1)*batch, num_envs)] - torch.slogdet(J)[1]

        return action_aug, log_probs

    def inference(self, observations):
        batch = 4096
        num_envs = observations.shape[0]

        action_aug_0 = torch.zeros(num_envs, self.a_dim * 2, device=observations.device)
        action_aug = torch.empty_like(action_aug_0)

        n = num_envs // batch
        n += int(num_envs % batch != 0)

        for i in range(n):
            action_aug[i*batch:min((i+1)*batch, num_envs)] = self._Heun_method_2(observations[i*batch:min((i+1)*batch, num_envs)],  action_aug_0[i*batch:min((i+1)*batch, num_envs)])

        return action_aug

    def inverse(self, observations, action_aug,  jac=False):
        batch = 2048*2
        num_envs = observations.shape[0]
        n = num_envs // batch
        n += int(num_envs % batch != 0)

        action_aug_0 = self._Heun_method_inverse(observations, action_aug)
        log_probs = self.dist.log_prob(action_aug_0).sum(dim=-1)
        for i in range(n):
            if jac:
                jacobian_fn = jacrev(self._Heun_method_inverse, argnums=1)
                batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0, 0))
                J = batched_jacobian_fn(observations[i*batch:min((i+1)*batch, num_envs)], action_aug[i*batch:min((i+1)*batch, num_envs)])
                logdet = torch.slogdet(J)[1]
                print(logdet)
                # log_probs[i*batch:min((i+1)*batch, num_envs)] = log_probs[i*batch:min((i+1)*batch, num_envs)] + logdet
                # print(J.requires_grad, log_probs.requires_grad)

        return log_probs

    def save_t_em(self):
        for i in range(self.N):
            t = torch.ones((1,), device=self.device) * i / self.N
            t_em = self.time_mlp(t)
            self.cache.append(t_em.detach().clone())
        self.saved = True

obs_dim = 200
action_dim = 10
# model = Flow(input_dim=2, output_dim=4, a_dim=2, actor_hidden_dim=[256,256],activation='mish', N=1, p=0.9, device='cpu')
model_mlp = MLP(input_dim=obs_dim+action_dim, output_dim=action_dim, hidden_dim=[256,256,256])
obs = torch.randn(10000,obs_dim)
x = torch.randn(10000,action_dim*2)
# y = model_mlp(obs, x)
# print(y)
jacobian_fn = jacrev(model_mlp, argnums=0)
batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0, 0))
J = batched_jacobian_fn(x, obs).detach()
print(J.shape)
J = J.view(10000,action_dim*2,action_dim*2)
print(slogdet(J)[1])