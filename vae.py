from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import argparse

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
from torch.autograd import grad as torchgrad

from utils.math_ops import log_normal, log_bernoulli, log_mean_exp
from utils.approx_posts import Flow


class VAE(nn.Module):
    """Generic VAE for MNIST and Fashion datasets."""
    def __init__(self, hps):
        super(VAE, self).__init__()

        self.z_size = hps.z_size
        self.has_flow = hps.has_flow
        self.use_cuda = hps.cuda
        self.act_func = hps.act_func
        self.n_flows = hps.n_flows
        self.hamiltonian_flow = hps.hamiltonian_flow

        self._init_layers(wide_encoder=hps.wide_encoder)

        if self.use_cuda:
            self.cuda()
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def _init_layers(self, wide_encoder=False):
        h_s = 500 if wide_encoder else 200

        self.fc1 = nn.Linear(784, h_s)  # assume flattened
        self.fc2 = nn.Linear(h_s, h_s)
        self.fc3 = nn.Linear(h_s, self.z_size*2)

        self.fc4 = nn.Linear(self.z_size, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)

        self.x_info_layer = nn.Linear(200, self.z_size)

        if self.has_flow:
            self.q_dist = Flow(self, n_flows=self.n_flows)
            if self.use_cuda:
                self.q_dist.cuda()

    def sample(self, mu, logvar, grad_fn=lambda x: 1, x_info=None):
        eps = Variable(torch.FloatTensor(mu.size()).normal_().type(self.dtype))
        z = eps.mul(logvar.mul(0.5).exp_()).add_(mu)
        logqz = log_normal(z, mu, logvar)

        if self.has_flow:
            z, logprob = self.q_dist.forward(z, grad_fn, x_info)
            logqz += logprob

        zeros = Variable(torch.zeros(z.size()).type(self.dtype))
        logpz = log_normal(z, zeros, zeros)

        return z, logpz, logqz

    def encode(self, net):
        net = self.act_func(self.fc1(net))
        net = self.act_func(self.fc2(net))
        x_info = self.act_func(self.x_info_layer(net))
        net = self.fc3(net)

        mean, logvar = net[:, :self.z_size], net[:, self.z_size:]

        return mean, logvar, x_info

    def decode(self, net):
        net = self.act_func(self.fc4(net))
        net = self.act_func(self.fc5(net))
        logit = self.fc6(net)

        return logit

    def forward(self, x, k=1, warmup_const=1.):
        x = x.repeat(k, 1)
        mu, logvar, x_info = self.encode(x)

        # posterior-aware inference
        def U(z):
            logpx = log_bernoulli(self.decode(z), x)
            logpz = log_normal(z)
            return -logpx - logpz  # energy as -log p(x, z)

        def grad_U(z):
            grad_outputs = torch.ones(z.size(0)).type(self.dtype)
            grad = torchgrad(U(z), z, grad_outputs=grad_outputs, create_graph=True)[0]
            # gradient clipping avoid numerical issue
            norm = torch.sqrt(torch.norm(grad, p=2, dim=1))
            # neither grad clip methods consistently outperforms the other
            grad = grad / norm.view(-1, 1)
            # grad = torch.clamp(grad, -10000, 10000)
            return grad.detach()

        if self.hamiltonian_flow:
            z, logpz, logqz = self.sample(mu, logvar, grad_fn=grad_U, x_info=x_info)
        else:
            z, logpz, logqz = self.sample(mu, logvar, x_info=x_info)

        logit = self.decode(z)
        logpx = log_bernoulli(logit, x)
        elbo = logpx + logpz - warmup_const * logqz  # custom warmup

        # need correction for Tensor.repeat
        elbo = log_mean_exp(elbo.view(k, -1).transpose(0, 1))
        elbo = torch.mean(elbo)

        logpx = torch.mean(logpx)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)

        return elbo, logpx, logpz, logqz
