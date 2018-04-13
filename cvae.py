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


class CVAE(nn.Module):
    """Convolutional VAE for CIFAR."""
    def __init__(self, hps):
        super(CVAE, self).__init__()

        self.z_size = hps.z_size
        self.has_flow = hps.has_flow
        self.hamiltonian_flow = hps.hamiltonian_flow
        self.n_flows = hps.n_flows
        self.use_cuda = hps.cuda
        self.act_func = hps.act_func

        self._init_layers(wide_encoder=hps.wide_encoder)

        if self.use_cuda:
            self.cuda()
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

    def _init_layers(self, wide_encoder=False):

        if wide_encoder:
            init_channel = 128
        else:
            init_channel = 64

        # encoder
        self.conv1 = nn.Conv2d(3, init_channel, 4, 2)
        self.conv2 = nn.Conv2d(init_channel, init_channel*2, 4, 2)
        self.conv3 = nn.Conv2d(init_channel*2, init_channel*4, 4, 2)
        self.fc_enc = nn.Linear(init_channel*4*2*2, self.z_size*2)

        self.bn_enc1 = nn.BatchNorm2d(init_channel)
        self.bn_enc2 = nn.BatchNorm2d(init_channel*2)
        self.bn_enc3 = nn.BatchNorm2d(init_channel*4)

        self.x_info_layer = nn.Linear(init_channel*4*2*2, self.z_size)

        # decoder
        self.fc_dec = nn.Linear(self.z_size, 256*2*2)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2)

        self.bn_dec1 = nn.BatchNorm2d(128)
        self.bn_dec2 = nn.BatchNorm2d(64)

        self.decoder_layers = []
        self.decoder_layers.append(self.deconv1)
        self.decoder_layers.append(self.deconv2)
        self.decoder_layers.append(self.deconv3)
        self.decoder_layers.append(self.fc_dec)
        self.decoder_layers.append(self.bn_dec1)
        self.decoder_layers.append(self.bn_dec2)

        if self.has_flow:
            self.q_dist = Flow(self, n_flows=self.n_flows)
            if self.use_cuda:
                self.q_dist.cuda()

    def encode(self, net):

        net = self.act_func(self.bn_enc1(self.conv1(net)))
        net = self.act_func(self.bn_enc2(self.conv2(net)))
        net = self.act_func(self.bn_enc3(self.conv3(net)))
        net = net.view(net.size(0), -1)
        x_info = self.act_func(self.x_info_layer(net))
        net = self.fc_enc(net)
        mean, logvar = net[:, :self.z_size], net[:, self.z_size:]

        return mean, logvar, x_info

    def decode(self, net):

        net = self.act_func(self.fc_dec(net))
        net = net.view(net.size(0), -1, 2, 2)
        net = self.act_func(self.bn_dec1(self.deconv1(net)))
        net = self.act_func(self.bn_dec2(self.deconv2(net)))
        logit = self.deconv3(net)

        return logit

    def sample(self, mu, logvar, grad_fn=lambda x: 1, x_info=None):
        # grad_fn default is identity, i.e. don't use grad info
        eps = Variable(torch.randn(mu.size()).type(self.dtype))
        z = eps.mul(logvar.mul(0.5).exp()).add(mu)
        logqz = log_normal(z, mu, logvar)

        if self.has_flow:
            z, logprob = self.q_dist.forward(z, grad_fn, x_info)
            logqz += logprob

        zeros = Variable(torch.zeros(z.size()).type(self.dtype))
        logpz = log_normal(z, zeros, zeros)

        return z, logpz, logqz

    def forward(self, x, k=1, warmup_const=1.):

        x = x.repeat(k, 1, 1, 1)  # for computing iwae bound
        mu, logvar, x_info = self.encode(x)

        # posterior-aware inference
        def U(z):
            logpx = log_bernoulli(self.decode(z), x)
            logpz = log_normal(z)
            return -logpx - logpz  # energy as -log p(x, z)

        def grad_U(z):
            grad_outputs = torch.ones(z.size(0)).type(self.dtype)
            grad = torchgrad(U(z), z, grad_outputs=grad_outputs, create_graph=True)[0]
            # gradient clipping by norm avoid numerical issue
            norm = torch.sqrt(torch.norm(grad, p=2, dim=1))
            grad = grad / norm.view(-1, 1)
            return grad.detach()

        if self.hamiltonian_flow:
            z, logpz, logqz = self.sample(mu, logvar, grad_fn=grad_U, x_info=x_info)
        else:
            z, logpz, logqz = self.sample(mu, logvar, x_info=x_info)

        logit = self.decode(z)
        logpx = log_bernoulli(logit, x)
        elbo = logpx + logpz - warmup_const * logqz  # custom warmup
        # correction for Tensor.repeat
        elbo = log_mean_exp(elbo.view(k, -1).transpose(0, 1))
        elbo = torch.mean(elbo)

        logpx = torch.mean(logpx)
        logpz = torch.mean(logpz)
        logqz = torch.mean(logqz)

        return elbo, logpx, logpz, logqz

    def reconstruct_img(self, x):

        # for visualization
        mu, logvar, x_info = self.encode(x)
        z, logpz, logqz = self.sample(mu, logvar)
        logit = self.decode(z)
        x_hat = torch.sigmoid(logit)

        return x_hat

    def freeze_decoder(self):
        # freeze so that decoder is not optimized
        for layer in self.decoder_layers:
            for param_name in layer._parameters:
                layer._parameters[param_name].requires_grad = False

    def unfreeze_decoder(self):
        # unfreeze so that decoder is optimized
        for layer in self.decoder_layers:
            for param_name in layer._parameters:
                layer._parameters[param_name].requires_grad = True

