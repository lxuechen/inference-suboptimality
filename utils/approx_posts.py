from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.math_ops import log_bernoulli, log_normal, log_mean_exp


class Flow(nn.Module):
    """A combination of R-NVP and auxiliary variables."""

    def __init__(self, model, n_flows=2):
        super(Flow, self).__init__()
        self.z_size = model.z_size
        self.n_flows = n_flows
        self._construct_weights()

    def forward(self, z, grad_fn=lambda x: 1, x_info=None):
        return self._sample(z, grad_fn, x_info)

    def _norm_flow(self, params, z, v, grad_fn, x_info):
        h = F.elu(params[0][0](torch.cat((z, x_info), dim=1)))
        mu = params[0][1](h)
        logit = params[0][2](h)
        sig = F.sigmoid(logit)

        # old CIFAR used the one below
        # v = v * sig + mu * grad_fn(z)

        # the more efficient one uses the one below
        v = v * sig - F.elu(mu) * grad_fn(z)
        logdet_v = torch.sum(logit - F.softplus(logit), 1)

        h = F.elu(params[1][0](torch.cat((v, x_info), dim=1)))
        mu = params[1][1](h)
        logit = params[1][2](h)
        sig = F.sigmoid(logit)

        z = z * sig + mu
        logdet_z = torch.sum(logit - F.softplus(logit), 1)
        logdet = logdet_v + logdet_z

        return z, v, logdet

    def _sample(self, z0, grad_fn, x_info):
        B = z0.size(0)
        z_size = self.z_size
        act_func = F.elu
        qv_weights, rv_weights, params = self.qv_weights, self.rv_weights, self.params

        out = torch.cat((z0, x_info), dim=1)
        for i in range(len(qv_weights)-1):
            out = act_func(qv_weights[i](out))
        out = qv_weights[-1](out)
        mean_v0, logvar_v0 = out[:, :z_size], out[:, z_size:]

        eps = Variable(torch.randn(B, z_size).type( type(out.data) ))
        v0 = eps.mul(logvar_v0.mul(0.5).exp_()) + mean_v0
        logqv0 = log_normal(v0, mean_v0, logvar_v0)

        zT, vT = z0, v0
        logdetsum = 0.
        for i in range(self.n_flows):
            zT, vT, logdet = self._norm_flow(params[i], zT, vT, grad_fn, x_info)
            logdetsum += logdet

        # reverse model, r(vT|x,zT)
        out = torch.cat((zT, x_info), dim=1)
        for i in range(len(rv_weights)-1):
            out = act_func(rv_weights[i](out))
        out = rv_weights[-1](out)
        mean_vT, logvar_vT = out[:, :z_size], out[:, z_size:]
        logrvT = log_normal(vT, mean_vT, logvar_vT)

        assert logqv0.size() == (B,)
        assert logdetsum.size() == (B,)
        assert logrvT.size() == (B,)

        logprob = logqv0 - logdetsum - logrvT

        return zT, logprob

    def _construct_weights(self):
        z_size = self.z_size
        n_flows = self.n_flows
        h_s = 200

        qv_arch = rv_arch = [z_size*2, h_s, h_s, z_size*2]
        qv_weights, rv_weights = [], []

        # q(v|x,z)
        id = 0
        for ins, outs in zip(qv_arch[:-1], qv_arch[1:]):
            cur_layer = nn.Linear(ins, outs)
            qv_weights.append(cur_layer)
            self.add_module('qz%d' % id, cur_layer)
            id += 1

        # r(v|x,z)
        id = 0
        for ins, outs in zip(rv_arch[:-1], rv_arch[1:]):
            cur_layer = nn.Linear(ins, outs)
            rv_weights.append(cur_layer)
            self.add_module('rv%d' % id, cur_layer)
            id += 1

        # nf
        params = []
        for i in range(n_flows):
            layer_grid = [
                [nn.Linear(z_size*2, h_s),
                 nn.Linear(h_s, z_size),
                 nn.Linear(h_s, z_size)],
                [nn.Linear(z_size*2, h_s),
                 nn.Linear(h_s, z_size),
                 nn.Linear(h_s, z_size)],
            ]

            params.append(layer_grid)

            id = 0
            for layer_list in layer_grid:
                for layer in layer_list:
                    self.add_module('flow%d_layer%d' % (i, id), layer)
                    id += 1

        self.qv_weights = qv_weights
        self.rv_weights = rv_weights
        self.params = params

        self.sanity_check_param = self.params[0][0][0]._parameters['weight']
