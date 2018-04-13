from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.math_ops import log_bernoulli, log_normal, log_mean_exp, safe_repeat
from utils.hparams import HParams

from loader import get_Larochelle_MNIST_loader, get_fashion_loader, get_cifar10_loader
from vae import VAE
from cvae import CVAE


parser = argparse.ArgumentParser(description='local_expressive')
# action configuration flags
parser.add_argument('--no-cuda', '-nc', action='store_true')
parser.add_argument('--debug', action='store_true', help='debug mode')

# model configuration flags
parser.add_argument('--z-size', '-zs', type=int, default=50)
parser.add_argument('--batch-size', '-bs', type=int, default=100)
parser.add_argument('--eval-path', '-ep', type=str, default='model.pth',
                    help='path to load evaluation ckpt (default: model.pth)')
parser.add_argument('--dataset', '-d', type=str, default='mnist',
                    choices=['mnist', 'fashion', 'cifar'], 
                    help='dataset to train and evaluate on (default: mnist)')
parser.add_argument('--has-flow', '-hf', action='store_true', help='inference uses FLOW')
parser.add_argument('--n-flows', '-nf', type=int, default=2, help='number of flows')
parser.add_argument('--wide-encoder', '-we', action='store_true',
                    help='use wider layer (more hidden units for FC, more channels for CIFAR)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def get_default_hparams():
    return HParams(
        z_size=args.z_size,
        act_func=F.elu,
        has_flow=args.has_flow,
        n_flows=args.n_flows,
        wide_encoder=args.wide_encoder,
        cuda=args.cuda,
        hamiltonian_flow=False
    )


def optimize_local_expressive(
    log_likelihood,
    model,
    data_var,
    k=100,
    check_every=100,
    sentinel_thres=10,
    n_flows=2,
    debug=False
):
    """data_var should be (cuda) variable."""

    def log_joint(x_logits, x, z):
        """log p(x,z)"""
        zeros = Variable(torch.zeros(z.size()).type(model.dtype))
        logpz = log_normal(z, zeros, zeros)
        logpx = log_likelihood(x_logits, x)

        return logpx + logpz

    def norm_flow(params, z, v):

        h = F.tanh(params[0][0](z))
        mew_ = params[0][1](h)
        logit_ = params[0][2](h)
        sig_ = F.sigmoid(logit_)

        v = v*sig_ + mew_
        # numerically stable: log (sigmoid(logit)) = logit - softplus(logit)
        logdet_v = torch.sum(logit_ - F.softplus(logit_), 1)

        h = F.tanh(params[1][0](v))
        mew_ = params[1][1](h)
        logit_ = params[1][2](h)
        sig_ = F.sigmoid(logit_)

        z = z*sig_ + mew_
        logdet_z = torch.sum(logit_ - F.softplus(logit_), 1)

        logdet = logdet_v + logdet_z

        return z, v, logdet

    def sample(mean_v0, logvar_v0):

        B = mean_v0.size()[0]
        eps = Variable(torch.FloatTensor(B, z_size).normal_().type(model.dtype))
        v0 = eps.mul(logvar_v0.mul(0.5).exp_()) + mean_v0
        logqv0 = log_normal(v0, mean_v0, logvar_v0)

        out = v0
        for i in range(len(qz_weights)-1):
            out = act_func(qz_weights[i](out))
        out = qz_weights[-1](out)
        mean_z0, logvar_z0 = out[:, :z_size], out[:, z_size:]

        eps = Variable(torch.FloatTensor(B, z_size).normal_().type(model.dtype))
        z0 = eps.mul(logvar_z0.mul(0.5).exp_()) + mean_z0
        logqz0 = log_normal(z0, mean_z0, logvar_z0)

        zT, vT = z0, v0
        logdetsum = 0.
        for i in range(n_flows):
            zT, vT, logdet = norm_flow(params[i], zT, vT)
            logdetsum += logdet

        # reverse model, r(vT|x,zT)
        out = zT
        for i in range(len(rv_weights)-1):
            out = act_func(rv_weights[i](out))
        out = rv_weights[-1](out)
        mean_vT, logvar_vT = out[:, :z_size], out[:, z_size:]
        logrvT = log_normal(vT, mean_vT, logvar_vT)

        logq = logqz0 + logqv0 - logdetsum - logrvT

        return zT, logq

    def get_params():

        all_params = []

        mean_v = Variable(torch.zeros(B*k, z_size).type(model.dtype), requires_grad=True)
        logvar_v = Variable(torch.zeros(B*k, z_size).type(model.dtype), requires_grad=True)

        all_params.append(mean_v)
        all_params.append(logvar_v)

        qz_weights = []  # q(z|x,v)
        for ins, outs in zip(qz_arch[:-1], qz_arch[1:]):
            cur_layer = nn.Linear(ins, outs)
            if args.cuda:
                cur_layer.cuda()
            qz_weights.append(cur_layer)
            all_params.append(cur_layer.weight)

        rv_weights = []  # r(v|x,z)
        for ins, outs in zip(rv_arch[:-1], rv_arch[1:]):
            cur_layer = nn.Linear(ins, outs)
            if args.cuda:
                cur_layer.cuda()
            rv_weights.append(cur_layer)
            all_params.append(cur_layer.weight)

        params = []
        for i in range(n_flows):
            layers = [
                [nn.Linear(z_size, h_s),
                 nn.Linear(h_s, z_size),
                 nn.Linear(h_s, z_size)],
                [nn.Linear(z_size, h_s),
                 nn.Linear(h_s, z_size),
                 nn.Linear(h_s, z_size)],
            ]

            params.append(layers)

            for sublist in layers:
                for item in sublist:
                    all_params.append(item.weight)
                    if args.cuda:
                        item.cuda()

        return (mean_v, logvar_v), all_params, params, qz_weights, rv_weights

    # the real shit
    B = data_var.size(0)
    z_size = args.z_size
    qz_arch = rv_arch = [args.z_size, 200, 200, args.z_size*2]
    h_s = 200
    act_func = F.elu

    data_var = safe_repeat(data_var, k)
    (mean_v, logvar_v), all_params, params, qz_weights, rv_weights = get_params()

    # tile input for IS
    optimizer = optim.Adam(all_params, lr=1e-3)
    best_avg, sentinel, prev_seq = 999999, 0, []

    # perform local opt
    time_ = time.time()
    for epoch in range(1, 999999):
        z, logqz = sample(mean_v, logvar_v)
        x_logits = model.decode(z)
        logpxz = log_joint(x_logits, data_var, z)

        optimizer.zero_grad()
        loss = -torch.mean(logpxz - logqz)
        loss_np = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        prev_seq.append(loss_np)
        if epoch % check_every == 0:
            last_avg = np.mean(prev_seq)
            if debug:  # debugging helper
                sys.stderr.write(
                    'Epoch %d, time elapse %.4f, last avg %.4f, prev best %.4f\n' % \
                    (epoch, time.time()-time_, -last_avg, -best_avg)
                )
            if last_avg < best_avg:
                sentinel, best_avg = 0, last_avg
            else:
                sentinel += 1
            if sentinel > sentinel_thres:
                break

            prev_seq = []
            time_ = time.time()

    # evaluation
    z, logqz = sample(mean_v, logvar_v)
    x_logits = model.decode(z)
    logpxz = log_joint(x_logits, data_var, z)
    elbo = logpxz - logqz

    vae_elbo = torch.mean(elbo)
    iwae_elbo = torch.mean(log_mean_exp(elbo.view(k, -1).transpose(0, 1)))

    return vae_elbo.data[0], iwae_elbo.data[0]


def main():
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        evaluate=True, batch_size=1
    )
    model = get_model(args.dataset, get_default_hparams())
    model.load_state_dict(torch.load(args.eval_path)['state_dict'])
    model.eval()

    vae_record, iwae_record = [], []
    time_ = time.time()
    for i, (batch, _) in tqdm(enumerate(train_loader)):
        batch = Variable(batch.type(model.dtype))
        elbo, iwae = optimize_local_expressive(
            log_bernoulli,
            model,
            batch,
            n_flows=args.n_flows, debug=args.debug
        )
        vae_record.append(elbo)
        iwae_record.append(iwae)
        print ('Local opt w/ flow, batch %d, time elapse %.4f, ELBO %.4f, IWAE %.4f' % \
            (i+1, time.time()-time_, elbo, iwae))
        print ('mean of ELBO so far %.4f, mean of IWAE so far %.4f' % \
            (np.nanmean(vae_record), np.nanmean(iwae_record)))
        time_ = time.time()

    print ('Finishing...')
    print ('Average ELBO %.4f, IWAE %.4f' % (np.nanmean(vae_record), np.nanmean(iwae_record)))


if __name__ == '__main__':
    main()
