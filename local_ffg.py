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
from utils.helper import get_model, get_loaders


parser = argparse.ArgumentParser(description='local_factorized_gaussian')
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


def optimize_local_gaussian(
    log_likelihood,
    model,
    data_var,
    k=100,
    check_every=100,
    sentinel_thres=10,
    debug=False
):
    """data_var should be (cuda) variable."""

    B = data_var.size()[0]
    z_size = model.z_size

    data_var = safe_repeat(data_var, k)
    zeros = Variable(torch.zeros(B*k, z_size).type(model.dtype))
    mean = Variable(torch.zeros(B*k, z_size).type(model.dtype), requires_grad=True)
    logvar = Variable(torch.zeros(B*k, z_size).type(model.dtype), requires_grad=True)

    optimizer = optim.Adam([mean, logvar], lr=1e-3)
    best_avg, sentinel, prev_seq = 999999, 0, []

    # perform local opt
    time_ = time.time()
    for epoch in range(1, 999999):

        eps = Variable(torch.FloatTensor(mean.size()).normal_().type(model.dtype))
        z = eps.mul(logvar.mul(0.5).exp_()).add_(mean)
        x_logits = model.decode(z)

        logpz = log_normal(z, zeros, zeros)
        logqz = log_normal(z, mean, logvar)
        logpx = log_likelihood(x_logits, data_var)

        optimizer.zero_grad()
        loss = -torch.mean(logpx + logpz - logqz)
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
    eps = Variable(torch.FloatTensor(B*k, z_size).normal_().type(model.dtype))
    z = eps.mul(logvar.mul(0.5).exp_()).add_(mean)

    logpz = log_normal(z, zeros, zeros)
    logqz = log_normal(z, mean, logvar)
    logpx = log_likelihood(model.decode(z), data_var)
    elbo = logpx + logpz - logqz

    vae_elbo = torch.mean(elbo)
    iwae_elbo = torch.mean(log_mean_exp(elbo.view(k, -1).transpose(0, 1)))

    return vae_elbo.data[0], iwae_elbo.data[0]


def main():
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        evaluate=True, batch_size=args.batch_size
    )
    model = get_model(args.dataset, get_default_hparams())
    model.load_state_dict(torch.load(args.eval_path)['state_dict'])
    model.eval()

    vae_record, iwae_record = [], []
    time_ = time.time()
    for i, (batch, _) in tqdm(enumerate(train_loader)):
        batch = Variable(batch.type(model.dtype))
        elbo, iwae = optimize_local_gaussian(log_bernoulli, model, batch, debug=args.debug)
        vae_record.append(elbo)
        iwae_record.append(iwae)
        print ('Local opt w/ ffg, batch %d, time elapse %.4f, ELBO %.4f, IWAE %.4f' % \
            (i+1, time.time()-time_, elbo, iwae))
        print ('mean of ELBO so far %.4f, mean of IWAE so far %.4f' % \
            (np.nanmean(vae_record), np.nanmean(iwae_record)))
        time_ = time.time()

    print ('Finishing...')
    print ('Average ELBO %.4f, IWAE %.4f' % (np.nanmean(vae_record), np.nanmean(iwae_record)))


if __name__ == '__main__':
    main()
