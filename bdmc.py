import numpy as np
import itertools
import time
import argparse

import torch
from torch.autograd import Variable
from torch.autograd import grad as torchgrad
import torch.nn.functional as F

from utils.ais import ais_trajectory
from utils.simulate import simulate_data
from utils.hparams import HParams
from utils.math_ops import sigmoidial_schedule
from utils.helper import get_model


parser = argparse.ArgumentParser(description='bidirectional_mc')
# action configuration flags
parser.add_argument('--n-ais-iwae', '-nai', type=int, default=100,
                    help='number of IMPORTANCE samples for AIS evaluation (default: 100). \
                          This is different from MC samples.')
parser.add_argument('--n-ais-dist', '-nad', type=int, default=10000,
                    help='number of distributions for AIS evaluation (default: 10000)')
parser.add_argument('--no-cuda', '-nc', action='store_true', help='force not use CUDA')

# model configuration flags
parser.add_argument('--z-size', '-zs', type=int, default=50,
                    help='dimensionality of latent code (default: 50)')
parser.add_argument('--batch-size', '-bs', type=int, default=100,
                    help='batch size (default: 100)')
parser.add_argument('--n-batch', '-nb', type=int, default=10,
                    help='total number of batches (default: 10)')
parser.add_argument('--eval-path', '-ep', type=str, default='model.pth',
                    help='path to load evaluation ckpt (default: model.pth)')
parser.add_argument('--dataset', '-d', type=str, default='mnist', choices=['mnist', 'fashion', 'cifar'], 
                    help='dataset to train and evaluate on (default: mnist)')
parser.add_argument('--wide-encoder', '-we', action='store_true',
                    help='use wider layer (more hidden units for FC, more channels for CIFAR)')
parser.add_argument('--has-flow', '-hf', action='store_true',
                    help='use flow for training and eval')
parser.add_argument('--hamiltonian-flow', '-hamil-f', action='store_true')
parser.add_argument('--n-flows', '-nf', type=int, default=2, help='number of flows')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def get_default_hparams():
    return HParams(
        z_size=args.z_size,
        act_func=F.elu,
        has_flow=args.has_flow,
        hamiltonian_flow=args.hamiltonian_flow,
        n_flows=args.n_flows,
        wide_encoder=args.wide_encoder,
        cuda=args.cuda,
    )


def bdmc(model, loader, forward_schedule=np.linspace(0., 1., 500), n_sample=100):
    """Bidirectional Monte Carlo. Integrate forward and backward AIS.
    The backward schedule is the reverse of the forward.

    Args:
        model (vae.VAE): VAE model
        loader (iterator): iterator to loop over pairs of Variables; the first 
            entry being `x`, the second being `z` sampled from the true 
            posterior `p(z|x)`
        forward_schedule (list or numpy.ndarray): forward temperature schedule;
            backward schedule is used as its reverse
        n_sample: number of importance (not simple MC) sample
    Returns:
        Two lists for forward and backward bounds on batchs of data
    """

    # iterator is exhaustable in py3, so need duplicate
    load, load_ = itertools.tee(loader, 2)

    # forward chain
    forward_logws = ais_trajectory(
        model, load,
        mode='forward', schedule=forward_schedule,
        n_sample=n_sample
    )

    # backward chain
    backward_schedule = np.flip(forward_schedule, axis=0)
    backward_logws = ais_trajectory(
        model, load_,
        mode='backward',
        schedule=backward_schedule,
        n_sample=n_sample
    )

    upper_bounds = []
    lower_bounds = []

    for i, (forward, backward) in enumerate(zip(forward_logws, backward_logws)):
        lower_bounds.append(forward.mean())
        upper_bounds.append(backward.mean())

    upper_bounds = np.mean(upper_bounds)
    lower_bounds = np.mean(lower_bounds)

    print ('Average bounds on simulated data: lower %.4f, upper %.4f' % (lower_bounds, upper_bounds))

    return forward_logws, backward_logws


def main():
    # sanity check
    model = get_model(args.dataset, get_default_hparams())
    model.load_state_dict(torch.load(args.eval_path)['state_dict'])
    model.eval()

    loader = simulate_data(model, batch_size=args.batch_size, n_batch=args.n_batch)
    schedule = sigmoidial_schedule(args.n_ais_dist)
    bdmc(model, loader, forward_schedule=schedule, n_sample=args.n_ais_iwae)


if __name__ == '__main__':
    main()
