from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import visdom

from utils.hparams import HParams
from utils.ais import ais_trajectory
from utils.math_ops import sigmoidial_schedule, linear_schedule
from utils.helper import get_model, get_loaders


parser = argparse.ArgumentParser(description='VAE')
# action configuration flags
parser.add_argument('--train', '-t', action='store_true')
parser.add_argument('--load-path', '-lp', type=str, default='NA',
                    help='path to load checkpoint to retrain')
parser.add_argument('--load-epoch', '-le', type=int, default=0,
                    help='epoch number to start recording when retraining')
parser.add_argument('--display-epoch', '-de', type=int, default=10,
                    help='print status every so many epochs (default: 10)')
parser.add_argument('--eval-iwae', '-ei', action='store_true')
parser.add_argument('--eval-ais', '-ea', action='store_true')
parser.add_argument('--n-iwae', '-ni', type=int, default=5000,
                    help='number of samples for IWAE evaluation (default: 5000)')
parser.add_argument('--n-ais-iwae', '-nai', type=int, default=100,
                    help='number of IMPORTANCE samples for AIS evaluation (default: 100). \
                          This is different from MC samples.')
parser.add_argument('--n-ais-dist', '-nad', type=int, default=10000,
                    help='number of distributions for AIS evaluation (default: 10000)')
parser.add_argument('--ais-schedule', type=str, default='linear', help='schedule for AIS')

parser.add_argument('--no-cuda', '-nc', action='store_true', help='force not use CUDA')
parser.add_argument('--visdom', '-v', action='store_true', help='visualize samples')
parser.add_argument('--port', '-p', type=int, default=8097, help='port for visdom')
parser.add_argument('--save-visdom', default='test', help='visdom save path')
parser.add_argument('--encoder-more', action='store_true', help='train the encoder more (5 vs 1)')
parser.add_argument('--early-stopping', '-es', action='store_true', help='apply early stopping')
parser.add_argument('--epochs', '-e', type=int, default=3280,
                    help='total num of epochs for training (default: 3280)')
parser.add_argument('--lr-schedule', '-lrs', action='store_true',
                    help='apply learning rate schedule')

# model configuration flags
parser.add_argument('--z-size', '-zs', type=int, default=50,
                    help='dimensionality of latent code (default: 50)')
parser.add_argument('--batch-size', '-bs', type=int, default=100,
                    help='batch size (default: 100)')
parser.add_argument('--save-name', '-sn', type=str, default='model.pth',
                    help='name to save trained ckpt (default: model.pth)')
parser.add_argument('--eval-path', '-ep', type=str, default='model.pth',
                    help='path to load evaluation ckpt (default: model.pth)')
parser.add_argument('--dataset', '-d', type=str, default='mnist',
                    choices=['mnist', 'fashion', 'cifar'], 
                    help='dataset to train and evaluate on (default: mnist)')
parser.add_argument('--wide-encoder', '-we', action='store_true',
                    help='use wider layer (more hidden units for FC, more channels for CIFAR)')
parser.add_argument('--has-flow', '-hf', action='store_true',
                    help='use flow for training and eval')
parser.add_argument('--hamiltonian-flow', '-hamil-f', action='store_true')
parser.add_argument('--n-flows', '-nf', type=int, default=2, help='number of flows')
parser.add_argument('--warmup', '-w', action='store_true',
                    help='apply warmup during training')

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


def train(
    model,
    train_loader,
    test_loader,
    k_train=1,  # num iwae sample for training
    k_eval=1,  # num iwae sample for eval
    epochs=3280,
    display_epoch=10,
    lr_schedule=True,
    warmup=True,
    warmup_thres=None,
    encoder_more=False,
    checkpoints=None,
    early_stopping=False,
    save=True,
    save_path='checkpoints/mnist/',
    patience=10  # for early-stopping
):
    print('Training')

    if args.load_path != 'NA':
        f = args.load_path
        model.load_state_dict(torch.load(f)['state_dict'])

    # default warmup schedule
    if warmup_thres is None:
        if 'cifar' in save_path:
            warmup_thres = 50.
        elif 'mnist' in save_path or 'fashion' in save_path:
            warmup_thres = 400.

    if checkpoints is None:  # save a checkpoint every display_epoch
        checkpoints = [1] + list(range(0, 3280, display_epoch))[1:] + [3280]

    time_ = time.time()

    if lr_schedule:
        current_lr = 1e-3
        pow = 0
        epoch_elapsed = 0
        # pth default: beta_1 = .9, beta_2 = .999, eps = 1e-8
        optimizer = optim.Adam(model.parameters(), lr=current_lr, eps=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-4, eps=1e-4)

    num_worse = 0  # compare against `patience` for early-stopping
    prev_valid_err = None

    for epoch in tqdm(range(1, epochs+1)):
        warmup_const = min(1., epoch / warmup_thres) if warmup else 1.
        # lr schedule from IWAE: https://arxiv.org/pdf/1509.00519.pdf
        if lr_schedule:
            if epoch_elapsed >= 3 ** pow:
                current_lr *= 10. ** (-1. / 7.)
                pow += 1
                epoch_elapsed = 0
                # correct way to do lr decay; also possible w/ `torch.optim.lr_scheduler`
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            epoch_elapsed += 1

        model.train()  # crucial for BN to work properly
        for _, (batch, _) in enumerate(train_loader):
            batch = Variable(batch)
            if args.cuda:
                batch = batch.cuda()

            # train the encoder more
            if encoder_more:
                model.freeze_decoder()
                for _ in range(10):
                    optimizer.zero_grad()
                    elbo, _, _, _ = model.forward(batch, k_train, warmup_const)
                    loss = -elbo
                    loss.backward()
                    optimizer.step()
                model.unfreeze_decoder()

            optimizer.zero_grad()
            elbo, _, _, _ = model.forward(batch, k_train, warmup_const)
            loss = -elbo
            loss.backward()
            optimizer.step()

        if epoch % display_epoch == 0:
            model.eval()  # crucial for BN to work properly

            train_logpx, test_logpx = [], []
            train_logpz, test_logpz = [], []
            train_logqz, test_logqz = [], []
            train_stats, test_stats = [], []
            for _, (batch, _) in enumerate(train_loader):
                batch = Variable(batch)
                if args.cuda:
                    batch = batch.cuda()
                elbo, logpx, logpz, logqz = model(batch, k=1)
                train_stats.append(elbo.data[0])
                train_logpx.append(logpx.data[0])
                train_logpz.append(logpz.data[0])
                train_logqz.append(logqz.data[0])

            for _, (batch, _) in enumerate(test_loader):
                batch = Variable(batch)
                if args.cuda:
                    batch = batch.cuda()
                # early stopping with iwae bound
                elbo, logpx, logpz, logqz = model(batch, k=k_eval)
                test_stats.append(elbo.data[0])
                test_logpx.append(logpx.data[0])
                test_logpz.append(logpz.data[0])
                test_logqz.append(logqz.data[0])
            print (
                'Train Epoch: [{}/{}]'.format(epoch, epochs),
                'Train set ELBO {:.4f}'.format(np.mean(train_stats)),
                'Test/Validation set IWAE {:.4f}'.format(np.mean(test_stats)),
                'Time: {:.2f}'.format(time.time()-time_),
            )
            time_ = time.time()

            if early_stopping:
                curr_valid_err = np.mean(test_stats)

                if prev_valid_err is None:  # don't have history yet
                    prev_valid_err = curr_valid_err
                elif curr_valid_err >= prev_valid_err:  # performance improved
                    prev_valid_err = curr_valid_err
                    num_worse = 0
                else:
                    num_worse += 1

                if num_worse >= patience:
                    break

        if save and epoch in checkpoints:
            torch.save({
                'epoch': epochs + args.load_epoch,
                'state_dict': model.state_dict(),
            }, '%s%d_%s' % (save_path, epoch + args.load_epoch, args.save_name))


def test_iwae(
    model,
    loader,
    k=5000,
    f='model.pth',
    print_res=True
):
    print('Testing with %d importance samples' % k)
    model.load_state_dict(torch.load(f)['state_dict'])
    model.eval()
    time_ = time.time()
    elbos = []
    for i, (batch, _) in enumerate(loader):
        batch = Variable(batch)
        if args.cuda:
            batch = batch.cuda()
        elbo, logpx, logpz, logqz = model(batch, k=k)
        elbos.append(elbo.data[0])

    mean_ = np.mean(elbos)
    if print_res:
        print(mean_, 'T:', time.time()-time_)
    return mean_


def run():
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        evaluate=args.eval_iwae or args.eval_ais,
        batch_size=args.batch_size
    )
    model = get_model(args.dataset, get_default_hparams())

    if args.train:
        save_path = 'checkpoints/%s/%s/%s%s/' % (
                        args.dataset,
                        'warmup' if args.warmup else 'no_warmup',
                        'wide_' if args.wide_encoder else '',
                        'hamiltonian_flow' if args.hamiltonian_flow else 
                            'flow' if args.has_flow else 'ffg'
                    )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        train(
            model, train_loader, test_loader,
            display_epoch=args.display_epoch, epochs=args.epochs,
            lr_schedule=args.lr_schedule,
            warmup=args.warmup,
            early_stopping=args.early_stopping,
            encoder_more=args.encoder_more,
            save=True, save_path=save_path
        )

    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=args.port)
        model.load_state_dict(torch.load(args.eval_path)['state_dict'])

        # plot original images
        batch, _ = train_loader.next()
        images = list(batch.numpy())
        win_samples = vis.images(images, 10, 2, opts={'caption': 'original images'}, win=None)

        # plot reconstructions
        batch = Variable(batch.type(model.dtype))
        reconstruction = model.reconstruct_img(batch)
        images = list(reconstruction.data.cpu().numpy())
        win_samples = vis.images(images, 10, 2, opts={'caption': 'reconstruction'}, win=None)

    if args.eval_iwae:
        # VAE bounds computed w/ 100 MC samples to reduce variance
        train_res, test_res = [], []
        for _ in range(100):
            test_iwae(model, train_loader, k=1, f=args.eval_path)
            test_iwae(model, test_loader, k=1, f=args.eval_path)
            train_res.append(train_res)
            test_res.append(test_res)

        print ('Training set VAE ELBO w/ 100 MC samples: %.4f' % np.mean(train_res))
        print ('Test set VAE ELBO w/ 100 MC samples: %.4f' % np.mean(test_res))

        # IWAE bounds
        test_iwae(model, train_loader, k=args.n_iwae, f=args.eval_path)
        test_iwae(model, test_loader, k=args.n_iwae, f=args.eval_path)

    if args.eval_ais:
        model.load_state_dict(torch.load(args.eval_path)['state_dict'])
        schedule_fn = linear_schedule if args.ais_schedule == 'linear' else sigmoidial_schedule
        schedule = schedule_fn(args.n_ais_dist)
        ais_trajectory(
            model, train_loader,
            mode='forward', schedule=schedule, n_sample=args.n_ais_iwae
        )


if __name__ == '__main__':
    run()
