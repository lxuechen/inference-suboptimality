from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from vae import VAE
from cvae import CVAE
from loader import get_Larochelle_MNIST_loader, get_fashion_loader, get_cifar10_loader


def get_loaders(dataset='mnist', evaluate=False, batch_size=100):
    if dataset == 'mnist':
        train_loader, test_loader = get_Larochelle_MNIST_loader(
            batch_size=batch_size,
            partial=evaluate, num=1000
        )
    elif dataset == 'fashion':
        train_loader, test_loader = get_fashion_loader(
            batch_size=batch_size,
            partial=evaluate, num=1000
        )
    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar10_loader(
            batch_size=batch_size,
            partial=evaluate, num=100
        )

    return train_loader, test_loader


def get_model(dataset, hps):
    if dataset == 'mnist' or dataset == 'fashion':
        model = VAE(hps)
    elif dataset == 'cifar':  # convolutional VAE for CIFAR
        model = CVAE(hps)

    return model
