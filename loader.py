import numpy as np
from scipy import io
import sys
import os
import time

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import collections
import pickle
from torch.autograd import Variable


class CIFAR10:

    def __init__(self,
                 part='train',
                 batch_size=128,
                 partial=None,
                 binarize=True,
                 valid_size=0.1,
                 num_workers=4,
                 pin_memory=False):

        transform_list = [transforms.ToTensor()]
        if binarize:
            transform_list.append(lambda x: x >= 0.5)
            transform_list.append(lambda x: x.float())

        data_transform = transforms.Compose(transform_list)
        train_set = datasets.CIFAR10('./datasets', train=True, download=True, transform=data_transform)
        valid_set = datasets.CIFAR10('./datasets', train=True, download=True, transform=data_transform)
        test_set  = datasets.CIFAR10('./datasets', train=False, download=True, transform=data_transform)

        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        self.loader = {
            'train': DataLoader(train_set,
                        batch_size=batch_size, sampler=SubsetRandomSampler(train_idx),
                        num_workers=num_workers, pin_memory=pin_memory, shuffle=False),
            'valid': DataLoader(valid_set,
                        batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx),
                        num_workers=num_workers, pin_memory=pin_memory, shuffle=False),
            'test':  DataLoader(test_set, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
        }[part]

        self.size = len(self.loader) if partial is None else partial // batch_size
        self._iter = iter(self.loader)
        self.batch_size = batch_size
        self.p = 0

    def __iter__(self):
        self.p = 0
        self._iter = iter(self.loader)
        return self

    def __next__(self):
        self.p += 1
        if self.p > self.size:
            raise StopIteration
        return next(self._iter)

    # due to inconsistency between py2 and py3
    def next(self):
        return self.__next__()


class Larochelle_MNIST:

    def __init__(self, part='train', batch_size=128, partial=1000):
        with open('datasets/mnist.pkl', 'rb') as f:
            if sys.version_info[0] < 3:
                mnist = pickle.load(f)
            else:
                mnist = pickle.load(f, encoding='latin1')
            self.data = {
                'train': np.concatenate((mnist[0][0], mnist[1][0])),
                'test': mnist[2][0],
                'partial_train': mnist[0][0][:partial],
                'partial_test': mnist[2][0][:partial],
            }[part]
        self.size = self.data.shape[0]
        self.batch_size = batch_size
        self._construct()

    def __iter__(self):
        return iter(self.batch_list)

    def _construct(self):
        self.batch_list = []
        for i in range(self.size // self.batch_size):
            batch = self.data[self.batch_size*i:self.batch_size*(i+1)]
            batch = torch.from_numpy(batch)
            # placeholder for second entry
            self.batch_list.append((batch, None))


class Binarized_Omniglot:

    def __init__(self, part='train', batch_size=128, partial=1000):
        omni_raw = io.loadmat('datasets/chardata.mat')
        reshape_data = lambda d: d.reshape(
            (-1, 28, 28)).reshape((-1, 28*28), order='fortran')

        def static_binarize(d):
            ids = d < 0.5
            d[ids] = 0.
            d[~ids] = 1.

        train_data = reshape_data(omni_raw['data'].T.astype('float32'))
        test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
        static_binarize(train_data)
        static_binarize(test_data)

        assert train_data.shape == (24345, 784)
        assert test_data.shape == (8070, 784)

        self.data = {
            'train': train_data,
            'test':  test_data,
            'partial_train': train_data[:partial],
            'partial_test': test_data[:partial],
        }[part]
        self.size = self.data.shape[0]
        self.batch_size = batch_size
        self._construct()

    def __iter__(self):
        return iter(self.batch_list)

    def _construct(self):
        self.batch_list = []
        for i in range(self.size // self.batch_size):
            batch = self.data[self.batch_size*i:self.batch_size*(i+1)]
            batch = torch.from_numpy(batch)
            self.batch_list.append((batch, None))


class Binarized_Fashion:

    def __init__(self, part='train', batch_size=128, partial=1000):

        from utils.mnist_reader import load_mnist
        train_raw, _ = load_mnist('datasets/fashion', kind='train')
        test_raw, _ = load_mnist('datasets/fashion', kind='t10k')

        grey_scale = lambda x: np.float32(x / 255.)

        def static_binarize(d):
            ids = d < 0.5
            d[ids] = 0.
            d[~ids] = 1.

        train_data = grey_scale(train_raw)
        test_data = grey_scale(test_raw)

        static_binarize(train_data)
        static_binarize(test_data)

        assert train_data.shape == (60000, 784)
        assert test_data.shape == (10000, 784)

        self.data = {
            'train': train_data[:55000],
            'valid': train_data[55000:],
            'test':  test_data,
            'partial_train': train_data[:partial],
            'partial_test': test_data[:partial],
        }[part]
        self.size = self.data.shape[0]
        self.batch_size = batch_size
        self._construct()

    def __iter__(self):
        return iter(self.batch_list)

    def _construct(self):
        self.batch_list = []
        for i in range(self.size // self.batch_size):
            batch = self.data[self.batch_size*i:self.batch_size*(i+1)]
            batch = torch.from_numpy(batch)
            self.batch_list.append((batch, None))


def get_default_mnist_loader():

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./datasets', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=128, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./datasets', train=False,
                       transform=transforms.ToTensor()),
        batch_size=100, shuffle=True, **kwargs)

    return train_loader, test_loader


def get_cifar10_loader(batch_size=100, partial=False, num=1000):

    if partial:
        train_loader = CIFAR10(part='train', batch_size=batch_size, partial=num)
        test_loader  = CIFAR10(part='test', batch_size=4)
    else:
        train_loader = CIFAR10(part='train', batch_size=batch_size)
        test_loader = CIFAR10(part='valid', batch_size=4)  # really validation set

    return train_loader, test_loader


def get_Larochelle_MNIST_loader(batch_size=100, partial=False, num=1000):

    if partial:
        train_loader = Larochelle_MNIST(part='partial_train', batch_size=batch_size, partial=num)
        test_loader = Larochelle_MNIST(part='partial_test')
    else:
        train_loader = Larochelle_MNIST(part='train', batch_size=batch_size)
        test_loader = Larochelle_MNIST(part='test', batch_size=batch_size)
    
    return train_loader, test_loader


def get_omniglot_loader(batch_size=100, partial=False, num=1000):

    if partial:
        train_loader = Binarized_Omniglot(part='partial_train', batch_size=batch_size, partial=num)
        test_loader = Binarized_Omniglot(part='partial_test')
    else:
        train_loader = Binarized_Omniglot(part='train', batch_size=batch_size)
        test_loader = Binarized_Omniglot(part='valid', batch_size=10)

    return train_loader, test_loader


def get_fashion_loader(batch_size=100, partial=False, num=1000):

    if partial:
        train_loader = Binarized_Fashion(part='partial_train', batch_size=batch_size, partial=num)
        test_loader = Binarized_Fashion(part='partial_test', batch_size=10)
    else:
        train_loader = Binarized_Fashion(part='train', batch_size=batch_size)
        test_loader = Binarized_Fashion(part='valid', batch_size=10)
    
    return train_loader, test_loader


if __name__ == '__main__':
    # sanity checking
    train_loader, test_loader = get_cifar10_loader()
    for i, (batch, _) in enumerate(train_loader):
        batch = Variable(batch)
        print (i)

    for i, (batch, _) in enumerate(train_loader):
        batch = Variable(batch)
        print (i)
