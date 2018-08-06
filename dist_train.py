"""Distributed training via PyTorch."""
# torch utils
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision

# MPI library:
from mpi4py import MPI

# convnet model that classifies MNIST:
from model import MNISTNetwork

# other utilities:
import numpy as np
import os
import argparse
from collections import OrderedDict


def build_mnist_dataloader(train=True, bsz=16):
    """Return a dataloader that lets us iterate through the MNIST dataset."""
    return data.DataLoader(
        torchvision.datasets.MNIST("./", train=train, download=True, transform=torchvision.transforms.ToTensor()),
        shuffle=True,
        num_workers=1,
        batch_size=bsz
    )


def run_worker_process(model, comm, rank, size, args):
    """Define the training process for a single epoch."""
    loader = build_mnist_dataloader(train=True, bsz=args.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for k, (xs, ys) in enumerate(loader):
        opt.zero_grad()
        loss_fn(model(xs), ys).backward()
        opt.step()
    comm.send(model.state_dict(), 0)


def run_master_process(model, comm, rank, size, args):
    """Wait until all state-dicts received; then average results and perform validation."""
    # --- wait for all worker processes to return a state dict:
    # note: recv is blocking, so this for loop will wait until all worker
    # processes have returned state dicts:
    print("* Waiting for {0} training processes to finish...".format(size-1))
    state_dicts = []
    for p in range(size-1):
        state_dicts.append(comm.recv())
        print("(Received a trained model from process {0} of {1} workers...)".format(p+1, size-1))
    # --- average all state dicts together and update averaged model:
    print("* Averaging models...")
    avg_state_dict = OrderedDict()
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum([sd[key] for sd in state_dicts]) / float(size-1)
    model.load_state_dict(avg_state_dict)
    # --- run validation loop:
    model.eval()
    with torch.no_grad():
        val_losses = []
        val_loader = build_mnist_dataloader(train=False, bsz=args.batch_size)
        loss_fn = nn.CrossEntropyLoss()
        for xs, ys in val_loader:
            val_losses.append(loss_fn(model(xs), ys).item())
        print("* Mean validation loss of averaged model: {}".format(np.mean(val_losses)))


def dist_train(args):
    """Schedule a distributed training job."""
    # fetch MPI environment settings:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # set the random seed to be different for each process:
    torch.manual_seed(rank)
    # decide what to do based on rank:
    if rank == 0:
        # build a fresh model:
        model = MNISTNetwork()
        # loop over some number of epochs:
        for t in range(args.nepochs):
            print("[ = = = = = Epoch {} = = = = = ]".format(t))
            [comm.send(model.state_dict(), k) for k in range(1,size)]
            run_master_process(model, comm, rank, size, args)
    else:
        for t in range(args.nepochs):
            model = MNISTNetwork()
            model.load_state_dict(comm.recv())
            run_worker_process(model, comm, rank, size, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MNIST network across multiple distributed processes.")
    parser.add_argument("--lr", dest="lr", default=0.001,
                        help="Learning rate for SGD optimizer. [0.9]")
    parser.add_argument("--momentum", dest="momentum", default=0.9,
                        help="Momentum for SGD optimizer [0.9].")
    parser.add_argument("--batch_size", dest="batch_size", default=16,
                        help="Batch size to use for each process.")
    parser.add_argument("--nepochs", dest="nepochs", default=10,
                        help="Number of epochs (times to loop through the dataset).")
    args = parser.parse_args()
    dist_train(args)
