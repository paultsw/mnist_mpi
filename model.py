"""
Neural network for MNIST classification.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim


class MNISTNetwork(nn.Module):
    def __init__(self):
        """
        Feedforward NN for MNIST classification.
        This model is adapted from the PyTorch example at:
          https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py#L29

        The 320-dimensional hidden size of the linear feedforward network is due to
        the assumption that MNIST images are of shape (1,28,28) ~ (nchannels, height, width).
        
        ** TODO: consider using the "One Weird Trick..." implementation of applying data-parallelism
        for self.convnet and model parallelism for self.feedforward.
        """
        super(MNISTNetwork, self).__init__()
        # network model, convnet base:
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        # network model, feedforward:
        self.feedforward = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(50, 10)
        )

        # initialize NN:
        for p in self.convnet.parameters():
            if len(p.shape) > 1: init.xavier_uniform_(p)
        for p in self.feedforward.parameters():
            if len(p.shape) > 1: init.xavier_uniform_(p)

    def forward(self, xs):
        """Run MNIST network on a batch of images: xs ~ (bsz, rows, cols)."""
        return self.feedforward(self.convnet(xs).view(-1, 320))
