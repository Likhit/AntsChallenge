import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import netutils
from ... import utils

class ConvNet(nn.Module):
    """
    A CNN featurizer.

    Args:
        input_shape (tuple): The shape of the input that will be
            fed to the net.
            Expected shape: (channels, height, width).
    """
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self._define_net()

    def _define_net(self):
        self.conv0 = nn.Conv2d(self.input_shape[0], 3, 1)
        shape = netutils.conv_output_shape(self.input_shape[1:], 1)
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, 3)
        shape = netutils.conv_output_shape(shape, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        shape = netutils.conv_output_shape(shape, 3)
        self.max_pool = nn.MaxPool2d(2, stride=1)
        shape = netutils.conv_output_shape(shape, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        shape = netutils.conv_output_shape(shape, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        shape = netutils.conv_output_shape(shape, 3)
        self.output_size = 32 * shape[0] * shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm1(x)
        x = self.max_pool(F.relu(self.conv3(x)))
        x = self.conv4(x)
        return x
