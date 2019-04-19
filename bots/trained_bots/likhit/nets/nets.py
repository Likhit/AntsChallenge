import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import netutils

class ConvEmbedding(nn.Module):
    """
    A CNN featurizer which transforms the input space into an
    embedding space by using 1x1 convolutions.

    Args:
        input_shape ([channels, height, width]): The shape of the input that
        will fed to the network.
        output_channels (int): The number of channels in the embedding space.
    """
    def __init__(self, input_shape, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_shape[0], output_channels, 1)
        shape = netutils.conv_output_shape(input_shape[1:], 1)
        self.output_shape = (output_channels, *shape)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlock(nn.Module):
    """
    A 4-layer CNN network with batch-norm and ReLU after each layer (except
    the last layer). Each convolutin is a 3x3 convolution with stride 1 and
    no padding. The last layer is a 1x1 convolution without any activation.

    Args:
        input_shape ([channels, height, width]): The shape of the input that
        will be fed to the net.
        num_filters ([int]): The number of output channels in each of the
        filters. Default 32 per layer.
    """
    def __init__(self, input_shape, num_filters=None):
        super().__init__()
        if not num_filters:
            num_filters = [32] * 4
        elif len(num_filters) < 4:
            extras = [32] * (4 - len(num_filters))
            num_filters += extras
        self._define_net(input_shape, num_filters)

    def _define_net(self, input_shape, num_filters):
        self.conv1 = nn.Conv2d(input_shape[0], num_filters[0], 3)
        self.batch_norm1 = nn.BatchNorm2d(num_filters[0])
        shape = netutils.conv_output_shape(input_shape[1:], 3)

        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], 3)
        self.batch_norm2 = nn.BatchNorm2d(num_filters[1])
        shape = netutils.conv_output_shape(shape, 3)

        self.conv3 = nn.Conv2d(num_filters[1], num_filters[2], 3)
        self.batch_norm3 = nn.BatchNorm2d(num_filters[2])
        shape = netutils.conv_output_shape(shape, 3)

        self.conv4 = nn.Conv2d(num_filters[2], num_filters[3], 1)
        shape = netutils.conv_output_shape(shape, 1)

        self.output_size = (num_filters[-1], *shape)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm1(x)

        x = F.relu(self.conv2(x))
        x = self.batch_norm2(x)

        x = F.relu(self.conv3(x))
        x = self.batch_norm3(x)
        return self.conv4(x)


class ShallowStandAlonePolicy(nn.Module):
    """
    A shallow policy network with an ConvEmbedding layer followed by a
    single ConvBlock followed 1 single linear layer which is transformed
    into the output shape.
    """
    NUMBER_OUTPUT = 5 # The number of moves possible for an ant.

    def __init__(self, input_shape, embedding_channels, num_filters=None):
        super().__init__()
        self.embeddings = ConvEmbedding(
            input_shape,
            embedding_channels
        )
        self.conv_block = ConvBlock(
            self.embeddings.output_shape,
            num_filters
        )

        num_feat = np.prod(self.conv_block.output_size)
        num_hidden = num_feat // 4
        self.linear1 = nn.Linear(num_feat, num_hidden)
        self.linear2 = nn.Linear(
            num_hidden, ShallowStandAlonePolicy.NUMBER_OUTPUT
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = self.conv_block(x)
        x = self.linear1(x.view(-1, self.linear1.in_features))
        x = self.linear2(x)
        return x
