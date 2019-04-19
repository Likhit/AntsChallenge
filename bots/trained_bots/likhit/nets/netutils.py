"""
Helpful functions.
"""
import numpy as np

from types import SimpleNamespace

def conv_output_shape(shape, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(shape) is not tuple:
        shape = (shape, shape)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (shape[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    w = (shape[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1

    return h, w

def convtransp_output_shape(shape, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(shape) is not tuple:
        shape = (shape, shape)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (shape[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (shape[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w

class WrapPadTransform(object):
    """
    Transform to pad an numpy array of shape (N, C, H, W) in wrap mode along the H, W dimensions.

    Args:
        pad (int): The pad size.
    """
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, x):
        return np.pad(x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), 'wrap')

def collate_namespace(namespace_list):
    """
    Colates a list of namespaces into a sigle namespace object.
    Used to create a batch out of a list of namespaces.
    """
    result = SimpleNamespace()
    if len(namespace_list) == 0:
        return result

    attrs = list(namespace_list[0].__dict__.keys())
    for attr in attrs:
        setattr(result, attr, [])
    for item in namespace_list:
        for attr in attrs:
            getattr(result, attr).append(getattr(item, attr))
    return result
