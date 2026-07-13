"""
This module handles tensor pooling.
"""
import torch
from torch import Tensor
import torch.nn.functional as nnf

from tensor_function import util

KERNEL_SIZE = (2, 2)
STRIDE = (2, 2)
POOL_TYPE = 'max'


def _pool(x: Tensor,
    pool_size=KERNEL_SIZE, pool_stride=STRIDE, pool_type=POOL_TYPE
) -> Tensor:
    """
    Pool the input tensor.

    Args:
        x (Tensor): The input tensor
        pool_size (tuple): The kernel/filter shape
        pool_stride (int): The window increment amount
        pool_type (str): The type of pooling operation to perform

    Return:
        A pooled tensor
    """
    # Check if performing average pooling
    if pool_type == "average":
        return nnf.avg_pool2d(x, kernel_size=pool_size, stride=pool_stride)
    # Else, return max pooling -- the default
    return nnf.max_pool2d(x, kernel_size=pool_size, stride=pool_stride)


def _unpool(
    upstream_grad: Tensor, pool_in: Tensor,
    pool_size=KERNEL_SIZE, pool_stride=STRIDE, pool_type=POOL_TYPE
) -> Tensor:
    """
    Apply unpooling on the upstream gradient tensor.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        relu_out (Tensor): The ReLU activated output tensor
        pool_size (tuple): The pool window dimensions
        pool_type (str): The type of pooling operation
        pool_stride (int): The window increment amount

    Return:
        The pooling input gradient
    """
    # Initialize the pooling input gradient
    pool_in_grad = torch.zeros_like(pool_in)

    # The 3rd and 4th dimensions of the ReLU activated output are the height
    #   and width of the prepooled Tensor
    h_dim, w_dim = (2, 3)

    # Get pool window height and width
    pool_h, pool_w = (pool_size)

    # Iterate over each position in the pool spatial grid
    for i in range(upstream_grad.shape[h_dim]):
        for j in range(upstream_grad.shape[w_dim]):
            h_start = i * pool_stride[0]
            h_stop = h_start + pool_h
            w_start = j * pool_stride[1]
            w_stop = w_start + pool_w

            # Get the pool window
            window = pool_in[:, :, h_start:h_stop, w_start:w_stop]
            # Get the upstream gradient window
            upstream_grad_window = upstream_grad[:, :, i:i+1, j:j+1]
            
            if pool_type == 'average':
                # Perform uniform distribution of the upstream gradient's
                #   average value across the pool window
                avg_value = upstream_grad_window / (pool_h * pool_w)
                pool_in_grad[:, :, h_start:h_stop, w_start:w_stop] += avg_value

            else: # unpool_type = 'max'
                # Perform routing of the upstream gradient's max value across
                #   the pool window
                max_values = window.amax(dim=(h_dim, w_dim), keepdim=True)
                mask = (window == max_values)

                pool_in_grad[:, :, h_start:h_stop, w_start:w_stop] += \
                    mask * upstream_grad_window

    # Return the pooling input gradient
    return pool_in_grad