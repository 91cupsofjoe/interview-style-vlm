"""
This module handles tensor pooling.
"""
import torch
from torch import Tensor
import torch.nn.functional as nnf


def pool(x: Tensor, kernel_size: tuple,
        stride: int, pool_type: str):
    """
    Pool the input tensor.

    Args:
        x (Tensor): The input tensor
        kernel_size (tuple): The kernel/filter shape
        stride (int): The window increment amount
        pool_type (str): The type of pooling operation to perform

    Return:
        A pooled tensor
    """
    if pool_type == "average":
        return nnf.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
    # Else, return max pooling -- the default
    return nnf.max_pool2d(x, kernel_size=kernel_size, stride=stride)

def unpool(
    x: Tensor, a_relu: Tensor,
    pool_size: tuple, unpool_type: str, stride: int
) -> Tensor:
    """
    Apply unpooling on the input tensor.

    Args:
        x (Tensor): The input tensor
        a_relu (Tensor): The ReLU activated tensor
        pool_size (tuple): The pool window dimensions
        pool_type (str): The type of pooling operation
        stride (int): The window increment amount

    Return:
        A ReLU activated gradient tensor
    """
    # The derivative of the loss wrt to the ReLU activated 
    d_relu = torch.zeros_like(a_relu)

    # The 3rd and 4th dimensions of the ReLU activated convolution output are
    #   the height and width of the prepooled Tensor
    h_dim, w_dim = (2, 3)
    # Get pool window height and width
    pool_h, pool_w = (pool_size)

    # Iterate over each position in the pooled spatial grid
    for i in range(x.shape[h_dim]):
        for j in range(x.shape[w_dim]):
            h_start = i * stride
            h_stop = h_start + pool_size[0]
            w_start = j * stride
            w_stop = w_start + pool_size[1]

            # Get the pool window
            window = a_relu[:, :, h_start:h_stop, w_start:w_stop]
            # Get the upstream gradient
            upstream_grad = x[:, :, i:i+1, j:j+1] 
            
            if unpool_type == 'average':
                # Perform uniform distribution across the pool window
                avg_value = upstream_grad / (pool_h * pool_w)
                d_relu[:, :, h_start:h_stop, w_start:w_stop] += avg_value

            else: # unpool_type = 'max'
                # Perform routing of the max value across the pool window
                max_values = window.amax(dim=(h_dim, w_dim), keepdim=True)
                mask = (window == max_values)

                d_relu[:, :, h_start:h_stop, w_start:w_stop] += \
                mask * upstream_grad

    # Return the derivative of the loss wrt the ReLU activated convolution output
    return d_relu