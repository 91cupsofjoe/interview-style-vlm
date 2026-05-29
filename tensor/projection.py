"""
This module handles tensor projection.
"""
from typing import Optional

from torch import Tensor

from tensor.activation import relu


def lin_proj(x: Tensor, W: Tensor, b: Optional[Tensor]=None) -> Tensor:
    """
    Perform linear projection on the input tensor.

    Args:
        x (Tensor): The input tensor
        W (Tensor): The projection weight matrix
        b (Tensor): The projection bias vector

    Return:
        A linearly projected tensor
    """
    projection = x @ W.T
    if b is not None:
        projection = x @ W.T + b
    return projection


def lin_proj_backward(
    x: Tensor,
    h: Tensor, W: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Apply linear projection backward on the input tensor

    Args:
        x (Tensor): The input tensor

    Return:
        The projection gradient
    """
    # Get linear projection backward parameters
    d_W = h.T @ x
    d_b = x.sum(dim=0) # Get sum of embedding values for each feature
    d_zproj = x @ W.T
    
    return d_W, d_b, d_zproj


def feed_forward(
    x: Tensor,
    W_1: Tensor, W_2: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Perform feed forwarding on the input tensor.

    Args:
        feed_forward_size (int): The size of the feed forward dimension

    Return:
        x_proj (Tensor): The result of initially projecting the input tensor

    """
    # Project the input tensor
    x_proj = lin_proj(x=x, W=W_1)

    # Apply ReLU activation to the projected input tensor
    x_relu = relu(x_proj)

    # Project the ReLU activated and projected input tensor
    x_final_proj = lin_proj(x=x_relu, W=W_2)

    # Return both the initial input projection and the ReLU activated projection
    return x_proj, x_final_proj