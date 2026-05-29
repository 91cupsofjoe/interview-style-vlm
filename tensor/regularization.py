"""
This module handles tensor regularization.
"""
import torch
from torch import Tensor


def dropout(
    x: Tensor, p: float
) -> Tensor:
    """
    Apply dropout on the input tensor using Bernoulli distribution.

    Args:
        x (Tensor): The input tensor
        p (float): The probability of zeroing a tensor element

    Return:
        Tensor with dropout applied
    """
    # Create a randomized dropout mask
    mask = (torch.rand_like(x) > p)

    # Return the dropout masked tensor
    return x * mask