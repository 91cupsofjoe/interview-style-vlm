"""
This module handles tensor normalization.
"""
from typing import Optional

from torch import Tensor

def layer_normalization(
    x: Tensor,
    gamma: float, beta: Optional[Tensor]=None
) -> Tensor:
    """
    Perform layer normalization on the input tensor.

    Args:
        x (Tensor): The input tensor
        gamma (float): The normalization scaling factor
        beta (Tensor): The normalization offset vector

    Return:
        The layer normalized tensor
    """
    # Get the mean over the last dimension (embedding size)
    #   Keep the embedding dimension (with embedding size = 1)
    mean = x.mean(dim=-1, keepdim=True)
    # Similarly get the standard deviation without using the unbiased estimator
    std = x.var(dim=-1, keepdim=True, unbiased=False)

    # Update bias if used
    bias = 0
    if beta is not None:
        bias = beta

    # Apply layer normalization to the input tensor and return the result
    return gamma * (x - mean) / std + bias