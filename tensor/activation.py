"""
This module handles tensor activation.
"""
import torch
from torch import Tensor


def relu(x: Tensor) -> Tensor:
    """
    Apply ReLU activation on the input tensor.

    Args:
        x (tensor): The input tensor
    
    Return:
        A ReLU activated tensor
    """
    return torch.clamp(x, min=0)


def relu_backward(x: Tensor, mask: Tensor) -> Tensor:
    """
    Perform gating/masking (hadamard product) on the input tensor

    Args:
        x (Tensor): The input tensor
        mask (Tensor): The gate/mask to apply

    Return:
        The gated/masked tensor
    """
    return x * (mask > 0)


def softmax(x: Tensor, dim: int) -> Tensor:
    """
    Perform sigmoid activation on input tensor along the specified dimension

    Args:
        x (Tensor): The input tensor
        dim (int): The dimension along which to apply softmax activation

    Return:
        A tensor of probabilities
    """

    # Subtract each element along the dimension by the dimension max value
    #   to work with smaller base tensor values
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def sigmoid(x: Tensor) -> Tensor:
    """
    Perform sigmoid activation across a tensor along the specified dimension
        and return the result

    Args:
        x (Tensor): Tensor of base values
        dim (int): Dimension along which to apply sigmoid activations

    Return:
        Tensor of sigmoid values
    """

    # Use conditional statement to handle "large" negative values
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),
        torch.exp(x) / (1 + torch.exp(-x))
    )