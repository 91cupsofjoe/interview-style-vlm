"""
This module handles tensor activation.
"""
import torch
from torch import Tensor


def relu(x: Tensor) -> Tensor:
    """
    Apply ReLU activation on the input tensor.

    Args:
        relu_in (tensor): The input tensor
    
    Return:
        A ReLU activated tensor
    """
    return torch.clamp(x, min=0)


def relu_backward(upstream_grad: Tensor, relu_in: Tensor) -> Tensor:
    """
    Perform ReLU backward on the upstream gradient tensor.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        relu_in (Tensor): The relu input tensor

    Return:
        The relu input gradient tensor
    """
    # Use the relu input as a binary mask
    return upstream_grad * (relu_in > 0)


def sigmoid(x: Tensor) -> Tensor:
    """
    Perform sigmoid activation on the input tensor.

    Args:
        x (Tensor): The input tensor

    Return:
        A sigmoid activated tensor
    """
    # Use conditional statement to handle "large" negative values
    return torch.where(
        x >= 0,
        1 / (1 + torch.exp(-x)),
        torch.exp(x) / (1 + torch.exp(-x))
    )


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