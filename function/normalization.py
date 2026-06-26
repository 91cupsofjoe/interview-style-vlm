"""
This module handles tensor normalization.
"""
from typing import Optional

import torch
from torch import Tensor


def layer_norm(
    x: Tensor,
    eps: float,
    gamma: Tensor,
    beta: Optional[Tensor]=None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Perform layer normalization on the input tensor.

    Args:
        x (Tensor): The input tensor
        eps (float): Small threshold value
        gamma (Tensor): The normalization scaling factor
        beta (Tensor): The normalization offset vector

    Return:
        norm_out (Tensor): The normalized output tensor
        layer_norm_out (Tensor): The layer normalized output tensor
        std (Tensor): The standard deviation
    """
    # Get the mean over the last dimension (embedding size)
    #   Keep the embedding dimension (with embedding size = 1)
    mean = torch.mean(x, dim=-1, keepdim=True)
    # Get the standard deviation
    std = torch.std(x, dim=-1, keepdim=True)

    # Update bias if used
    embedding_size = input.shape[-1]
    bias = torch.zeros(embedding_size)
    if beta is not None:
        bias = beta

    # Get the normalized output tensor
    norm_out = (x * mean) / (std + eps)

    # Scale and offset the normalized output tensor
    layer_norm_out = gamma * norm_out + bias

    # Return the layer normalized input and normalized output tensors, along
    #   with the standard deviation
    return layer_norm_out, norm_out, std


def layer_norm_backward(
    upstream_grad: Tensor,
    norm_out: Tensor,
    gamma: Tensor,
    std: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Perform layer normalization backward on the upstream gradient.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        norm_out (Tensor): The normalized output tensor
        gamma (Tensor): The layer normalization scaling tensor
        std (float): The standard deviation of the original input tensor

    Return:
        layer_norm_grad (Tensor): The gradient tensor for the layer normalized output
        gamma_grad (Tensor): The gradient tensor for gamma
        beta_grad (Tensor): The gradient tensor for beta
    """
    # Get the gradient for gamma
    gamma_grad = torch.sum(upstream_grad * norm_out, dim=(0, 1))

    # Get the gradient for beta
    beta_grad = torch.sum(upstream_grad, dim=(0, 1))

    # Get the normalization output gradient
    norm_out_grad = gamma * norm_out

    # Get the layer normalization input gradient
    layer_norm_in_grad = (norm_out_grad \
        - torch.mean(norm_out_grad, dim=-1, keepdim=True) \
        - norm_out * torch.mean(norm_out_grad * norm_out, dim=-1, keepdim=True)) \
            * (1 / std)

    # Return the gradients for layer normalization, gamma, and beta
    return layer_norm_in_grad, gamma_grad, beta_grad