"""
This module handles tensor regularization.
"""
from typing import Optional

import torch
from torch import Tensor

DROPOUT = .1
REG_TYPE = 'ridge'
REG_STRENGTH = 1.0

def dropout(
    x: Tensor,
    dropout=DROPOUT
) -> Tensor:
    """
    Apply droput to the input tensor.

    Args:
        x (Tensor): The input tensor
        dropout (float): The chance of dropping (zeroing out) an element

    Return:
        The dropout output tensor
    """
    # Get Bernoulli probability mask using the dropout value
    keep_prob = 1 - dropout
    mask = (torch.rand_like(x) < keep_prob).float()

    # Apply the dropout mask with dropout inversion to match expected values
    #   with the original input values, and return the result
    return x * mask / keep_prob


def dropout_backward(
    upstream_grad: Tensor,
    dropout_mask: Tensor,
    dropout=DROPOUT
) -> Tensor:
    """
    Apply dropout backward to the upstream gradient using the original dropout mask

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        dropout (float): The chance of dropping (zeroing out) an element
        dropout_mask (Tensor): The original dropout mask

    Return:
        The dropout gradient
    """
    keep_prob = 1 - dropout
    return upstream_grad * dropout_mask / keep_prob


def ridge_regression(
    x: Tensor,
    reg_strength=REG_STRENGTH,
    weights: Optional[list[Tensor]]=None,
) -> Tensor:
    """
    Perform ridge regression on the input tensor.

    Args:
        x (Tensor): The input tensor
        reg_strength (float): The regularization strength
        weights (list[Tensor]): List of weight tensors

    Return:
        The input tensor with ridge regression applied
    """
    # Get the squared L2 norm
    squared_l2_norm = 0
    
    # Check if weights were provided
    if weights is not None:
        for weight in weights:
            squared_l2_norm += (weight**2).sum()

    # Return the input tensor with ridge regression applied
    return x + reg_strength * squared_l2_norm