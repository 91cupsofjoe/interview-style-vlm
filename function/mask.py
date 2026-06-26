"""
This module handles tensor masking.
"""
import torch
from torch import Tensor


def get_causal_mask(grid_length: int) -> Tensor:
    """
    Return a causal mask, where True values along grid columns incrementally grow
        along grid rows.

    Args:
        grid_length (int): The grid length

    Return:
        A causal mask tensor
    """
    future_mask = torch.triu(
        torch.ones(grid_length, grid_length), diagonal=1).bool()
    return (~future_mask).reshape(1, 1, *future_mask.shape)


def get_padding_mask(x: Tensor, pad_value: int) -> Tensor:
    """
    Return a padding mask, which masks pad values from the input tensor.

    Args:
        x (Tensor): The input tensor
        pad_value (int): The value to mask

    Return:
        The padding mask
    """
    return x != pad_value