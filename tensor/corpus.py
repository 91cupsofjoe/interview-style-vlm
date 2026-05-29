"""
This module is for tensor operations related to corpi.
"""
from typing import Union
import torch
from torch import Tensor


def get_tokens_tensor(
    corpus: Union[str, list[str]]
) -> Tensor:
    """
    Convert a body of text (corpus) into a tensor of tokens.

    Args:
        corpus
    """
    # 
    return torch.zeros()