"""
This module handles tensor reshaping.
"""
from typing import Optional

from torch import Tensor, Size


def flatten(x) -> Tensor:
    """
    Flatten the input tensor.

    Args:
        x (Tensor): The input tensor

    Return:
        A flattened tensor
    """
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)


def unflatten(x: Tensor, unflatten_shape: Size) -> Tensor:
    """
    Unflatten the input Tensor.

    Args:
        x (Tensor): The input tensor
        unflatten_shape (Size): The target unflattened shape

    Return:
        An unflattened tensor
    """
    return x.reshape(unflatten_shape)


def split_tensor(
    x: Tensor, num_split_heads: int, dim: int=-1,
    transpose_dims: Optional[tuple[int, int]]=None
) -> Optional[Tensor]:
    """
    Return the input tensor split at a dimension.

    Args:
        x (Tensor): The input tensor
        num_split_heads (int): The number of split heads
        dim (int): The dimension to split at

    Return:
        The split tensor
    """
    # Get the size of the splitting dimension
    dim_size = x.shape[dim]

    # Isolate the leading dimensions
    leading_dims = None
    if dim != 0 and -1 * dim < len(x.shape):
        leading_dims = x.shape[:dim]

    # Isolate the trailing dimensions
    trailing_dims = None
    if dim + 1 < len(x.shape) and dim != -1:
        trailing_dims = x.shape[dim+1:]

    # Make sure the split size is divisible by the number of splits
    if dim_size % num_split_heads == 0:

        # Get the new shape
        new_shape = (num_split_heads, dim_size // num_split_heads)
        if leading_dims is not None:
            new_shape = leading_dims + new_shape
        if trailing_dims is not None:
            new_shape += trailing_dims
        
        # Reshape the input tensor (and transpose it, if specified)
        x = x.reshape(new_shape)
        if transpose_dims is not None:
            x = x.transpose(transpose_dims[0], transpose_dims[1])

        # Return the split tensor
        return x

    # Else, return None since the input tensor could not be split
    return None