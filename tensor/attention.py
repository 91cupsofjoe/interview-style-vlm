"""
This module handles tensor attention.
"""
from typing import Optional

import torch
from torch import Tensor

from tensor.reshape import split_tensor
from tensor.projection import lin_proj
from tensor.activation import softmax


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor]=None
) -> tuple[Tensor, Tensor]:
    """
    Return the context vector and attention weights from the scaled dot
        product between the queries and keys matrices.

    Args:
        Q (Tensor): The queries matrix tensor
        K (Tensor): The keys matrix tensor
        V (Tensor): The values matrix tensor
        mask (Tensor): The mask tensor

    Return:
        context_vector (Tensor): The context vector tensor
        attention_weights (Tensor): The attention weights tensor
    """
    # The shape of Q, K, and V =
    #   [batch size, number of attention heads, sequence length, embedding size]
    # Get embedding_size
    embedding_size = Q.shape[-1]

    # Get attention scores by performing matrix multiplication between Q and K
    attention_scores = Q @ K.transpose(-2, -1)

    # Scale the scores by the embedding size
    # The shape of Q, K, and V =
    #   [batch size, number of attention heads, sequence length, embedding size]
    embedding_size = Q.shape[-1]
    attention_scores /= embedding_size

    # Apply the mask, if applicable
    if mask is not None:
        attention_scores *= mask

    # Get the attention weights vector from performing softmax activation on
    #   the attention scores -- use the embeddings dimension
    attention_weights = softmax(attention_scores, dim=3)

    # Get the context vector by multiplying the attention weights with the
    #   values matrix
    context_vector = attention_weights @ V

    # Return the context vector and attention weights
    return context_vector, attention_weights
    

def multi_head_attention(
    Q: Tensor, K: Tensor, V: Tensor,
    W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor,
    num_heads: int, mask: Optional[Tensor]=None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Return the context vector and attention weights by performing multi-head
        attention calculations.

    Args:
        Q (tensor): The queries matrix tensor
        K (tensor): The keys matrix tensor
        V (tensor): The value matrix tensor
        W_q (tensor): The queries projection weight
        W_k (tensor): The keys projection weight
        W_v (tensor): The values projection weight
        W_o (tensor): The output projection weight
        num_heads (int): The number of attention heads
        mask (Tensor): The mask tensor

    Return:
        Q_proj (Tensor): The projected queries matrix
        K_proj (Tensor): The projected keys matrix
        V_proj (Tensor): The projected values matrix
        O_proj (Tensor): The projected context vector
    """
    # First project the queries, keys, and values matrices
    Q_proj = lin_proj(x=Q, W=W_q)
    K_proj = lin_proj(x=K, W=W_k)
    V_proj = lin_proj(x=V, W=W_v)

    # Then split the matrices
    Q_split = split_tensor(x=Q, num_split_heads=num_heads)
    K_split = split_tensor(x=K, num_split_heads=num_heads)
    V_split = split_tensor(x=V, num_split_heads=num_heads)

    # Get context vector by performing the scaled dot product on the split
    #   matrices
    if Q_split is not None \
            and K_split is not None \
            and V_split is not None:
        context_vector, attention_weights = scaled_dot_product_attention(
            Q=Q_split, K=K_split, V=V_split, mask=mask
        )
        # 'Unsplit' the context vector
        context_vector = context_vector.transpose(-3, -2).reshape
        assert(isinstance(context_vector, Tensor))
        # Isolate and collapse the last two dimensions
        first_dims = context_vector.shape[:-2]
        context_vector = context_vector.reshape(*first_dims, -1)

    # Get the output projection of the context vector
    O_proj = lin_proj(x=context_vector, W=W_o)

    # Return the projected matrices and projected context vector
    return Q_proj, K_proj, V_proj, O_proj