"""
This module handles tensor attention.
"""
from typing import Optional
import math

from torch import Tensor

from function.activation import softmax, softmax_backward
from function.mask import get_causal_mask, get_padding_mask
from function.projection import lin_proj, lin_proj_backward
from function.reshape import split_tensor


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, attn_mask: Optional[Tensor]=None
) -> tuple[Tensor, Tensor]:
    """
    Return the context vector and attention weights from the scaled dot
        product between the queries and keys matrices.

    Args:
        Q (Tensor): The queries matrix tensor
        K (Tensor): The keys matrix tensor
        V (Tensor): The values matrix tensor
        attn_mask (Tensor): The attention mask tensor

    Return:
        context_vector (Tensor): The context vector tensor
        attention_weights (Tensor): The attention weights tensor
    """
    # NOTE: The shape of Q = [batch size, query length, queries embedding size],
    #   the shape of K = [batch size, key length, keys embedding size],
    #   and the shape of V = [batch size, key length, values embedding size]
    # Get embedding_size
    embedding_size = Q.shape[-1]

    # Get attention scores by performing matrix multiplication between Q and K
    # NOTE: The keys and queries embedding sizes are usually the same,
    #   and the shape of attention scores = [batch size, query length, key length]
    attention_scores = Q @ K.transpose(-2, -1)

    # Scale the scores by the queries embedding size
    embedding_size = Q.shape[-1]
    attention_scores /= embedding_size

    # Apply the mask, if applicable
    if attn_mask is not None:
        # Since softmax is applied to the masked attention scores, set zero
        #   elements to negative infinity (e^-inf = 0)
        attention_scores.masked_fill(attn_mask == 0, float('-inf'))

    # Get the attention weights vector from performing softmax activation on
    #   the attention scores -- use the embeddings dimension
    attention_weights = softmax(attention_scores, dim=-1)

    # Get the context vector by multiplying the attention weights with the
    #   values matrix
    # NOTE: The shape of attention_weights = [batch size, query length, key length]
    context_vector = attention_weights @ V

    # Return the context vector and attention weights
    # NOTE: The shape of the context vector =
    #   [batch size, query length, values embedding size]
    return context_vector, attention_weights


def scaled_dot_product_attention_backward(
    upstream_grad: Tensor,
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    attention_weights: Tensor,
    attn_mask: Optional[Tensor]=None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Return the gradients for the queries, keys, and values.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        Q (Tensor): The queries tensor
        K (Tensor): The keys tensor
        V (Tensor): The values tensor
        attention_weights (Tensor): The attention weights tensor
        attn_mask (Tensor): The mask tensor

    Return:
        Q_grad (Tensor): The queries gradient tensor
        K_grad (Tensor): The keys gradient tensor
        V_grad (Tensor): The values gradient tensor
    """
    # Transpose the values matrix
    V_T = V.transpose(-2, -1)

    # Get the attention weights gradient
    attention_weights_grad = upstream_grad @ V_T

    # Get the scaled scores gradient
    scaled_scores_grad = softmax_backward(
        upstream_grad=attention_weights_grad,
        softmax_out=attention_weights,
        dim=-1)
    
    # Apply the attention mask (if provided) to the scaled scores gradient
    if attn_mask is not None:
        scaled_scores_grad = scaled_scores_grad.masked_fill(attn_mask == 0, 0)

    # Get the queries gradient
    # First get the keys embedding size
    keys_embedding_size = Q.shape[-1]
    Q_grad = scaled_scores_grad @ K / math.sqrt(keys_embedding_size)

    # Get the keys gradient
    K_grad = scaled_scores_grad.transpose(-2, -1) @ \
                    Q / math.sqrt(keys_embedding_size)
    
    # Get the values gradient
    V_grad = attention_weights.transpose(-2, -1) @ upstream_grad
    
    # Return the queries, keys, and values gradients
    return Q_grad, K_grad, V_grad

NUM_ATTN_HEADS = 4

def multi_head_attention(
    Q: Tensor, K: Tensor, V: Tensor,
    W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor,
    num_attn_heads=NUM_ATTN_HEADS,
    b_q: Optional[Tensor]=None, b_k: Optional[Tensor]=None,
    b_v: Optional[Tensor]=None, b_o: Optional[Tensor]=None,
    attn_mask: Optional[Tensor]=None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Return the context vector and attention weights by performing multi-head
        attention calculations.

    Args:
        Q (Tensor): The queries tensor
        K (Tensor): The keys tensor
        V (Tensor): The value tensor
        W_q (Tensor): The queries projection weight tensor
        W_k (Tensor): The keys projection weight tensor
        W_v (Tensor): The values projection weight tensor
        W_o (Tensor): The output projection weight tensor
        num_attn_heads (int): The number of attention heads
        b_q (Tensor): The queries projection bias tensor
        b_k (Tensor): The keys projection bias tensor
        b_v (Tensor): The values projection bias tensor
        b_o (Tensor): The output projection bias tensor
        attn_mask (Tensor): The attention mask tensor

    Return:
        context_vector (Tensor): The context vector tensor
        attention_weights (Tensor): The attention weights tensor
        Q_proj (Tensor): The projected queries tensor
        K_proj (Tensor): The projected keys tensor
        V_proj (Tensor): The projected values tensor
        O_proj (Tensor): The projected context vector tensor
    """
    # First project the queries, keys, and values matrices
    Q_proj = lin_proj(x=Q, W=W_q, b=b_q)
    K_proj = lin_proj(x=K, W=W_k, b=b_k)
    V_proj = lin_proj(x=V, W=W_v, b=b_v)

    # Then split the projected matrices
    Q_split = split_tensor(x=Q_proj, num_split_heads=num_attn_heads)
    K_split = split_tensor(x=K_proj, num_split_heads=num_attn_heads)
    V_split = split_tensor(x=V_proj, num_split_heads=num_attn_heads)
    assert(Q_split is not None)
    assert(K_split is not None)
    assert(V_split is not None)

    # Get context vector by performing the scaled dot product on the split
    #   projected matrices
    context_vector, attention_weights = scaled_dot_product_attention(
        Q=Q_split, K=K_split, V=V_split, attn_mask=attn_mask
    )
    # 'Unsplit' the context vector
    context_vector = context_vector.transpose(1, 2).reshape
    assert(isinstance(context_vector, Tensor))
    # Isolate and collapse the last two dimensions
    first_dims = context_vector.shape[:-2]
    context_vector = context_vector.reshape(*first_dims, -1)

    # Project the context vector
    context_vector = lin_proj(x=context_vector, W=W_o, b=b_o)

    # Return the projected matrices and context vector, along with the
    #   attention weights and attention mask
    return context_vector, attention_weights, Q_proj, K_proj, V_proj, attn_mask


def multi_head_attention_backward(
    upstream_grad: Tensor,
    Q: Tensor, K: Tensor, V: Tensor,
    Q_proj: Tensor, K_proj: Tensor, V_proj: Tensor,
    W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor,
    context_vector: Tensor,
    attention_weights: Tensor,
    num_attn_heads=NUM_ATTN_HEADS,
    b_q: Optional[Tensor]=None,
    b_k: Optional[Tensor]=None,
    b_v: Optional[Tensor]=None,
    b_o: Optional[Tensor]=None,
    attn_mask: Optional[Tensor]=None
) -> tuple[
    Tensor, Tensor, Optional[Tensor],
    Tensor, Tensor, Optional[Tensor],
    Tensor, Tensor, Optional[Tensor],
    Tensor, Optional[Tensor]
]:
    """
    Perfom multi-head attention backward on the upstream gradient.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        Q (Tensor): The queries tensor
        K (Tensor): The keys tensor
        V (Tensor): The value tensor
        Q_proj (Tensor): The projected queries tensor
        K_proj (Tensor): The projected keys tensor
        V_proj (Tensor): The projected values tensor
        W_q (Tensor): The queries projection weight tensor
        W_k (Tensor): The keys projection weight tensor
        W_v (Tensor): The values projection weight tensor
        W_o (Tensor): The output projection weight tensor
        context_vector (Tensor): The context vector tensor
        attention_weights (Tensor): The attention weights tensor
        num_attn_heads (int): The number of attention heads
        b_q (Tensor): The queries projection bias tensor
        b_k (Tensor): The keys projection bias tensor
        b_v (Tensor): The values projection bias tensor
        b_o (Tensor): The output projection bias tensor
        attn_mask (Tensor): The attention mask tensor

    Return:
        Q_in_grad (Tensor): The queries input gradient tensor
        W_q_grad (Tensor): The queries projection weight gradient tensor
        b_q_grad (Tensor): The queries projection bias gradient tensor
        K_in_grad (Tensor): The keys input gradient tensor
        W_k_grad (Tensor): The keys projection weight gradient tensor
        b_k_grad (Tensor): The keys projection bias gradient tensor
        V_in_grad (Tensor): The values input gradient tensor
        W_v_grad (Tensor): The values projection weight gradient tensor
        b_v_grad (Tensor): The values projection bias gradient tensor
        W_o_grad (Tensor): The final projection weight gradient tensor
        b_o_grad (Tensor): The final projection bias gradient tensor
    """
    # Get the final projection weight, bias, and input gradients
    O_in_grad, W_o_grad, b_o_grad = lin_proj_backward(
        upstream_grad=upstream_grad,
        proj_in=context_vector,
        W=W_o,
        b=b_o
    )

    # Get the context heads (projection input gradient split) gradient
    # First get the dims of the context vector gradient
    batch_size, query_len, embedding_size = O_in_grad.shape

    # Then get the embedding size per attention head
    head_size = embedding_size // num_attn_heads
    assert(head_size * num_attn_heads == embedding_size)

    context_heads_grad = O_in_grad.reshape(
        batch_size,
        query_len,
        num_attn_heads,
        head_size
    ).transpose(1, 2)

    # Get the queries, keys, and values split projection outputs
    Q_split = split_tensor(x=Q_proj, num_split_heads=num_attn_heads)
    K_split = split_tensor(x=K_proj, num_split_heads=num_attn_heads)
    V_split = split_tensor(x=V_proj, num_split_heads=num_attn_heads)
    assert(Q_split is not None)
    assert(K_split is not None)
    assert(V_split is not None)

    # Get the queries, keys, and values head gradients
    Q_heads_grad, K_heads_grad, V_heads_grad = scaled_dot_product_attention_backward(
        upstream_grad=context_heads_grad,
        Q=Q_split,
        K=K_split,
        V=V_split,
        attention_weights=attention_weights,
        attn_mask=attn_mask
    )

    # Reshape the head gradients into their projection shapes
    Q_proj_grad = Q_heads_grad.transpose(1, 2).reshape(
            batch_size, query_len, embedding_size
        )
    # Get the key length for keys and values
    _, key_len, _ = K.shape
    K_proj_grad = K_heads_grad.transpose(1, 2).reshape(
            batch_size, key_len, embedding_size
        )
    V_proj_grad = V_heads_grad.transpose(1, 2).reshape(
            batch_size, key_len, embedding_size
        )
    
    # Get the queries projection weight, bias, and input gradients
    Q_in_grad, W_q_grad, b_q_grad = lin_proj_backward(
        upstream_grad=Q_proj_grad,
        proj_in=Q,
        W=W_q,
        b=b_q
    )

    # Get the keys projection weight, bias, and input gradients
    K_in_grad, W_k_grad, b_k_grad = lin_proj_backward(
        upstream_grad=K_proj_grad,
        proj_in=K,
        W=W_k,
        b=b_k
    )

    # Get the values projection weight, bias, and input gradients
    V_in_grad, W_v_grad, b_v_grad = lin_proj_backward(
        upstream_grad=V_proj_grad,
        proj_in=V,
        W=W_v,
        b=b_v
    )

    # Return the weight, bias, and input gradients for the keys, queries, and values
    return Q_in_grad, W_q_grad, b_q_grad, \
            K_in_grad, W_k_grad, b_k_grad, \
            V_in_grad, W_v_grad, b_v_grad, \
            W_o_grad, b_o_grad


def multi_head_masked_self_attention(
    Q: Tensor, K: Tensor, V: Tensor,
    W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor,
    num_attn_heads=NUM_ATTN_HEADS,
    b_q: Optional[Tensor]=None, b_k: Optional[Tensor]=None,
    b_v: Optional[Tensor]=None, b_o: Optional[Tensor]=None,
    causal_mask: Optional[Tensor]=None,
    sequence_length: Optional[int]=None,
    padding_mask: Optional[Tensor]=None,
    pad_value: Optional[int]=None
) -> Optional[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]]:
    """
    Perform multi-head masked self attention on the input.
    NOTE: See multi_head_attention() for the full implementation details.

    Additional Args:
        causal_mask (Tensor): The causal mask tensor
        sequence_length (int): The sequence length
        padding_mask (Tensor): The padding mask tensor
        pad_value (int): The value to mask
    """
    # Check if a causal mask was not provided
    if causal_mask is None:
        # If the sequence length was provided, use it for the causal mask
        if sequence_length is not None:
            causal_mask = get_causal_mask(sequence_length)

    # Check if a padding mask was not provided
    if padding_mask is None:
        # If the padding value was provided, use it for the padding mask
        if pad_value is not None:
            padding_mask = get_padding_mask(Q, pad_value)

    # Only run multi-head attention if both the causal and padding masks exist
    if causal_mask is not None and padding_mask is not None:
        # Copy the local parameters
        kwargs=locals().copy()
        # Adjust the causal and padding masks keys and remove sequence
        #   length and pad value
        kwargs['attn_mask'] = kwargs.pop('causal_mask') | kwargs.pop('padding_mask')
        kwargs.pop('sequence_length')
        kwargs.pop('pad_value')

        # Return multi head attention
        return multi_head_attention(**kwargs)
    
    # Else return None


def multi_head_cross_attention(
    Q: Tensor, K: Tensor, V: Tensor,
    W_q: Tensor, W_k: Tensor, W_v: Tensor, W_o: Tensor,
    num_attn_heads=NUM_ATTN_HEADS,
    b_q: Optional[Tensor]=None, b_k: Optional[Tensor]=None,
    b_v: Optional[Tensor]=None, b_o: Optional[Tensor]=None,
    padding_mask: Optional[Tensor]=None,
    pad_value: Optional[int]=None,
) -> Optional[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]]:
    """
    Perform multi-head cross attention on the input.
    NOTE: See multi_head_attention() for the full implementation details.

    Additional Args:
        padding_mask (Tensor): The padding mask tensor
        pad_value (int): The value to mask
    """
    # Check if a padding mask was not provided
    if padding_mask is None:
        # If the pad value was provided, use it for the padding mask
        if pad_value is not None:
            padding_mask = get_padding_mask(Q, pad_value)

    # Only run multi-head attention if the padding mask exists
    if padding_mask is not None:
        # Copy the local parameters
        kwargs=locals().copy()
        # Adjust the padding mask key and remove sequence length
        kwargs['attn_mask'] = kwargs.pop('padding_mask')
        kwargs.pop('pad_value')

        # Return multi head attention
        return multi_head_attention(**kwargs)
    
    # Else return None