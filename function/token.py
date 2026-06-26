"""
This module handles tensor tokenization.
"""
from typing import Optional, Any
from collections.abc import Callable
import math

import torch
from torch import Tensor, Size


PUNCTUATION_SEPARATORS = [
    '.', '?', ',', "'"
]

def get_token_ids(
    sentence: Optional[str]=None,
    sentences: Optional[list[str]]=None,
    sentence_lists: Optional[list[list[str]]]=None,
    token_ids: Optional[dict[str, int]]=None
) -> dict[str, int]:
    """
    Return a dict of token ids from the list or lists of sentences.

    Args:
        sentence (str): A sentence with tokens
        sentences (list[str]): A list of sentences
        sentence_lists (list[list[str]]): A list of sentence lists
        token_ids (dict[str, int]): Dict of token ids

    Return:
        Dict of token ids
    """
    # Get the sentence lists
    sentence_lists = get_sentence_lists(
        sentence=sentence,
        sentences=sentences,
        sentence_lists=sentence_lists
    )

    # Get set of tokens
    tokens = set()

    # Iterate through the sentence lists
    for sentence_list in sentence_lists:
        # Iterate through the sentences in the sentence list
        for sentence in sentence_list:
            # Convert the sentence into a tuple of tokens
            tokens_tuple = parse_sentence_into_tuple(
                sentence=sentence,
                punctuation_separators=PUNCTUATION_SEPARATORS
            )
            # Iterate through the tokens in the tokens tuple
            for token in tokens_tuple:
                # Append unique tokens to the tokens set
                tokens.add(token)

    # Initialize the token ids dict if not provided
    if token_ids is None:
        token_ids = {}

    # Iterate through the tokens in the token set to assign it an id
    num_id = len(token_ids)
    for token in tokens:
        # Add the token in the tokens set to the token ids dict
        token_ids[token] = num_id
        num_id += 1

    # Return the token ids dict
    return token_ids


def get_sentence_lists(
    sentence: Optional[str]=None,
    sentences: Optional[list[str]]=None,
    sentence_lists: Optional[list[list[str]]]=None
) -> list[list[str]]:
    """
    Return a list of sentence lists from a sentence or list of sentences.

    Args:
        sentence (str): A sentence with tokens
        sentences (list[str]): A list of sentences
        sentence_lists (list[list[str]]): A list of sentence lists

    Return:
        List of sentence lists
    """
    # Check if a sentence was provided
    if sentence is not None:
        # Convert the sentence to a list of sentences
        sentences = [sentence]

    # Check if sentences were provided
    if sentences is not None:
        # Convert the list of sentences into a list of sentence lists
        sentence_lists = [sentences]

    # Initialize the sentence lists if not provided
    if sentence_lists is None:
        sentence_lists = [[]]

    return sentence_lists


def parse_sentence_into_tuple(
    sentence: str,
    punctuation_separators: list[str]
) -> tuple[str, ...]:
    """
    Parse a sentence by the specified punctuation separators.

    Args:
        token_tuple (tuple[str]): The tuple of tokens
        separators (list[str]): The separators to look for

    Return:
        The new token tuple
    """
    # Convert the sentence into a tuple of tokens
    tokens_tuple = tuple(sentence.split(' '))
    # Initialized the parsed token tuple
    parsed_tokens_tuple = tuple()

    # Iterate through the tokens in the token tuple
    for token in tokens_tuple:
        # Treat the token as a split token tuple
        split_token_tuple = (token,)
        # Iterate through the separators
        for separator in punctuation_separators:
            # Iterate through the split tokens
            split_token_index = 0
            while split_token_index < len(split_token_tuple):
                split_token = split_token_tuple[split_token_index]

                # Check if the separator exists in the split token
                if separator in split_token and separator != split_token:
                    # Use the separator to split the split token tuple into parts
                    split_index = split_token.index(separator)
                    split_token_parts = (
                        split_token_tuple[:split_index] + (separator,)
                    )
                    if split_index < len(split_token):
                        split_token_parts += (split_token_tuple[split_index+1:])
                    # Add the split token tuple prefix if applicable
                    if split_token_index > 0:
                        split_token_parts = split_token_tuple[:split_token_index] \
                                        + split_token_parts
                    # Add the split token tuple suffix if applicable
                    if split_token_index < len(split_token_tuple):
                        split_token_parts += split_token_tuple[split_token_index+1:]
                    # Update the split token tuple
                    split_token_tuple = split_token_parts

                else:
                    # The separator wasn't found, so increment the split token
                    #   index by one
                    split_token_index += 1

        # Add the split token tuple to the parsed token tuple
        parsed_tokens_tuple += split_token_tuple

    # Return the parsed token tuple
    return parsed_tokens_tuple


def get_tokens_tensor(
    #corpus_text: Optional[str]=None,
    sentence: Optional[str]=None,
    sentences: Optional[list[str]]=None,
    sentence_lists: Optional[list[list[str]]]=None,
    token_ids: Optional[dict[str, int]]=None
) -> Optional[Tensor]:
    """
    Convert a list or lists of sentences into a token tensor and return it.

    Args:
        sentence (str): A sentence with tokens
        sentences (list[str]): A list of sentences
        sentence_lists (list[list[str]]): A list of sentence lists
        token_ids (dict[str, int]): Dict of token ids

    Return:
        The tokens tensor
    """
    # Get the token ids dict from converting the sentence, sentences list, or
    #   sentence lists into tokens
    # NOTE: Passing in only the token ids dict will return it without modification
    token_ids = get_token_ids(
        sentence=sentence,
        sentences=sentences,
        sentence_lists=sentence_lists,
        token_ids=token_ids
    )

    # Get the sentence lists
    sentence_lists = get_sentence_lists(
        sentence=sentence,
        sentences=sentences,
        sentence_lists=sentence_lists
    )

    # Initialize the list of sentence list tensors
    sentence_list_tensors = []

    # Use the token ids to convert each list of sentences into a tokens tensor
    # Iterate through the sentence lists
    for sentence_list in sentence_lists:
        # Initialize the list of sentence vectors
        sentence_tensors = []
        # Iterate through the sentences in the sentence list
        for sentence in sentence_list:
            # Parse the sentence into a tuple of tokens
            tokens_tuple = parse_sentence_into_tuple(
                sentence=sentence,
                punctuation_separators=PUNCTUATION_SEPARATORS
            )
            # Add the tokens tuple as a sentence vector
            token_ids_tuple = [
                token_ids[token] for token in tokens_tuple
            ]
            sentence_tensors.append(
                torch.tensor(token_ids_tuple)
            )

        # Stack the sentence vectors into a single tensor
        sentence_list_tensor = torch.stack(tuple(sentence_tensors))
        # Add the sentence tensor to the sentence_list_tensors
        sentence_list_tensors.append(sentence_list_tensor)

    # Stack the sentence list tensors into a single tensor and return it
    return torch.stack(tuple(sentence_list_tensors))


def get_vocab_size(
    token_ids: dict[str, Any],
    omit_tokens: Optional[tuple[str]]=None
) -> int:
    """
    Return the number of unique tokens ids.

    Args:
        token_ids (dict[str, Any]): The token ids dict
        omit_tokens (tuple[str]): The tuple of tokens to omit from the count

    Return:
        Int of unique token ids
    """
    # Initialize omit tokens tuple if not provided
    if omit_tokens is None:
        omit_tokens = ("",)

    return len(
        [k for k in token_ids.keys()
            if k not in omit_tokens]
    )


def get_token_embeddings(
    token_ids: dict[str, int],
    embedding_size: int
) -> Tensor:
    """
    Get a token embeddings tensor.

    Args:
        token_ids (dict[str, Any]): The token ids dict

    Return:
        The token embeddings tensor
    """
    # Get the vocab size of the tokens tensor
    vocab_size = get_vocab_size(token_ids)

    # Create and return a random tensor using the vocab and embeddings sizes
    return torch.rand((vocab_size, embedding_size))


def get_sinusoidal_encodings(height: int, width: int) -> Tensor:
    """
    Get a sinusoidal encodings tensor.

    Args:
        height (Tensor): The height of the tensor
        width (Tensor): The width of the tensor

    Return:
        sinusoidal_encodings (Tensor): The sinusoial encodings tensor
    """
    # Positional encoding formula (pos along height and dim along width):
    #   PE(pos, 2 * dim) = sin( pos / 10000^(2*dim/width) )
    #   PE(pos, 2 * dim + 1) = cos( pos / 10000^(2*dim/width) )
    # NOTE: Divide dim by 2 since using 2*dim and 2*dim+1 for indices
    positions = torch.arange(height).unsqueeze(1)
    dimensions = 10000 ** (2 * (torch.arange(width).unsqueeze(0) // 2) / width)

    # Get the sinusoidal encodings
    sinusoidal_encodings = positions / dimensions
    sinusoidal_encodings[:, 0::2] = math.sin(sinusoidal_encodings[:, 0::2])
    sinusoidal_encodings[:, 0::2] = math.cos(sinusoidal_encodings[:, 1::2])

    # Return the sinusoidal encodings
    return sinusoidal_encodings


def get_positional_encodings(
    shape: Size,
    positional_encoding_type: Optional[str]=None
) -> Tensor:
    """
    Get a positional encodings tensor.
    NOTE: For now, only sinusoidal encoding is available.

    Args:
        shape (Size): The tensor shape
        positional_encoding_type (str): The type of positional encoding

    Return:
        The position encodings tensor
    """
    # if positional_encoding_type = 'sinusoidal':
    return get_sinusoidal_encodings(shape[-2], shape[-1])


def get_embedded_tokens(
    tokens: Tensor,
    token_embeddings: Optional[Tensor]=None,
    token_ids: Optional[dict[str, Any]]=None,
    embedding_size: Optional[int]=None,
    positional_encodings: Optional[Tensor]=None,
    positional_encoding_type: Optional[str]=None,
    use_positional_encodings=False
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    Embed the input tensor with token embeddings and positional encodings (optional).

    Args:
        tokens (Tensor): The input tokens tensor
        token_embeddings (Tensor): The token embeddings tensor
        token_ids (dict[str, Any]): The token ids dict
        positional_encodings (Tensor): The positional embeddings tensor
        positional_encoding_type (str): The type of positional encoding
        use_positional_encodings (bool): The boolean indicating if using
            positional encodings

    Return:
        The embedded tokens tensor
    """
    # Get the token embeddings if not provided
    if token_embeddings is None:
        # Make sure token ids and embedding size were provided
        if token_ids is not None and embedding_size is not None:
            token_embeddings = get_token_embeddings(token_ids, embedding_size)

    # If using positional encodings, get the positional encodings if not provided
    if use_positional_encodings and positional_encodings is None:
        positional_encodings = get_positional_encodings(
            shape=tokens.shape,
            positional_encoding_type=positional_encoding_type
        )

    # Apply the token embeddings if they exist
    if token_embeddings is not None:
        tokens = tokens @ token_embeddings
        # Scale the embedded tokens tensor using the embedding size
        if embedding_size is None:
            embedding_size = token_embeddings.shape[-1]
        tokens = tokens * math.sqrt(embedding_size)

    # If using positional encodings, apply the positional encodings if they exist
    if use_positional_encodings and positional_encodings is not None:
        tokens = tokens + positional_encodings

    # Return the embedded tokens tensor, token embeddings, and positional
    #   encodings tensors
    return tokens, token_embeddings, positional_encodings


def get_embedded_tokens_backward(
    upstream_grad: Tensor,
    tokens_in: Tensor,
    token_embeddings: Optional[Tensor]=None,
    are_token_embeddings_learnable=False,
    positional_encodings: Optional[Tensor]=None,
    are_positional_encodings_learnable=False
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    Apply token embedding backward on the upstream gradient.

    Args:
        upstream_grad (Tensor): The upstream gradient tensor
        tokens_in (Tensor): The preembedded tokens tensor
        token_embeddings (Tensor): The token embeddings tensor
        are_token_embeddings_learnable (bool): The boolean indicating if
            token embeddings are learnable
        position_embeddings (Tensor): The position embeddings tensor
        are_positional_encodings_learnable (bool): The boolean indicating if
            positional encodings are learnable

    Return:
        The embedded tokens input gradient tensor
    """
    # Initialize the positional embeddings input gradient
    pos_in_grad = None

    # If positional encodings are used and learnable, update the positional
    #   embeddings input gradient
    if positional_encodings is not None and are_positional_encodings_learnable:
        pos_in_grad = positional_encodings + upstream_grad

    # Initialize the token embeddings input gradient
    embed_in_grad = None

    # If token embeddings are used and learnable, update the token embeddings
    #   input gradient
    if token_embeddings is not None and are_token_embeddings_learnable:
        # Scale the upstream gradient by the embedding size
        upstream_grad = upstream_grad * math.sqrt(token_embeddings.shape[-1])

        # Get the token embeddings input gradient
        embed_in_grad = token_embeddings[tokens_in]
        embed_in_grad = embed_in_grad + upstream_grad

    # Return the upstream, positional encoding, and token embedding gradients
    return upstream_grad, pos_in_grad, embed_in_grad