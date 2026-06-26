from typing import Union, Optional, Any
from collections.abc import Callable

import torch
from torch import Tensor, Size

from function import attention as attn, convolution as conv, pool, \
    regularization as reg
from function.tensor_function import get_tensor_function
from model import model as ml
from log import logger as log

# ================================== LAYER ====================================

class Layer:
    """
    This is the base layer class. Each Layer object contains its own attributes,
        learnable parameters, and both forward and backward function sets.
    """
    def __init__(self, 
        layer_parameters: dict[str, Any],
        layer_pass_function_names: list[str],
        layer_update_function_name: Optional[str]=None,
        layer_update_weights_function_name: Optional[str]=None,
        object_name: Optional[str]=None, has_log_id=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.LAYER)

        # Set the layer parameters
        self.parameters = layer_parameters

        # Set the layer pass functions
        self.pass_functions = []
        # Iterate through the layer pass function names
        for function_name in layer_pass_function_names:
            # Get the pass function (a PassFunction object)
            pass_function = get_tensor_function(
                tensor_function_name=function_name,
                tensor_function_cache_parameters=self.parameters,
                tensor_update_function_name=layer_update_function_name,
                tensor_update_weight_function_name=layer_update_weights_function_name
            )
            # Only append the pass function if it exists
            if pass_function is not None:
                self.pass_functions.append(pass_function)

    def forward(self,
        x: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Run the layer's forward function on the input.

        x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (dict[str, Any]): Dict of output values
        """
        # Iterate through the forward functions
        for function in self.pass_functions:
            # Get the output values
            output_values = function.forward(
                x=x,
                kwargs=kwargs,
                output_keys=output_keys
            )
            # Get the input from the output values
            x = output_values[0]

        # Return the output values
        return output_values

    def backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Run the layer's backward function on the upstream gradient

        x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (dict[str, Any]): Dict of output values
        """
        # Iterate through the backward functions
        for function in self.pass_functions:
            # Get the output values
            output_values = function.backward(
                upstream_grad=upstream_grad,
                kwargs=kwargs,
                output_keys=output_keys
            )
            # Get the upstream gradient from the output values
            upstream_grad = output_values[0]

        # Return the output values
        return output_values

    def update(self) -> bool:
        """
        Update all of the layer's learnable parameters.

        Args:
            None

        Return:
            update_success (boolean): Boolean indicating success with updating
        """
        update_success = True

        # Iterate through the pass functions
        for function in self.pass_functions:
            # Store the boolean result of updating the learnable parameters
            # All tensor function updates should be successful, otherwise return False
            if not function.update():
                update_success = False

        # Return the boolean result of updating the learnable parameters
        return update_success


# ==================== TRANSFORMER ENCODER/DECODER BLOCKS =====================

class TransformerBlock(Layer):
    """
    This is the transformer encoder/decoder block (layer) class.
    """
    def __init__(self,
        layer_pass_function_names: list[str],
        layer_update_function_name: Optional[str]=None,
        layer_update_weights_function_name: Optional[str]=None,
        num_in_tokens=ml.NUM_IN_TOKENS, num_out_classes=ml.NUM_OUT_CLASSES,
        embedding_size=ml.TRANSFORMER_EMBEDDING_SIZE,
        feed_fwd_size=ml.FEED_FWD_SIZE,
        max_seq_len=ml.MAX_SEQ_LEN,
        num_attn_heads=attn.NUM_ATTN_HEADS, dropout=reg.DROPOUT,
        object_name: Optional[str]=None, has_log_id=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.TRANSFORMER_BLOCK)

        # Get the layer parameters for the transformer block
        layer_parameters = {
            'num_in_tokens': num_in_tokens,
            'num_out_classes': num_out_classes,
            'max_seq_len': max_seq_len,
            'num_attn_heads': num_attn_heads,
            'embedding_size': embedding_size,
            'feed_fwd_size': feed_fwd_size,
            'dropout': dropout
        }

        super().__init__(
            layer_parameters=layer_parameters,
            layer_pass_function_names=layer_pass_function_names,
            layer_update_function_name=layer_update_function_name,
            layer_update_weights_function_name=layer_update_weights_function_name
        )

    
# ============================ CONVOLUTION LAYER ==============================
    
class ConvolutionLayer(Layer):
    """
    This is the convolution layer class.
    """
    def __init__(self,
        layer_pass_function_names: list[str],
        layer_update_function_name: Optional[str]=None,
        layer_update_weights_function_name: Optional[str]=None,
        num_in_channels=ml.NUM_IN_CHANNELS, num_out_features=ml.NUM_OUT_FEATURES,
        kernel_size=conv.KERNEL_SIZE,
        stride=conv.STRIDE, padding=conv.PADDING,
        pool_type=pool.POOL_TYPE,
        pool_size=pool.KERNEL_SIZE, pool_stride=pool.STRIDE,
        object_name: Optional[str]=None, has_log_id=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.CONVOLUTION_LAYER)

        # Get the layer parameters for the convolution layer
        layer_parameters = {
            'num_in_channels': num_in_channels,
            'num_out_channels': num_out_features,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'pool_size': pool_size,
            'pool_stride': pool_stride,
            'pool_type': pool_type
        }

        super().__init__(
            layer_parameters=layer_parameters,
            layer_pass_function_names=layer_pass_function_names,
            layer_update_function_name=layer_update_function_name,
            layer_update_weights_function_name=layer_update_weights_function_name
        )


# ============================= PROJECTION LAYER ==============================

NUM_PATCHES = 512 # Default number of patches/feature maps
EMBEDDING_SIZE = 756 # Default embedding size for projection

class ProjectionLayer(Layer):
    """
    This is the projection layer class.
    """
    def __init__(self,
        layer_pass_function_names: list[str],
        layer_update_function_name: Optional[str]=None,
        layer_update_weights_function_name: Optional[str]=None,
        pre_embedding_size=ml.TRANSFORMER_EMBEDDING_SIZE,
        embedding_size=ml.TRANSFORMER_FINAL_EMBEDDING_SIZE,
        object_name: Optional[str]=None, has_log_id=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.PROJECTION_LAYER)

        # Get the static parameters for the projection layer
        layer_parameters = {
            'pre_embedding_size': pre_embedding_size,
            'embedding_size': embedding_size
        }

        super().__init__(
            layer_parameters=layer_parameters,
            layer_pass_function_names=layer_pass_function_names,
            layer_update_function_name=layer_update_function_name,
            layer_update_weights_function_name=layer_update_weights_function_name
        )


# =============================== LAYER LOOKUP ================================

layer_subclasses = {
    'convolution_layer' : ConvolutionLayer,
    'projection_layer' : ProjectionLayer,
    'transformer_block' : TransformerBlock
}

def get_layer(
    layer_type: str,
    layer_parameters: dict[str, Any],
    layer_pass_function_names: list[str],
    layer_update_function_name: Optional[str]=None,
    layer_update_weights_function_name: Optional[str]=None
) -> Layer:
    """
    Return a Layer subclass object with the specified layer parameters.

    Args:
        layer_type (str): The type of layer
        layer_parameters (dict[str, Any]): The layer parameters

    Return:
        The Layer subclass object
    """
    # Get the layer subclass with its layer parameters
    layer_subclass = layer_subclasses[layer_type]
    return layer_subclass(
        layer_pass_function_names=layer_pass_function_names,
        layer_update_function_name=layer_update_function_name,
        layer_update_weights_function_name=layer_update_weights_function_name,
        **layer_parameters,
    )