from typing import Union, Optional, Any
from collections.abc import Callable

import torch
from torch import Tensor, Size

from tensor_function import \
    attention as attn, convolution as conv, pool, regularization as reg, \
        update, util
from tensor_function.tensor_function import _get_tensor_function

from log import logger as log

# ================================== LAYER ====================================

class Layer:
    """
    This is the base layer class. Each Layer object contains its own attributes,
        learnable parameters, and both forward and backward function sets.
    """
    def __init__(self, 
        layer_hyperparameters: dict[str, Any],
        layer_pass_function_names: list[str],
        layer_update_function_name: Optional[str]=None,
        do_print_layer_attr: Union[bool, dict[str, bool], None]=None,
        do_print_tensor_function_attr: Union[bool, dict[str, bool], None]=None,
        object_name: Optional[str]=None, has_log_id=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.LAYER)

        # Initialize the print attributes dict if not provided and
        #   set the print attributes dict
        if do_print_layer_attr is None:
            self.do_print_attr = {
                'forward': False,
                'backward': False,
                'update': False
            }

        elif isinstance(do_print_layer_attr, bool):
            self.do_print_attr = {
                'forward': do_print_layer_attr,
                'backward': do_print_layer_attr,
                'update': do_print_layer_attr
            }

        else:
            # do_print_layer_attr is a dict
            self.do_print_attr = do_print_layer_attr

        # Initialize the layer name if not provided
        if object_name is None:
            object_name = 'UNNAMED LAYER'

        # Set the layer name
        self.name = object_name

        # Set the layer parameters
        self.hyperparameters = layer_hyperparameters

        # Set the layer pass functions
        self.pass_functions = []

        # Keep track of the tensor functions used
        tensor_functions = {}
        
        # Iterate through the layer pass function names
        for function_name in layer_pass_function_names:
            # Add the function name to the tensor functions
            if function_name not in tensor_functions:
                tensor_functions[function_name] = 0
            tensor_function_count = tensor_functions[function_name]

            # Increment the function name count
            tensor_function_count += 1

            # Get the pass function (a PassFunction object)
            pass_function = _get_tensor_function(
                tensor_function_name=function_name,
                tensor_function_cache_parameters=self.hyperparameters,
                tensor_update_function_name=layer_update_function_name,
                tensor_function_seq_num=tensor_function_count,
                do_print_tensor_function_attr=do_print_tensor_function_attr
            )
            # Only append the pass function if it exists
            if pass_function is not None:
                self.pass_functions.append(pass_function)

        # Get the pass functions reverse
        self.pass_functions_reverse = self.pass_functions.copy()
        self.pass_functions_reverse.reverse()

        # Set the layer update function name
        self.update_function_name = layer_update_function_name

        # SANITY CHECK
        # self._print()

    def _print(self,
        print_types: Union[str, list[str], None]=None,
        kwargs: Optional[dict[str, Any]]=None
    ) -> None:
        """
        Print the layer attributes.

        Args:
            print_types (list[str]): The types of layer attributes to print
                NOTE: Passing in None for print_types prints all attributes.
            kwargs (dict[str, Any]): The keyword arguments dict

        Return
            None:
        """
        # Convert a single print type to a list
        if print_types is not None and isinstance(print_types, str):
            print_types = [print_types]

        print(f"\n{util.HARD_BANNER}\nPrinting attributes for {self.name}...")

        # Check if printing the general attributes
        if print_types is None or 'hyperparameters' in print_types:

            # Print the layer hyperparameters
            util._print_dict(self.hyperparameters, 'layer hyperparameters')

        # Check if printing the current function attributes
        if print_types is None \
                or 'current' in print_types \
                or 'function' in print_types \
                or 'current_function' in print_types:

            # Print the layer current function keyword arguments
            util._print_dict(kwargs, 'layer current function keyword arguments')

        # Check if printing the functions attributes
        if print_types is None or 'functions' in print_types:

            # Print the layer pass function names
            util._print_list_tuple(
                list(self.pass_functions), 'layer pass function names'
            )

            # Print the layer update function name
            util._print_element(self.update_function_name, 'layer update function name')

    def _update_cache(self,
        cache_updates: dict[str, Any],
        ignore_none=False
    ) -> None:
        """
        Update the cache for all layer pass functions.

        Args:
            cache_updates (dict): Dict of cache updates
            ignore_none (bool): Boolean indicating if ignoring values equal to None

        Return:
            None
        """
        for pass_function in self.pass_functions:
            # Update the pass function cache
            pass_function._update_cache(cache_updates, ignore_none)

    def _get_learnable_parameters(self,
        do_return_dict=False,
        get_weights_only=False
    ) -> Union[list, dict]:
        """
        Return the layer learnable parameters.

        Args:
            do_return_dict (bool): Boolean indicating if returning a dict instead
                of list
            get_weights_only (bool): Boolean indicating if only getting learnable
                weights

        Return:
            Dict or list of learnable parameters
        """
        # Check if returning a dict
        if do_return_dict:

            # Check if getting learnable weights only
            if get_weights_only:
                # Return the learnable weights for each pass function as a dict
                return {
                    f'pass function {i+1}': self.pass_functions[i].learnable_weights
                        for i in range(len(self.pass_functions))
                }
            else:
                # Return the learnable parameters for each pass function as a dict
                return {
                    f'pass function {i+1}': self.pass_functions[i].learnable_parameters
                        for i in range(len(self.pass_functions))
                }
        else:
            # Initialize the learnable parameters list
            learnable_parameters = []

            # Return the learnable weights for each pass function as a list
            for pass_function in self.pass_functions:
                # Check if getting learnable weights only
                if get_weights_only:
                    learnable_parameters += \
                                    list(pass_function.learnable_weights.values())
                else:
                    learnable_parameters += \
                                    list(pass_function.learnable_parameters.values())

            return learnable_parameters

    def _forward(self,
        x: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Run the layer's forward function on the input.

        x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (dict[str, Any]): Dict of output values
        """
        # Make debug log for performing the layer forward function
        log._log_debug(
            f"Performing {util._get_print_name(self.name)} forward function...",
            self.log_id
        )
        
        # Check if printing layer forward function attributes
        if self.do_print_attr['forward']:
            if kwargs is not None:
                self._print(
                    print_types='current',
                    kwargs=kwargs | {'x':x}
                )
            else:
                self._print(
                    print_types='current',
                    kwargs={'x':x}
                )

        # Iterate through the forward functions
        for function in self.pass_functions:
            # Get the output values
            output_values = function._forward(
                x=x,
                kwargs=kwargs,
                output_keys=output_keys,
                do_return_dict=do_return_dict
            )

            # Get the input from the output values
            # Check if output values is a dict
            if isinstance(output_values, dict):
                # Get the x by values position
                x = list(output_values.values())[0]
            else:
                # Get x by position
                x = output_values[0]

        # Make debug log for successfully performing the layer forward function
        log._log_debug(
            f"Successfully performed {util._get_print_name(self.name)} "
            "forward function!",
            self.log_id
        )

        # Return the output values
        return output_values

    def _backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Run the layer's backward function on the upstream gradient

        x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (dict[str, Any]): Dict of output values
        """
        # Make debug log for performing the layer backward function
        log._log_debug(
            f"Performing {util._get_print_name(self.name)} backward function...",
            self.log_id
        )

        # Check if printing layer backward function attributes
        if self.do_print_attr['backward']:
            if kwargs is not None:
                self._print(
                    print_types='current',
                    kwargs=kwargs | {'upstream_grad':upstream_grad}
                )
            else:
                self._print(
                    print_types='current',
                    kwargs={'upstream_grad':upstream_grad}
                )
    
        # Iterate through the backward functions
        for function in self.pass_functions_reverse:
            # Get the output values
            output_values = function._backward(
                upstream_grad=upstream_grad,
                kwargs=kwargs,
                output_keys=output_keys,
                do_return_dict=do_return_dict
            )

            # Get the upstream gradient from the output values
            # Check if output values is a dict
            if isinstance(output_values, dict):
                # Get the upstream gradient by values position
                upstream_grad = list(output_values.values())[0]
            else:
                # Get the upstream gradient by position
                upstream_grad = output_values[0]

        # Make debug log for successfully performing the layer backward function
        log._log_debug(
            f"Successfully performed {util._get_print_name(self.name)} "
            "backward function!",
            self.log_id
        )

        # Return the output values
        return output_values

    def _update(self) -> bool:
        """
        Update all of the layer's learnable parameters.

        Args:
            None

        Return:
            update_success (boolean): Boolean indicating success with updating
        """
        # Make debug log for performing the layer update function if provided
        if self.update_function_name is not None:
            log._log_debug(
                f"Performing {util._get_print_name(self.update_function_name)} "
                "function...",
                self.log_id
            )

        # Check if printing layer update function attributes
        if self.do_print_attr['update']:
            self._print()

        update_success = True

        # Iterate through the pass functions
        for function in self.pass_functions:
            # Store the boolean result of updating the learnable parameters
            # All tensor function updates should be successful, otherwise return False
            if not function._update():
                update_success = False

        # Check if updating layer learnable parameters was successful
        if update_success:
            # Make debug log for successfully  performing the layer update function
            #   if provided
            if self.update_function_name is not None:
                log._log_debug(
                    "Successfully performed "
                    f"{util._get_print_name(self.update_function_name)} function!",
                    self.log_id
                )


        # Return the boolean result of updating the learnable parameters
        return update_success


# ==================== TRANSFORMER ENCODER/DECODER BLOCKS =====================

NUM_IN_TOKENS = 64
NUM_OUT_CLASSES = 128
MAX_SEQ_LEN = 12
TRANSFORMERBLOCK_EMBEDDING_SIZE = 256
FEED_FWD_SIZE = 1024

class TransformerBlock(Layer):
    """
    This is the transformer encoder/decoder block (layer) class.
    """
    def __init__(self,
        # Transformer block parameters
        layer_pass_function_names: list[str],

        # Base layer hyperparameters
        reg_type=reg.REG_TYPE, reg_strength=reg.REG_STRENGTH,
        learning_rate=update.LEARNING_RATE,

        # Transformer block hyperparameters
        num_in_tokens=NUM_IN_TOKENS, num_out_classes=NUM_OUT_CLASSES,
        embedding_size=TRANSFORMERBLOCK_EMBEDDING_SIZE,
        feed_fwd_size=FEED_FWD_SIZE,
        max_seq_len=MAX_SEQ_LEN,

        # Transformer block parameters cont'd
        layer_update_function_name: Optional[str]=None,
        do_print_layer_attr=False,
        do_print_tensor_function_attr=False,
        num_attn_heads=attn.NUM_ATTN_HEADS, dropout=reg.DROPOUT,
        object_name: Optional[str]=None, has_log_id=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.TRANSFORMER_BLOCK)

        # Initialize the transformer block name
        transformer_block_name = ''

        # Update the transformer block name if the object name is provided
        if object_name is not None:
            transformer_block_name = f' [{object_name}]'

        # Make debug log for loading the transformer block
        log._log_debug(
            f"Loading the transformer block{transformer_block_name}...",
            log.LAYER_MODULE
        )

        # Get the transformer block hyperparameters
        layer_hyperparameters = {
            # Base layer hyperparameters
            'reg_type': reg_type,
            'reg_strength': reg_strength,
            'learning_rate': learning_rate,

            # Transformer block hyperparameters
            'num_in_tokens': num_in_tokens,
            'num_out_classes': num_out_classes,
            'max_seq_len': max_seq_len,
            'num_attn_heads': num_attn_heads,
            'embedding_size': embedding_size,
            'feed_fwd_size': feed_fwd_size,
            'dropout': dropout
        }

        super().__init__(
            object_name=object_name,
            layer_hyperparameters=layer_hyperparameters,
            layer_pass_function_names=layer_pass_function_names,
            layer_update_function_name=layer_update_function_name,
            do_print_layer_attr=do_print_layer_attr,
            do_print_tensor_function_attr=do_print_tensor_function_attr
        )

        # Make debug log for successfully loading the transformer block
        log._log_debug(
            f"Successfully loaded the transformer block!",
            self.log_id
        )

    
# ============================ CONVOLUTION LAYER ==============================

NUM_IN_CHANNELS = 3
NUM_OUT_FEATURES = 64 
    
class ConvolutionLayer(Layer):
    """
    This is the convolution layer class.
    """
    def __init__(self,
        # Convolution layer parameters
        layer_pass_function_names: list[str],

        # Base layer hyperparameters
        reg_type=reg.REG_TYPE, reg_strength=reg.REG_STRENGTH,
        learning_rate=update.LEARNING_RATE,

        # Convolution layer hyperparameters
        num_in_channels=NUM_IN_CHANNELS, num_out_features=NUM_OUT_FEATURES,
        kernel_size=conv.KERNEL_SIZE,
        stride=conv.STRIDE, padding=conv.PADDING,
        pool_size=pool.KERNEL_SIZE,
        pool_stride=pool.STRIDE, pool_type=pool.POOL_TYPE,

        # Convolution layer parameters cont'd
        layer_update_function_name: Optional[str]=None,
        do_print_layer_attr=False,
        do_print_tensor_function_attr=False,
        object_name: Optional[str]=None, has_log_id=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.CONVOLUTION_LAYER)

        # Initialize the convolution layer name
        conv_layer_name = ''

        # Update the convolution layer name if the object name is provided
        if object_name is not None:
            conv_layer_name = f' [{object_name}]'

        # Make debug log for loading the convolution layer
        log._log_debug(
            f"Loading the convolution layer{conv_layer_name}...",
            log.LAYER_MODULE
        )

        # Set kernel size, pool size, and pool stride to tuple
        kernel_size = util._get_tuple(kernel_size)
        pool_size = util._get_tuple(pool_size)
        pool_stride = util._get_tuple(pool_stride)
            
        # Get the Convolution layer hyperparameters
        layer_hyperparameters = {
            # Base layer hyperparameters
            'reg_type': reg_type,
            'reg_strength': reg_strength,
            'learning_rate': learning_rate,

            # Convolution layer hyperparameters
            'num_in_channels': num_in_channels,
            'num_out_features': num_out_features,
            'kernel_size': kernel_size,
            'kernel_height': kernel_size[0],
            'kernel_width': kernel_size[1],
            'stride': stride,
            'padding': padding,
            'pool_size': pool_size,
            'pool_stride': pool_stride,
            'pool_type': pool_type,
        }

        super().__init__(
            object_name=object_name,
            layer_hyperparameters=layer_hyperparameters,
            layer_pass_function_names=layer_pass_function_names,
            layer_update_function_name=layer_update_function_name,
            do_print_layer_attr=do_print_layer_attr,
            do_print_tensor_function_attr=do_print_tensor_function_attr
        )

        # Make debug log for successfully loading the convolution layer
        log._log_debug(
            f"Successfully loaded the convolution layer!",
            log.LAYER_MODULE
        )


# ============================= PROJECTION LAYER ==============================

PROJ_PRE_EMBEDDING_SIZE = 512 # Default pre-embedding size for projection
PROJ_EMBEDDING_SIZE = 756 # Default embedding size for projection

class ProjectionLayer(Layer):
    """
    This is the projection layer class.
    """
    def __init__(self,
        # Projection layer parameters
        layer_pass_function_names: list[str],

        # Base layer hyperparameters
        reg_type=reg.REG_TYPE, reg_strength=reg.REG_STRENGTH,
        learning_rate=update.LEARNING_RATE,

        # Projection layer hyperparameters
        pre_embedding_size=PROJ_PRE_EMBEDDING_SIZE,
        embedding_size=PROJ_EMBEDDING_SIZE,

        # Projection layer parameters cont'd
        layer_update_function_name: Optional[str]=None,
        do_print_layer_attr=False,
        do_print_tensor_function_attr=False,
        object_name: Optional[str]=None, has_log_id=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.PROJECTION_LAYER)

        # Initialize the projection layer name
        proj_layer_name = ''

        # Update the projection layer name if the object name is provided
        if object_name is not None:
            proj_layer_name = f' [{object_name}]'

        # Make debug log for loading the projection layer
        log._log_debug(
            f"Loading the projection layer{proj_layer_name}...",
            log.LAYER_MODULE
        )

        # Get the Convolution layer hyperparameters
        layer_hyperparameters = {
            # Base layer parameters
            'reg_type': reg_type,
            'reg_strength': reg_strength,
            'learning_rate': learning_rate,

            # Convolution layer hyperparameters
            'pre_embedding_size': pre_embedding_size,
            'embedding_size': embedding_size
        }

        super().__init__(
            object_name=object_name,
            layer_hyperparameters=layer_hyperparameters,
            layer_pass_function_names=layer_pass_function_names,
            layer_update_function_name=layer_update_function_name,
            do_print_layer_attr=do_print_layer_attr,
            do_print_tensor_function_attr=do_print_tensor_function_attr
        )

        # Make debug log for successfully loading the projection layer
        log._log_debug(
            f"Successfully loaded the projection layer!",
            log.LAYER_MODULE
        )


# =============================== LAYER LOOKUP ================================

layer_subclasses = {
    'layer': (
        Layer,
        None
    ),
    'convolution_layer' : (
        ConvolutionLayer,
        [
            'reg_type',
            'reg_strength',
            'learning_rate',

            'num_in_channels',
            'num_out_features',
            'kernel_size',
            'kernel_height',
            'kernel_width',
            'stride',
            'padding',
            'pool_size',
            'pool_stride',
            'pool_type',
        ]
    ),
    'transformer_block' : (
        TransformerBlock,
        [
            'reg_type',
            'reg_strength',
            'learning_rate',

            'num_in_tokens',
            'num_out_classes',
            'max_seq_len',
            'num_attn_heads',
            'embedding_size',
            'feed_fwd_size',
            'dropout'
        ]
    ),
    'projection_layer' : (
        ProjectionLayer,
        [
            'reg_type',
            'reg_strength',
            'learning_rate',

            'pre_embedding_size',
            'embedding_size'
        ]
    )
}

def _get_layer(
    layer_type: str,
    layer_hyperparameters: dict[str, Any],
    layer_pass_function_names: list[str],
    layer_update_function_name: Optional[str]=None,
    layer_seq_num=-1,
    do_print_layer_attr=False,
    do_print_tensor_function_attr=False
) -> Union[
    Layer, ConvolutionLayer, TransformerBlock, ProjectionLayer
]:
    """
    Return a Layer subclass object with the specified layer parameters.

    Args:
        layer_type (str): The type of layer
        layer_hyperparameters (dict[str, Any]): The layer hyperparameters
        layer_pass_function_names (list[str]): List of layer pass function names
        layer_update_function_name (str): Layer update function name
        layer_seq_num (int): Layer sequence number
        do_print_layer_attr (bool): Boolean indicating if printing the layer attributes
        do_print_tensor_function_attr (bool): Boolean indicating if printing the
            tensor/pass function attributes

    Return:
        The Layer subclass object
    """

    # Get the layer subclass with its layer hyperparameters
    layer_subclass, subclass_hyperparameter_keys = layer_subclasses[layer_type]

    # Prune the subclass hyperparameters for invalid hyperparameters
    subclass_hyperparameters = {
        k: v for k, v in layer_hyperparameters.items()
            if k in subclass_hyperparameter_keys
    }

    # Check if pass functions exist in the layer hyperparameters
    if 'pass_functions' in layer_hyperparameters.keys():
        layer_pass_function_names = layer_hyperparameters.pop('pass_functions')

    # Get the layer name
    layer_name = f'{layer_type}'

    # Check if a valid sequence number is provided
    if layer_seq_num > 0:
        layer_name += f' #{layer_seq_num}'

    return layer_subclass(
        layer_pass_function_names=layer_pass_function_names,
        **subclass_hyperparameters,
        layer_update_function_name=layer_update_function_name,
        object_name=layer_name,
        do_print_layer_attr=do_print_layer_attr,
        do_print_tensor_function_attr=do_print_tensor_function_attr,
    )