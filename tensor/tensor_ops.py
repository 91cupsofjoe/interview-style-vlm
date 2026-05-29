"""
This module is for fetching tensor operations.
"""
from typing import Any

from torch import Tensor

from tensor import \
    activation, attention, convolution, image, \
    loss, normalization, pool, projection, regularization, reshape


# ========================= TENSOR FUNCTION LOOKUP ============================


# A dict of tensor functions and their respective parameter keys
tensor_functions = {
    # Activation functions
    'relu': (
        activation.relu,
        {
            'input': ('input', None),
            'output': ('output')
        }
    ),
    'relu_backward': (
        activation.relu_backward,
        {
            'input': ('input', 'z_conv'),
            'output': ('output')
        }
    ),

    # Attention functions
    'multi_head_attention': (
        attention.multi_head_attention,
        {
            'input': ('input',
                      'Wq_proj', 'Wk_proj', 'Wv_proj', 'W_proj',
                      'num_heads', 'mask'
                    ),
            'output': ('output')
        }
    ),

    # Convolution functions
    'conv2d': (
        convolution.conv2d,
        {
            'input': ('input', 'W_conv', 'b_conv', 'stride', 'padding'),
            'output': ('output')
        }
    ),
    'conv2d_backward': (
        convolution.conv2d_backward,
        {
            'input': ('input', 'x', 'W_conv', 'kernel_size', 'stride', 'padding'),
            'output': ('output')
        }
    ),

    # Image functions are not used by tensor lookup

    # Loss functions
    'cross_entropy_loss': (
        loss.cross_entropy_loss,
        {
            'input': ('input',
                      'true_labels', 'weights',
                      'reg_type', 'reg_strength'
                    ),
            'output': ('output')
        }
    ),

    # Normalization functions
    'layer_norm': (
        normalization.layer_normalization,
        {
            'input': ('input', 'gamma', 'beta'),
            'output': ('output')
        }
    ),

    # Pooling functions
    'pool': (
        pool.pool,
        {
        'input': ('input', 'pool_type', 'kernel_size', 'stride'),
        'output': ('output')
        }
    ),
    'unpool': (
        pool.unpool,
        {
            'input': ('input', 'a_relu', 'pool_size', 'pool_type', 'stride'),
            'output': ('output')
        }
    ),

    # Projection functions
    'lin_proj': (
        projection.lin_proj,
        {
            'input': ('input', 'W_proj', 'b_proj'),
            'output': ('output')
        }
        
    ),
    'lin_proj_backward': (
        projection.lin_proj_backward,
        {
            'input': ('input', 'h', 'W_proj'),
            'output': ('output')
        }
    ),
    'feed_forward': (
        projection.feed_forward,
        {
            'input': ('input', 'Wff_proj_1', 'Wff_proj_2'),
            'output': ('output')
        }
    ),

    # Regularization functions
    'regularization': (
        regularization.dropout,
        {
            'input': ('input', 'dropout'),
            'output': ('output')
        }
    ),

    # Reshaping functions
    'flatten': (
        reshape.flatten,
        {
            'input': ('input', None),
            'output': ('output')
        }
    ),
    'unflatten': (
        reshape.unflatten,
        {
            'input': ('input', 'z_conv'),
            'output': ('output')
        }
    )
}


def apply_tensor_function(x: Tensor,
    function_name: str, function_params: dict[str, Any]
) -> dict[str, Any]:
    """
    Apply a tensor transformation on the input tensor.

    Args:
        x (Tensor): The input tensor
        transform_name (str): The name of the transformation function to apply
        params (dict[str, Any]): Dict of transformation function parameters

    Return:
        A transformed tensor
    """
    # Get the tensor function and its parameter keys
    tensor_function, function_param_keys = tensor_functions[function_name]
    function_input_keys = function_param_keys['input']
    function_output_keys = function_param_keys['output']

    # Get the function output(s)
    function_output = tensor_function(
        x,
        (function_params[key] for key in function_input_keys
            if key in function_params)
    )
    
    # Check if the function output is a singular item
    if not isinstance(function_output, tuple):
        # Convert the function output to a tuple
        function_output = (function_output)
    
    # Return the function output(s) as a dict
    return {
        keys: values for keys, values
            in zip(function_output_keys, function_output)
        }