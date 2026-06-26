"""
This module handles tensor function wrapping.
"""
from __future__ import annotations
from typing import Optional, Any, Union
from collections.abc import Callable

import torch
from torch import Tensor

from function import \
    image, token, \
    attention, convolution, \
    activation, normalization, pool, regularization, reshape, residual, \
    projection, loss, update
from log import logger as log
    

class TensorFunction:
    """
    Wrapper class for tensor functions.
    """
    def __init__(self,
        function: Optional[Callable]=None,
        function_parameters: Optional[dict[str, Any]]=None,
        cache_parameters: Optional[dict[str, Any]]=None,
        object_name=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.TENSOR_FUNCTION)

        # Set the tensor function
        self.function = function

        # Initialize the cache
        self.cache = {}

        # Update the cache from the cache parameters, if provided
        if cache_parameters is not None:
            for key, value in cache_parameters:
                self.cache[key] = value

        # Initialize the function parameters if not provided
        if function_parameters is None:
            function_parameters = {}

        (
            self.kwargs,
            self.learnable_parameters,
            self.learnable_weights,
            self.return_values,
            self.output_keys
        ) = self.parse_function_parameters(
                function_parameters=function_parameters,
                do_init_learnable_parameters=False
            )

    def parse_function_parameters(self,
        function_parameters: dict[str, Any],
        do_init_learnable_parameters: bool
    ) -> tuple[
        dict[str, Any],
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Any],
        tuple[Any, ...]
    ]:
        """
        Parse the function parameters, returning the function keyword
            arguments and return values.

        Args:
            function_parameters (dict[str, Any]): The function parameters dict
            do_init_learnable_parameters (bool): Boolean for initializing
                learnable parameters

        Return:
            kwargs (dict[str, Any]): The keyword arguments dict
            learnable_parameters (dict[str, Tensor]): The learnable parameters dict
            learnable_weights (dict[str, Tensor]): The dict of learnable weights
            return_values (dict[str, Any]): The return values dict
            return_value_keys (tuple[str, ...]): The return value keys tuple
        """
        # Get the argument keys, argument key tuples, and return value keys
        #   from the function parameters if provided
        if function_parameters is not None:
            (
                argument_keys,
                argument_key_tuples,
                learnable_parameter_shapes,
                learnable_weight_keys,
                return_value_keys
            ) = function_parameters.values()
            
        # Set the keyword arguments from the argument keys
        kwargs: dict[str, Any] = {
            k: None for k in argument_keys
        }

        # Update the keyword arguments from the cache
        for key in kwargs:
            if key in self.cache:
                kwargs[key] = self.cache[key]

        # Link keyword argument keys as specified from the argument key tuples
        if argument_key_tuples is not None:
            for reference_key, key in argument_key_tuples:
                kwargs[key] = kwargs[reference_key]

        # Initialize the learnable parameters dict
        learnable_parameters = {}

        # Set the learnable parameters if using learnable parameters
        if do_init_learnable_parameters:
            learnable_parameters = self.get_init_learnable_parameters(
                learnable_parameter_shapes
            )

        # Initialize the learnable weights
        learnable_weights = {}

        # Check if the learnable weight keys were provided
        if learnable_weight_keys is not None:
            # Update learnable weights using the learnable parameters and
            #   the learnable weight keys
            for key in learnable_weight_keys:
                if key in learnable_parameters:
                    learnable_weights[key] = learnable_parameters[key]

        # Set the return values from the return value keys
        return_values: dict[str, Any] = {
            k: None for k in return_value_keys
        }

        # Return the keyword arguments and return values dict, along with the
        #   original return value keys tuple
        return kwargs, learnable_parameters, learnable_weights, \
                        return_values, return_value_keys
    
    def get_init_learnable_parameters(self,
        learnable_parameter_shapes: dict[str, tuple[str, ...]]
    ) -> dict[str, Tensor]:
        """
        Initialize and return learnable parameters.

        Args:
            learnable_parameter_shapes (dict): Dict of learnable parameter shapes

        Return:
            Dict of learnable parameters
        """
        # Get the init tensor shapes
        init_learnable_parameters = {}
        for lp_name, dimension_keys in learnable_parameter_shapes:
            # Check if dimension keys exist
            if dimension_keys is not None:
                # Check if all dimension keys exist in the cache
                if not any([key not in self.cache.keys()
                                for key in dimension_keys]):
                    lp_dims = tuple(
                        self.cache[key] for key in dimension_keys
                    )
                    init_learnable_parameters[lp_name] = torch.rand(lp_dims)

        # Return the initialized learnable parameters
        return init_learnable_parameters
    
    def get_learnable_parameters(self) -> Optional[dict[str, Any]]:
        """
        Return a dict of learnable parameters from the tensor function.

        Args:
            None

        Return:
            Dict of learnable parameters
        """
        return self.learnable_parameters

    def update_cache(self,
        cache_parameters: dict[str, Any]
    ) -> None:
        """
        Update arguments for the TensorFunction arguments dict.

        Args:
            arguments (dict): Dict of arguments

        Return:
            None
        """
        # Iterate through the provided arguments
        for key, value in cache_parameters:
            # Update the TensorFunction arguments dict
            self.cache[key] = value

    def run(self,
        x: Optional[Tensor]=None,
        upstream_grad: Optional[Tensor]=None,
        function: Optional[Callable]=None,
        kwargs: Optional[dict]=None,
        return_values: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Update the TensorFunction by applying the tensor function on input
            tensor(s), and return the updated return values.
        NOTE: May return an empty tuple.

        Args:
            x (Tensor): The input tensor
            upstream_grad (Tensor): The upstream gradient tensor
            function (Callable): The function pointer
            kwargs (dict): Dict of keyword arguments
            output_keys (tuple[str, ...]): The output keys tuple

        Return:
            Tuple of output values
        """
        # Initialize the function if not provided
        if function is None:
            function = self.function

        # Make sure the function exists
        if function is not None:
            # Initialize the function keyword arguments if not provided
            if kwargs is None:
                kwargs = self.kwargs

            # Initialize the function return values if not provided
            if return_values is None:
                return_values = self.return_values

            # Initialize the function output keys if not provided
            if output_keys is None:
                output_keys = self.output_keys

            # Set the input if provided
            # NOTE: Only forward functions take in 'x' as input
            if x is not None:
                kwargs['x'] = x

            # Set the upstream gradient if provided
            # NOTE: Only backward functions take in 'upstread_grad' as input
            if upstream_grad is not None:
                kwargs['upstream_grad'] = upstream_grad

            # Get the function output values from running the function
            output_values = function(**kwargs)

            # Make sure the function output is a tuple
            if not isinstance(output_values, tuple):
                output_values = (output_values,)

            # Zip the output keys with the function output
            outputs = {
                k: v for k, v in zip(
                    self.output_keys,
                    output_values
                )
            }

            # Update the function return values with the outputs dict
            for key, value in outputs:
                return_values[key] = value

            # Return the function output values
            return output_values
        
        # Else log error and return empty tuple since the function wasn't set
        log.log_error(
            "Couldn't run the tensor function because the function wasn't set!",
            self.log_id
        )
        return tuple()


class PassFunction(TensorFunction):
    """
    This class is for a tensor function used in forward pass and backpropagation.
    """
    def __init__(self,
        forward_function: Callable,
        forward_function_parameters: dict[str, Any],
        backward_function: Callable,
        backward_function_parameters: dict[str, Any],
        update_function: Callable,
        update_function_parameters: dict[str, Any],
        update_weight_function: Optional[Callable]=None,
        update_weight_function_parameters: Optional[dict[str, Any]]=None,
        cache_parameters: Optional[dict]=None,
        object_name=None
    ) -> None:
        # Set the log id for the pass function
        self.log_id = log.set_log_id(object_name, log.PASS_FUNCTION)

        # Initialize the base TensorFunction
        super().__init__(
            cache_parameters=cache_parameters,
            object_name=object_name, has_log_id=True
        )
        
        # Set the forward function attributes
        self.forward_function = forward_function
        (
            self.forward_kwargs,
            self.learnable_parameters,
            self.learnable_weights,
            self.forward_return_values,
            self.forward_output_keys
        ) = self.parse_function_parameters(
            function_parameters=forward_function_parameters,
            do_init_learnable_parameters=True
        )

        # Set the backward function attributes
        self.backward_function = backward_function
        (
            self.backward_kwargs,
            _,
            _,
            self.backward_return_values,
            self.backward_output_keys
        ) = self.parse_function_parameters(
            function_parameters=backward_function_parameters,
            do_init_learnable_parameters=False
        )

        # Set the update function attributes
        self.update_function = update_function
        (
            self.update_kwargs,
            _,
            _,
            self.update_return_values,
            self.update_output_keys
        ) = self.parse_function_parameters(
            function_parameters=update_function_parameters,
            do_init_learnable_parameters=False
        )

        # Initialize the update weight function
        self.update_weight_function = None

        # Set the update weight function attributes if provided
        if update_weight_function is not None \
                        and update_weight_function_parameters is not None:
            self.update_weight_function = update_weight_function
            (
                self.update_weight_kwargs,
                _,
                _,
                self.update_weight_return_values,
                self.update_weight_output_keys
            ) = self.parse_function_parameters(
                function_parameters=update_weight_function_parameters,
                do_init_learnable_parameters=False
            )

    def forward(self,
        x: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> Optional[tuple[Any, ...]]:
        """
        Perform the tensor forward function on the input tensor.

        Args:
            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            The forward function output values tuple
        """
        # Initialize the keyward arguments if not provided
        if kwargs is None:
            kwargs = self.forward_kwargs

        # Initialize the output keys if not provided
        if output_keys is None:
            output_keys = self.forward_output_keys

        return self.run(
            x=x,
            function=self.forward_function,
            kwargs=self.forward_kwargs,
            return_values=self.forward_return_values,
            output_keys=self.forward_output_keys
        )
    
    def backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> Optional[tuple[Any, ...]]:
        """
        Perform the tensor backward function on the upstream gradient tensor.

        Args:
            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            The backward function output values tuple
        """
        # Initialize the keyward arguments if not provided
        if kwargs is None:
            kwargs = self.backward_kwargs

        # Initialize the output keys if not provided
        if output_keys is None:
            output_keys = self.backward_output_keys

        return self.run(
            upstream_grad=upstream_grad,
            function=self.backward_function,
            kwargs=kwargs,
            return_values=self.backward_return_values,
            output_keys=output_keys
        )
    
    def update(self) -> bool:
        """
        Perform the tensor forward function on the input tensor.

        Args:
            update_function (Callable): The update function pointer

        Return:
            Boolean indicating success with updating
        """
        # Iterate through the learnable parameters
        for lp_name, learnable_parameter in self.learnable_parameters:
            # Get the corresponding gradient to the learnable parameter
            #NOTE: Gradients are returned from the tensor backward functions
            gradient_name = lp_name+'_grad'

            # Check if the gradient name doesn't exist in the backward return values
            if gradient_name in self.backward_return_values.keys():
                # Log error and return False since the gradient doesn't exist
                log.log_error(
                    f"Couldn't update {lp_name} since {gradient_name} doesn't exist!",
                    self.log_id
                )
                return False
            
            # Else, get the gradient
            gradient = self.backward_return_values[gradient_name]

            # Check if the learnable parameter is a learnable weight
            if lp_name in self.learnable_weights:
                # Update the learnable parameter and gradient in the update
                #   function keyword arguments
                self.update_weight_kwargs['learnable_weight'] = learnable_parameter
                self.update_weight_kwargs['weight_gradient'] = gradient

                # Run the update function
                self.run(
                    function=self.update_weight_function,
                    kwargs=self.update_weight_kwargs,
                    return_values=self.update_weight_return_values,
                    output_keys=self.update_weight_output_keys
                )

                # Update the learnable weight from the return values
                self.learnable_weights[lp_name] = \
                        self.update_weight_return_values['updated_learnable_weight']

            # Else, the learnable parameter is not a learnable weight
            else:
                # Update the learnable parameter and gradient in the update
                #   function keyword arguments
                self.update_kwargs['learnable_parameter'] = learnable_parameter
                self.update_kwargs['gradient'] = gradient

                # Run the update function
                self.run(
                    function=self.update_function,
                    kwargs=self.update_kwargs,
                    return_values=self.update_return_values,
                    output_keys=self.update_output_keys
                )

                # Update the learnable parameter from the return values
                self.learnable_parameters[lp_name] = \
                        self.update_return_values['updated_learnable_parameter']
            
        # Return True since all learnable parameters were successfully updated
        return True
            

# ========================= TENSOR FUNCTION LOOKUP ============================


# A dict of tensor functions and their respective parameter keys
tensor_functions = {
    # Activation functions
    'relu': (
        activation.relu,
        {
            'argument_keys': ('x'),
            'argument_key_tuples': ( ('x', 'relu_in'), ),
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('relu_out',),
            'backward_function_name': 'relu_backward'
        }
    ),
    'relu_backward': (
        activation.relu_backward,
        {
            'arugment_keys': ('upstream_grad', 'relu_in'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('relu_in_grad',)
        }
    ),

    # Attention functions
    'multi_head_attention': (
        attention.multi_head_attention,
        {
            'argument_keys': (
                'Q', 'K', 'V',
                'W_q', 'W_k', 'W_v', 'W_o', # learnable parameters
                'num_attn_heads',
                'b_q', 'b_k', 'b_v', 'b_o', # learnable parameters
                'attn_mask'
            ),
            'argument_key_tuples': None,

            'learnable_parameter_shapes': {
                # Init W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o
                'W_q': ('keys_embedding_size', 'embedding_size'),
                'W_k': ('keys_embedding_size', 'embedding_size'),
                'W_v': ('values_embedding_size', 'embedding_size'),
                'W_o': ('embedding_size', 'embedding_size'),
                'b_q': ('embedding_size',),
                'b_k': ('embedding_size',),
                'b_v': ('embedding_size',),
                'b_o': ('embedding_size',)
            },
            'learnable_weights': ('W_q', 'W_k', 'W_v', 'W_o'),

            'return_value_keys': (
                'context_vector', 'attention_weights',
                'Q_proj', 'K_proj', 'V_proj',
            ),
            'backward_function_name': 'multi_head_attention_backward'
        }
    ),
    'multi_head_attention_backward': (
        attention.multi_head_attention_backward,
        {
            'argument_keys': (
                'upstream_grad',
                'Q', 'K', 'V',
                'Q_proj', 'K_proj', 'V_proj',
                'W_q', 'W_k', 'W_v', 'W_o',
                'context_vector', 'attention_weights',
                'num_attn_heads',
                'b_q', 'b_k', 'b_v', 'b_o',
                'attn_mask'
            ),
            'argument_key_tuples': None,

            'learnable_parameter_shapes': None,
            'learnable_weights': None,

            'return_value_keys': (
                'Q_in_grad', 'W_q_grad', 'b_q_grad',
                'K_in_grad', 'W_k_grad', 'b_k_grad',
                'V_in_grad', 'W_v_grad', 'b_v_grad',
                'W_o_grad', 'b_o_grad',
                'attn_mask'
            )
        }
    ),

    'multi_head_cross_attention': (
        attention.multi_head_cross_attention,
        {
            'argument_keys': (
                'Q', 'K', 'V',
                'W_q', 'W_k', 'W_v', 'W_o', # learnable parameters
                'num_attn_heads',
                'b_q', 'b_k', 'b_v', 'b_o', # learnable parameters
                'padding_mask', 'pad_value'
            ),
            'argument_key_tuples': None,

            'learnable_parameter_shapes': {
                # Init W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o
                'W_q': ('keys_embedding_size', 'embedding_size'),
                'W_k': ('keys_embedding_size', 'embedding_size'),
                'W_v': ('values_embedding_size', 'embedding_size'),
                'W_o': ('embedding_size', 'embedding_size'),
                'b_q': ('embedding_size',),
                'b_k': ('embedding_size',),
                'b_v': ('embedding_size',),
                'b_o': ('embedding_size',)
            },
            'learnable_weights': ('W_q', 'W_k', 'W_v', 'W_o'),

            'return_value_keys': (
                'context_vector', 'attention_weights',
                'Q_proj', 'K_proj', 'V_proj',
                'attn_mask'
            ),
            'backward_function_name': 'multi_head_attention_backward'
        }
    ),

    'multi_head_masked_self_attention': (
        attention.multi_head_masked_self_attention,
        {
            'argument_keys': (
                'Q', 'K', 'V',
                'W_q', 'W_k', 'W_v', 'W_o', # learnable parameters
                'num_attn_heads',
                'b_q', 'b_k', 'b_v', 'b_o', # learnable parameters
                'causal_mask', 'sequence_length',
                'padding_mask', 'pad_value'
            ),
            'argument_key_tuples': None,

            'learnable_parameter_shapes': {
                # Init W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o
                'W_q': ('keys_embedding_size', 'embedding_size'),
                'W_k': ('keys_embedding_size', 'embedding_size'),
                'W_v': ('values_embedding_size', 'embedding_size'),
                'W_o': ('embedding_size', 'embedding_size'),
                'b_q': ('embedding_size',),
                'b_k': ('embedding_size',),
                'b_v': ('embedding_size',),
                'b_o': ('embedding_size',)
            },
            'learnable_weights': ('W_q', 'W_k', 'W_v', 'W_o'),

            'return_value_keys': (
                'context_vector', 'attention_weights',
                'Q_proj', 'K_proj', 'V_proj',
                'attn_mask'
            ),
            'backward_function_name': 'multi_head_attention_backward'
        }
    ),

    # Convolution functions
    'conv2d': (
        convolution.conv2d,
        {
            'argument_keys': (
                'x',
                'W', 'b', # learnable parameters
                'stride', 'padding'
            ),
            'argument_key_tuples': ( ('x', 'conv_in'), ),

            'learnable_parameter_shapes': { # Init W, b
                'W': (
                    'num_out_features', 'num_in_channels',
                    'kernel_height', 'kernel_width'
                ),
                'b': ('num_out_features')
            },
            'learnable_weights': ('W',),
                
            'return_value_keys': ('conv_out',),
            'backward_function_name': 'conv2d_backward'
        }
    ),
    'conv2d_backward': (
        convolution.conv2d_backward,
        {
            'argument_keys': (
                'upstream_grad', 'conv_in',
                'W',
                'kernel_size', 'stride', 'padding'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('conv_in_grad', 'W_grad', 'b_grad')
        }
    ),

    # Image functions
    'get_image_tensor': (
        image.get_image_tensor,
        {
            'argument_keys': ('image_filepath',),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('image_tensor',),
            'backward_function_name': None
        }
    ),
    'get_images_tensor': (
        image.get_image_tensor,
        {
            'argument_keys': ('image_filepaths',),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('images_tensor',),
            'backward_function_name': None
        }
    ),

    # Loss functions
    'binary_cross_entropy_loss': (
        loss.binary_cross_entropy_loss,
        {
            'argument_keys': (
                'predictions', 'true_labels',
                'loss_reduction_type',
                'reg_type', 'reg_strength',
                'weights'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('bce_loss',),
            'backward_function_name': None
        }
    ),
    'binary_cross_entropy_loss_backward': (
        loss.binary_cross_entropy_loss_backward,
        {
            'argument_keys': (
                'upstream_grad',
                'sigmoid_out', 'true_labels',
                'loss_reduction_type'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('logits_grad',)
        }
    ),

    'cross_entropy_loss': (
        loss.cross_entropy_loss,
        {
            'argument_keys': (
                'predictions', 'true_labels',
                'loss_reduction_type',
                'reg_type', 'reg_strength',
                'weights'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('ce_loss',),
            'backward_function_name': None
        }
    ),
    'cross_entropy_loss_backward': (
        loss.cross_entropy_loss_backward,
        {
            'argument_keys': (
                'upstream_grad',
                'softmax_out', 'true_labels',
                'loss_reduction_type'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('logits_grad',)
        }
    ),

    # Normalization functions
    'layer_norm': (
        normalization.layer_norm,
        {
            'argument_keys': ('x', 'eps', 'gamma', 'beta'),
            'argument_key_tuples': None,

            'learnable_parameter_shapes': { # Init gamma, beta
                'gamma': ('embedding_size',),
                'beta': ('embedding_size',)
            },
            'learnable_weights': ('gamma',),

            'return_value_keys': ('layer_norm_out', 'norm_out', 'std'),
            'backward_function_name': 'layer_norm_backward'
        }
    ),
    'layer_norm_backward': (
        normalization.layer_norm_backward,
        {
            'argument_keys': ('upstream_grad', 'norm_out', 'gamma', 'std'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('layer_norm_in_grad', 'gamma_grad, beta_grad')
        }
    ),

    # Pooling functions
    'pool': (
        pool.pool,
        {
            'argument_keys': ('x', 'pool_size', 'pool_stride', 'pool_type'),
            'argument_key_tuples': ( ('x', 'pool_in'), ),
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('pool_out',),
            'backward_function_name': 'unpool'
        }
    ),
    'unpool': (
        pool.unpool,
        {
            'argument_keys': (
                'upstream_grad', 'pool_in',
                'pool_size', 'pool_stride', 'pool_type'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('pool_in_grad',)
        }
    ),

    # Projection functions
    'feed_forward': (
        projection.feed_forward,
        {
            'argument_keys': ('x', 'W_1', 'W_2', 'b_1', 'b_2'),
            'argument_key_tuples': ( ('x', 'ff_in'), ),

            'learnable_parameter_shapes': { # Init W_1, W_2, b_1, b_2
                'W_1': ('embedding_size', 'ff_embedding_size'),
                'W_2': ('ff_mebedding_size', 'embedding_size'),
                'b_1': ('embedding_size',),
                'b_2': ('embedding_size',)
            },
            'learnable_weights': ('W_1', 'W_2'),

            'cache': None,
            'return_value_keys': ('ff_out', 'relu_out', 'relu_in'),
            'backward_function_name': 'feed_forward_backward'
        }
    ),
    'feed_forward_backward': (
        projection.feed_forward_backward,
        {
            'argument_keys': (
                'upstream_grad',
                'W_1', 'W_2',
                'relu_out', 'relu_in',
                'ff_in',
                'b_1', 'b_2'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': (
                'ff_in_grad',
                'W_1_grad', 'b_1_grad',
                'relu_out_grad',
                'W_2_grad', 'b_2_grad'
            )
        }
    ),

    'lin_proj': (
        projection.lin_proj,
        {
            'argument_keys': ('x', 'W', 'b'),
            'argument_key_tuples': ( ('x', 'proj_in'), ),

            'learnable_parameter_shapes': { # Init W, b
                'W': ('pre_embedding_size', 'embedding_size'),
                'b': ('embedding_size',)
            },
            'learnable_weights': ('W',),

            'cache': None,
            'return_value_keys': ('proj_out',),
            'backward_function_name': 'lin_proj_backward'
        }
        
    ),
    'lin_proj_backward': (
        projection.lin_proj_backward,
        {
            'argument_keys': ('upstream_grad', 'proj_input', 'W'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('proj_in_grad', 'W_grad', 'b_grad')
        }
    ),

    # Regularization functions
    'dropout': (
        regularization.dropout,
        {
            'argument_keys': ('x', 'dropout'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('dropout_out',),
            'backward_function_name': 'dropout_backward'
        }
    ),
    'dropout_backward': (
        regularization.dropout_backward,
        {
            'argument_keys': ('upstream_grad', 'dropout', 'dropout_mask'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('dropout_in_grad',)
        }
    ),

    'ridge_regression': (
        regularization.ridge_regression,
        {
            'argument_keys': ('x', 'reg_strength', 'weights'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('ridge_out',)
        }
    ),

    # Reshaping functions
    'flatten': (
        reshape.flatten,
        {
            'argument_keys': ('x'),
            'argument_key_tuples': ( ('x', 'flat_in'), ),
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('flat_out',),
            'backward_function_name': 'unflatten'
        }
    ),
    'unflatten': (
        reshape.unflatten,
        {
            'argument_keys': ('upstream_grad', 'flat_in'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('flat_in_grad',)
        }
    ),

    # Residual functions
    'residual_add': (
        residual.residual_add,
        {
            'argument_keys': ('x', 'res_addend'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('res_out',),
            'backward_function_name': 'residual_add_backward'
        }
    ),

    'residual_add_backward': (
        residual.residual_add_backward,
        {
            'argument_keys': ('upstream_grad',),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('res_in_grad', 'res_addend_grad')
        }
    ),

    # Token functions
    'get_tokens_tensor': (
        token.get_tokens_tensor,
        {
            'argument_keys': (
                'sentence', 'sentences', 'sentence_list',
                'token_ids'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('tokens_tensor',),
            'backward_function_name': None
        }
    ),

    'get_embedded_tokens': (
        token.get_embedded_tokens,
        {
            'argument_keys': (
                'tokens',
                'token_embeddings', 'token_ids',
                'embedding_size',
                'positional_encodings', 'positional_encoding_type',
                'use_positional_encodings'
            ),
            'argument_key_tuples': (('tokens', 'tokens_in'),),
            
            'learnable_parameter_shapes': {
                'token_embeddings': None,
                'positional_embeddings': None
            },
            'learnable_weights': None,

            'return_value_keys': (
                'tokens',
                'token_embeddings',
                'positional_encodings'
            ),
            'backward_function_name': 'get_embedded_tokens_backward'
        }
    ),
    'get_embedded_tokens_backward': (
        token.get_embedded_tokens_backward,
        {
            'argument_keys': (
                'upstream_grad',
                'tokens_in'
                'token_embeddings', 'are_token_embeddings_learnable',
                'positional_encodings', 'are_positional_encodings_learnable',
            ),
            'argument_key_tuples': None,
            
            'learnable_parameter_shapes': None,
            'learnable_weights': None,

            'return_value_keys': (
                'upstream_grad',
                'pos_in_grad',
                'embed_in_grad'
            )
        }
    ),

    # Update functions
    'basic_update': (
        update.basic_update,
        {
            'argument_keys': ('learnable_parameter', 'gradient', 'learning_rate'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('updated_learnable_parameter',),
            'backward_function_name': None
        }
    ),

    'ridge_regression_update': (
        update.ridge_regression_update,
        {
            'argument_keys': (
                'learnable_weight', 'weight_gradient',
                'reg_strength', 'learning_rate'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('updated_learnable_weight',),
            'backward_function_name': None
        }
    )
}


update_function_names = {
    'ridge_regression_update': ('basic_update', 'ridge_regression_update')
}

def get_tensor_function(
    tensor_function: Optional[Callable]=None,
    tensor_function_name: Optional[str]=None,
    tensor_update_function_name: Optional[str]=None,
    tensor_function_cache_parameters: Optional[dict[str, Any]]=None
) -> Union[Optional[TensorFunction], Optional[PassFunction]]:
    """
    Return a TensorFunction object.
    NOTE: If a tensor function doesn't have a corresponding backward function,
        then it can't make use of the update function.

    Args:
        tensor_function (Tensor): The tensor function pointer
        tensor_function_name (str): The tensor function name
        tensor_update_function_name (str): The tensor update function name
        tensor_function_cache (dict[str, Any]) The tensor function cache

    Return:
        A TensorFunction object
    """
    # Get the tensor function by function name if provided
    if tensor_function is None and tensor_function_name is not None:
        tensor_function, tensor_function_parameters = \
                        tensor_functions[tensor_function_name]
        assert(tensor_function is not None)

        # Get the tensor backward function by backward function name if provided
        tensor_backward_function_name = \
                        tensor_function_parameters['backward_function_name']
        if tensor_backward_function_name is not None:
            tensor_backward_function, tensor_backward_function_parameters = \
                            tensor_functions[tensor_backward_function_name]

            # Get the tensor update function name if not provided
            if tensor_update_function_name is None:
                tensor_update_function_name = 'basic_update'

            # Separate the standard update from the weights-specific update
            tensor_update_function_name, tensor_update_weights_function_name = \
                            update_function_names[tensor_function_name]
            

            # Get the tensor update function and function parameters
            tensor_update_function, tensor_update_function_parameters = \
                            tensor_functions[tensor_update_function_name]
            
            # Initialize the tensor weight update function and its parameters
            #   if not provided
            if tensor_update_weights_function_name is None:
                tensor_update_weights_function = None
                tensor_update_weights_function_parameters = None
            else:
                tensor_update_weights_function, \
                    tensor_update_weights_function_parameters = \
                                    tensor_functions[tensor_update_weights_function_name]
            
            # Return the TensorFunction with the specified forward, backward,
            #   and update functions
            return PassFunction(
                forward_function=tensor_function,
                forward_function_parameters=tensor_function_parameters,
                backward_function=tensor_backward_function,
                backward_function_parameters=tensor_backward_function_parameters,
                update_function=tensor_update_function,
                update_function_parameters=tensor_update_function_parameters,
                update_weight_function=tensor_update_weights_function,
                update_weight_function_parameters=\
                    tensor_update_weights_function_parameters,
                cache_parameters=tensor_function_cache_parameters
            )

    # Return a TensorFunction only if the function exists
    if tensor_function is not None:
        return TensorFunction(
            function=tensor_function,
            function_parameters=tensor_function_parameters,
            cache_parameters=tensor_function_cache_parameters
        )
    
    # Else, return None