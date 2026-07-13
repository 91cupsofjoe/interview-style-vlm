"""
This module handles tensor function wrapping.
"""
from __future__ import annotations
from typing import Optional, Any, Union
from collections.abc import Callable
from copy import deepcopy
import math

import torch
from torch import Tensor

from tensor_function import \
    image, token, \
    attention, convolution, \
    activation, normalization, pool, regularization, reshape, residual, \
    projection, loss, update, \
    util
from log import logger as log
    

class TensorFunction:
    """
    Wrapper class for tensor functions.
    """
    def __init__(self,
        function_name: Optional[str]=None,
        function_ptr: Optional[Callable]=None,
        function_arguments: Optional[dict[str, Any]]=None,
        function_return_value_keys: Optional[tuple[str, ...]]=None,
        function_parameters: Optional[dict[str, Any]]=None,
        cache_parameters: Optional[dict[str, Any]]=None,
        do_print_tensor_function_attr: Union[bool, dict[str, bool], None]=None,
        object_name: Optional[str]=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.TENSOR_FUNCTION)

        # Set the bool for printing tensor function attributes
        self.do_print_tensor_attr = do_print_tensor_function_attr

        # Initialize the function name if not provided
        if function_name is None:
            # Check if the object name exists
            if object_name is not None:
                function_name = object_name
            else:
                function_name = f'{log.UNLABELED} TENSORFUNCTION'

        # Initialize the cache
        self.cache = {}

        # Update the cache from the cache parameters, if provided
        if cache_parameters is not None:
            for key, value in cache_parameters.items():
                self.cache[key] = value

        # Check if the function pointer was provided
        if function_ptr is not None:
            # This instantiates a strict TensorFunction instead of a PassFunction

            # Set the tensor function name
            self.tensor_name = function_name

            # Set the tensor function pointer
            self.function = function_ptr

            # Initialze the function arguments if not provided
            if function_arguments is None:
                function_arguments = {}

            # Update the cache with the function arguments
            for key, value in function_arguments.items():
                    self.cache[key] = value

            # Initialize the function parameters if not provided
            if function_parameters is None:

                function_parameters = {
                    'argument_keys': function_arguments.keys(),
                    'argument_key_tuples': None,
                    'learnable_parameter_shapes': None,
                    'learnable_weights': None,
                    'return_value_keys': function_return_value_keys
                }
                
                (
                    self.kwargs,
                    self.ref_kwargs,
                    self.learnable_parameters,
                    _,
                    self.return_values,
                    self.output_keys
                ) = self._parse_function_parameters(
                        function_parameters=function_parameters,
                        do_init_learnable_parameters=False
                    )
            
    def _print(self,
        print_types: Union[str, list[str], None]=None,
        kwargs: Optional[dict[str, Any]]=None
    ) -> None:
        """
        Print the tensor function attributes.

        Args:
            kwargs (dict[str, Any]): The keyword arguments
            print_types (str | list[str]): The print type(s)
            NOTE: Print types are used in the subclass _print() function

        Return:
            None
        """
        # Convert single str print_types to list
        if isinstance(print_types, str):
            print_types = [print_types]

        print(f"\n{util.HARD_BANNER}\nPrinting attributes for "
              f"{util._get_print_name(self.tensor_name)}...")
        
        # Check if printing the cache
        if print_types is None or 'cache' in print_types:

            # Print the cache
            util._print_dict(self.cache, 'cache')

        if print_types is None or 'general' in print_types:

            # Print the reference keyword arguments
            util._print_dict(
                self._get_ref_kwargs(),
                'reference keyword arguments'
            )

            # Print the return values
            util._print_dict(self.return_values, 'return values')

            # Print the output keys
            util._print_list_tuple(self.output_keys, 'output keys')
        
        if print_types is None or 'current' in print_types:

            # Initialize the kwargs if not provided
            if kwargs is None:
                kwargs = self.kwargs

            # Print the keyword arguments
            util._print_dict(kwargs, 'keyword arguments')

        print(f'{util.HARD_BANNER}\n')

    def _update_cache(self,
        cache_updates: dict[str, Any],
        ignore_none=False
    ) -> None:
        """
        Update the tensor function cache.

        Args:
            cache_updates (dict): Dict of cache updates
            ignore_none (bool): Boolean indicating if ignoring values equal to None

        Return:
            None
        """
        # Update the cache
        self.cache |= util._get_update_dict(cache_updates, ignore_none=ignore_none)

        # Update the keyword arguments using the update cache
        self.kwargs |= util._get_update_dict(
            self.cache, self.kwargs, ignore_none
        )

    def _parse_function_parameters(self,
        function_parameters: dict[str, Any],
        do_init_learnable_parameters=False
    ) -> tuple[
        dict[str, Any],
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
            ref_kwargs (dict[str, Any]): The reference keyword arguments dict
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
        kwargs = {
            k: None for k in argument_keys
        }

        # Update the keyword arguments from the cache
        kwargs = util._get_update_dict(self.cache, kwargs)

        # Initialize the reference keyword arguments
        ref_kwargs = {}

        # Link keyword argument keys as specified from the argument key tuples
        if argument_key_tuples is not None:
            for ref_key, key in argument_key_tuples:
                ref_kwargs[ref_key] = (key, kwargs[ref_key])

        # Initialize the learnable parameters
        learnable_parameters = {}

        # Set the learnable parameters if using learnable parameters
        if do_init_learnable_parameters:
            learnable_parameters = self._get_init_learnable_parameters(
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

        # Initialize the return value keys if not set
        if return_value_keys is None:
            return_value_keys = ()

        # Set the return values from the return value keys
        return_values: dict[str, Any] = {
            k: None for k in return_value_keys
        }

        # Return the keyword arguments, reference keyword arguments,
        #   learnable parameters, learnable weights,
        #   return values, and return value keys
        return kwargs, ref_kwargs, \
            learnable_parameters, learnable_weights, \
                return_values, return_value_keys
    
    def _get_init_learnable_parameters(self,
        learnable_parameter_shapes: Optional[dict[str, tuple[str, ...]]]=None
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

        # Initialize learnable parameter shapes if not set
        if learnable_parameter_shapes is None:
            learnable_parameter_shapes = {}
        
        # Iterate through the learnable parameters shapes to initialize the
        #   learnable parameters
        for lp_name, dimension_keys in learnable_parameter_shapes.items():
            # Check if dimension keys exist
            if dimension_keys is not None:

                # Check if all dimension keys exist in the cache
                if not any([key not in self.cache.keys()
                                for key in dimension_keys]):
                    # Get the learnable parameters dimensions
                    lp_dims = tuple(
                        self.cache[key] for key in dimension_keys
                    )

                    # Get the He/Kaiming standard deviation
                    in_size = 0
                    for dim in lp_dims[1:]:
                        in_size += dim

                    std = 1
                    if in_size > 0:
                        std = math.sqrt(2.0 / in_size)

                    init_learnable_parameters[lp_name] = torch.randn(lp_dims) * std

        '''SANITY CHECK
        for lp_name, lp in init_learnable_parameters.items():
            print(f"Learnable parameter {lp_name} has shape = {lp.shape}")
        '''

        # Return the initialized learnable parameters
        return init_learnable_parameters
    
    def _get_learnable_parameters(self) -> Optional[dict[str, Any]]:
        """
        Return a dict of learnable parameters for the tensor function.

        Args:
            None

        Return:
            Dict of learnable parameters
        """
        return self.learnable_parameters

    def _update_ref_kwargs(self,
        update_kwargs: dict[str, Any]
    ) -> None:
        """
        Update keyword arguments for the reference keyword arguments.

        Args:
            update_kwargs (dict[str, Any]): The updated keyword arguments

        Return:
            None
        """
        for key, value in update_kwargs.items():
            if key in self.ref_kwargs.keys():
                ref_key, _ = self.ref_kwargs[key]
                self.ref_kwargs[key] = (ref_key, value)

    def _get_ref_kwargs(self,
        base_kwargs: Optional[dict[str, Any]]=None
    ) -> dict[str, Any]:
        """
        Return the reference keyword arguments dict that uses reference keys
            for lookup.

        Args:
            base_kwargs (dict[str, Any]): The base keyword arguments dict 

        Return:
            Dict of reference keyword arguments
        """
        # Get the reference keyword arguments
        ref_kwargs = {
            rf: v for k, (rf, v) in self.ref_kwargs.items()
        }
        
        # Check if base keyword arguments were provided
        if base_kwargs is not None:
            return {
                k: v for k, v in ref_kwargs.items()
                    if k in base_kwargs.keys()
            }
        else:
            return ref_kwargs

    def _run(self,
        x: Optional[Tensor]=None,
        upstream_grad: Optional[Tensor]=None,
        function_ptr: Optional[Callable]=None,
        kwargs: Optional[dict[str, Any]]=None,
        base_kwargs: Optional[dict[str, Any]]=None,
        return_values: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        function_name: Optional[str]=None,
        print_types: Union[str, list[str], None]=None,
        do_return_dict=False
    ) -> Union[tuple[Any, ...], dict[str, Any]]:
        """
        Update the TensorFunction by applying the tensor function on input
            tensor(s), and return the updated return values.
        NOTE: May return an empty tuple.

        Args:
            x (Tensor): The input tensor
            upstream_grad (Tensor): The upstream gradient tensor
            function_ptr (Callable): The function pointer
            kwargs (dict[str, Any]): Dict of keyword arguments
            base_kwargs (dict[str, Any]): Dict of base function keyword arguments
            output_keys (tuple[str, ...]): The output keys tuple
            function_name (str): The name of the tensor function
            print_types (str | list[str]): The print type(s)
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            Tuple of output values
        """
        # Initialize the function name if not provided
        if function_name is None:
            function_name = self.tensor_name

        # Make debug log for running the tensor function if function name is provided
        if function_name is not None:
            log._log_debug(
                f"Running {util._get_print_name(function_name)}...",
                self.log_id
            )

        # Initialize the function if not provided
        if function_ptr is None:
            function_ptr = self.function

        # Make sure the function exists
        if function_ptr is not None:
            # Initialize the base keyword arguments if not provided
            if base_kwargs is None:
                # Check if the tensor function keyword arguments exist
                if self.kwargs is not None:
                    base_kwargs = self.kwargs

                else:
                    base_kwargs = {}

            # Update the base keyword arguments with the learnable parameters
            # base_kwargs = util._update_dict(self.learnable_parameters, base_kwargs)
                
            # Initialize the keyword arguments if not provided
            if kwargs is None:
                kwargs = {}

            # Update the base keyword arguments with the keyword arguments
            kwargs = base_kwargs | kwargs

            # Update the keyword arguments with the reference keyword arguments
            kwargs |= self._get_ref_kwargs(kwargs)

            # Initialize the function output keys if not provided
            if output_keys is None:
                output_keys = self.output_keys

            # Initialize the function return values if not provided
            if return_values is None:
                return_values = self.return_values

            # Initialize the updated keyword arguments
            updated_kwargs = {}

            # Set the input if provided
            # NOTE: Only forward functions take in 'x' as input
            if x is not None:
                updated_kwargs = {'x':x}
                self._update_ref_kwargs(updated_kwargs)

            # Set the upstream gradient if provided
            # NOTE: Only backward functions take in 'upstread_grad' as input
            if upstream_grad is not None:
                updated_kwargs = {'upstream_grad':upstream_grad}
                self._update_ref_kwargs(updated_kwargs)

            kwargs = util._get_update_dict(updated_kwargs, kwargs)

            # Check if print types were provided
            if print_types is not None or self.do_print_tensor_attr:
                self._print(
                    print_types=print_types,
                    kwargs=kwargs
                )

            # Get the function output values from running the function
            output_values = function_ptr(**kwargs)

            # Make sure the function output is a tuple
            if not isinstance(output_values, tuple):
                output_values_tuple = (output_values,)
            else:
                output_values_tuple = output_values

            # Zip the output keys with the function output
            outputs = {
                k: v for k, v in zip(
                    output_keys,
                    output_values_tuple
                )
            }

            # Update the reference keyword arguments using the updated keyword
            #   arguments and return values
            self._update_ref_kwargs(
                update_kwargs=updated_kwargs | return_values
            )

            # Make debug log for successfully running the tensor function if
            #   name is provided
            if function_name is not None:
                log._log_debug(
                    f"Successfully ran {util._get_print_name(function_name)}!",
                    self.log_id
                )

            # Check if returning a dict of value
            if do_return_dict:
                # Update the function return values with the outputs dict
                return_values = util._get_update_dict(outputs, return_values)

                # Return the return values dict
                return return_values
            
            # Else, return the function output values
            return output_values_tuple
        
        # Else log error and return empty tuple since the function wasn't set
        log._log_error(
            "Couldn't run the tensor function because the function wasn't set!",
            self.log_id
        )
        return tuple()


class PassFunction(TensorFunction):
    """
    This class is for a tensor function used in forward pass and backpropagation.
    """
    def __init__(self,
        forward_function_name: str,
        forward_function_ptr: Callable,
        forward_function_parameters: dict[str, Any],
        backward_function_name: str,
        backward_function_ptr: Callable,
        backward_function_parameters: dict[str, Any],
        update_function_name: str,
        update_function_ptr: Callable,
        update_function_parameters: dict[str, Any],
        update_weights_function_name: str,
        update_weights_function_ptr: Callable,
        update_weights_function_parameters: dict[str, Any],
        cache_parameters: Optional[dict]=None,
        object_name: Optional[str]=None,
        do_print_pass_function_attr: Union[bool, dict[str, bool], None]=None
    ) -> None:
        # Set the log id for the pass function
        self.log_id = log._set_log_id(object_name, log.PASS_FUNCTION)

        # Check if do_print_pass_function_attr was not provided
        if do_print_pass_function_attr is None:
            self.do_print_pass_attr = {
                'forward': False,
                'backward': False,
                'update': False
            }

        # Else, check if do_print_pass_function_attr is a bool
        elif isinstance(do_print_pass_function_attr, bool):
            self.do_print_pass_attr = {
                'forward': do_print_pass_function_attr,
                'backward': do_print_pass_function_attr,
                'update': do_print_pass_function_attr
            }

        else:
            # do_print_pass_function_attr is a dict
            self.do_print_pass_attr = do_print_pass_function_attr

        # Initialize the base TensorFunction
        super().__init__(
            cache_parameters=cache_parameters,
            object_name=object_name, has_log_id=True
        )

        # Set the pass function name
        self.pass_name = forward_function_name
        
        # Set the forward function attributes
        self.forward_name = forward_function_name
        self.forward_function = forward_function_ptr
        (
            self.forward_kwargs,
            self.ref_kwargs,
            self.learnable_parameters,
            self.learnable_weights,
            self.forward_return_values,
            self.forward_output_keys
        ) = self._parse_function_parameters(
            function_parameters=forward_function_parameters,
            do_init_learnable_parameters=True
        )

        # Update the forward keyword arguments with the learnable parameters
        self.forward_kwargs |= util._get_update_dict(
            self.learnable_parameters, self.forward_kwargs
        )

        # Set the backward function attributes
        self.backward_name = backward_function_name
        self.backward_function = backward_function_ptr
        (
            self.backward_kwargs,
            _,
            _,
            _,
            self.backward_return_values,
            self.backward_output_keys
        ) = self._parse_function_parameters(
            function_parameters=backward_function_parameters,
            do_init_learnable_parameters=False
        )

        # Update the backward keyword arguments with the learnable parameters
        self.backward_kwargs |= util._get_update_dict(
            self.learnable_parameters, self.backward_kwargs
        )

        # Set the update function attributes
        self.update_name = update_function_name
        self.update_function = update_function_ptr
        (
            self.update_kwargs,
            _,
            _,
            _,
            self.update_return_values,
            self.update_output_keys
        ) = self._parse_function_parameters(
            function_parameters=update_function_parameters,
            do_init_learnable_parameters=False
        )

        # Set the update weight function attributes if provided
        self.update_weights_name = update_weights_function_name
        self.update_weights_function = update_weights_function_ptr
        (
            self.update_weights_kwargs,
            _,
            _,
            _,
            self.update_weights_return_values,
            self.update_weights_output_keys
        ) = self._parse_function_parameters(
            function_parameters=update_weights_function_parameters,
            do_init_learnable_parameters=False
        )

        # Initialize the gradients dict
        self.gradients = {}

        '''
        # SANITY CHECK
        if do_print_pass_function_attr:
            self._print(['cache', 'functions'])
        '''

    def _print(self,
        print_types: Union[str, list[str], None]=None,
        forward_kwargs: Optional[dict[str, Any]]=None,
        backward_kwargs: Optional[dict[str, Any]]=None,
        update_kwargs: Optional[dict[str, Any]]=None,
        kwargs: Optional[dict[str, Any]]=None
    ) -> None:
        """
        Print the pass function attributes.

        Args:
            print_types (str | list[str]): The types of pass function attributes
                to print
                NOTE: Passing in None for print_types prints all attributes.
            forward_kwargs (dict[str, Any]): The forward keyword arguments
            backward_kwargs (dict[str, Any]): The backward keyword arguments
            update_kwargs (dict[str, Any]): The update keyword arguments
            kwargs: (dict[str, Any]): The keyword arguments for the current function

        Return:
            None
        """
        # Convert single str print_types to list
        if isinstance(print_types, str):
            print_types = [print_types]

        print(f"\n{util.HARD_BANNER}\nPrinting attributes for {self.pass_name}...")

        # Check if printing the cache
        if print_types is None or 'cache' in print_types:

            # Print the cache
            util._print_dict(self.cache, 'cache')
        
        # Check if printing the general attributes
        if print_types is None or 'general' in print_types:
            # Print the reference keyword arguments
            util._print_dict(
                {
                    rf: v for k, (rf, v) in self.ref_kwargs.items()
                },
                'reference keyword arguments'
            )

            # Print the learnable parameters
            util._print_dict(self.learnable_parameters, 'learnable_parameters')

            # Print the learnable weights
            util._print_dict(self.learnable_weights, 'learnable weights')

        # Check if printing pass function names
        if print_types is None \
                        or 'functions' in print_types \
                        or 'function names' in print_types:
            print('\nforward function name: ' \
                f'{util._get_print_name(self.forward_name)}')
            print('\nbackward function name: ' \
                f'{util._get_print_name(self.backward_name)}')
            print('\nupdate function name: ' \
                f'{util._get_print_name(self.update_name)}')
            print('\nupdate weights function name: ' \
                f'{util._get_print_name(self.update_weights_name)}')

        # Check if printing the forward function attributes
        if print_types is None or 'forward' in print_types:

            # Print the forward function attributes
            print("\nforward function attributes...")

            # Initialize the forward keyword arguments if not provided
            if forward_kwargs is None:
                forward_kwargs = self.forward_kwargs

            # Print the forward keyword arguments
            util._print_dict(forward_kwargs, 'forward keyword arguments')

            # Print the forward return values
            util._print_dict(self.forward_return_values, 'forward return values')

            # Print the forward output keys
            util._print_list_tuple(self.forward_output_keys, 'forward output keys')

        # Check if printing the backward function attributes
        if print_types is None or 'backward' in print_types:

            # Print the backward function attributes
            print("\nbackward function attributes...")

            # Initialize the backward keyword arguments if not provided
            if backward_kwargs is None:
                backward_kwargs = self.backward_kwargs

            # Print the backward arguments
            util._print_dict(backward_kwargs, 'backward keyword arguments')

            # Print the backward return values
            util._print_dict(self.backward_return_values, 'backward return values')

            # Print the backward output keys
            util._print_list_tuple(self.backward_output_keys, 'backward output keys')

        # Check if printing the update function attributes
        if print_types is None or 'update' in print_types:

            # Print the update function attributes
            print("\nupdate function attributes...")

            # Initialize the update keyword arguments if not provided
            if update_kwargs is None:
                update_kwargs = self.update_kwargs

            # Print the update keyword arguments
            util._print_dict(update_kwargs, 'update keyword arguments')

        # Check if printing the current function attributes
        if print_types is None \
                    or 'current' in print_types \
                    or 'current function' in print_types:

            # Print the pass function current function keyword arguments
            util._print_dict(kwargs, 'layer current function keyword arguments')

            '''SANITY CHECK
            if kwargs is not None:
                for k, v in kwargs.items():
                    if isinstance(v, Tensor):
                        print(f'\n{k} max: {v.max().item()}')
                        print(f'\n{k} min: {v.min().item()}')
            '''
        if print_types is None \
                        or 'input' in print_types \
                        or 'input tensor' in print_types:
            if kwargs is not None:
                in_tensor = None
                if 'x' in kwargs.keys():
                    in_tensor = kwargs['x']
                elif 'upstream_grad' in kwargs.keys():
                    in_tensor = kwargs['upstream_grad']
                if in_tensor is not None:
                    print(f'\nInput tensor max: {in_tensor.max().item()}')
                    print(f'\nInput tensor min: {in_tensor.min().item()}')

        print(f'{util.HARD_BANNER}\n')

    def _update_cache(self,
        cache_updates: dict[str, Any],
        ignore_none=False
    ) -> None:
        """
        Update the pass function cache.

        Args:
            cache_updates (dict): Dict of cache updates
            ignore_none (bool): Boolean indicating if ignoring values equal to None

        Return:
            None
        """
        # Update the cache
        self.cache |= util._get_update_dict(cache_updates, ignore_none=ignore_none)

        # Update all keyword arguments dicts using the update cache
        self.forward_kwargs |= util._get_update_dict(
            self.cache, self.forward_kwargs, ignore_none
        )
        self.backward_kwargs |= util._get_update_dict(
            self.cache, self.backward_kwargs, ignore_none
        )
        self.update_kwargs |= util._get_update_dict(
            self.cache, self.update_kwargs, ignore_none
        )
        self.update_weights_kwargs |= util._get_update_dict(
            self.cache, self.forward_kwargs, ignore_none
        )

    def _forward(self,
        x: Optional[Tensor]=None,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False,
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform the forward pass function on the input tensor.

        Args:
            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            forward_output_values (tuple | dict): The forward function output values
                tuple or dict
        """
        # Initialize print types
        print_types = None

        # Set print types
        if self.do_print_pass_attr['forward']:
            print_types = 'input'
        
        # Initialize the output keys if not provided
        if output_keys is None:
            output_keys = self.forward_output_keys

        forward_output_values = self._run(
            function_name=self.forward_name,
            print_types=print_types,
            x=x,
            function_ptr=self.forward_function,
            kwargs=kwargs,
            base_kwargs=self.forward_kwargs,
            return_values=self.forward_return_values,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )

        # Update the forward return values
        if isinstance(forward_output_values, dict):
            self.forward_return_values = deepcopy(forward_output_values)
        else:
            self.forward_return_values = {
                k:v for k, v in zip(
                    self.forward_output_keys,
                    forward_output_values
                )
            }

        return forward_output_values
    
    def _backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False,
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform the backward pass function on the upstream gradient tensor.

        Args:
            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            backward_outpput_values (tuple | dict): The backward function output values
                tuple or dict
        """
        # Initialize print types
        print_types = None

        # Set print types
        if self.do_print_pass_attr['backward']:
            print_types = 'input'

        # Initialize the output keys if not provided
        if output_keys is None:
            output_keys = self.backward_output_keys

        backward_output_values = self._run(
            function_name=self.backward_name,
            print_types=print_types,
            upstream_grad=upstream_grad,
            function_ptr=self.backward_function,
            kwargs=kwargs,
            base_kwargs=self.backward_kwargs,
            return_values=self.backward_return_values,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )

        # Update the backward return values
        if isinstance(backward_output_values, dict):
            self.backward_return_values = deepcopy(backward_output_values)
        else:
            self.backward_return_values = {
                k:v for k, v in zip(
                    self.backward_output_keys,
                    backward_output_values
                )
            }

        # Use the backward return values to update the gradients
        self.gradients |= {
            k: v for k, v in self.backward_return_values.items()
                if '_grad' in k
        }
    
        return backward_output_values
    
    def _update(self,
        kwargs: Optional[dict[str, Any]]=None
    ) -> bool:
        """
        Perform the learnable parameters updates for the pass function.

        Args:
            kwargs (dict[str, Any]): Keyword arguments dict

        Return:
            Boolean indicating success with updating learnable parameters
        """
        # Make debug log for running the update function
        log._log_debug(
            f"Updating {util._get_print_name(self.forward_name)} "
            "learnable parameters...",
            self.log_id
        )

        # Initialize print types
        print_types = None

        # Set print types
        if self.do_print_pass_attr['update']:
            print_types = None

        # Iterate through the learnable parameters
        for lp_name, learnable_parameter in self.learnable_parameters.items():
            # Get the corresponding gradient to the learnable parameter
            #NOTE: Gradients are returned from the pass function backward function
            gradient_name = lp_name+'_grad'

            # Check if the gradient name doesn't exist in the backward return values
            if gradient_name not in self.gradients.keys():
                # Log error and return False since the gradient doesn't exist
                log._log_error(
                    f"Couldn't update {lp_name} since {gradient_name} doesn't exist!",
                    self.log_id
                )
                return False
            
            # Else, get the gradient
            gradient = self.backward_return_values[gradient_name]

            # Initialize the update output values
            update_output_values = None

            # Check if the update and update weights functions have different names
            #   and the learnable parameter is a learnable weight
            if self.update_name != self.update_weights_name \
                and lp_name in self.learnable_weights.keys():
                # Update the learnable parameter and gradient in the update
                #   function keyword arguments
                self.update_weights_kwargs['learnable_weight'] = learnable_parameter
                self.update_weights_kwargs['weight_gradient'] = gradient

                # Run the update function
                update_output_values = self._run(
                    print_types=print_types,
                    function_name=self.update_weights_name,
                    function_ptr=self.update_weights_function,
                    kwargs=kwargs,
                    base_kwargs=self.update_weights_kwargs,
                    return_values=self.update_weights_return_values,
                    output_keys=self.update_weights_output_keys
                )

                # Initialize the updated learnable weight
                updated_learnable_weight = None

                # Get the updated learnable weight
                if isinstance(update_output_values, dict):
                    updated_learnable_weight = \
                                    update_output_values['updated_learnable_weight']
                else:
                   updated_learnable_weight = \
                                    update_output_values[0]

                # Update the learnable weight from the output values
                self.learnable_parameters[lp_name] = updated_learnable_weight
                self.learnable_weights[lp_name] = updated_learnable_weight

            # Else, the learnable parameter is not a learnable weight
            else:
                # Update the learnable parameter and gradient in the update
                #   function keyword arguments
                self.update_kwargs['learnable_parameter'] = learnable_parameter
                self.update_kwargs['gradient'] = gradient

                # Run the update function
                update_output_values = self._run(
                    function_name=self.update_name,
                    function_ptr=self.update_function,
                    kwargs=kwargs,
                    base_kwargs=self.update_kwargs,
                    return_values=self.update_return_values,
                    output_keys=self.update_output_keys
                )

                # Initialize the updated learnable weight
                updated_learnable_parameter = None

                # Update the updated learnable parameter
                if isinstance(update_output_values, dict):
                    updated_learnable_parameter = \
                                    update_output_values['updated_learnable_parameter']
                else:
                   updated_learnable_parameter = \
                                    update_output_values[0]

                # Update the learnable parameter from the output values
                self.learnable_parameters[lp_name] = updated_learnable_parameter

                # Check if the learnable parameter is in learnable weights
                if lp_name in self.learnable_weights.keys():
                    self.learnable_weights[lp_name] = updated_learnable_parameter

            # Update the update return values
            if isinstance(update_output_values, dict):
                self.update_return_values = deepcopy(update_output_values)
            else:
                self.update_return_values = {
                    k:v for k, v in zip(
                        self.update_output_keys,
                        update_output_values
                    )
                }
                
        # Make debug log for successfully running the update function
        log._log_debug(
            f"Successfully updated the learnable parameters!",
            self.log_id
        )
            
        # Return True since all learnable parameters were successfully updated
        return True
            

# ========================= TENSOR FUNCTION LOOKUP ============================


# A dict of tensor functions and their respective parameter keys
tensor_functions = {
    # Activation functions
    'relu': (
        activation._relu,
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
        activation._relu_backward,
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
        attention._multi_head_attention,
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
                'W_q': ('embedding_size', 'keys_embedding_size'),
                'W_k': ('embedding_size', 'keys_embedding_size'),
                'W_v': ('embedding_size', 'values_embedding_size'),
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
        attention._multi_head_attention_backward,
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
        attention._multi_head_cross_attention,
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
                'W_q': ('embedding_size', 'keys_embedding_size'),
                'W_k': ('embedding_size', 'keys_embedding_size'),
                'W_v': ('embedding_size', 'values_embedding_size'),
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
        attention._multi_head_masked_self_attention,
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
                'W_q': ('embedding_size', 'keys_embedding_size'),
                'W_k': ('embedding_size', 'keys_embedding_size'),
                'W_v': ('embedding_size', 'values_embedding_size'),
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
        convolution._conv2d,
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
                'b': ('num_out_features',)
            },
            'learnable_weights': ('W',),
                
            'return_value_keys': ('conv_out',),
            'backward_function_name': 'conv2d_backward'
        }
    ),
    'conv2d_backward': (
        convolution._conv2d_backward,
        {
            'argument_keys': (
                'upstream_grad', 'conv_in',
                'W', 'b',
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
        image._get_image_tensor,
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
        image._get_images_tensor,
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
        loss._binary_cross_entropy_loss,
        {
            'argument_keys': (
                'logits', 'true_labels',
                'loss_reduction_type',
                'reg_type', 'reg_strength',
                'learnable_weights'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': (
                'scalar_loss', 'probabilities', 'sigmoid_out'),
            'backward_function_name': 'binary_cross_entropy_loss_backward'
        }
    ),
    'binary_cross_entropy_loss_backward': (
        loss._binary_cross_entropy_loss_backward,
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
        loss._cross_entropy_loss,
        {
            'argument_keys': (
                'logits', 'true_labels',
                'loss_reduction_type',
                'reg_type', 'reg_strength',
                'learnable_weights'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': (
                'scalar_loss', 'probabilities', 'softmax_out'),
            'backward_function_name': 'cross_entropy_loss_backward'
        }
    ),
    'cross_entropy_loss_backward': (
        loss._cross_entropy_loss_backward,
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
        normalization._layer_norm,
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
        normalization._layer_norm_backward,
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
        pool._pool,
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
        pool._unpool,
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
        projection._feed_forward,
        {
            'argument_keys': ('x', 'W_1', 'W_2', 'b_1', 'b_2'),
            'argument_key_tuples': ( ('x', 'ff_in'), ),

            'learnable_parameter_shapes': { # Init W_1, W_2, b_1, b_2
                'W_1': ('ff_embedding_size', 'embedding_size'),
                'W_2': ('embedding_size', 'ff_embedding_size'),
                'b_1': ('embedding_size',),
                'b_2': ('embedding_size',)
            },
            'learnable_weights': ('W_1', 'W_2'),

            'return_value_keys': ('ff_out', 'relu_out', 'relu_in'),
            'backward_function_name': 'feed_forward_backward'
        }
    ),
    'feed_forward_backward': (
        projection._feed_forward_backward,
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

    'linear_projection': (
        projection._lin_proj,
        {
            'argument_keys': ('x', 'W', 'b'),
            'argument_key_tuples': ( ('x', 'proj_in'), ),

            'learnable_parameter_shapes': { # Init W, b
                'W': ('embedding_size', 'pre_embedding_size'),
                'b': ('embedding_size',)
            },
            'learnable_weights': ('W',),

            'return_value_keys': ('proj_out',),
            'backward_function_name': 'linear_projection_backward'
        }
        
    ),
    'linear_projection_backward': (
        projection._lin_proj_backward,
        {
            'argument_keys': ('upstream_grad', 'proj_in', 'W', 'b'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('proj_in_grad', 'W_grad', 'b_grad')
        }
    ),

    # Regularization functions
    'dropout': (
        regularization._dropout,
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
        regularization._dropout_backward,
        {
            'argument_keys': ('upstream_grad', 'dropout', 'dropout_mask'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('dropout_in_grad',)
        }
    ),

    'ridge_regression': (
        regularization._ridge_regression,
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
        reshape._flatten,
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
        reshape._unflatten,
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
        residual._residual_add,
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
        residual._residual_add_backward,
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
        token._get_tokens_tensor,
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
        token._get_embedded_tokens,
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
        token._get_embedded_tokens_backward,
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
        update._basic_update,
        {
            'argument_keys': ('learnable_parameter', 'gradient', 'learning_rate'),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('updated_learnable_parameter',)
        }
    ),

    'ridge_regression_update': (
        update._ridge_regression_update,
        {
            'argument_keys': (
                'learnable_weight', 'weight_gradient',
                'reg_strength', 'learning_rate'
            ),
            'argument_key_tuples': None,
            'learnable_parameter_shapes': None,
            'learnable_weights': None,
            'return_value_keys': ('updated_learnable_weight',)
        }
    )
}

reg_update_function_names = {
    'ridge': 'ridge_regression_update'
}

update_and_weights_update_function_names = {
    'basic_update': ('basic_update', 'basic_update'),
    'ridge_regression_update': ('basic_update', 'ridge_regression_update')
}

def _get_update_function_name(reg_type: Optional[str]=None) -> Optional[str]:
    """
    Get the update function name for the regularization type.

    Args:
        reg_type (str): The regularization type

    Return:
        The update function name
    """
    # Check if the regularization type was not provided
    if reg_type is None:
        return None
    
    # Check if the regularization type is valid
    if reg_type in reg_update_function_names.keys():
        return reg_update_function_names[reg_type]
    
    # Else, return None

def _get_update_and_update_weights_function_names(
        update_function_name: Optional[str]=None
) -> Optional[tuple[str, str]]:
    """
    Get the update and weights update function names for the update function.

    Args:
        update_function_name (str): The name of the update function

    Return:
        The update and update weights function names
    """
    # Check if the update function name was not provided
    if update_function_name is None:
        return None
    
    # Check if the regularization type is valid
    if update_function_name in update_and_weights_update_function_names.keys():
        return update_and_weights_update_function_names[update_function_name]
    
    # Else, return None

def _get_tensor_function(
    # If getting the TensorFunction from pointer and arguments
    tensor_function_ptr: Optional[Callable]=None,
    tensor_function_args: Optional[dict[str, Any]]=None,
    tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

    # If getting the TensorFunction from name (usually pass functions)
    tensor_function_name: Optional[str]=None,
    tensor_update_function_name: Optional[str]=None,
    tensor_function_cache_parameters: Optional[dict[str, Any]]=None,

    tensor_function_seq_num=-1,
    do_print_tensor_function_attr: Union[bool, dict[str, bool], None]=None
) -> Union[
    TensorFunction,
    PassFunction,
None]:
    """
    Return a TensorFunction object.
    NOTE: If a tensor function doesn't have a corresponding backward function,
        then it can't make use of the update function.

    Args:
        tensor_function (Tensor): The tensor function pointer
        tensor_function_arguments (dict[str, Any]): Dict of tensor function arguments
        tensor_function_return_value_keys (tuple[str, ...]): Tuple of tensor
            function return value keys
        tensor_function_name (str): The tensor function name
        tensor_update_function_name (str): The tensor update function name
        tensor_function_cache_parameters (dict[str, Any]): The tensor function
            cache parameters
        tensor_function_num (int): The tensor function number (to distinguish
            tensor functions with the same name)
        do_print_tensor_function_attr (bool | dict[str, bool]): Boolean or dict
            of booleans indicating if printing tensor function attributes

    Return:
        A TensorFunction object
    """
    # Initialize the tensor function parameters
    tensor_function_parameters = None

    # Get the tensor function by function name if provided
    if tensor_function_ptr is None and tensor_function_name is not None:
        tensor_function_ptr_params = get_tf_pointer_parameters(tensor_function_name)
        if tensor_function_ptr_params is not None:
            tensor_function_ptr, tensor_function_parameters = \
                            tensor_function_ptr_params
        assert(tensor_function_ptr is not None)
        assert(tensor_function_parameters is not None)

        # Get the tensor backward function by backward function name if provided
        tensor_backward_function_name = None
        if 'backward_function_name' in tensor_function_parameters.keys():
            tensor_backward_function_name = \
                            tensor_function_parameters.pop('backward_function_name')
            
        # Set the backward and update functions if applicable
        if tensor_backward_function_name is not None:
            tensor_backward_function_ptr, tensor_backward_function_parameters = \
                            tensor_functions[tensor_backward_function_name]

            # Get the tensor update function name if not provided
            if tensor_update_function_name is None:
                tensor_update_function_name = 'basic_update'

            # Separate the standard update from the weights-specific update
            tensor_update_function_names = \
                            _get_update_and_update_weights_function_names(
                                tensor_update_function_name
                            )
            
            if tensor_update_function_names is not None:
                tensor_update_function_name, tensor_update_weights_function_name = \
                                tensor_update_function_names
                
            else:
                tensor_update_function_name, tensor_update_weights_function_name = \
                            update_and_weights_update_function_names['basic_update']
            
            # Get the tensor update function and function parameters
            tensor_update_function_ptr, tensor_update_function_parameters = \
                            tensor_functions[tensor_update_function_name]
            
            # Get the tensor update weights function and function parameters
            tensor_update_weights_function_ptr, \
                            tensor_update_weights_function_parameters = \
                                tensor_functions[tensor_update_weights_function_name]
                
            # Get the pass function label
            pass_function_label = \
                            f'{util._get_print_name(tensor_function_name)} function'

            # Check if a valid tensor function num was provided
            if tensor_function_seq_num > 0:
                pass_function_label += f' #{tensor_function_seq_num}'
            
            # Return the PassFunction with the specified forward, backward,
            #   and update functions
            return PassFunction(
                object_name=pass_function_label,
                forward_function_name=tensor_function_name,
                forward_function_ptr=tensor_function_ptr,
                forward_function_parameters=tensor_function_parameters,
                backward_function_name=tensor_backward_function_name,
                backward_function_ptr=tensor_backward_function_ptr,
                backward_function_parameters=tensor_backward_function_parameters,
                update_function_name=tensor_update_function_name,
                update_function_ptr=tensor_update_function_ptr,
                update_function_parameters=tensor_update_function_parameters,
                update_weights_function_name=tensor_update_weights_function_name,
                update_weights_function_ptr=tensor_update_weights_function_ptr,
                update_weights_function_parameters=\
                    tensor_update_weights_function_parameters,
                cache_parameters=tensor_function_cache_parameters,
                do_print_pass_function_attr=do_print_tensor_function_attr
            )
        
    # Get the tensor function label from name
    # Check if the tensor function name was not provided
    if tensor_function_name is None:
        tensor_function_name = f'{log.UNLABELED} TENSOR FUNCTION'
    tensor_function_label = util._get_print_name(tensor_function_name)

    # Check if a valid tensor function num was provided
    if tensor_function_seq_num > 0:
        tensor_function_label += f' #{tensor_function_seq_num}'

    # Return a TensorFunction only if the function exists
    if tensor_function_ptr is not None:
        return TensorFunction(
            object_name=tensor_function_label,
            function_name=tensor_function_name,
            function_ptr=tensor_function_ptr,
            function_arguments=tensor_function_args,
            function_return_value_keys=tensor_function_return_value_keys,
            function_parameters=tensor_function_parameters,
            cache_parameters=tensor_function_cache_parameters,
            do_print_tensor_function_attr=do_print_tensor_function_attr
        )
    
    # Else, return None


def get_tf_pointer_parameters(
    tensor_function_name: str
) -> Optional[tuple[Callable, dict[str, Any]]]:
    """
    Get tensor function pointer and parameters from name.

    Args:
        tensor_function_name (str): The name of the tensor function

    Return:
        Tuple of tensor function pointer and parameters dict
    """
    if tensor_function_name in tensor_functions:
        # Get the tensor function pointer and parameters
        tensor_function_ptr, tensor_function_params = \
                        tensor_functions[tensor_function_name]
        # Copy the tensor function parameters
        tensor_function_params_copy = {
            k: v for k, v in tensor_function_params.items()
        }
        return tensor_function_ptr, tensor_function_params_copy