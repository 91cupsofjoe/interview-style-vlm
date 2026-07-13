"""
This module contains the DataSet class and its children.
"""
from torch.utils.data import Dataset
from typing import Optional, Any
from collections.abc import Callable

import torch
from torch import Tensor

from log import logger as log
from tensor_function import util
from tensor_function.tensor_function import TensorFunction, _get_tensor_function


# ============================== DATASET CLASS ================================

class DataSet(Dataset):
    """
    This is the base dataset class.
    """
    def __init__(self,
        # If loading data sources
        data_sources: Optional[tuple[Any, ...]]=None,

        # If loading examples data
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        do_load_examples_data=True,

        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,

        # If loading labels data
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,
        do_load_labels_data=True,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None,

        # General parameters
        training_test_split=-1.0,
        use_test_only=False,
        object_name=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.DATASET)

        # Make debug log for loading the base dataset
        log._log_debug(
            f"Loading the base dataset...",
            log.DATASET_MODULE
        )

        # Initialize the examples data, tensors, and tensor function
        self.training_examples_data: Optional[list[Any]] = None
        self.training_examples_tensor: Optional[Tensor] = None
        self.test_examples_data: Optional[list[Any]] = None
        self.test_examples_tensor: Optional[Tensor] = None
        self.examples_data_tensor_function = None

        # Initialize the labels data, tensors, and tensor function
        self.training_labels_data: Optional[list[Any]] = None
        self.training_labels_tensor: Optional[Tensor] = None
        self.test_labels_data: Optional[list[Any]] = None
        self.test_labels_tensor: Optional[Tensor] = None
        self.labels_data_tensor_function = None

        # Initialize the load success bool
        load_data_success = False

        # Check if data sources, examples data, or labels data are provided
        if data_sources is not None \
                        or examples_data is not None \
                        or labels_data is not None:
            # Get keyword arguments without the object name and log id bool
            kwargs = locals()
            kwargs.pop('object_name')
            kwargs.pop('has_log_id')
            # Don't forget to pop self
            kwargs.pop('self')

            # Load data into the base dataset
            load_data_success = self.load_data(**kwargs)

        # Check if successfully loaded data into the base dataset
        if load_data_success:
            # Make debug log for successfully loading the base dataset with data
            log._log_debug(
                "Successfully loaded the base dataset with data!",
                log.DATASET_MODULE
            )

        else:
            # Make debug log for successfully loading the base dataset without data
            log._log_debug(
                f"Successfully loaded the base dataset without data!",
                self.log_id
            )

    def __len__(self):
        """
        Return the number of training examples in the dataset.
        """
        if self.training_examples_data is not None:
            return len(self.training_examples_data)

    def __getitem__(self,
        key: Optional[Any]=None,
        use_tensors=False,
        use_test_data=False
    ) -> Optional[tuple]:
        """
        Return an example and label pair from the dataset by index.
        """
        # Set key to ... if not provided
        if key is None:
            key = ...

        # Return the tensors if specified
        if use_tensors:
            
            # Return from test tensors if specified
            # Make sure the test tensors exist
            if use_test_data and \
                (self.test_examples_tensor is not None \
                        and self.test_labels_tensor is not None):
                
                # Make debug log for successfully getting test tensors
                log._log_debug(
                    "Successfully retrieved test data tensors!",
                    self.log_id
                )

                # Check if key is int or slice
                if isinstance(key, (int, slice)):
                    return self.test_examples_tensor[key], \
                                    self.test_labels_tensor[key]
                else:
                    # Key is Ellipsis, return the entire tensors
                    return self.test_examples_tensor, \
                                    self.test_labels_tensor
            
            # Else, return from training tensors
            # Make sure the training tensors exist
            elif not use_test_data \
                    and (self.training_examples_tensor is not None \
                        and self.training_labels_tensor is not None):
                
                # Make debug log for successfully getting training tensors
                log._log_debug(
                    "Successfully retrieved training data tensors!",
                    self.log_id
                )

                # Check if key is int or slice
                if isinstance(key, (int, slice)):
                    return self.training_examples_tensor[key], \
                                    self.training_labels_tensor[key]
                else:
                    # Key is Ellipsis, return the entire tensors
                    return self.training_examples_tensor, \
                                    self.training_labels_tensor
                        
            # Else, return the data
            elif not use_tensors:

                # Return from test data if specified
                # Make sure the test data exists
                if use_test_data and \
                    (self.test_examples_data is not None \
                            and self.test_labels_data is not None):
                    
                    # Make debug log for successfully getting test data
                    log._log_debug(
                        "Successfully retrieved test data!",
                        self.log_id
                    )

                    # Check if key is int or slice
                    if isinstance(key, (int, slice)):
                        return self.test_examples_data[key], \
                                        self.test_labels_data[key]
                    else:
                        # Key is Ellipsis, return the entire data
                        return self.test_examples_data, \
                                        self.test_labels_data
                
                # Else, return from training data
                # Make sure the training data exists
                elif not use_test_data \
                        and (self.training_examples_data is not None \
                            and self.training_labels_data is not None):
                    
                    # Make debug log for successfully getting training data
                    log._log_debug(
                        "Successfully retrieved training data!",
                        self.log_id
                    )

                    # Check if key is int or slice
                    if isinstance(key, (int, slice)):
                        return self.training_examples_data[key], \
                                        self.training_labels_data[key]
                    else:
                        # Key is Ellipsis, return the entire tensors
                        return self.test_examples_data, \
                                        self.test_labels_data
                    
        # Make debug log for failure to get data / tensors
        log._log_debug(
            "Failed to get any data or tensors!",
            self.log_id
        )
            
        # Return None
        return None
    
    def get_training_data(self,
        key: Optional[Any]=None,
        use_tensors=True,

        # If loading the training data from files
        data_sources: Optional[tuple[Any, Any]]=None,

        # If loading the training data from lists
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,

        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None,

        # General parameters
        training_test_split=-1.0,
    ) -> Optional[tuple]:
        """
        Return training examples and training labels tensors
            by index, slice, or Ellipsis.

        Args:
            key (Any): Index, slice, or Ellipsis
            use_tensors (bool): Boolean indicating if getting training data tensors
            NOTE: See load_data for the info on the rest of the args

        Return:
            Tuple of tensors
        """
        # Set the training data name
        data_name = 'training data'

        # Check if getting the training data tensors
        if use_tensors:
            data_name += ' tensors'
        
        # Make debug log for getting the training data / tensors
        log._log_debug(
            f"Getting {data_name}...",
            self.log_id
        )

        # Check if data sources, examples data, or labels data were provided
        if data_sources is not None \
                or (examples_data is not None or labels_data is not None):
            # Load the training data and check if it fails
            if not self._load_training_test_data(
                data_sources=data_sources,
                examples_data=examples_data,
                examples_data_name=examples_data_name,
                examples_data_tensor_function_ptr=examples_data_tensor_function_ptr,
                examples_data_tensor_function_args=\
                    examples_data_tensor_function_args,
                examples_data_tensor_function_return_value_keys=\
                    examples_data_tensor_function_return_value_keys,
                examples_data_tensor_function_name=examples_data_tensor_function_name,
                labels_data=labels_data,
                labels_data_name=labels_data_name,
                labels_data_tensor_function_ptr=labels_data_tensor_function_ptr,
                labels_data_tensor_function_args=labels_data_tensor_function_args,
                labels_data_tensor_function_return_value_keys=\
                    labels_data_tensor_function_return_value_keys,
                labels_data_tensor_function_name=labels_data_tensor_function_name,
                training_test_split=training_test_split
            ):
                # Make error log for failure to get training data since loading
                #   the input data failed
                log._log_error(
                    f"Could not retrieve {data_name} since loading the "
                    "input data failed!",
                    self.log_id
                )
                return None

        # Return the training data (lists or tensors)
        return self.__getitem__(
            key=key,
            use_tensors=use_tensors
        )
        
    def get_test_data(self,
        key: Optional[Any]=None,
        use_tensors=True,

        # If loading the training data from files
        data_sources: Optional[tuple[Any, Any]]=None,

        # If loading the training data from lists
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,

        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None
    ) -> Optional[tuple]:
        """
        Return test examples and test labels tensors
            by index, slice, or Ellipsis.

        Args:
            key (Any): Index, slice, or Ellipsis
            use_tensors (bool): Boolean indicating if getting training data tensors
            NOTE: See load_data for the info on the rest of the args

        Return:
            Tuple of tensors
        """
        # Set the test data name
        data_name = 'test data'

        # Check if getting the test data tensors
        if use_tensors:
            data_name += ' tensors'
        
        # Make debug log for getting the test data / tensors
        log._log_debug(
            f"Getting {data_name}...",
            self.log_id
        )

        # Check if data sources, examples data, or labels data were provided
        if data_sources is not None \
                or (examples_data is not None or labels_data is not None):
            # Load test data and check if it fails
            if not self.load_test_data(
                data_sources=data_sources,
                examples_data=examples_data,
                examples_data_name=examples_data_name,
                examples_data_tensor_function_ptr=examples_data_tensor_function_ptr,
                examples_data_tensor_function_args=\
                    examples_data_tensor_function_args,
                examples_data_tensor_function_return_value_keys=\
                    examples_data_tensor_function_return_value_keys,
                examples_data_tensor_function_name=examples_data_tensor_function_name,
                labels_data=labels_data,
                labels_data_name=labels_data_name,
                labels_data_tensor_function_ptr=labels_data_tensor_function_ptr,
                labels_data_tensor_function_args=labels_data_tensor_function_args,
                labels_data_tensor_function_return_value_keys=\
                    labels_data_tensor_function_return_value_keys,
                labels_data_tensor_function_name=labels_data_tensor_function_name
            ):
                # Make error log for failure to get test data since loading the
                #   input data failed
                log._log_error(
                    f"Could not retrieve {data_name} since loading the "
                    "input data failed!",
                    self.log_id
                )
                return None

        # Return the test data (lists or tensors)
        return self.__getitem__(
            key=key,
            use_tensors=use_tensors,
            use_test_data=True
        )

    def _set_data_tensor_functions(self,
        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None
    ) -> bool:
        """
        Set either the examples tensor function, labels tensor function, or
            both for the dataset.
        NOTE: This must be done in addition to loading sample data if the
            sample data is loaded post dataset initialization.

        Args:
            NOTE: See load_data for info on the args.

        Return:
            Boolean indicating success with setting the data tensor function(s)
        """
        set_success = False

        # Check if setting the examples data tensor function
        if examples_data_tensor_function_ptr is not None \
                        or examples_data_tensor_function_name is not None:
            
            # Log setting the examples data tensor function
            examples_function_name = ''
            if examples_data_tensor_function_name is not None:
                examples_function_name = f' [{examples_data_tensor_function_name}]'

            # Make debug log for setting the examples data tensor function
            log._log_debug(
                "Setting the examples data tensor function"
                f"{util._get_print_name(examples_function_name)}...",
                self.log_id
            )
        
            examples_data_tensor_function = _get_tensor_function(
                tensor_function_ptr=examples_data_tensor_function_ptr,
                tensor_function_args=examples_data_tensor_function_args,
                tensor_function_return_value_keys=\
                    examples_data_tensor_function_return_value_keys,
                tensor_function_name=examples_data_tensor_function_name
            )
            # Check if the examples data tensor exists
            if examples_data_tensor_function_ptr is not None:
                # Set the examples data tensor and set set_success to True
                self.examples_data_tensor_function = examples_data_tensor_function
                set_success = True

                # Make debug log for successfully setting the examples data
                #   tensor function
                log._log_debug(
                    f"Successfully set the examples data tensor function!",
                    self.log_id
                )

            else:
                # Make warning log for failure to set the examples data tensor
                #   function, and return False
                log._log_warning(
                    "Failed to set the examples data tensor function!",
                    self.log_id
                )
                return False

        # Check if setting the labels data tensor function
        if labels_data_tensor_function_ptr is not None \
                        or labels_data_tensor_function_name is not None:
            
            # Log setting the labels data tensor function
            labels_function_name = ''
            if labels_data_tensor_function_name is not None:
                labels_function_name = f' [{labels_data_tensor_function_name}]'

            # Make debug log for setting the labels data tensor function
            log._log_debug(
                "Setting the labels data tensor function"
                f"{util._get_print_name(labels_function_name)}...",
                self.log_id
            )

            labels_data_tensor_function = _get_tensor_function(
                tensor_function_ptr=labels_data_tensor_function_ptr,
                tensor_function_args=labels_data_tensor_function_args,
                tensor_function_return_value_keys=\
                    labels_data_tensor_function_return_value_keys,
                tensor_function_name=labels_data_tensor_function_name
            )
            # Check if the labels data tensor exists
            if labels_data_tensor_function is not None:
                # Set the labels data tensor and set set_success to True
                self.labels_data_tensor_function = labels_data_tensor_function
                set_success = True

                # Make debug log for successfully setting the labels data tensor function
                log._log_debug(
                    f"Successfully set the labels data tensor function!",
                    self.log_id
                )

            else:
                # Make warning log for failure to set the labels data tensor
                #   function, and return False
                log._log_warning(
                    "Failed to set the labels data tensor function!",
                    self.log_id
                )
                return False

        return set_success
    
    def _read_data(self,
        data_sources: tuple[Any, ...]
    ) -> tuple[
            Optional[Any],
            Optional[Any]
        ]:
        """
        Get data samples from reading the data sources. The dataset subclass
            will overwrite this method.
        """
        return None, None
    
    def load_data(self,
        # If loading data sources
        data_sources: Optional[tuple[Any, Any]]=None,

        # If loading examples data
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        do_load_examples_data=True,

        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,

        # If loading labels data
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,
        do_load_labels_data=True,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None,

        # General parameters
        training_test_split=-1.0,
        use_test_only=False
    ) -> bool:
        """
        Load training data.
        NOTE: This will set the lists and tensors for the training examples and
            training labels data.

        Args:
            data_sources (tuple[Any]): The data sources to load data from

            NOTE: examples data args:

            examples_data (list[Any]): List of examples
            examples_data_name (str): The name for the examples data
            do_load_examples_data (bool): Boolean indicating if loading the
                examples data

            examples_data_tensor_function_ptr (Callable): The examples data tensor
                function pointer
            examples_data_tensor_function_args (dict[str, Any]): Dict of
                examples tensor function arguments
            examples_data_tensor_function_return_value_keys (tuple[str, ...]): Tuple
                of examples tensor function return value keys

            examples_data_tensor_function_name (str): The name for the examples
                data tensor function

            NOTE: labels data args:

            labels_data (list[Any]): List of labels
            labels_data_name (str): The name for the labels
            do_load_labels_data (bool): Boolean indicating if loading the
                labels data

            labels_data_tensor_function_ptr (Callable): The labels data tensor
                function pointer
            labels_data_tensor_function_args (dict[str, Any]): Dict of
                labels tensor function arguments
            labels_data_tensor_function_return_value_keys (tuple[str, ...]): Tuple
                of labels tensor function return value keys
            
            labels_data_tensor_function_name (str): The name for the labels
                data tensor function

            NOTE: General args:
            
            training_test_split (int): The splitting point between the
                training and test data
            use_test_only (bool): Boolean indicating if strictly loading test data

        Return:
            Boolean indicating if loading data was successful
        """
        # Make debug log for loading sample data
        log._log_info(
            "Loading the sample data...",
            self.log_id
        )
        load_success = False

        # If data sources are provided, get examples and labels data from them
        if data_sources is not None:
            examples_data, labels_data = self._read_data(data_sources)

        # Load the training examples and test examples from the examples data
        if do_load_examples_data and examples_data is not None:

            # Check if loading examples data is unsuccessful
            if not self._load_examples_data(
                examples_data=examples_data,
                examples_data_name=examples_data_name,
                examples_data_tensor_function_ptr=examples_data_tensor_function_ptr,
                examples_data_tensor_function_args=\
                    examples_data_tensor_function_args,
                examples_data_tensor_function_return_value_keys=\
                    examples_data_tensor_function_return_value_keys,
                examples_data_tensor_function_name=examples_data_tensor_function_name,
                training_test_split=training_test_split,
                use_test_only=use_test_only
            ):
                # Make warning log for failure to load examples data, and return False
                log._log_warning(
                    "Failed to load examples data!",
                    self.log_id
                )
                return False
            
            # Else, set load_success to True
            load_success = True

        # Load the training labels and test labels from the labels data
        if do_load_labels_data and labels_data is not None:

            # Check if loading labels data is unsuccessful
            if not self._load_labels_data(
                labels_data=labels_data,
                labels_data_name=labels_data_name,
                labels_data_tensor_function_ptr=labels_data_tensor_function_ptr,
                labels_data_tensor_function_args=labels_data_tensor_function_args,
                labels_data_tensor_function_return_value_keys=\
                    labels_data_tensor_function_return_value_keys,
                labels_data_tensor_function_name=labels_data_tensor_function_name,
                training_test_split=training_test_split,
                use_test_only=use_test_only
            ):   
                # Make warning log for failure to load labels data, and return False
                log._log_warning(
                    "Failed to load labels data!",
                    self.log_id
                )
                return False
            
            # Else, set load_success to True
            load_success = True
            
        # Return True if examples and/or labels data successfully loaded,
        #   False otherwise
        return load_success
    
    def load_training_data(self,
        # If loading data sources
        data_sources: Optional[tuple[Any, Any]]=None,

        # If loading examples data
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        do_load_examples_data=True,

        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,

        # If loading labels data
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,
        do_load_labels_data=True,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None,
    ) -> bool:
        """
        Load training data.
        NOTE: This will set the lists and tensors for the training examples and
            training labels data.

        Args:
            NOTE: See load_data for info on the args.

        Return:
            load_success (bool): Boolean indicating if loading data was successful
        """
        # Make debug log for loading training data
        log._log_debug(
            "Loading training data...",
            self.log_id
        )

        load_success = self.load_data(**locals())

        # Make debug log for successfully loading training data
        if load_success:
            log._log_debug(
                "Successfully loaded training data!",
                self.log_id
            )

        else:
            # Make warning log for failure to load training data
            log._log_warning(
                "Failed to load training data!",
                self.log_id
            )

        return load_success
    
    def load_test_data(self,
        # If loading data sources
        data_sources: Optional[tuple[Any, Any]]=None,

        # If loading examples data
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        do_load_examples_data=True,

        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,

        # If loading labels data
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,
        do_load_labels_data=True,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None
    ) -> bool:
        """
        Load test data.
        NOTE: This will set the lists and tensors for the test examples and
            test labels data.

        Args:
            NOTE: See load_data for info on the args.

        Return:
            load_success (bool): Boolean indicating if loading data was successful
        """
        # Make debug log for loading test data
        log._log_debug(
            "Loading test data...",
            self.log_id
        )

        load_success = self.load_data(
            **locals(),
            training_test_split=-1,
            use_test_only=True
        )

        # Make debug log for successfully loading test data
        if load_success:
            log._log_debug(
                "Successfully loaded test data!",
                self.log_id
            )

        else:
            # Make warning log for failure to load test data
            log._log_warning(
                "Failed to load test data!",
                self.log_id
            )

        return load_success
    
    def _load_training_test_data(self,
        # If loading data sources
        data_sources: Optional[tuple[Any, Any]]=None,

        # If loading examples data
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        do_load_examples_data=True,

        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,

        # If loading labels data
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,
        do_load_labels_data=True,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None,
        training_test_split=.6
    ) -> bool:
        """
        Load training and test data.
        NOTE: This will split the input data into training and test data sets.

        Args:
            NOTE: See load_data for info on the args.

        Return:
            load_success (bool): Boolean indicating if loading data was successful
        """
        # Initialize the data name for the training/test data
        data_name = 'training and test data'

        # Make debug log for loading training and test data
        log._log_debug(
            "Loading training and/or test data...",
            self.log_id
        )

        # Load training data if training test split is -1
        if training_test_split == -1:
            load_success = self.load_training_data(**locals())

        else:
            # Make debug log for loading training and test data
            log._log_debug(
                "Loading training and test data...",
                self.log_id
            )

            load_success = self.load_data(**locals())

            # Make debug log for successfully loading training and test data
            if load_success:
                log._log_debug(
                    "Successfully loaded training and/test data!",
                    self.log_id
                )

            else:
                # Make warning log for failure to load training and test data
                log._log_warning(
                    "Failed to load training and test data!",
                    self.log_id
                )

        return load_success
    
    def _load_examples_data(self,
        examples_data: list[Any],
        examples_data_name: Optional[str]=None,

        # If loading examples data by tensor function pointer
        examples_data_tensor_function_ptr: Optional[Callable]=None,
        examples_data_tensor_function_args: Optional[dict[str, Any]]=None,
        examples_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,
        
        # If loading examples data by tensor function name
        examples_data_tensor_function_name: Optional[str]=None,
        
        training_test_split=-1.0,
        use_test_only=False
    ) -> bool:
        """
        Load the examples training and test data tensors.

        Args:
            NOTE: See load_data for info on the args.

        Return:
            Boolean indicating success with loading the examples data
        """
        # Set the data name
        data_name = 'examples data'
        if examples_data_name is not None:
            data_name = examples_data_name

        # Make info log for loading examples data
        log._log_info(
            f"Loading examples data [{util._get_print_name(data_name)}]...",
            self.log_id
        )

        # If only loading test data, set the training test split to -1
        if use_test_only:
            training_test_split = -1

        # Check if setting the examples data tensor function failed
        if not self._set_data_tensor_functions(
            examples_data_tensor_function_ptr=examples_data_tensor_function_ptr,
            labels_data_tensor_function_args=examples_data_tensor_function_args,
            labels_data_tensor_function_return_value_keys=\
                examples_data_tensor_function_return_value_keys,
            labels_data_tensor_function_name=examples_data_tensor_function_name
        ):
            # Make warning log for failure to load examples data since setting the
            #   examples data tensor function failed, and return False
            log._log_error(
                "Could not load examples data since setting the examples data "
                "tensor function failed!",
                self.log_id
            )
            return False

        # Get the examples data tensors
        examples_data_tensors = self._get_training_test_tensors(
            sample_data=examples_data,
            sample_data_name=examples_data_name,
            sample_data_tensor_function=self.examples_data_tensor_function,
            training_test_split=training_test_split
        )
        
        # Check if the examples data tensors exist
        if examples_data_tensors is not None:
            # Get the examples data training and test tensors
            examples_data_training_tensor, examples_data_test_tensor = \
                            examples_data_tensors
            # Make sure the examples data training tensor exists
            if examples_data_training_tensor is not None:
                # Check if only loading the test data
                if use_test_only:
                    # Store the test examples data
                    self.test_examples_data = examples_data
                    # Store the test examples tensor
                    # NOTE: The training tensor becomes the test tensor
                    # Check if the examples data test tensor exists
                    if examples_data_test_tensor is not None:
                        self.test_examples_tensor = torch.stack([
                            examples_data_training_tensor, examples_data_test_tensor
                        ], dim=0)
                    else:
                        self.test_examples_tensor = examples_data_training_tensor

                else:
                    # Split the examples data
                    training_start, training_stop, test_start, test_stop = \
                                    util._get_collection_indices_by_split(
                                        examples_data, training_test_split
                                    )
                    self.training_examples_data = \
                                    examples_data[training_start:training_stop]
                    self.test_examples_data = examples_data[test_start:test_stop]

                    # Store the examples data tensors
                    self.training_examples_tensor = examples_data_training_tensor
                    
                    # Check if the examples data test tensor exists
                    if examples_data_test_tensor is not None:
                        self.test_examples_tensor = examples_data_test_tensor
                        
                    else:
                        # Make warning log for failure to load examples data test
                        #   tensor
                        log._log_warning(
                            "Failed to load examples data test tensor!",
                            self.log_id
                        )
                    
                # Make info log for successfully loading the examples data
                log._log_info(
                    f"Successfully loaded examples data!",
                    self.log_id
                )
                return True

            else:
                # Make warning log for failure to load training examples tensor
                log._log_error(
                    "Failed to load training examples tensor!",
                    self.log_id
                )

        else:
            # Make warning log for failure to load training and test examples tensors
            #   since the examples data tensors failed to load
            log._log_error(
                "Could not load training and test examples tensors since the "
                "examples data tensors failed to load!",
                self.log_id
            )

        # Return False since loading the examples data tensors failed
        return False
    
    def _load_labels_data(self,
        labels_data: list[Any],
        labels_data_name: Optional[str]=None,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None,

        training_test_split=-1.0,
        use_test_only=False
    ) -> bool:
        """
        Load the labels training and test data tensors.

        Args:
            NOTE: See load_data for info on the args.

        Return:
            Boolean indicating success with loading the examples data
        """
        # Set the data name
        data_name = 'labels data'
        if labels_data_name is not None:
            data_name = labels_data_name

        # Make info log for loading the examples data
        log._log_info(
            f"Loading labels data [{util._get_print_name(data_name)}]...",
            self.log_id
        )

        # If only loading test data, set the training test split to -1
        if use_test_only:
            training_test_split = -1
         
        # Check if setting the labels data tensor function failed
        if not self._set_data_tensor_functions(
            labels_data_tensor_function_ptr=labels_data_tensor_function_ptr,
            labels_data_tensor_function_args=labels_data_tensor_function_args,
            labels_data_tensor_function_return_value_keys=\
                labels_data_tensor_function_return_value_keys,
            labels_data_tensor_function_name=labels_data_tensor_function_name
        ):
            # Make warning log for failure to load labels data since setting the
            #   labels data tensor function failed, and return False
            log._log_error(
                "Could not load labels data since setting the labels data "
                "tensor function failed!",
                self.log_id
            )
            return False

        # Get the labels data tensors
        labels_data_tensors = self._get_training_test_tensors(
            sample_data=labels_data,
            sample_data_name=labels_data_name,
            sample_data_tensor_function=self.labels_data_tensor_function,
            training_test_split=training_test_split
        )
        
        # Check if the labels data tensors exist
        if labels_data_tensors is not None:
            # Get the labels data training and test tensors
            labels_data_training_tensor, labels_data_test_tensor = \
                            labels_data_tensors
            # Make sure the labels data training tensor exists
            if labels_data_training_tensor is not None:
                # Check if only loading the test data
                if use_test_only:
                    # Store the test labels data
                    self.test_labels_data = labels_data
                    # Store the test labels tensor
                    # NOTE: The training tensor becomes the test tensor
                    self.test_labels_tensor = labels_data_training_tensor

                else:
                    # Split the labels data
                    training_start, training_stop, test_start, test_stop = \
                                    util._get_collection_indices_by_split(
                                        labels_data, training_test_split
                                    )
                    self.training_labels_data = \
                                    labels_data[training_start:training_stop]
                    self.test_labels_data = labels_data[test_start:test_stop]

                    # Store the labels data tensors
                    self.training_labels_tensor = labels_data_training_tensor

                    # Check if the labels data test tensor exists
                    if labels_data_test_tensor is not None:
                        self.test_labels_tensor = labels_data_test_tensor
                        
                    else:
                        # Make warning log for failure to load labels data test
                        #   tensor
                        log._log_warning(
                            "Failed to load labels data test tensor!",
                            self.log_id
                        )
                    
                # Make info log for successfully loading the labels data
                log._log_info(
                    f"Successfully loaded labels data!",
                    self.log_id
                )
                return True

            else:
                # Make warning log for failure to load training labels tensor
                log._log_error(
                    "Failed to load training labels tensor!",
                    self.log_id
                )

        else:
            # Make warning log for failure to load training and test labels tensors
            #   since the labels data tensors failed to load
            log._log_error(
                "Could not load training and test labels tensors since the "
                "labels data tensors failed to load!",
                self.log_id
            )

        # Return False since loading the labels data tensors failed
        return False
    
    def _get_training_test_tensors(self,
        sample_data: list[Any],
        sample_data_name: Optional[str]=None,

        # If loading sample data by wrapped tensor function
        sample_data_tensor_function: Optional[TensorFunction]=None,

        # If loading sample data by tensor function pointer
        sample_data_tensor_function_ptr: Optional[Callable]=None,
        sample_data_tensor_function_args: Optional[dict[str, Any]]=None,
        sample_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading sample data by tensor function pointer
        sample_data_tensor_function_name: Optional[str]=None,

        training_test_split=-1.0,
    ) -> Optional[tuple[Optional[Tensor], Optional[Tensor]]]:
        """
        Get training and test tensors for the sample data.

        Args:
            sample_data (list[Any]): The sample data list
            sample_data_name (str): The sample data name

            sample_data_tensor_function (TensorFunction): The sample data wrapped
                tensor function

            sample_data_tensor_function_ptr (Callable): The sample data tensor
                function pointer
            sample_data_tensor_function_args (dict[str, Any]): Dict of
                examples tensor function arguments
            sample_data_tensor_function_return_value_keys (tuple[str, ...]): Tuple
                of examples tensor function return value keys
            sample_data_tensor_function_name (str): The name for the examples
                data tensor function

            training_test_split (int): The index where to split the data
                tensor for training and testing

        Return:
            sample_data_training_tensor (Tensor): The sample data training tensor
            sample_data_test_tensor (Tensor): The sample data test tensor
        """
        # Set the data name
        data_name = 'sample data'
        if sample_data_name is not None:
            data_name = sample_data_name

        # Make debug log for loading the sample data
        log._log_info(
            f"Loading sample data [{util._get_print_name(data_name)}]...",
            self.log_id
        )

        # Check if the wrapped tensor function for the sample data was not provided
        if sample_data_tensor_function is None:
            # Get the sample data tensor function from function pointer or function name
            sample_data_tensor_function = _get_tensor_function(
                tensor_function_ptr=sample_data_tensor_function_ptr,
                tensor_function_args=sample_data_tensor_function_args,
                tensor_function_return_value_keys=\
                    sample_data_tensor_function_return_value_keys,
                tensor_function_name=sample_data_tensor_function_name,
            )
            
        # Check if the sample data tensor function doesn't exist
        if sample_data_tensor_function is None:
            # Make error log for failure to get training / test tensors since
            #   loading the tensor function failed
            log._log_error(
                f"Could not load {data_name} training / test tensors since "
                "getting the tensor function failed!",
                self.log_id
            )
            return None

        # Get sample data tensor
        sample_data_tensor = self._get_data_tensor(
            tensor_function=sample_data_tensor_function,
            sample_data=sample_data,
            sample_data_name=sample_data_name
        )

        if sample_data_tensor is None:
            # Make error log for failure to get training / test tensors since
            #   loading the sample data tensor failed
            log._log_error(
                f"Could not load {data_name} training / test tensors since "
                "loading the sample data failed!",
                self.log_id
            )
            return None

        else:
            # Split the sample data tensor to get the training and test sets
            (
                training_sample_data_start,
                training_sample_data_stop,
                test_sample_data_start,
                test_sample_data_stop
            ) = util._get_collection_indices_by_split(
                    collection=sample_data,
                    split_index=training_test_split
                )
            sample_data_training_tensor = \
                sample_data_tensor[training_sample_data_start:training_sample_data_stop]
            sample_data_test_tensor = \
                sample_data_tensor[test_sample_data_start:test_sample_data_stop]
            
        # Convert empty sample data tensors to None
        if sample_data_training_tensor.shape == [0]:
            sample_data_training_tensor = None
        if sample_data_test_tensor.shape == [0]:
            sample_data_test_tensor = None

        # Make debug log for successfully getting training / test tensors
        log._log_debug(
            "Successfully retrieved the training and test data tensors "
            f"for {data_name}!",
            self.log_id
        )

        return sample_data_training_tensor, sample_data_test_tensor

    def _item_to_tensor(self,
        example: Optional[Any]=None,
        label: Optional[Any]=None
    ) -> Optional[Tensor]:
        """
        Convert and example or label to a tensor.

        Args:
            example (Any): The example
            label (Any): The label

        Return:
            The example or label tensor
        """
        # If an example is provided, check that the example tensor function exists
        if example is not None and self.examples_data_tensor_function is not None:
            # Get the example tensor using the example tensor function
            return self._get_data_tensor(
                tensor_function=self.examples_data_tensor_function,
                sample_data=example,
                sample_data_name='example'
            )
        
        # Check if the example was provided but the examples tensor function
        #   doesn't exist
        elif example is not None and self.examples_data_tensor_function is None:
            # Make error log for failure to convert item to tensor since the
            #   examples data tensor function was not set, and return None
            log._log_error(
                f"Failed to convert item to tensor since the examples data tensor "
                "function was not set!",
                self.log_id
            )
            return None
        
        # If a label is provided, check that the label tensor function exists
        elif label is not None and self.labels_data_tensor_function is not None:
            # Get the example tensor using the example tensor function
            return self._get_data_tensor(
                tensor_function=self.labels_data_tensor_function,
                sample_data=label,
                sample_data_name='label'
            )
        
        # Check if the label was provided but the labels tensor function
        #   doesn't exist
        elif label is not None and self.labels_data_tensor_function is None:
            # Make error log for failure to convert item to tensor since the
            #   examples data tensor function was not set, and return None
            log._log_error(
                f"Failed to convert item to tensor since the examples data tensor "
                "function was not set!",
                self.log_id
            )
            return None
        
        # Else, make error log for failure to convert item to tensor since neither
        #   an example nor label was provided
        log._log_error(
            "No item was provided to convert to tensor!",
            self.log_id
        )

    def _items_to_tensor(self,
        examples: Optional[list[Any]]=None,
        labels: Optional[list[Any]]=None
    ) -> Optional[Tensor]:
        """
        Convert a list of examples or labels to a tensor.

        Args:
            examples (list[Any]): The list of examples
            labels (list[Any]): The list of labels

        Return:
            The examples or labels tensor
        """
        # Make debug log for converting items to tensor
        log._log_debug(
            "Converting items to tensor...",
            self.log_id
        )

        # Initialize the list of tensors
        tensors_list = []

        # Check if examples were provided
        if examples is not None:
            # Iterate through the examples to append example tensors
            for example in examples:
                # Get the example tensor
                example_tensor = self._item_to_tensor(example=example)
                # Check if the example tensor is None
                if example_tensor is None:
                    # Make log error for failure to convert examples to tensor,
                    #   and return None
                    log._log_error(
                        "Failed to convert examples to tensor!",
                        self.log_id
                    )
                    return None
                
                # Else, append the example tensor to the tensors list
                tensors_list.append(example_tensor)

        # Check if labels were provided
        elif labels is not None:
            # Iterate through the labels to append label tensors
            for label in labels:
                # Get the label tensor
                label_tensor = self._item_to_tensor(label=label)
                # Check if the label tensor is None
                if label_tensor is None:
                    # Make log error for failure to convert labels to tensor,
                    #   and return None
                    log._log_error(
                        "Failed to convert labels to tensor!",
                        self.log_id
                    )
                    return None
                
                # Else, append the label tensor to the tensors list
                tensors_list.append(label_tensor)

        else:
            # Make error log for failure to convert item to tensor since neither
            #   an examples nor labels was provided
            log._log_error(
                "No items were provided to convert to tensor!",
                self.log_id
            )
            return None
        
        # Else, make debug log for successfully converting items to tensor
        log._log_debug(
            "Successfully converted items to tensor!",
            self.log_id
        )

        # Stack and return the tensors in the tensors list as one tensor
        return torch.stack(tensors_list)
    
    def _get_data_tensor(self,
        tensor_function: TensorFunction,
        sample_data: list[Any],
        sample_data_name: Optional[str]=None
    ) -> Optional[Tensor]:
        """
        Load data into a tensor.

        Args:
            tensor_function (TensorFunction): The wrapped tensor function
            sample_data (Any): The sample data
            sample_data_name (str): The sample data name
            data_tensor (Tensor): The sample data tensor

        Return:
            Boolean indicating success with loading the training examples
        """
        # Log getting the data tensor
        data_name = 'sample data'
        if sample_data_name:
            data_name = sample_data_name

        # Make debug log for getting data tensor
        log._log_debug(
            f"Getting data tensor for {util._get_print_name(data_name)}...",
            self.log_id
        )

        # Initialize sample data name if not provided
        if sample_data_name is None:
            sample_data_name = 'x'

        # Set the keyward arguments for the tensor function
        kwargs = {
            sample_data_name: sample_data
        }

        # Run the wrapped tensor function on the sample data
        try:
            # Initialize the data tensor
            data_tensor = None

            # Get the output values
            output_values = tensor_function._run(
                kwargs=kwargs
            )

            # Make sure the output values are a tuple
            assert(isinstance(output_values, tuple))

            # Get the data tensor
            data_tensor = output_values[0]
            
        except:
            # Make error log for failure to get data tensor since running the
            #   wrapped tensor function on the data failed
            log._log_error(
                "Could not get data tensor since running the wrapped tensor function "
                f"on {data_name} failed!",
                self.log_id
            )

        # Check if the data tensor does not exist
        if data_tensor is None:
            # Make error log for failure to get data tensor
            log._log_error(
                f"Failed to retrieve data tensor!",
                self.log_id
            )
            return None
        
        # Else, make debug log for successfully retrieving data tensor
        log._log_debug(
            f"Successfully retrieved data tensor!",
            self.log_id
        )

        # Return the data tensor
        return data_tensor
    

# ============================= IMAGE DATASET =================================

class ImageDataSet(DataSet):
    """
    This is a dataset class for images. It uses images as training examples and
        the classification type for training labels.

    Methods to overwrite:
        _read_data
    """
    def __init__(self,
        # If loading data source items
        images_filename: Optional[str]=None,
        classification_filename: Optional[str]=None,

        # If loading labels data by tensor function pointer
        labels_data_tensor_function_ptr: Optional[Callable]=None,
        labels_data_tensor_function_args: Optional[dict[str, Any]]=None,
        labels_data_tensor_function_return_value_keys: Optional[tuple[str, ...]]=None,

        # If loading labels data by tensor function name
        labels_data_tensor_function_name: Optional[str]=None,

        training_test_split=-1.0,
        use_test_only=False,
        object_name=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.IMAGEDATASET)

        # Make debug log for loading the image dataset
        log._log_debug(
            f"Loading the image dataset...",
            log.DATASET_MODULE
        )

        # Set the data sources if the images filename and/or the classification
        #   filename are provided
        data_sources = None
        if images_filename is not None or classification_filename is not None:
            data_sources = (images_filename, classification_filename)

        # Initialize the base dataset
        super().__init__(
            data_sources=data_sources,
            examples_data_tensor_function_name='get_images_tensor',
            labels_data_tensor_function_ptr=labels_data_tensor_function_ptr,
            labels_data_tensor_function_name=\
                            labels_data_tensor_function_name,
            labels_data_tensor_function_args=labels_data_tensor_function_args,
            labels_data_tensor_function_return_value_keys=\
                labels_data_tensor_function_return_value_keys,
            training_test_split=training_test_split,
            use_test_only=use_test_only,
            object_name=object_name, has_log_id=True
        )

        # Initialize the preprocessor for the image dataset
        self.preprocessor = None

        # Initialize the mean and standard deviation for RGB values in preprocessing
        self.mean = None
        self.std = None

        # Check if there are examples and labels data for the image dataset
        if self.training_examples_data is not None \
                        and self.test_examples_data is not None:
            # Make debug log for successfully loading the image dataset
            log._log_debug(
                f"Successfully loaded the image dataset!",
                log.DATASET_MODULE
            )

        else:
            # Make warning log for loading the image dataset
            log._log_warning(
                "Partially loaded the image dataset with missing training data!",
                log.DATASET_MODULE
            )

    
    # NOTE: Overwritten method
    def _read_data(self,
        images_jsonfile=None, classification_jsonfile=None
    ) -> tuple[Optional[list[str]], Optional[list[Any]]]:
        """
        Get training examples and training labels from the images and captions
            JSON files. Updating either JSON file refreshes the training data.

        Args:
            images_jsonfile (str): The name of the images JSON file
            captions_jsonfile (str): The name of the captions JSON file

        Return:
            image_filepaths (list[str]): List of image filepaths
            image_captions (list[list[str]]): List of image caption lists
        """
        # Make debug log for reading input JSON files
        log._log_debug(
            f"Reading input JSON files...",
            self.log_id
        )

        # Get image filepaths from the images JSON file
        image_filepaths = None
        if images_jsonfile:
            image_filepaths = util._load_json(
                filename=images_jsonfile,
                format='values'
            )

            # Check if the image filepaths exist
            if image_filepaths is not None:
                # Make debug log for successfully loading image filepaths from
                #   JSON file
                log._log_debug(
                    f"Successfully loaded image filepaths from JSON file!",
                    self.log_id
            )
                
            else:
                # Make error log for failure to load image filepaths from JSON file
                log._log_warning(
                    "Failed to load image filepaths from JSON file!",
                    self.log_id
                )

        else:
            # Make warning log for failure to load image filepaths since no images
            #   JSON file was provided
            log._log_warning(
                "Failed to load image filepaths since no images JSON file "
                "was provided!",
                self.log_id
            )

        # Get the image captions from the captions JSON file
        classification_items = None
        if classification_jsonfile:
            classification_items = util._load_json(
                filename=classification_jsonfile,
                format='values'
            )

            # Check if the classification items exist
            if classification_items is not None:
                # Make debug log for successfully loading classification items
                #   from JSON file
                log._log_debug(
                    f"Successfully loaded classification items from JSON file!",
                    self.log_id
            )
                
            else:
                # Make error log for failure to load classification items from JSON file
                log._log_warning(
                    "Failed to load classification items from JSON file!",
                    self.log_id
                )

        else:
            # Make warning log for failure to load classification items since no
            #   classification items JSON file was provided
            log._log_warning(
                "Failed to load image filepaths since no classification items "
                "JSON file was provided!",
                self.log_id
            )

        # Return the image filepaths and classification items
        return image_filepaths, classification_items
    

# ============================= CAPTIONS DATASET ==============================

class CaptionDataSet(DataSet):
    """
    This is a dataset class for captions. It uses captions for both the
        training examples and training labels.

    Methods to overwrite:
        _read_data
    """
    def __init__(self,
        captions_filename: Optional[str]=None,
        training_test_split=-1.0,
        use_test_only=False,
        object_name=None, has_log_id=False
    ):     
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.IMAGEDATASET)

        # Make debug log for loading the caption dataset
        log._log_debug(
            f"Loading the caption dataset...",
            log.DATASET_MODULE
        )

        # Set the data sources if the captions filename exists
        data_sources = None
        if captions_filename is not None:
            data_sources = (captions_filename,)

        # Initialize the base dataset
        super().__init__(
            data_sources=data_sources,
            examples_data_tensor_function_name='get_tokens_tensor',
            labels_data_tensor_function_name='get_tokens_tensor',
            training_test_split=training_test_split,
            use_test_only=use_test_only,
            object_name=object_name, has_log_id=True
        )

        # Check if there are examples and labels data for the caption dataset
        if self.training_examples_data is not None \
                        and self.test_examples_data is not None:
            # Make debug log for successfully loading the caption dataset
            log._log_debug(
                f"Successfully loaded the caption dataset!",
                log.DATASET_MODULE
            )

        else:
            # Make warning log for loading the caption dataset
            log._log_warning(
                "Partially loaded the caption dataset with missing training data!",
                log.DATASET_MODULE
            )
    
    # NOTE: Overwritten method
    def _read_data(self,
        captions_jsonfile=None
    ) -> tuple[ Optional[list[list[str]]], Optional[list[list[str]]] ]:
        """
        Get training examples and training labels from the captions JSON file.
            Updating the JSON file refreshes the training data.

        Args:
            captions_jsonfile (str): The name of the captions JSON file

        Return:
            Tuple of captions, captions
        """
        # Make debug log for reading input JSON files
        log._log_debug(
            f"Reading input JSON files...",
            self.log_id
        )

        # Get the caption lists from the images JSON file
        # NOTE: Each caption list contains several captions
        if captions_jsonfile:
            caption_lists = util._load_json(
                filename=captions_jsonfile,
                format='values'
            )

            # Check if the caption lists exist
            if caption_lists is not None:
                # Make debug log for successfully loading caption lists from
                #   JSON file
                log._log_debug(
                    f"Successfully loaded caption lists from JSON file!",
                    self.log_id
            )
                
            else:
                # Make error log for failure to load caption lists from JSON file
                log._log_warning(
                    "Failed to load caption lists from JSON file!",
                    self.log_id
                )

        # Return caption lists twice for both examples and labels
        return caption_lists, caption_lists