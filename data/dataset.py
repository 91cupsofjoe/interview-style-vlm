"""
This module contains the DataSet class and its children.
"""
from torch.utils.data import Dataset
from typing import Optional, Any, Union
from collections.abc import Callable

import torch
from torch import Tensor

from log import logger as log
from function import util
from function.tensor_function import TensorFunction, get_tensor_function


# ============================== DATASET CLASS ================================

class DataSet(Dataset):
    """
    This is the base dataset class.
    """
    def __init__(self,
        data_sources: Optional[tuple[Any, ...]],
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        examples_data_tensor_function_name: Optional[str]=None,
        examples_data_tensor_function_arg_keys: Optional[list[str]]=None,
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,
        labels_data_tensor_function_name: Optional[str]=None,
        labels_data_tensor_function_arg_keys: Optional[list[str]]=None,
        training_test_split=-1.0,
        object_name=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.DATASET)

        # Initialize the examples data, tensors, and tensor function
        self.examples_data: Optional[list[Any]] = None
        self.training_examples_tensor: Optional[Tensor] = None
        self.test_examples_tensor: Optional[Tensor] = None
        self.examples_data_tensor_function = None

        # Initialize the labels data, tensors, and tensor function
        self.labels_data: Optional[list[Any]] = None
        self.training_labels_tensor: Optional[Tensor] = None
        self.test_labels_tensor: Optional[Tensor] = None
        self.labels_data_tensor_function = None

         # Initialize the dataframe (this is the secondary option for loading
        #   and retrieving both examples and labels data)
        self.dataframe = None

        # If data sources are provided, load the data for the dataset
        if data_sources is not None:
            # Get keyword arguments without the object name and log id bool
            kwargs = locals().pop('object_name').pop('has_log_id')
            # Load the data
            self.load_data(kwargs)

    def __len__(self):
        """
        Return the number of training examples in the dataset.
        """
        if self.examples_data is not None:
            return len(self.examples_data)

    def __getitem__(self,
        key: Optional[Any]=None,
        use_tensors=False,
        use_test_data=False
    ) -> Union[
        tuple[Any, Optional[Any]],
        tuple[list[Any], Optional[list[Any]]],
        tuple[Tensor, Optional[Tensor]],
        None
    ]:
        """
        Return an example and label pair from the dataset by index.
        """
        # Set key to ... if not provided
        if key is None:
            key = ...

        # Check if key is int or slice
        if isinstance(key, int) or isinstance(key, slice):

            # Return the example and label pair(s) by key
            # Return from tensors if specified
            if use_tensors:

                # Return from test tensors if specified
                # Make sure the test tensors exist
                if use_test_data and \
                    (self.test_examples_tensor is not None \
                            and self.test_labels_tensor is not None):
                    return self.test_examples_tensor[key], \
                                    self.test_labels_tensor[key]
                
                # Else, return from training tensors
                # Make sure the training tensors exist
                elif not use_test_data \
                        and (self.training_examples_tensor is not None \
                            and self.training_labels_tensor is not None):
                    return self.training_examples_tensor, \
                                    self.training_labels_tensor
                        
            # Else, use the examples data and/or labels data
            elif not use_tensors:
                # Check if examples data was already loaded
                if self.examples_data is not None:

                    # Check if labels data was already loaded
                    if self.labels_data is not None:
                    
                        # Return from the examples data and labels data
                        return self.examples_data[key], self.labels_data[key]
                    
                    else:
                        # Return from the examples data
                        return self.examples_data[key], None
                
        # Else, check if key is an Ellipsis
        elif key is Ellipsis:
            # Return tensors if specified
            if use_tensors:

                # Return test tensors if specified
                # Make sure the test tensors exist
                if use_test_data \
                    and (self.test_examples_tensor is not None
                            and self.test_labels_tensor is not None):
                    return self.test_examples_tensor, \
                                    self.test_labels_tensor
                
                # Else, return training tensors
                # Make sure the training tensors exist
                elif not use_test_data \
                    and (self.training_examples_tensor is not None
                            and self.training_labels_tensor is not None):
                    return self.training_examples_tensor, \
                                    self.training_labels_tensor
                
            # Else, return the examples data and/or labels data
            elif not use_tensors:
                # Check if the examples data was already loaded
                if self.examples_data is not None:

                    # Check if the labels data was already loaded
                    if self.labels_data is not None:
                        
                        # Return the examples data and labels data
                        return self.examples_data, self.labels_data
                    
                    # Else, return the examples data
                    return self.examples_data, None
            
        # Return None
        return None

    def get_training_data(self,
        key: Optional[Any]=None
    ) -> Union[
        tuple[Any, Optional[Any]],
        tuple[list[Any], Optional[list[Any]]],
        None
    ]:
        """
        Return training examples and training labels by index, slice, or Ellipsis.

        Args:
            key (Any): Index, slice, or Ellipsis

        Return:
            List element, tuple of list elements, list, or tuple of lists
        """
        return self.__getitem__(
            key=key
        )
    
    def get_test_data(self,
        key: Optional[Any]=None
    ) -> Union[
        tuple[Any, Optional[Any]],
        tuple[list[Any], Optional[list[Any]]],
        None
    ]:
        """
        Return test examples and test labels by index, slice, or Ellipsis.

        Args:
            key (Any): Index, slice, or Ellipsis

        Return:
            List element, tuple of list elements, list, or tuple of lists
        """
        return self.__getitem__(
            key=key,
            use_test_data=True
        )
    
    def get_training_tensors(self,
        # If using preloaded data
        key: Optional[Any]=None,

        # If loading the training data from files
        data_sources: Optional[tuple[Any, Any]]=None,

        # If loading the training data from lists
        examples_data: Optional[list[Any]]=None,
        labels_data: Optional[list[Any]]=None,

        # Data loading params
        examples_data_name: Optional[str]=None,
        examples_data_tensor_function_name: Optional[str]=None,
        examples_data_tensor_function_arg_keys: Optional[list[str]]=None,
        labels_data_name: Optional[str]=None,
        labels_data_tensor_function_name: Optional[str]=None,
        labels_data_tensor_function_arg_keys: Optional[list[str]]=None,
        training_test_split=-1.0
    ) -> Optional[tuple[Tensor, Optional[Tensor]]]:
        """
        Return training examples and training labels tensors
            by index, slice, or Ellipsis.

        Args:
            key (Any): Index, slice, or Ellipsis
            NOTE: See load_data for the info on the rest of the args

        Return:
            Tensor or tuple of Tensors
        """
        # Check if data sources, examples data, or labels data were provided
        if data_sources is not None \
                or (examples_data is not None or labels_data is not None):
            # Load the training data
            self.load_data(
                data_sources=data_sources,
                examples_data=examples_data,
                examples_data_name=examples_data_name,
                examples_data_tensor_function_name=examples_data_tensor_function_name,
                examples_data_tensor_function_arg_keys=\
                                examples_data_tensor_function_arg_keys,
                labels_data=labels_data,
                labels_data_name=labels_data_name,
                labels_data_tensor_function_name=labels_data_tensor_function_name,
                labels_data_tensor_function_arg_keys=\
                                labels_data_tensor_function_arg_keys,
                training_test_split=training_test_split
            )

        # Get the training tensors
        training_tensors = self.__getitem__(
            key=key,
            use_tensors=True
        )
        # Check if training tensors exist
        if training_tensors is not None:
            # Get the examples and labels tensors, and make sure they are tensors
            #   if they do exist
            examples_tensor, labels_tensor = training_tensors
            assert(isinstance(examples_tensor, Tensor))
            if labels_tensor is not None:
                assert(isinstance(labels_tensor, Tensor))

            return examples_tensor, labels_tensor
        
        # Else, return None
        
    def get_test_tensors(self,
        key: Optional[Any]=None
    ) -> Optional[tuple[Tensor, Optional[Tensor]]]:
        """
        Return test examples and test labels tensors
            by index, slice, or Ellipsis.

        Args:
            key (Any): Index, slice, or Ellipsis

        Return:
            Tensor or tuple of Tensors
        """
        # Get the test tensors
        test_tensors = self.__getitem__(
            key=key,
            use_tensors=True
        )

        # Check if test tensors exist
        if test_tensors is not None:

            # Get the examples and labels tensors, and make sure they are tensors
            #   if they do exist
            examples_tensor, labels_tensor = test_tensors
            assert(isinstance(examples_tensor, Tensor))
            if labels_tensor is not None:
                assert(isinstance(labels_tensor, Tensor))

            return examples_tensor, labels_tensor
        
        # Else, return None

    def get_dataframe(self) -> Optional[Tensor]:
        """
        Return the dataframe, which contains the training examples, training
            labels, test examples, and test labels packed into one tensor.
        """
        return self.dataframe
    
    def get_training_test_tensors(self,
        sample_data: list[Any],
        sample_data_tensor_function: Optional[TensorFunction]=None,
        sample_data_tensor_function_name: Optional[str]=None,
        sample_data_tensor_function_arg_keys: Optional[list[str]]=None,
        sample_data_name: Optional[str]=None,
        training_test_split=-1.0
    ) -> Optional[tuple[Optional[Tensor], Optional[Tensor]]]:
        """
        Get training and test tensors for the sample data.

        Args:
            sample_data (list[Any]): The sample data list
            sample_data_tensor_function (TensorFunction): The tensor function
                to convert sample data to tensor
            sample_tensor_function_arg_keys (list[str]): The argument keys for
                the tensor function
            sample_data_name (str): The sample data name
            training_test_split (int): The index where to split the data
                tensor for training and testing

        Return:
            sample_data_training_tensor (Tensor): The sample data training tensor
            sample_data_test_tensor (Tensor): The sample data test tensor
        """
        # Check if the sample data tensor function was provided
        if sample_data_tensor_function is None:
            # Check if the sample data tensor function name was provided
            if sample_data_tensor_function_name is not None:
                # Get the sample data tensor function from its name
                sample_data_tensor_function = get_tensor_function(
                    tensor_function_name=sample_data_tensor_function_name
                )
            
            else:
                # Log error and return False since the sample data tensor
                #   function nor its name were provided
                log.log_error(
                    "Could not load sample data since the tensor function nor"
                    "its name were provided!",
                    self.log_id
                )
                return None

        assert(sample_data_tensor_function is not None)
        # Get sample data tensor
        sample_data_tensor = self.get_data_tensor(
            tensor_function=sample_data_tensor_function,
            tensor_function_arg_keys=sample_data_tensor_function_arg_keys,
            sample_data=sample_data,
            sample_data_name=sample_data_name
        )

        if sample_data_tensor is None:
            # Log error and return False since loading the sample data failed
            log.log_error(
                f"Failed to load the same sample data tensor!",
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
            ) = util.get_collection_indices_by_split(
                    collection=sample_data,
                    split_index=training_test_split
                )
            sample_data_training_tensor = \
                sample_data_tensor[training_sample_data_start:training_sample_data_stop]
            sample_data_test_tensor = \
                sample_data_tensor[test_sample_data_start:test_sample_data_stop]
            
        # Convert empty sample data tensors to None
        if len(sample_data_training_tensor) == 0:
            sample_data_training_tensor = None
        if len(sample_data_test_tensor) == 0:
            sample_data_test_tensor = None

        return sample_data_training_tensor, sample_data_test_tensor
    
    def read_data(self,
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
        data_sources: Optional[tuple[Any, Any]]=None,
        examples_data: Optional[list[Any]]=None,
        examples_data_name: Optional[str]=None,
        examples_data_tensor_function_name: Optional[str]=None,
        examples_data_tensor_function_arg_keys: Optional[list[str]]=None,
        do_load_examples_data=True,
        labels_data: Optional[list[Any]]=None,
        labels_data_name: Optional[str]=None,
        labels_data_tensor_function_name: Optional[str]=None,
        labels_data_tensor_function_arg_keys: Optional[list[str]]=None,
        do_load_labels_data=True,
        training_test_split=-1.0
    ) -> bool:
        """
        Load data from the data sources into the dataset.
        NOTE: This will set the training examples, training labels, test
            examples, and test labels members.

        Args:
            data_sources (tuple[Any]): The data sources to load data from
            examples_data (list[Any]): List of examples
            examples_data_name (str): The name for the examples
            examples_data_tensor_function_name (str): The name for the examples
                data tensor function lookup
            examples_data_tensor_function_arg_keys (list[str]): List of
                additional examples data tensor function args keys
            do_load_examples_data (bool): Boolean indicating if the examples
                data is being loaded
            labels_data (list[Any]): List of labels
            labels_data_name (str): The name for the labels
            labels_data_tensor_function_name (str): The name for the labels
                data tensor function lookup
            labels_data_tensor_function_arg_keys (list[str]): List of
                additional labels data tensor function args keys
            do_load_labels_data (bool): Boolean indicating if the labels data
                is being loaded
            training_test_split (int): The splitting point between the
                training and test data

        Return:
            load_success (bool): Boolean indicating if loading data was successful
        """
        load_success = False

        # If data sources are provided, get examples and labels data from them
        if data_sources is not None:
            examples_data, labels_data = self.read_data(data_sources)

        # Load the training examples and test examples from the examples data
        if do_load_examples_data and examples_data is not None:

            # Check if loading examples data is unsuccessful
            if not self.load_examples_data(
                examples_data=examples_data,
                examples_data_tensor_function_name=examples_data_tensor_function_name,
                examples_data_tensor_function_arg_keys=examples_data_tensor_function_arg_keys,
                examples_data_name=examples_data_name,
                training_test_split=training_test_split
            ):
                
                # Log error and return False since the examples data failed to load
                log.log_error(
                    "Failed to load examples data!",
                    self.log_id
                )
                return False
            
            # Else, set load_success to True
            load_success = True

        # Load the training labels and test labels from the labels data
        if do_load_labels_data and labels_data is not None:

            # Check if loading labels data is unsuccessful
            if not self.load_labels_data(
                labels_data=labels_data,
                labels_data_tensor_function_name=labels_data_tensor_function_name,
                labels_data_tensor_function_arg_keys=labels_data_tensor_function_arg_keys,
                labels_data_name=labels_data_name,
                training_test_split=training_test_split
            ):
                    
                # Log error and return False since the labels data failed to load
                log.log_error(
                    "Failed to load labels data!",
                    self.log_id
                )
                return False
            
            # Else, set load_success to True
            load_success = True
            
        # Return True if examples and/or labels data successfully loaded,
        #   False otherwise
        return load_success
    
    def load_examples_data(self,
        examples_data: list[Any],
        examples_data_tensor_function: Optional[TensorFunction]=None,
        examples_data_tensor_function_name: Optional[str]=None,
        examples_data_tensor_function_arg_keys: Optional[list[str]]=None,
        examples_data_name: Optional[str]=None,
        training_test_split=-1.0
    ) -> bool:
        """
        Load the examples training and test data tensors.

        Args:
            examples_data (list[Any]): The sample data list
            examples_data_tensor_function (TensorFunction): The tensor function
                to convert sample data to tensor
            examples_tensor_function_arg_keys (list[str]): The argument keys for
                the tensor function
            examples_data_name (str): The sample data name
            training_test_split (int): The index where to split the data
                tensor for training and testing

        Return:
            examples_training_data_tensor (Tensor): The sample training data tensor
            examples_test_data_tensor (Tensor): The sample test data tensor
        """
        # Set the examples data tensor function is not provided
        if examples_data_tensor_function is None:
            # Check if the examples data tensor function name was not provided
            if examples_data_tensor_function_name is None:
                # Set the examples data tensor function from its name
                self.set_data_tensor_functions(
                    examples_data_tensor_function_name=\
                                    examples_data_tensor_function_name
                )
                
            # Use the stored examples data tensor function name
            examples_data_tensor_function = self.examples_data_tensor_function

        # Get the examples data tensors
        examples_data_tensors = self.get_training_test_tensors(
            sample_data=examples_data,
            sample_data_tensor_function=examples_data_tensor_function,
            sample_data_tensor_function_name=examples_data_tensor_function_name,
            sample_data_tensor_function_arg_keys=\
                examples_data_tensor_function_arg_keys,
            sample_data_name=examples_data_name,
            training_test_split=training_test_split
        )
        
        # Check if the examples data tensors exist
        if examples_data_tensors is not None:
            # Get the examples data training and test tensors
            examples_data_training_tensor, examples_data_test_tensor = \
                            examples_data_tensors
            # Make sure the examples data training tensor exists
            if examples_data_training_tensor is not None:
                # Store the examples data
                self.examples_data = examples_data

                # Store the examples data training and test tensors
                self.training_examples_tensor = examples_data_training_tensor
                self.test_examples_tensor = examples_data_test_tensor

            else:
                # Log error and return False since the examples data training
                #   tensor failed to load
                log.log_error(
                    "Could not load the training examples tensor since the "
                    "examples data training tensor failed to load!",
                    self.log_id
                )

        else:
            # Log error and return False since the examples data tensors
            #   failed to load
            log.log_error(
                "Could not load the training and test examples tensors since "
                "the examples data tensors failed to load!",
                self.log_id
            )

        # Return True since the training and test examples tensors loaded successfully
        return True
    
    def load_labels_data(self,
        labels_data: list[Any],
        labels_data_tensor_function: Optional[TensorFunction]=None,
        labels_data_tensor_function_name: Optional[str]=None,
        labels_data_tensor_function_arg_keys: Optional[list[str]]=None,
        labels_data_name: Optional[str]=None,
        training_test_split=-1.0
    ) -> bool:
        """
        Load the labels training and test data tensors.

        Args:
            labels_data (list[Any]): The sample data list
            labels_data_tensor_function (TensorFunction): The tensor function
                to convert sample data to tensor
            labels_tensor_function_arg_keys (list[str]): The argument keys for
                the tensor function
            labels_data_name (str): The sample data name
            training_test_split (int): The index where to split the data
                tensor for training and testing

        Return:
            labels_training_data_tensor (Tensor): The sample training data tensor
            labels_test_data_tensor (Tensor): The sample test data tensor
        """
        # Set the labels data tensor function is not provided
        if labels_data_tensor_function is None:
            # Check if the labels data tensor function name was provided
            if labels_data_tensor_function_name is not None:
                # Set the labels data tensor function from its name
                self.set_data_tensor_functions(
                    labels_data_tensor_function_name=\
                                    labels_data_tensor_function_name
                )
                
            # Use the stored labels data tensor function name
            labels_data_tensor_function = self.labels_data_tensor_function


        # Get the labels data tensors
        labels_data_tensors = self.get_training_test_tensors(
            sample_data=labels_data,
            sample_data_tensor_function=labels_data_tensor_function,
            sample_data_tensor_function_arg_keys=\
                labels_data_tensor_function_arg_keys,
            sample_data_name=labels_data_name,
            training_test_split=training_test_split
        )
        
        # Check if the labels data tensors exist
        if labels_data_tensors is not None:
            # Get the labels data training and test tensors
            labels_data_training_tensor, labels_data_test_tensor = \
                            labels_data_tensors
            # Make sure the labels data training tensor exists
            if labels_data_training_tensor is not None:
                # Store the labels data
                self.labels_data = labels_data

                # Store the labels data training and test tensors
                self.training_labels_tensor = labels_data_training_tensor
                self.test_labels_tensor = labels_data_test_tensor

            else:
                # Log error and return False since the labels data training
                #   tensor failed to load
                log.log_error(
                    "Could not load the training labels tensor since the "
                    "labels data training tensor failed to load!",
                    self.log_id
                )

        else:
            # Log error and return False since the labels data tensors
            #   failed to load
            log.log_error(
                "Could not load the training and test labels tensors since "
                "the labels data tensors failed to load!",
                self.log_id
            )

        # Return True since the training and test labels tensors loaded successfully
        return True
    
    def set_data_tensor_functions(self,
        examples_data_tensor_function_name: Optional[str]=None,
        labels_data_tensor_function_name: Optional[str]=None
    ):
        """
        Set either the examples tensor function, labels tensor function, or
            both for the dataset.
        NOTE: This must be done in addition to loading sample data if the
            sample data is loaded post dataset initialization.
        """
        if examples_data_tensor_function_name is not None:
            self.examples_data_tensor_function = get_tensor_function(
                tensor_function_name=examples_data_tensor_function_name
            )

        if labels_data_tensor_function_name is not None:
            self.labels_data_tensor_function = get_tensor_function(
                tensor_function_name=labels_data_tensor_function_name
            )

    def item_to_tensor(self,
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
            return self.get_data_tensor(
                tensor_function=self.examples_data_tensor_function,
                sample_data=example,
                sample_data_name='example'
            )
        
        # Check if the example was provided but the examples tensor function
        #   doesn't exist
        elif example is not None and self.examples_data_tensor_function is None:
            # Log error and return None since the examples tensor function
            #   doesn't exist
            log.log_error(
                f"Could not get the examples tensor because the examples tensor"
                f" function was not set!",
                self.log_id
            )
            return None
        
        # If a label is provided, check that the label tensor function exists
        elif label is not None and self.labels_data_tensor_function is not None:
            # Get the example tensor using the example tensor function
            return self.get_data_tensor(
                tensor_function=self.labels_data_tensor_function,
                sample_data=label,
                sample_data_name='example'
            )
        
        # Check if the label was provided but the labels tensor function
        #   doesn't exist
        elif label is not None and self.labels_data_tensor_function is None:
            # Log error and return None since the labels tensor function
            #   doesn't exist
            log.log_error(
                f"Could not get the labels tensor because the labels tensor "
                f" function was not set!",
                self.log_id
            )
            return None
        
        # Else, log error and return None since an example or label was not
        #   provided
        log.log_error(
            "Neither an example or label was provided for tensor conversion!",
            self.log_id
        )

    def items_to_tensor(self,
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
        # Initialize the list of tensors
        tensors_list = []

        # Check if examples were provided
        if examples is not None:
            # Iterate through the examples to append example tensors
            for example in examples:
                # Get the example tensor
                example_tensor = self.item_to_tensor(example=example)
                # Check if the example tensor is None
                if example_tensor is None:
                    # Log error and return None since one of the examples could
                    #   not be converted to a tensor
                    log.log_error(
                        "Error with converting the examples to a tensor!",
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
                label_tensor = self.item_to_tensor(label=label)
                # Check if the label tensor is None
                if label_tensor is None:
                    # Log error and return None since one of the labels could
                    #   not be converted to a tensor
                    log.log_error(
                        "Error with converting the labels to a tensor!",
                        self.log_id
                    )
                    return None
                
                # Else, append the label tensor to the tensors list
                tensors_list.append(label_tensor)

        else:
            # Log error and return None since neither examples nor labels were
            #   provided
            log.log_error(
                f"Could not get the tensor since neither examples nor labels "
                f"were provided!",
                self.log_id
            )
            return None
        
        # Stack and return the tensors in the tensors list as one tensor
        return torch.stack(tensors_list)
    
    def get_data_tensor(self,
        tensor_function: TensorFunction,
        tensor_function_arg_keys: Optional[list[str]]=None,
        sample_data: Optional[Any]=None,
        sample_data_name: Optional[str]=None,
        data_tensor: Optional[Tensor]=None
    ) -> Optional[Tensor]:
        """
        Load data into a tensor.

        Args:
            tensor_function (Callable): The tensor functions

        Return:
            Boolean indicating success with loading the training examples
        """
        # Load training examples from the image filepaths if no image tensor
        #   is provided
        if data_tensor is None:

            # Get the data tensor from the samples data
            if sample_data is not None:

                # Initialize the tensor function argument keys if not provided
                if tensor_function_arg_keys is None:
                    tensor_function_arg_keys = []

                # Get the tensor function arguments
                tensor_function_args = {
                    k: v for k, v
                        in zip(
                            tensor_function_arg_keys,
                            sample_data
                        )
                }

                data_tensor = tensor_function.run(
                    x=sample_data,
                    kwargs=tensor_function_args
                )[0]

                # Check if an images tensor couldn't be loaded
                if data_tensor is None:
                    # Log error and return None since the data tensor
                    #   could not be loaded from the sample data
                    if sample_data_name is None:
                        sample_data_name = 'sample data'
                    log.log_error(
                        f"Could not load data tensor from the "
                        f"{sample_data_name}!",
                        self.log_id
                    )
                    return None

            else:
                # Log error and return None since no valid sample data
                #   was provided
                if sample_data_name is None:
                    sample_data_name = 'sample data'
                log.log_error(
                    f"Could not load data tensor since {sample_data_name} "
                    f"not provided!",
                    self.log_id
                )
                return None
            
        # Return the data tensor
        return data_tensor
    

# ============================= IMAGE DATASET =================================

class ImageDataSet(DataSet):
    """
    This is a dataset class for images. It uses images as training examples and
        the classification type for training labels.

    Methods to overwrite:
        read_data
    """
    def __init__(self,
        images_filename: Optional[str]=None,
        classification_filename: Optional[str]=None,
        labels_data_tensor_function_name: Optional[str]=None,
        labels_data_tensor_function_arg_keys: Optional[list[str]]=None,
        training_test_split=-1.0,
        object_name=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.IMAGEDATASET)

        # Initialize the tensor function arg

        # Initialize the base dataset
        super().__init__(
            data_sources=(images_filename, classification_filename),
            examples_data_tensor_function_name='get_image_tensor',
            labels_data_tensor_function_name=\
                            labels_data_tensor_function_name,
            labels_data_tensor_function_arg_keys=\
                            labels_data_tensor_function_arg_keys,
            training_test_split=training_test_split,
            object_name=object_name, has_log_id=True
        )

        # Initialize the preprocessor for the image dataset
        self.preprocessor = None

        # Initialize the mean and standard deviation for RGB values in preprocessing
        self.mean = None
        self.std = None
    
    # NOTE: Overwritten method
    def read_data(self,
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
        # Get the image filepaths from the images JSON file
        if images_jsonfile:
            image_filepaths = util.load_json(
                filename=images_jsonfile,
                format='values'
            )

        # Get the image captions from the captions JSON file
        if classification_jsonfile:
            classification_items = util.load_json(
                filename=classification_jsonfile,
                format='values'
            )

        # Return the image filepaths and classification items
        return image_filepaths, classification_items
    

# ============================= CAPTIONS DATASET ==============================

class CaptionDataSet(DataSet):
    """
    This is a dataset class for captions. It uses captions for both the
        training examples and training labels.

    Methods to overwrite:
        read_data
    """
    def __init__(self,
        captions_filename: Optional[str]=None,
        training_test_split=-1.0,
        object_name=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.IMAGEDATASET)

        # Initialize the base dataset
        super().__init__(
            data_sources=(captions_filename,),
            examples_data_tensor_function_name='get_tokens_tensor',
            labels_data_tensor_function_name='get_tokens_tensor',
            training_test_split=training_test_split,
            object_name=object_name, has_log_id=True
        )
    
    # NOTE: Overwritten method
    def read_data(self,
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
        # Get the caption lists from the images JSON file
        # NOTE: Each caption list contains several captions
        if captions_jsonfile:
            caption_lists = util.load_json(
                filename=captions_jsonfile,
                format='values'
            )

        # Return caption lists twice for both examples and labels
        return caption_lists, caption_lists