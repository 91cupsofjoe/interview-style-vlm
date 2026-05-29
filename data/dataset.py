"""
This module contains the DataSet class and its children.
"""

from matplotlib.pylab import std
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor, Size
from torchvision import transforms

from log import logger as log
from tensor import image, corpus, util


# ============================== DATASET CLASS ================================

class DataSet(Dataset):
    """
    This is the base dataset class.
    """
    def __init__(self,
        object_name=None,
        data_sources: Optional[tuple[Any, ...]]=None,
        examples_name: Optional[str]=None,
        examples_tensor_function: Optional[Callable]=None,
        labels_name: Optional[str]=None,
        labels_tensor_function: Optional[Callable]=None
    ):
        # Set the logger ID for the DataSet
        if object_name is None:
            object_name = "UNNAMED_DATASET"
        self.log_id = log.set_log_id(object_name=object_name)

        # Initialize the training examples and test examples, as as well as the
        #   examples tensor function (primary data loading/retrieving option)
        self.training_examples = None
        self.test_examples = None
        self.examples_tensor_function = None

        # Initialize the training labels and test labels, as well as the labels
        #   tensor function (primary data loading/retrieving option)
        self.training_labels = None
        self.test_labels = None
        self.labels_tensor_function = None

         # Initialize the dataframe (this is the secondary option for loading
        #   and retrieving both examples and labels data)
        self.dataframe = None

        # If data sources are provided, load the data for the dataset
        if data_sources is not None \
                        and self.examples_tensor_function is not None \
                        and self.labels_tensor_function is not None:
            self.load_data(
                data_sources=data_sources,
                examples_name=examples_name,
                examples_tensor_function=examples_tensor_function,
                labels_name=labels_name,
                labels_tensor_function=labels_tensor_function,

            )

    def __len__(self):
        """
        Return the number of training examples in the dataset.
        """
        assert(self.training_examples is not None)
        return self.training_examples.shape[0]
    
    def __getitem__(self, idx):
        """
        Return a training example, training label pair from the dataset by index.
        """
        assert(self.training_examples is not None)
        assert(self.training_labels is not None)
        return self.training_examples[idx], self.training_labels[idx]
    
    def get_dataframe(self) -> Optional[Tensor]:
        """
        Return the dataframe, which contains the training examples, training
            labels, test examples, and test labels packed into one tensor.
        """
        return self.dataframe
    
    def get_training_data(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Return the training data.
        """
        return self.training_examples, self.training_labels
    
    def get_test_data(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Return the test data
        """
        return self.test_examples, self.test_labels
    
    def set_tensor_ops(self,
        examples_tensor_function: Optional[Callable]=None,
        labels_tensor_function: Optional[Callable]=None
    ):
        """
        Set either the examples tensor function, labels tensor function, or
            both for the dataset.
        NOTE: This must be done in addition to loading sample data if the
            sample data is loaded post dataset initialization.
        """
        if examples_tensor_function is not None:
            self.examples_tensor_function = examples_tensor_function

        if labels_tensor_function is not None:
            self.labels_tensor_function = labels_tensor_function
    
    def load_data(self,
        data_sources: tuple[Any, ...],
        examples_name: Optional[str]=None,
        do_load_examples_data=True,
        examples_tensor_function: Optional[Callable]=None,
        labels_name: Optional[str]=None,
        do_load_labels_data=True,
        labels_tensor_function: Optional[Callable]=None,
        training_test_split_index=-1
    ) -> bool:
        """
        Load data from the data sources into the dataset.
        NOTE: This will set the training examples, training labels, test
            examples, and test labels members.

        Args:
            data_sources (tuple[Any]): The data sources to load data from
            do_load_examples_data (bool): Boolean indicating if the examples
                data is being loaded
            do_load_labels_data (bool): Boolean indicating if the labels data
                is being loaded
            training_test_split_index (int): The index which functions as a
                splitting point between the training and test data

        Return:
            load_success (bool): Boolean indicating if loading data was successful
        """
        load_success = True

        # Get the examples and labels from the data samples
        examples_data, labels_data = self.read_data(data_sources)

        # Load the training examples and test examples from the examples data
        #   and update the boolean indicating data loading success
        if do_load_examples_data and examples_data is not None:

            # If no examples tensor function is provided, use the preset
            #   examples tensor function
            if examples_tensor_function is None:
                examples_tensor_function = self.examples_tensor_function

            # Make sure the examples tensor function exists
            if examples_tensor_function is not None:

                # Get training examples tensor
                examples_tensor = self.get_data_tensor(
                    sample_data=examples_data,
                    sample_data_name=examples_name,
                    tensor_function=examples_tensor_function
                )

                if examples_tensor is None:
                    # Log error since loading training data failed
                    log.log_error(
                        f"Could not load training examples and test examples "
                        f"from the examples data!",
                        self.log_id
                    )
                    # Update load success to False
                    load_success = not do_load_examples_data

                else:
                    # Split the examples tensor to get the training examples and
                    #   test examples
                    (
                        training_examples_start,
                        training_examples_stop,
                        test_examples_start,
                        test_examples_stop
                    ) = util.get_collection_indices_by_split(
                            collection=examples_data,
                            split_index=training_test_split_index
                        )
                    self.training_examples = \
                        examples_tensor[training_examples_start:training_examples_stop]
                    self.test_examples = \
                        examples_tensor[test_examples_start:test_examples_stop]
                    
            else:
                # Log error since the examples tensor function doesn't exist
                log.log_error(
                    f"Could not load training and test examples since the "
                    f"the examples tensor function hasn't been set!",
                    self.log_id
                )

        # Load the training labels and test labels from the examples data and
        #   update the boolean indicating data loading success
        if do_load_labels_data and labels_data is not None:

            # If no labels tensor function is provided, use the preset
            #   labels tensor function
            if labels_tensor_function is None:
                labels_tensor_function = self.labels_tensor_function

            # Make sure the labels tensor function exists
            if labels_tensor_function is not None:
                # Get the labels tensor
                labels_tensor = self.get_data_tensor(
                    sample_data=labels_data,
                    sample_data_name=labels_name,
                    tensor_function=labels_tensor_function
                )

                if labels_tensor is None:
                    # Log error since loading training data failed
                    log.log_error(
                        f"Could not load training and test labels from the "
                        f"examples data!",
                        self.log_id
                    )
                    # Update load success to False
                    load_success = not do_load_labels_data

                else:
                    # Split the examples tensor to get the training labels and
                    #   test labels
                    (
                        training_labels_start,
                        training_labels_stop,
                        test_labels_start,
                        test_labels_stop
                    ) = util.get_collection_indices_by_split(
                                collection=examples_data,
                                split_index=training_test_split_index
                            )
                    self.training_labels = \
                        labels_tensor[training_labels_start:training_labels_stop]
                    self.test_labels = \
                        labels_tensor[test_labels_start:test_labels_stop]
                    
            else:
                # Log error since the labels tensor function doesn't exist
                log.log_error(
                    f"Could not load training and test labels since the "
                    f"the labels tensor function hasn't been set!",
                    self.log_id
                )
            
        # Return False if any of the intended data loading failed, True otherwise
        return load_success
    
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
    
    def get_data_tensor(self,
        tensor_function: Callable,
        sample_data: Optional[list]=None,
        sample_data_name: Optional[str]=None,
        data_tensor: Optional[Tensor]=None
    ) -> Optional[Tensor]:
        """
        Load training examples from image filepaths or a tensor of images data.

        Args:
            image_filepaths (str): List of image filepaths
            images_tensor (Tensor): Tensor of images data

        Return:
            Boolean indicating success with loading the training examples
        """
        # Load training examples from the image filepaths if no image tensor
        #   is provided
        if data_tensor is None:

            # Get the data tensor from the samples data
            if sample_data is not None:

                data_tensor = tensor_function(sample_data)

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

    Methods to add:
        load_dataframe
    """
    def __init__(self,
        object_name=None,
        images_filename: Optional[str]=None,
        classification_filename: Optional[str]=None,
        examples_tensor_function: Optional[Callable]=None,
        labels_tensor_function: Optional[Callable]=None
    ):
        # Set the logger ID for the ImageDataSet
        if object_name is None:
            object_name = "UNNAMED_IMAGEDATASET"
        self.log_id = log.set_log_id(object_name=object_name)

        # Initialize the base dataset
        super().__init__(
            object_name=object_name,
            data_sources=(images_filename, classification_filename),
            examples_tensor_function=examples_tensor_function,
            labels_tensor_function=labels_tensor_function
        )

        # Initialize the preprocessor for the image dataset
        self.preprocessor = None

        # Initialize the mean and standard deviation for RGB values in preprocessing
        self.mean = None
        self.std = None
    
    # NOTE: Overwritten method
    def read_data(self,
        images_jsonfile=None, classification_file=None
    ) -> tuple[Optional[Any], Optional[Any]]:
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
        if classification_file:
            classification_items = util.load_json(
                filename=classification_file,
                format='values'
            )

        # Return the image filepaths and classification items
        return image_filepaths, classification_items
    

# ========================= IMAGE CAPTION DATASET =============================

class ImageCaptionDataSet(ImageDataSet):
    """
    This is the image caption dataset class. Images are the training examples,
        and their captions are the training labels.

    Methods to overwrite: read_data, load_training_tensors
    """
    def __init__(self,
        object_name=None,
        images_jsonfile=None,
        captions_jsonfile=None
    ):
        # Set the logger ID for the ImageCaptionerDataSet
        if object_name is None:
            object_name = "UNNAMED_IMAGECAPTIONERDATASET"
        self.log_id = log.set_log_id(object_name=object_name)

        # Get the examples and labels tensor functions
        examples_tensor_function = image.get_images_tensor
        labels_tensor_function = corpus.get_tokens_tensor

        # Initialize the base image dataset
        super().__init__(
            object_name=object_name,
            images_filename=images_jsonfile,
            classification_filename=captions_jsonfile,
            examples_tensor_function=examples_tensor_function,
            labels_tensor_function=labels_tensor_function
        )