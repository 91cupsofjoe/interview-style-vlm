"""
This module handles testing the image encoder.
"""
from typing import Optional, Any
from pathlib import Path

from torch import Tensor

from data import data_reader as dr
from tensor_function import \
    convolution as conv, pool, regularization as reg, update
from log import logger as log
from model import model as ml
from model.image_encoder import get_image_encoder


TRAIN_FRACTION = .01

def _train_no_init_files_test(
    images_filename: str,
    context_vectors_filename: str,
    do_print_messages=True
) -> Optional[bool]:
    """
    Test the image encoder's training method with input files.
    NOTE: This entails not supplying input filenames when instantiating the
        the image encoder.

    Args:

    Return:
        Boolean indicating if training test succeeds or fails
    """
    image_encoder = get_image_encoder()

    # Make sure the image encoder loaded successfully
    if image_encoder is None:
        # Log error and return False since the image encoder failed to load
        log._log_error(
            "Could not test the image encoder since it failed to load!",
            log.TEST_MODULES
        )
        return None

    return image_encoder.train(
        data_sources=(images_filename, context_vectors_filename),
        training_test_split=.6,
        do_print_messages=do_print_messages,
        train_fraction=TRAIN_FRACTION
    )


def _train_no_init_input_data_test(
    image_filepaths: list[str],
    context_data: list[dict[str, int]],
    do_measure_accuracy=True,
    do_print_messages=True
) -> Optional[bool]:
    """
    Test training the image encoder initialized without input files or data.
    NOTE: This entails not supplying input as parameters when instantiating the
        image encoder.

    Args:

    Return:
        Boolean indicating if training test succeeds or fails
    """
    image_encoder = get_image_encoder()
    
    # Make sure the image encoder loaded successfully
    if image_encoder is None:
        # Log error and return False since the image encoder failed to load
        log._log_error(
            "Could not test the image encoder since it failed to load!",
            log.TEST_MODULES
        )
        return None

    return image_encoder.train(
        examples_data=image_filepaths,
        labels_data=context_data,
        training_test_split=.6,
        do_print_messages=do_print_messages,
        train_fraction=TRAIN_FRACTION
    )


def _train_no_init_data_tensors_test(
    examples_data_tensor: Tensor,
    labels_data_tensor: Tensor,
    do_print_messages=True
) -> Optional[bool]:
    """
    Test the image encoder's training method with data tensors.
    NOTE: This entails not supplying input parameteres when instantiating the
        the image encoder. Only supply data tensors to the train method.

    Args:

    Return:
        Boolean indicating if training test succeeds or fails
    """
    image_encoder = get_image_encoder()

    # Make sure the image encoder loaded successfully
    if image_encoder is None:
        # Log error and return False since the image encoder failed to load
        log._log_error(
            "Could not test the image encoder since it failed to load!",
            log.TEST_MODULES
        )
        return None

    return image_encoder.train(
        training_examples_tensor=examples_data_tensor,
        target_labels_tensor=labels_data_tensor,
        training_test_split=.6,
        do_print_messages=do_print_messages,
        train_fraction=TRAIN_FRACTION
    )


def _train_init_input_data_test(
    image_filepaths: list[str],
    context_data: list[dict[str, int]],
    do_print_messages=True,
    do_print_layer_attr=False,
    do_print_tensor_function_attr=False
) -> Optional[bool]:
    """
    Test the image encoder's training method with input files.
    NOTE: This entails supplying input data as parameters when instantiating
        the image encoder.

    Args:

    Return:
        Boolean indicating if training test succeeds or fails
    """
    image_encoder = get_image_encoder(
        object_name='image_encoder',
        batch_size=12,
        image_filepaths=image_filepaths,
        context_data=context_data,
        training_test_split=.8,
        do_print_layer_attr=do_print_layer_attr,
        do_print_tensor_function_attr=do_print_tensor_function_attr
    )

    # Make sure the image encoder loaded successfully
    if image_encoder is None:
        # Log error and return False since the image encoder failed to load
        log._log_error(
            "Could not test the image encoder since it failed to load!",
            log.TEST_MODULES
        )
        return None

    return image_encoder.train(
        do_print_messages=do_print_messages,
        train_fraction=TRAIN_FRACTION
    )


DATA_DIR = f'{Path(__file__).parent}/../data/'

if __name__ == '__main__':
    # Use the test log
    log._use_logger(
        logger_name='test_log',
        console_level=log.INFO,
        do_output_to_console=True
    )

    # Load the MS Coco dataset
    ms_coco_dataset_items = dr._get_ms_coco_dataset('validation_2017')

    # Check if the MS Coco dataset items exist
    if ms_coco_dataset_items is not None:
        # Get the names of the formatted MS Coco JSON files
        ms_coco_dataset_filenames = (
            f'{DATA_DIR}ms_coco_image_filepaths.json',
            f'{DATA_DIR}ms_coco_context_data.json',
            f'{DATA_DIR}ms_coco_captions.json'
        )

        # Save the MS Coco dataset items to their respective dataset JSON files
        ms_coco_filenames = dr._save_dataset_items(
            ms_coco_dataset_items, ms_coco_dataset_filenames
        )

        # Get the input data
        image_filepaths, context_data, _ = ms_coco_dataset_items

        # Get the input filenames
        images_filename, context_data_filename, _ = ms_coco_dataset_filenames

        # print(f"\nThe image filepaths:\n\n{image_filepaths}")
        # print(f"\nThe context data:\n\n{context_data}")

        # Test the image encoder training use input data
        print(f"\nImage encoder training using input data: "
              f"{_train_init_input_data_test(
                  image_filepaths, context_data,
                  do_print_tensor_function_attr=True)}")

    else:
        # Log error since the MS Coco dataset items failed to load
        log._log_error(
            "Could not test image encoder since the MS Coco dataset items " 
            "failed to load!",
            log.TEST_MODULES
        )

    # Flush the logger
    log._flush()