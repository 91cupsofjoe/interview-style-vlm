"""
This module is for the Model classes.
"""
import random

import torch
from torch import Tensor
from typing import Optional, Any, Union
from collections.abc import Callable
from pathlib import Path

from data.dataset import DataSet
from model import layer
from model.layer import Layer, _get_layer
from tensor_function import \
    tensor_function as tf, \
    attention as attn, convolution as conv, loss, pool, regularization as reg, \
        update, util
from tensor_function.tensor_function import _get_tensor_function, PassFunction
from log import logger as log


# ================================= MODEL =====================================

BATCH_SIZE = 32
NUM_PREDICTIONS = 1
PROBABILITY_THRESHOLD = .5
NUM_FOLDS = 5
NUM_EPOCHS = 100

default_model_hyperparams = {
    'batch_size': BATCH_SIZE,
    'num_predictions': NUM_PREDICTIONS,
    'num_folds': NUM_FOLDS,
    'num_epochs': NUM_EPOCHS,
    'eps': loss.EPS,
    'patience': loss.PATIENCE,
    'reg_type': reg.REG_TYPE,
    'reg_strength': reg.REG_STRENGTH,
    'learning_rate': update.LEARNING_RATE,
}

class Model:
    """
    This is the base model class, which performs training and prediction.
    """
    def __init__(self,
        model_layers: list[Layer],
        model_forward_pass_functions: dict[str, Callable],
        model_backpropagation_functions: dict[str, Callable],
        model_loss_function_name: Optional[str]=None,
        model_update_function_name: Optional[str]=None,
        dataset: Optional[DataSet]=None,
        model_hyperparameters: Optional[dict]=None,
        model_data_filename: Optional[str]=None,
        object_name: Optional[str]=None, has_log_id: Optional[bool]=False,
        do_print_tensor_function_attr=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.MODEL)

        # Make debug log of loading the base model
        log._log_debug(
            "Loading the base model...",
            self.log_id
        )

        # Initialize the model log filename
        self.log_filename = None

        # Set the model attributes if no model data filename was provided or
        #   the model data fails to load
        if model_data_filename is None or not self.load(model_data_filename):

            # Initialize the model name
            if object_name is None:
                object_name = "model"

            # Set the model name
            self.name = object_name

            # Set the model layers
            self.layers = model_layers

            # Initialize the model hyperparameters if not provided
            if model_hyperparameters is None:
                model_hyperparameters = {}

            # Update the model hyperparameters with the default model hyperparameters
            #   for missing keys and values
            for key, value in default_model_hyperparams.items():
                if key not in model_hyperparameters \
                                or model_hyperparameters[key] is None:
                    model_hyperparameters[key] = value
            
            # Set the model hyperparameters
            self.hyperparameters = model_hyperparameters

            # Set the model forward pass and backpropagation functions
            self.forward_pass_functions = model_forward_pass_functions
            self.backpropagation_functions = model_backpropagation_functions

            # Set the model loss and update functions
            self.loss_function = (
                model_loss_function_name,
                _get_tensor_function(
                    tensor_function_name=model_loss_function_name,
                    tensor_function_cache_parameters=self.hyperparameters,
                    do_print_tensor_function_attr=do_print_tensor_function_attr
                )
            )

            '''
            # Update all of the model layers cache with the model hyperparameters
            for layer in self.layers:
                layer._update_cache(self.hyperparameters, ignore_none=True)
            '''

            # NOTE: The model does not hold the update function pointer -- rather
            #   its layers do
            self.update_function = (
                model_update_function_name,
                None
            )

        # If a dataset was provided, use that dataset
        # NOTE: This overrides the dataset from the model data
        if dataset is not None:
            self.set_dataset(dataset)

        # Make debug log with loading the base model
        log._log_debug(
            f"Successfully loaded the base model!",
            self.log_id
        )


    # ========================== MODEL USER METHODS ===========================

    model_attribute_names = {
        'name',
        'layers',
        'dataset',
        'hyperparameters',
        'forward_pass_functions',
        'backpropagation_functions',
        'loss_function'
    }

    MODEL_DATA_DIR = f'{Path(__file__).parent}/../data/'

    def load(self,
        model_data_filename: str
    ) -> bool:
        """
        Load the model data.

        Args:
            model_filename (str): The name for the model data file

        Return:
            Boolean indicating if loading the model data was successful
        """
        # Make info log of loading the model data from file
        log._log_info(
            "Loading the model data from file...",
            self.log_id
        )

        try:
            # Load the model data
            model_data = torch.load(self.MODEL_DATA_DIR + model_data_filename)

            # Parse the model data
            # Initialize the missing keys list
            missing_keys = []

            # Try setting the model attributes from the model data
            for attr_name in self.model_attribute_names:
                key = f"model_{attr_name}"
                # Check if the key exists in the model data
                if key in model_data:
                    # Set the model attribute to model data value
                    setattr(self, attr_name, model_data[key])
                else:
                    # Append the missing key to the missing keys list
                    missing_keys.append(key)

            # If there are any missing keys, log error and return False
            if len(missing_keys) > 0:
                missing_keys_str = ''
                for key in missing_keys:
                    missing_keys_str += key + ', '

                # Make log warning for failure to load some of the model data, and
                #   return False
                log._log_warning(
                    "Failed to load the following from the model data:\n"
                    f"{missing_keys_str[:-2]}",
                    self.log_id
                )
                return False
            
            # Else, make info log for successfully loading the model data from file
            #   and return True
            log._log_info(
                "Successfully loaded the model data from file!",
                self.log_id
            )
            return True

        except:
            # Make warning log for failure to load the model data file, and
            #   return False
            log._log_warning(
                "Could not load the model data from the provided file!",
                self.log_id
            )
            return False

    def save(self,
        model_data_filename: Optional[str]=None
    ) -> bool:
        """
        Save the model data.

        Args:
            model_filename (str): The name for the model data file

        Return:
            Boolean indicating if saving the model data was successful
        """
        # Make info log for saving the model data to file
        log._log_info(
            "Saving the model data to file...",
            self.log_id
        )

        # Initialize the model data file if filename not provided
        if model_data_filename is None:
            model_data_filename = log._get_object_name(self.log_id)
            print(f'\nModel data file: {model_data_filename}')
            if model_data_filename is not None:
                model_data_filename += '.pt'

        # Load the model data only if the model filename exists
        if model_data_filename is not None:
            # Initialize the model data dict
            model_data = {}

            # Use model attribute names for the model data keys
            for attr_name in self.model_attribute_names:
                key = f"model_{attr_name}"
                # Store the model attribute in the model data
                model_data[key] = getattr(self, attr_name)
            
            try:
                # Store the model data in the model data file
                torch.save(model_data, self.MODEL_DATA_DIR + model_data_filename)
                # Make info log for successfully saving the model data and return True
                log._log_info(
                    "Successfully saved the model data to file!",
                    self.log_id
                )
                return True

            except:
                # Make warning log for failure to save model data to the specified
                #   file, and return False
                log._log_warning(
                    "Could not save the model data to the specified file!",
                    self.log_id
                )
                return False

        else:
            # Make warning log for failure to save model data since the filename was
            #   not specified, and return False
            log._log_warning(
                "Could not save the model data since the model data filename was "
                "not provided!",
                self.log_id
            )
            return False
        
    def _make_log(self,
        messages: Union[str, list[str]],
        filename: Optional[str]=None,
        mode: Optional[str]=None
    ) -> bool:
        """
        Make a log for the model.

        Args:
            filename (str): The name of the log file
            mode (str): The write mode

        Return:
            Boolean indicating if logging was successful
        """
        # Check if a filename was not provided
        if filename is None:
            filename = self.log_filename

        # Check if the filename exists
        if filename is not None:
            
            # Check if a valid write mode was not provided
            if mode is None or mode != 'a' or mode != 'w':
                mode = 'a'

            try:
                with open(filename, mode) as logfile:
                    # Make debug log for writing to the specified file,
                    log._log_debug(
                        f"Writing to {filename}...",
                        self.log_id
                    )

                    # If messages is a str, make it a list
                    if isinstance(messages, str):
                        messages = [messages]

                    # Log the messages to the specified file, and return True
                    for message in messages:
                        logfile.write(message)
                    return True

            except:
                # Make warning log for failure to write to the specified file,
                #   and return False
                log._log_warning(
                    f"Failed to write to {filename}!",
                    self.log_id
                )
                return False
            
        # Else, make warning log for failure to log the message since the
        #   log filename wasn't provided, and return False
        log._log_warning(
            "Failed to log the message since the log filename was not provided!",
            self.log_id
        )
        return False

    def set_dataset(self, dataset: DataSet):
        """
        Set the dataset for the model.

        Args:
            dataset (DataSet): The dataset for the model

        Return:
            None
        """
        self.dataset = dataset

        # Make info log for setting the model dataset
        log._log_info(
            "Set the model dataset!",
            self.log_id
        )

    def update_sequence_parameters(self,
        sequence_name: str,
        sequence: list,
        update_parameters: dict[str, Any]
    ) -> bool:
        """
        Update parameters for a sequence of layers.

        Args:
            sequence (list): List of layers
            update_parameters (dict[str, Any]): Dict of parameters to update

        Return:
            Boolean indicating if updating layer parameters is successful
        """
        # Make debug log for updating sequence parameters
        log._log_debug(
            f"Updating model {sequence_name} sequence parameters",
            self.log_id
        )

        for layer in sequence:
            # Check if updating layer parameters was not successful
            if not layer.update_parameters(update_parameters):
                # Make warning log for failure to update model sequence parameters
                #   since a layer failed to update, and return False
                log._log_warning(
                    "Could not update model sequence parameters since failed to update "
                    "layer parameters!",
                    self.log_id
                )
                return False

        # Make debug log for successfully updating all layer's parameters,
        #   and return True
        log._log_debug(
            "Successfully updated model sequence parameters!",
            self.log_id
        )
        return True

    def train(self,
        # If loading the training data from files
        data_sources: Optional[tuple[Any, ...]]=None,

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

        # If using preloaded training data
        training_data: Optional[Tensor]=None,
        training_examples_tensor: Optional[Tensor]=None,
        target_labels_tensor: Optional[Tensor]=None,

        # General data loading params
        training_test_split=-1.0,
        load_test_only=False,
        use_test_labels=False,

        # Prediction params
        num_predictions=NUM_PREDICTIONS,
        probability_threshold=PROBABILITY_THRESHOLD,

        # Crossvalidation
        num_folds=-1,

        # General params
        train_fraction=-1.0,
        num_epochs=-1,
        batch_size=-1,
        eps=-1,
        patience=-1,
        use_patience=False,
        training_results_log_filename: Optional[str]=None,
        do_print_messages: Optional[bool]=False
    ) -> bool:
        """
        Train the model on the given data.

        Args:
            NOTE: See DataSet.load_data for more info on data loading args

            training_data (Tensor): The training data tensor (if using a
                single tensor for both training examples and training labels)
            training_examples_tensor (Tensor): The training examples tensor
            training_labels_tensor (Tensor): The training labels tensor
            num_predictions (int): The number of predictions to make
            probability_threshold (float): The threshold for a probability to be true
            train_fraction (float): The factor to scale down the number of
                sample data entries
            num_epochs (int): The number of epochs
            batch_size (int): The batch size
            eps (float): The training threshold
            patience (int): The threshold activation count
            training_results_log_filename (str): The name of the log file for the
                training results
            do_print_messages (bool): Boolean indicating to print training messages

        Return:
            Boolean indicating training success
        """
        # Make info log for training the model
        log._log_info(
            "Training the model...",
            self.log_id
        )
        
        # If training data is provided as a single tensor, parse out the
            # training examples and training labels
        if training_data is not None:
            training_examples_tensor = training_data[:training_test_split]
            target_labels_tensor = training_data[training_test_split:]

        # If training examples and training labels are not provided, load them
        #   from the model's dataset
        if self.dataset is not None and \
                (training_examples_tensor is None or target_labels_tensor is None):
            # Check if loading data sources, examples data, or labels data was provided
            if data_sources is not None \
                            or examples_data is not None \
                            or labels_data is not None:

                if load_test_only:
                    # Load test data
                    load_success = self.dataset.load_test_data(
                        data_sources=data_sources,
                        examples_data=examples_data,
                        examples_data_name=examples_data_name,

                        examples_data_tensor_function_ptr=\
                            examples_data_tensor_function_ptr,

                        examples_data_tensor_function_args=\
                            examples_data_tensor_function_args,

                        examples_data_tensor_function_return_value_keys=\
                            examples_data_tensor_function_return_value_keys,

                        examples_data_tensor_function_name=\
                            examples_data_tensor_function_name,

                        labels_data=labels_data,
                        labels_data_name=labels_data_name,

                        labels_data_tensor_function_ptr=\
                            labels_data_tensor_function_ptr,

                        labels_data_tensor_function_args=\
                            labels_data_tensor_function_args,

                        labels_data_tensor_function_return_value_keys=\
                            labels_data_tensor_function_return_value_keys,

                        labels_data_tensor_function_name=labels_data_tensor_function_name
                    )

                else:
                    # Load training data
                    load_success = self.dataset._load_training_test_data(
                        data_sources=data_sources,
                        examples_data=examples_data,
                        examples_data_name=examples_data_name,

                        examples_data_tensor_function_ptr=\
                            examples_data_tensor_function_ptr,

                        examples_data_tensor_function_args=\
                            examples_data_tensor_function_args,

                        examples_data_tensor_function_return_value_keys=\
                            examples_data_tensor_function_return_value_keys,

                        examples_data_tensor_function_name=\
                            examples_data_tensor_function_name,

                        labels_data=labels_data,
                        labels_data_name=labels_data_name,

                        labels_data_tensor_function_ptr=\
                            labels_data_tensor_function_ptr,

                        labels_data_tensor_function_args=\
                            labels_data_tensor_function_args,

                        labels_data_tensor_function_return_value_keys=\
                            labels_data_tensor_function_return_value_keys,

                        labels_data_tensor_function_name=\
                            labels_data_tensor_function_name,

                        training_test_split=training_test_split
                    )

                # Check if loading data failed
                if not load_success:
                    # Make error log for failure to train model since loading the
                    #   training data failed, and return False
                    log._log_error(
                        "Could not train the model since loading the data failed!",
                        self.log_id
                    )
                    return False
                
            # Set the training examples tensor and target labels tensor from
            #   the dataset
            training_tensors = self.dataset.get_training_data(
                use_tensors=True
            )

            # Check if training tensors exist
            if training_tensors is not None:
                # Set the training examples and target labels tensors
                training_examples_tensor, target_labels_tensor = \
                                training_tensors
                
                # Check if using test labels
                if use_test_labels:

                    test_tensors = self.dataset.get_test_data()
                    # Check if test tensors exist
                    if test_tensors is not None:
                        # Update the target labels tensor to the test labels
                        target_labels_tensor = test_tensors[1]

                    else:
                        # Make error log for failure to train the model since getting
                        #   the test tensors failed, and return False
                        log._log_error(
                            "Could not train the model since getting the test "
                            "tensors failed!",
                            self.log_id
                        )
                        return False

                # Check if the target labels tensor failed to load
                if target_labels_tensor is None:
                    # Make error log for failure to train the model since getting the
                    #   target labels tensor failed, and return False
                    log._log_error(
                        "Could not train the model since getting the target labels "
                        "tensor failed!",
                        self.log_id
                    )
                    return False

                '''SANITY CHECK
                if training_examples_tensor is not None:
                    print("\nTraining examples tensor shape = "
                        f"{training_examples_tensor.shape}")
                    
                if target_labels_tensor is not None:
                    print("\nTarget labels tensor shape = "
                        f"{target_labels_tensor.shape}")
                '''

            else:
                # Make error log for failure to train the model since getting the
                #   training tensors failed, and return False
                log._log_error(
                    "Could not train the model since the getting the training "
                    "tensors failed!",
                    self.log_id
                )
                
        elif self.dataset is None:
            # Make error log for failure to train the model since the dataset was
            #   not initialized, and return False
            log._log_error(
                "Could not train the model since the dataset was not initialized!",
                self.log_id
            )
            return False

        # Only run the training loop if training examples and training labels
        #   tensors are provided
        if training_examples_tensor is not None and target_labels_tensor is not None:

            # Make debug log for successfully getting the training examples and
            #   target labels tensors
            log._log_debug(
                "Successfully retrieved the training examples and target "
                "labels tensors!",
                self.log_id
            )
            # Get the total number of training examples/target labels
            num_training_examples = training_examples_tensor.shape[0]

            # Check if valid training fraction was provided
            if train_fraction > 0 and train_fraction <= 1:
                # Get the new number of training examples/target labels
                num_training_examples = int(train_fraction * num_training_examples)

                # Reduce the training examples and target labels tensors by the
                #   train fraction
                training_examples_tensor = \
                                training_examples_tensor[:num_training_examples]
                target_labels_tensor = \
                                target_labels_tensor[:num_training_examples]

            # Get the model training hyperparameters
            if num_epochs < 0: num_epochs = self.hyperparameters['num_epochs']
            if batch_size < 0: batch_size = self.hyperparameters['batch_size']
            if num_predictions < 0:
                num_predictions = self.hyperparameters['num_predictions']
            if num_folds < 0: num_folds = self.hyperparameters['num_folds']
            if eps == float('-inf'): eps = self.hyperparameters['eps']
            if patience < 0: patience = self.hyperparameters['patience']

            # Get the crossvalidation and non-crossvalidation training examples
            #   and target labels tensor sets
            training_sets = self._get_cross_validation_sets(
                training_examples=training_examples_tensor,
                target_labels=target_labels_tensor,
                num_folds=num_folds
            )
            non_cv_training_examples = training_sets[0]
            non_cv_target_labels = training_sets[1]
            cv_training_examples = None
            cv_target_labels = None

            # Check if there are more than 2 cv sets
            if len(training_sets) > 2:
                cv_training_examples = training_sets[2]
                cv_target_labels = training_sets[3]

            # Update the number of training examples
            num_training_examples = non_cv_training_examples.shape[0]

            # Run the model training loop for the specified number of epochs
            e = 0 # epoch index
            b = 0 # batch index
            p = 0 # Patience count
            scalar_loss = float('inf')

            # Record the training accuracy metrics
            correct_predictions = 0
            total_predictions = 0
            correct_labels = 0
            total_labels = 0
            true_pos = 0
            false_pos = 0
            true_neg = 0
            false_neg = 0

            # Initialize the model training results messages log
            training_results_log = []

            print(f"\nTraining the model on {num_training_examples} examples "
                  f"\nwith a batch size of {batch_size}...")

            while e < num_epochs:

                # Record the epoch accuracy metrics
                epoch_correct_predictions = 0
                epoch_total_predictions = 0
                epoch_correct_labels = 0
                epoch_total_labels = 0
                epoch_true_pos = 0
                epoch_false_pos = 0
                epoch_true_neg = 0
                epoch_false_neg = 0
                    
                while b < num_training_examples:

                    # Get the current training and target label batches
                    start = b
                    stop = min(b + batch_size, num_training_examples)
                    training_batch = non_cv_training_examples[start:stop]
                    target_batch = non_cv_target_labels[start:stop]

                    # Get the training loop output values
                    training_loop_output_values = \
                        self._run_training_loop(
                            training_batch=training_batch,
                            target_batch=target_batch,
                            num_predictions=num_predictions,
                            probability_threshold=probability_threshold,
                            do_return_dict=True
                        )
                    
                    # Make sure the training loop output values exist
                    if training_loop_output_values is not None:
                        assert(isinstance(training_loop_output_values, dict))

                        # Get the scalar loss, prediction labels, and accuracy metrics
                        scalar_loss = training_loop_output_values['scalar_loss']
                        probabilities = \
                                training_loop_output_values['probabilities']
                        accuracy_metrics = \
                            training_loop_output_values['accuracy_metrics']
                        
                        # Update the epoch accuracy metrics
                        epoch_correct_predictions += \
                                        accuracy_metrics['correct predictions']
                        epoch_total_predictions += \
                                        accuracy_metrics['total predictions']
                        # Check if labels were measured
                        if 'correct_labels' in accuracy_metrics:
                            epoch_correct_labels += \
                                            accuracy_metrics['correct labels']
                            epoch_total_labels += \
                                            accuracy_metrics['total labels']
                            epoch_true_pos += \
                                            accuracy_metrics['true positives']
                            epoch_false_pos += \
                                            accuracy_metrics['false positives']
                            epoch_true_neg += \
                                            accuracy_metrics['true negatives']
                            epoch_false_neg += \
                                            accuracy_metrics['false negatives']
                            
                        # Update the training accuracy metrics
                        correct_predictions += epoch_correct_predictions
                        total_predictions += epoch_total_predictions
                        correct_labels += epoch_correct_labels
                        total_labels += epoch_total_labels
                        true_pos += epoch_true_pos
                        false_pos += epoch_false_pos
                        true_neg += epoch_true_neg
                        false_neg += epoch_false_neg
                                
                        # Check if the scalar loss is below epsilon
                        if scalar_loss < eps:
                            # If using patience, update and check patience index
                            if use_patience:
                                p += 1
                                # Stop training if the patience threshold is reached
                                if p == patience:
                                    # Make info log for stopping model training since
                                    #   the patience threshold was met, and stop epoch
                                    log._log_info(
                                        "Model training stopped due to reaching the "
                                        "patience threshold!",
                                        self.log_id
                                    )
                                    break
                        
                            else:
                                # Make info log for stopping model training since the
                                #   loss threshold was met, and stop epoch
                                log._log_info(
                                    "Training stopped due to reaching the "
                                    "loss threshold!",
                                    self.log_id
                                )
                                break
                        else:
                            # Reset the patience level since the scalar loss was above
                            #   the loss threshold
                            p = 0

                    # Else, the training loop failed to produce output values
                    else:
                        # Make error log for failure to train the model since the
                        #   training loop produced no output values
                        log._log_error(
                            "Could not train the model since the training loop "
                            "failed to produce output values!",
                            self.log_id
                        )

                    b += batch_size
                
                # End of the epoch

                '''
                # Initialize the crossvalidation accuracy metrics
                cv_accuracy_metrics = None

                # Check if crossvalidation training examples and target labels were provided
                if cv_training_examples is not None and cv_target_labels is not None:
                    # Get the model crossvalidation prediction labels
                    cv_predictions = self.predict(
                        query_examples_tensor=cv_training_examples,
                        num_predictions=num_predictions,

                        )
                '''
                    
                # Get, log, and print the epoch accuracy metrics
                prediction_accuracy = round(
                    epoch_correct_predictions / epoch_total_predictions, 4
                )

                epoch_msg = f"\nEpoch #{e+1}: " \
                                + f"Prediction accuracy = {prediction_accuracy}"
                
                training_results_log.append(epoch_msg)

                if do_print_messages:
                    print(epoch_msg, end='')

                # Check if label metrics were measured for the epoch
                if epoch_total_labels > 0:
                    labels_accuracy = round(
                        epoch_correct_labels / epoch_total_labels, 4
                    )

                    precision = round(
                        epoch_true_pos / epoch_true_pos + epoch_false_pos, 4
                    )

                    recall = round(
                        epoch_true_pos / epoch_true_pos + epoch_false_neg, 4
                    )

                    specificity = round(
                        epoch_true_neg / epoch_true_neg + epoch_false_pos, 4
                    )

                    f1_score = round(
                        2 * precision * recall / (precision + recall), 4
                    )

                    labels_msg = f"\n\tLabels accuracy = {labels_accuracy}, " \
                                  +  f"Precision = {precision}, " \
                                  +  f"Recall = {recall}, " \
                                  +  f"Specificity = {specificity}, " \
                                  +  f"F1 Score = {f1_score}, "
                    
                    training_results_log.append(labels_msg)

                    if do_print_messages:
                        print(labels_msg, end='')
                        
                # Stop training if the loss or patience thresholds were met
                if scalar_loss < eps or p == patience:
                    return True
                
                # Else
                e += 1

                # Randomize the training data if running another epoch
                if e < num_epochs:
                    rand_nums = torch.randperm(num_training_examples)
                    non_cv_training_examples = non_cv_training_examples[rand_nums]
                    non_cv_target_labels = non_cv_target_labels[rand_nums]

            # End of all epochs

            # Get, log, and print the training accuracy metrics
            prediction_accuracy = round(
                correct_predictions / total_predictions, 4
            )

            training_msg = f"\nTraining results (averaged over all epochs): " \
                            + f"Prediction accuracy = {prediction_accuracy}"
            
            training_results_log.insert(1, training_msg)

            if do_print_messages:
                print(f'\ntraining_msg', end='')

            # Check if label metrics were measured for the epoch
            if total_labels > 0:
                labels_accuracy = round(
                    correct_labels / total_labels, 4
                )

                precision = round(
                    true_pos / true_pos + false_pos, 4
                )

                recall = round(
                    true_pos / true_pos + false_neg, 4
                )

                specificity = round(
                    true_neg / true_neg + false_pos, 4
                )

                f1_score = round(
                    2 * precision * recall / (precision + recall), 4
                )

                labels_msg = f"\n\tlabels accuracy = {labels_accuracy}, " \
                                +  f"precision = {precision}, " \
                                +  f"recall = {recall}, " \
                                +  f"Specificity = {specificity}, " \
                                +  f"F1 Score = {f1_score}, "
                
                training_results_log.insert(2, labels_msg)

                if do_print_messages:
                    print(labels_msg, end='')

            # Check if training results log filename was not provided
            if training_results_log_filename is None:
                # Initialize the training results log filename using the model name
                training_results_log_filename = f'{self.name}.training_log.log'
            
            # Log the training results
            self._make_log(training_results_log, training_results_log_filename)
                
        else:
            # Make error log for failure to train the model since getting the training
            #   tensors were not initialized, and return False
            log._log_error(
                "Could not train the model since the training tensors were not "
                "initialized!",
                self.log_id
            )
            return False
    
        # Make info log for success with model training
        log._log_info(
            "Successfully trained the model!",
            self.log_id
        )
        return True

    def predict(self,
        query_examples: Optional[list[Any]]=None,
        query_examples_tensor: Optional[Tensor]=None,
        probabilities: Optional[Tensor]=None,
        true_labels: Optional[Tensor]=None,
        true_labels_data: Optional[list[Any]]=None,
        use_test_labels: Optional[bool]=False,
        num_predictions=NUM_PREDICTIONS,
        probability_threshold=PROBABILITY_THRESHOLD
    ) -> Union[
        list[Any],
        Tensor,
        None
    ]:
        """
        Predict the label for the provided example or set of probabilities.

        Args:
            query_examples (Any): The query examples data
            query_examples_tensor (Tensor): The query examples tensor
            probabilities (Tensor): The probabilities tensors
            true_labels (Tensor): The true labels tensor
            true_labels_data (list[Any]): The true labels data
            use_test_labels (bool): The boolean indicating if using test labels
            num_predictions (int): The number of predictions to make
            probability_threshold (float): The threshold for a probability to be true

        Return:
            predicted_labels (Tensor): The predicted labels tensor
            probabilities (Tensor): The probabilities tensor
        """
        # Make info log for performing model prediction
        log._log_info(
            f"Performing model prediction...",
            self.log_id
        )

        # If probabilities are not provided but query examples are, run the
        #   query examples through the forward pass and loss calculation to get
        #   the probabilities
        if probabilities is None and not \
            (query_examples is None and query_examples_tensor is None):
            # Make debug log for getting the probabilities since they were not
            #   initially provided
            log._log_debug(
                f"Probabilities not provided, getting the probabilities...",
                self.log_id
            )

            # If the query example tensor is not provided, get it from the
            #   query example data
            if query_examples_tensor is None and self.dataset is not None:
                query_example_tensor = self.dataset._item_to_tensor(
                    example=query_examples
                )

            # Make sure the query examples tensor exists
            if query_examples_tensor is not None:
                # Make debug log for getting the logits
                log._log_debug(
                    f"Getting the logits...",
                    self.log_id
                )

                forward_pass_output_values = self._forward_pass(
                    x=query_examples_tensor
                )

                # Get the logits
                assert(isinstance(forward_pass_output_values, tuple))
                logits = forward_pass_output_values[0]

            else:
                # Make error log for failure to perform model prediction since the
                #   query examples could not be converted to tensor, and return None
                log._log_error(
                    f"Could not perform model prediction since the query examples "
                    f"could not be converted to tensor!",
                    self.log_id
                )
                return None

            # Check if the true labels were not provided
            if true_labels is None:

                # Check if the true labels data was not provided
                if true_labels_data is None:

                    # Make sure the model dataset exists
                    if self.dataset is not None:
                        # If using test labels, set the labels data to the dataset
                        #   test labels
                        if use_test_labels:
                            test_data = self.dataset.get_test_data(
                                use_tensors=False
                            )

                            # Make sure the test data and labels data exists
                            if test_data is not None and test_data[1] is not None:
                                assert(isinstance(test_data[1], list))
                                true_labels_data = test_data[1]
                            
                            # Make sure the true labels data exists
                            if true_labels_data is None:
                                # Make error log for failure to make model prediction
                                #   since the test data was not initialized,
                                #   and return None
                                log._log_error(
                                    "Could not perform model prediction since the "
                                    "test data was not initialized!",
                                    self.log_id
                                )
                                return None

                        # Else, use the training labels
                        else:
                            training_data = self.dataset.get_training_data()

                            # Make sure the test data and labels data exists
                            if training_data is not None \
                                            and training_data[1] is not None:
                                labels_data = training_data[1]

                            # Make sure the labels data exists
                            if labels_data is None:
                                # Make error log for failure to make model prediction
                                #   since the training data was not initialized,
                                #   and return None
                                log._log_error(
                                    "Could not perform model prediction since the "
                                    "training data was not initialized!",
                                    self.log_id
                                )
                                return None

                    else:
                        # Make error log for failure to make model prediction since
                        #   the dataset was not initialized, and return None
                        log._log_error(
                            "Could not perform model prediction since the dataset "
                            "was not initialized!",
                            self.log_id
                        )
                        return None

                    # # Else, get the true labels tensor
                    true_labels = self.dataset._items_to_tensor(
                        labels=true_labels_data
                    )

                # Check if the true labels tensor does not exist
                if true_labels is None:
                    # Make error log for failure to perform model prediction since
                    #   converting the target labels data to tensor failed, and
                    #   return None
                    log._log_error(
                        "Could not perform model prediction since converting the "
                        "target labels data to tensor failed!",
                        self.log_id
                    )
                    return None

                # Else, get the probabilities from the logits
                loss_output_values = self._calculate_loss(
                    logits=logits,
                    true_labels=true_labels,
                    do_return_dict=True
                )

                # Check if the loss output values exist
                if loss_output_values is not None:
                    assert(isinstance(loss_output_values, dict))
                    # Get the probabilities
                    probabilities = loss_output_values['probabilities']

                else:
                    # Make error log for failure to perform model prediction since
                    #   the loss calculation produced no output values, and return None
                    log._log_error(
                        "Could not perform model prediction since the loss "
                        "calculation returned no values!",
                        self.log_id
                    )
                    return None

        # Make sure the true labels exist
        if true_labels is None:
            # Make error log for failure to perform model prediction since the true
            #   labels weren't provided, and return None
            log._log_error(
                "Could not perform model prediction since the true labels were "
                "not provided!",
                self.log_id
            )
            return None
        
        assert(probabilities is not None)
        # Get the loss function name
        loss_function_name = self.loss_function[0]
        
        # Check if using binary cross entropy loss
        if loss_function_name == 'binary_cross_entropy_loss':

            # Get the predictions using the predictions threshold
            predictions = (probabilities >= probability_threshold).to(
                                true_labels.dtype
                            )

        # Else, the model is using cross entropy loss
        else:
            
            # Get the indices of the top predictions
            top_preds_indices = probabilities.topk(
                num_predictions, dim=-1
            ).indices

            # Get the predictions
            predictions = \
                (top_preds_indices == true_labels.unsqueeze(-1)).any(dim=-1)
            
        # Return predicted labels if the true labels data exists
        # NOTE: This is the case if query examples or true labels data were provided
        #   as arguments
        if true_labels_data is not None:
            return true_labels_data[predictions]
        
        # Else, just return the predictions
        return predictions

    # ======================== MODEL DATASET METHODS ==========================

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
            NOTE: See dataset.get_training_data for info on the args.

        Return:
            model_training_data (tuple): Tuple of tensors or lists
        """
        # Set the training data name
        data_name = 'training data'

        # Check if getting the training data tensors
        if use_tensors:
            data_name += ' tensors'
        
        # Make debug log for getting the training data / tensors
        log._log_debug(
            f"Getting the {data_name}...",
            self.log_id
        )

        # Check if the dataset was set
        if self.dataset is not None:
            model_training_data = self.dataset.get_training_data(**locals())

            # Check if the model training data exists
            if model_training_data is not None:
                # Make debug log for successfully getting the training
                #   data / tensors
                log._log_debug(
                    f"Successfully retrieved the {data_name}!",
                    self.log_id
                )
                
                return model_training_data
        
        # Else, make error log for failure to get the training data / tensors since 
        #   the dataset was not initialized, and return None
        log._log_error(
            f"Could not get the {data_name} since the dataset was not initialized!",
            self.log_id
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
            NOTE: See dataset.get_test_data for info on the args.

        Return:
            model_test_data (tuple): Tuple of tensors or lists
        """
        # Set the test data name
        data_name = 'test data'

        # Check if getting the test data tensors
        if use_tensors:
            data_name += ' tensors'
        
        # Make debug log for getting the test data / tensors
        log._log_debug(
            f"Getting the {data_name}...",
            self.log_id
        )

        # Check if the dataset was set
        if self.dataset is not None:
            model_test_data = self.dataset.get_test_data(**locals())

            # Check if the model test data exists
            if model_test_data is not None:
                # Make debug log for successfully getting the test data / tensors
                log._log_debug(
                    f"Successfully retrieved the {data_name}!",
                    self.log_id
                )
                
                return model_test_data
        
        # Else, make error log for failure to get the test data / tensors since 
        #   the dataset was not initialized, and return None
        log._log_error(
            f"Could not get the {data_name} since the dataset was not initialized!",
            self.log_id
        )

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
            NOTE: See dataset.load_data for info on the args.

        Return:
            Boolean indicating if loading data was successful
        """
        # Make info log for loading the sample data
        log._log_info(
            "Loading the sample data...",
            self.log_id
        )

        # Check if the dataset was set
        if self.dataset is not None:
            load_success = self.dataset.load_data(**locals())

            # Check if loading the sample data was successful
            if load_success:
                # Make info log for successfully loading the sample data
                log._log_info(
                    f"Successfully loaded the sample data!",
                    self.log_id
                )
    
            else:
                # Make error log for failure to load the sample data
                log._log_error(
                    f"Failed to load the sample data!",
                    self.log_id
                )

            return load_success

        # Else, make error log for failure to load the sample data since the
        #   dataset was not initialized, and return False
        log._log_error(
            "Could not load the sample data since the dataset was not "
            "initialized!",
            self.log_id
        )
        return False
    
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
            NOTE: See dataset.load_training_data for info on the args.

        Return:
            Boolean indicating if loading data was successful
        """
        # Log loading the training data
        log._log_info(
            "Loading the test data...",
            self.log_id
        )

        # Check if the dataset was set
        if self.dataset is not None:
            load_success = self.dataset.load_training_data(**locals())

            # Check if loading the training data was successful
            if load_success:
                # Make info log for successfully loading the training data
                log._log_info(
                    f"Successfully loaded the training data!",
                    self.log_id
                )
    
            else:
                # Make error log for failure to load the training data
                log._log_error(
                    f"Failed to load the training data!",
                    self.log_id
                )

            return load_success

        # Else, make error log for failure to load the training data since the
        #   dataset was not initialized, and return False
        log._log_error(
            "Could not load the training data since the dataset was not "
            "initialized!",
            self.log_id
        )
        return False
    
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
            NOTE: See dataset.load_test_data for info on the args.

        Return:
            Boolean indicating if loading test data was successful
        """
        # Log loading the test data
        log._log_info(
            "Loading the test data...",
            self.log_id
        )

        # Check if the dataset was set
        if self.dataset is not None:
            load_success = self.dataset.load_test_data(**locals())

            # Check if loading the test data was successful
            if load_success:
                # Make info log for successfully loading the test data
                log._log_info(
                    f"Successfully loaded the test data!",
                    self.log_id
                )
    
            else:
                # Make error log for failure to load the test data
                log._log_error(
                    f"Failed to load the test data!",
                    self.log_id
                )

            return load_success

        # Else, make error log for failure to load the test data since the
        #   dataset was not initialized, and return False
        log._log_error(
            "Could not load the test data since the dataset was not "
            "initialized!",
            self.log_id
        )
        return False
    

    # ========================= MODEL HELPER METHODS ==========================

    sequence_to_layer_types = {
        'transition': 'layer',
        'convolution' : 'convolution_layer',
        'transformer_encoding' : 'transformer_block',
        'transformer_decoding' : 'transformer_block',
        'projection' : 'projection_layer'
    }

    def _get_seq_layers(self,
        sequence_type: str,
        model_sequence_parameters: dict[str, Any],
        model_update_function_name: Optional[str]=None,
        model_init_hyperparameters: Optional[dict[str, Any]]=None,
        model_hyperparameters: Optional[dict[str, Any]]=None,
        model_final_hyperparameters: Optional[dict[str, Any]]=None,
        model_settings: Optional[dict[str, Any]]=None,
        override_model_settings=False,
        do_print_layer_attr=False,
        do_print_tensor_function_attr=False
    ) -> list[Layer]:
        """
        Return a list of the specified sequence layers

        Args:
            sequence_type (str): The sequence type
            model_sequence_parameters (dict[str, Any]): The model sequence
                parameters dict
            model_update_function_name (str): The name of the model update fuction
            model_init_hyperparameters (dict[str, Any]): The model hyperparamters used
                for the first layer
            model_hyperparameters (dict[str, Any]): The model hyperparameters used
                for all the layers
            model_final_hyperparameters (dict[str, Any]): The model hyperparameters
                used for the last layer
            model_settings (dict[str, Any]): Dict of model settings
            override_model_settings (bool): Boolean indicating whether or not
                to override the model settings with layer hyperparameters
            do_print_layer_attr (bool): Boolean indicating whether or not
                to print the layer attributes
            do_print_tensor_function_attr (bool): Boolean indicating whether or
                not to print the tensor/pass function attributes

        Return:
            List of Layer objects
        """
        # Make debug log for getting the sequence layers
        log._log_debug(
            f"Getting layers for the model {sequence_type} sequence...",
            self.log_id
        )

        # Create the sequence layers
        sequence_layers = []

        # Get layer parameter updates from the model hyperparameters
        # Initialize the model hyperparameters if not provided
        if model_hyperparameters is None:
            model_hyperparameters = {}

        # Set the layer sequence parameters to those of the specified sequence type
        layer_sequence_parameters = model_sequence_parameters[sequence_type]

        # Check if the sequence type is valid
        if sequence_type in self.sequence_to_layer_types:
            # Get layer type, forward functions list, backward functions list,
            #   and layers list
            layer_type = self.sequence_to_layer_types[sequence_type]
            layer_pass_function_names = layer_sequence_parameters['pass_functions']
            layers = layer_sequence_parameters['layers']
            # Iterate through the list of layers to append layers
            for i in range(len(layers)):
                # Get the layer hyperparameters
                layer_hyperparameters = layers[i]

                # Initialize the layer parameter updates
                layer_hyperparameter_updates = {}

                # Update the first layer of the sequence with the model
                #   init hyperparameters
                if i == 0 and model_init_hyperparameters is not None:
                    layer_hyperparameter_updates = model_init_hyperparameters

                # Update all layers with the model hyperparameters
                layer_hyperparameter_updates |= model_hyperparameters

                # Update the final layer of the sequence with the model
                #   final parameters
                if i == len(layers) - 1 and model_final_hyperparameters is not None:
                    layer_hyperparameter_updates |= model_final_hyperparameters
                
                # Check if not overriding the model settings
                if not override_model_settings:
                    if model_settings is not None:
                        # Update the layer hyperparameter updates with the
                        #   model settings
                        layer_hyperparameter_updates |= model_settings

                    # Update the layer hyperparameter updates with the
                    #   layer hyperparameters
                    layer_hyperparameter_updates |= layer_hyperparameters

                else:
                    # Update the layer hyperparameter updates with the model settings
                    #   for missing values only
                    if model_settings is not None:
                        layer_hyperparameter_updates |= util._get_update_dict(
                            model_settings, layer_hyperparameter_updates,
                            update_none_only=True
                        )
                    
                # Update the layer hyperparameters with the layer hyperparameter
                #   updates
                layer_hyperparameters |= layer_hyperparameter_updates

                # Append layer with the specified layer type and parameters
                sequence_layers.append(_get_layer(
                    layer_type=layer_type,
                    layer_hyperparameters=layer_hyperparameters,
                    layer_pass_function_names=layer_pass_function_names,
                    layer_update_function_name=model_update_function_name,
                    layer_seq_num=i+1,
                    do_print_layer_attr=do_print_layer_attr,
                    do_print_tensor_function_attr=do_print_tensor_function_attr
                ))

        # Make debug log for successfully getting the sequence layers
        log._log_debug(
            f"Successfully retrieved the sequence layers!",
            self.log_id
        )

        # Return the sequence layers
        return sequence_layers
    
    def _get_learnable_parameters(self,
        do_return_dict=False,
        get_weights_only=False
    ) -> Union[dict, list]:
        """
        Return the model learnable parameters.

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

            # Initialize the learnable parameters dict
            learnable_parameters = {}

            # Get the learnable parameters for each layer as a dict
            for i in range(len(self.layers)):
                learnable_parameters[f'layer {i+1}'] = \
                    self.layers[i]._get_learnable_parameters(
                        do_return_dict=True,
                        get_weights_only=get_weights_only
                    )

        else:
            # Initialize the learnable parameters list
            learnable_parameters = []

            # Get the learnable parameters for each layer as a list
            for layer in self.layers:
                learnable_parameters += layer._get_learnable_parameters(
                    do_return_dict=False, get_weights_only=get_weights_only
                )
                
        return learnable_parameters



    def _run_pass(self,
        layers: list, # List of Layer, ConvolutionLayer,
                            # TransformerBlock, or ProjectionLayer
        x: Optional[Tensor]=None,
        upstream_grad: Optional[Tensor]=None,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[Any, ...]]=None,
        do_return_dict=False
    ) -> tuple[Any, ...]:
        """
        Feed either an input or upstream gradient tensor through the model
            forward pass or backpropagation, respectively.
        NOTE: Supplying x will run the forward pass, while instead supplying
            upstream_grad will run backpropagation.

        Args:
            x (Tensor): The input tensor (for the forward pass)
            upstream_grad (Tensor): The upstream gradient tensor (for backpropagation)
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values
            do_print_attr (bool): Boolean indicating whether or not to print attributes

        Return:
            output_values (Tensor): The output values tuple
        """
        '''
        # Make debug log for running the model pass
        log._log_debug(
            "Running the model pass...",
            self.log_id
        )
        '''

        # Initialize the output values
        output_values = tuple()
        
        # Iterate through the layers
        for layer in layers:
            # Run the forward pass if x was provided
            if x is not None:
                output_values = layer._forward(
                    x=x,
                    kwargs=kwargs,
                    output_keys=output_keys,
                    do_return_dict=do_return_dict
                )
                
                # Get the input from the output values
                # Check if returning a dict
                if do_return_dict:
                    # Get x by values position
                    x = list(output_values.values())[0]
                else:
                    # Get x by position
                    x = output_values[0]

            # Else, run backpropagation if the upstream gradient was provided
            elif upstream_grad is not None:
                output_values = layer._backward(
                    upstream_grad=upstream_grad,
                    kwargs=kwargs,
                    output_keys=output_keys,
                    do_return_dict=do_return_dict
                )
                
                # Get the upstream gradient from the output values
                # Check if returning a dict
                if do_return_dict:
                    # Get the upstream gradient by values position
                    upstream_grad = list(output_values.values())[0]
                else:
                    # Get the upstream gradient by position
                    upstream_grad = output_values[0]

            else:
                # Make warning log for failure to run the model pass since no input
                #   nor upstream gradient tensor was provided 
                log._log_warning(
                    f"Could not run forward pass nor backpropagation since no "
                    f"input or upstream gradient tensor was provided!",
                    self.log_id
                )
                return tuple()

        # Return the output values
        return output_values

    def _get_cross_validation_sets(self,
        training_examples: Tensor,
        target_labels: Tensor,
        num_folds=-1,
        do_random=True
    ) -> Union[
        tuple[Tensor, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor]
    ]:
        """
        Return the crossvalidation and non-cv tensor sets.

        Args:
            training_examples (Tensor): The training examples tensor
            target_labels (Tensor): The target labels tensor
            num_folds (int): The number of crossvalidation folds

        Return:
            cv_training_examples (Tensor): The crossvalidation training
                examples tensor
            cv_target_labels (Tensor): The crossvalidation target labels tensor
            non_cv_training_examples (Tensor): The non-crossvalidation
                training examples tensor
            non_cv_target_labels (Tensor): The non-crossvalidation target
                labels tensor
        """
        # Make debug log for getting the crossvalidation sets
        log._log_debug(
            "Getting the crossvalidation sets...",
            self.log_id
        )

        # Check if the number of folds is less than or equal 1
        if num_folds <= 1:
            # Return the training examples and target labels
            return training_examples, target_labels

        # Get the number of training examples / target labels
        num_training_examples = training_examples.shape[0]

        # Randomize the training examples and target labels
        if do_random:
            rand_nums = torch.randperm(num_training_examples)
            rand_training_examples = training_examples[rand_nums]
            rand_target_labels = target_labels[rand_nums]
        else:
            rand_training_examples = training_examples.clone()
            rand_target_labels = target_labels.clone()

        # Separate the training examples and target labels based on the
        #   number of folds
        # Add another layer of randomization
        cv_num = random.randint(0, num_folds - 1)
        # Get the crossvalidation set size
        cv_size = num_training_examples // num_folds
        # Get the start and stop indices for the crossvalidation set
        cv_start = cv_num * cv_size
        cv_stop = min((cv_num + 1) * cv_size, num_training_examples)

        # Get the crossvalidation and non-cv sets
        cv_training_examples = rand_training_examples[cv_start:cv_stop]
        cv_target_labels = rand_target_labels[cv_start:cv_stop]
        
        non_cv_training_examples = rand_training_examples[0:cv_start]
        non_cv_training_target_labels = rand_target_labels[0:cv_start]

        # Check if the last cv index is less than the number of training examples
        if cv_stop < num_training_examples:
            non_cv_examples = torch.cat(
                (non_cv_training_examples,
                 rand_training_examples[cv_stop:num_training_examples])
            )
            non_cv_target_labels = torch.cat(
                (non_cv_training_target_labels,
                 rand_target_labels[cv_stop:num_training_examples])
            )

        # Make debug log for successfully getting the crossvalidation sets
        log._log_debug(
            "Successfully retrieved the crossvalidation sets!",
            self.log_id
        )

        # Check if the last cv index is less than the number of training examples
        if cv_stop < num_training_examples:
            # Return the crossvalidation and non-cv sets
            return non_cv_examples, non_cv_target_labels, \
                cv_training_examples, cv_target_labels
        
        # Else, just return the cv set
        return cv_training_examples, cv_target_labels

    def _run_training_loop(self,
        training_batch: Tensor,
        target_batch: Tensor,
        num_predictions=NUM_PREDICTIONS,
        probability_threshold=PROBABILITY_THRESHOLD,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any],
        None
    ]:
        """
        Run the model training loop on the given data, performing these steps:
            forward pass --> loss --> backpropagation --> update.

        Args:
            training_batch (Tensor): The training examples batch tensor
            target_batch (Tensor): The target labels batch tensor
            cv_training_examples (Tensor): The crossvalidation training examples tensor
            cv_target_labels (Tensor): The crossvalidation target labels tensor
            num_predictions (int): The number of predictions to make
            probability_threshold (float): The threshold for a probability to be true
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            Tuple of float (scalar loss) and boolean to indicate training
                loop success
        """
        # Make debug log for running the training loop
        log._log_debug(
            "Running the training loop...",
            self.log_id
        )

        # Clone the training and target batches
        cloned_training_batch = training_batch.clone()
        cloned_target_batch = target_batch.clone()

        # Feed the training examples (clone) through the forward pass
        forward_pass_output_values = self._forward_pass(
            cloned_training_batch, do_return_dict=False
        )

        # Check if using return dict
        # NOTE: Can't use dict here since there is no 'logits' key
        # TODO: Fix the dict option for getting logits in the output values
        if False: #do_return_dict:
            assert(isinstance(forward_pass_output_values, dict))
            # Get the logits by name
            logits = forward_pass_output_values['logits']

        else:
            assert(isinstance(forward_pass_output_values, tuple))
            # Get the logits by position
            logits = forward_pass_output_values[0]

        # Get the learnable weights for regularization functions
        learnable_weights = self._get_learnable_parameters(
            get_weights_only=True
        )

        # Perform the loss calculation on the logits
        loss_output_values = self._calculate_loss(
            logits=logits,
            true_labels=cloned_target_batch,
            kwargs={'learnable_weights':learnable_weights},
            do_return_dict=do_return_dict
        )

        # Make sure the loss output values exist
        if loss_output_values is not None:

            # Check if using return dict
            if do_return_dict:
                assert(isinstance(loss_output_values, dict))
                # Get the scalar loss and probabilities by name
                scalar_loss = loss_output_values.pop('scalar_loss')
                probabilities = loss_output_values.pop('probabilities')

            else:
                assert(isinstance(loss_output_values, tuple))
                # Get the scalar loss and probabilities by position
                scalar_loss = loss_output_values[0]
                probabilities = loss_output_values[1]

        else:
            # Make error log for failure to run training loop since there were
            #   no loss function output values, and return None
            log._log_error(
                "Failed to run model training loop since the loss function "
                "produced no output values!",
                self.log_id
            )
            return None

        # Get additional loss backpropagation keyword arguments
        # Check if using return dict
        if do_return_dict:
            assert(isinstance(loss_output_values, dict))
            addl_kwargs = loss_output_values

        else:
            addl_kwargs = {}

        # Perform loss backpropagation on the scalar loss
        backpropagation_output_values = self._loss_backpropagation(
            upstream_gradient=torch.Tensor([1.0]),
            true_labels=cloned_target_batch,
            do_return_dict=False,
            kwargs=addl_kwargs
        )

        # Get the logits gradient and feed it through backpropagation
        assert(isinstance(backpropagation_output_values, tuple))
        logits_grad = backpropagation_output_values[0]

        # Perform backpropagation on the logits gradient
        # NOTE: Don't worry about gettting backpropagation output values
        self._backpropagation(
            upstream_grad=logits_grad,
            do_return_dict=True
        )

        # Make sure the update is successful, otherwise return None
        if not self._update():
            return None
        
        # Get the accuracy metrics
        accuracy_metrics = self._get_pred_accuracy(
            probabilities=probabilities,
            true_labels=cloned_target_batch,
            num_predictions=num_predictions,
            probability_threshold=probability_threshold
        )

        # Make debug log for successfully running the training loop
        log._log_debug(
            "Successfully ran the training loop!",
            self.log_id
        )

        util._print_dict(accuracy_metrics, 'accuracy metrics')
        
        # Check if returning a return values dict
        if do_return_dict:
            # Return the scalar loss, probabilities, and accuracy metrics as a dict
            return {
                'scalar_loss': scalar_loss.item(),
                'probabilities': probabilities,
                'accuracy_metrics': accuracy_metrics
            }
                
        # Else, return those values as a tuple
        return scalar_loss.item(), probabilities, accuracy_metrics

    # ---------------------- MODEL TRAINING LOOP METHODS ----------------------

    def _forward_pass(self,
        x: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform the forward pass on the input.

        Args:
            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            output_values (tuple[Any, ...] | dict[str, Any]): The model forward pass
                output values tuple or dict
        """
        # Make debug log for running model forward pass
        log._log_debug(
            "Running model forward pass...",
            self.log_id
        )

        for name, function in self.forward_pass_functions.items():
            output_values = function(
                x=x,
                kwargs=kwargs,
                output_keys=output_keys,
                do_return_dict=do_return_dict
            )

            # Get x from the output values
            # Check if output values is a dict
            if isinstance(output_values, dict):
                # Get x by values position
                x = list(output_values.values())[0]
            else:
                # Get x by position
                x = output_values[0]

        # Make debug log for successfully running model forward pass
        log._log_debug(
            "Successfully performed model forward pass!",
            self.log_id
        )
        
        return output_values
    
    def _calculate_loss(self,
        logits: Tensor,
        true_labels: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any],
        None
    ]:
        """
        Calculate the loss between the model prediction (output) and target labels

        Args:
            logits (Tensor): The model logits tensor
            true_labels (Tensor): The true labels tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            output_values (tuple[Any, ...] | dict[str, Any]): The model loss
                function output values tuple or dict
        """
        # Make debug log for performing model loss calculation
        log._log_debug(
            "Performing model loss calculation...",
            self.log_id
        )

        # Get the loss function name and pointer
        name, function = self.loss_function
        
        # Return the loss calculation from the prediction and target_labels
        if function is not None:
            
            # Initialize the keyword arguments if not provided
            if kwargs is None:
                kwargs = {}

            # Add logits and target labels to the keyword arguments
            kwargs['logits'] = logits
            kwargs['true_labels'] = true_labels

            # Get the loss output values
            assert(isinstance(function, PassFunction))
            output_values = function._forward(
                kwargs=kwargs,
                output_keys=output_keys,
                do_return_dict=do_return_dict
            )

            # Make debug log for successfully performing model loss calculation
            log._log_debug(
                f"Successfully performed model loss calculation!",
                self.log_id
            )

            return output_values
        
        # Else, make error log for failure to perform model loss calculation since
        #   the loss function was not initialized
        log._log_error(
            "Could not perform model loss calculation since the loss function was "
            "not initialized!",
            self.log_id
        )

    def _loss_backpropagation(self,
        upstream_gradient: Tensor,
        true_labels: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any],
        None
    ]:
        """
        Backpropagate the scalar loss to get the loss gradient.

        Args:
            scalar_loss (Tensor): The scalar loss tensor
            true_labels (Tensor): The true labels tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            output_values (tuple[Any, ...] | dict[str, Any]): The model backpropagate
                loss function output values tuple or dict
        """
        # Make debug log for performing model loss backpropagation
        log._log_debug(
            "Performing model loss backpropagation...",
            self.log_id
        )

        # Get the loss function name and function pointer
        name, function = self.loss_function

        # Return the loss calculation from the prediction and target_labels
        if function is not None:
            # Initialize the keyword arguments if not provided
            if kwargs is None:
                kwargs = {}

            # Add true labels to the keyword arguments
            kwargs['true_labels'] = true_labels

            # Get the loss output values
            assert(isinstance(function, PassFunction))
            backpropagate_loss_function_output_values = function._backward(
                upstream_grad=upstream_gradient,
                kwargs=kwargs,
                output_keys=output_keys,
                do_return_dict=do_return_dict
            )
        
            # Make debug log for successfully performing model loss backpropagation
            log._log_debug(
                f"Successfully performed model loss backpropagation!",
                self.log_id
            )

            return backpropagate_loss_function_output_values
        
        # Else, make error log for failure to perform model loss backpropagation since
        #   the loss function was not initialized
        log._log_error(
            "Could not perform model loss calculation since the loss function was "
            "not initialized!",
            self.log_id
        )
    
    def _backpropagation(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform the backpropagation on the upstream gradient.

        Args:
            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            output_values (tuple[Any, ...] | dict[str, Any]): The model backpropagation
                output values tuple or dict
        """
        # Make debug log for performing model loss backpropagation
        log._log_debug(
            "Performing model loss backpropagation...",
            self.log_id
        )
        
        for name, function in self.backpropagation_functions.items():
            output_values = function(
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

        # Make debug log for successfully performing model backpropagation
        log._log_debug(
            "Successfully performed model backpropagation!",
            self.log_id
        )
        
        return output_values
    
    def _update(self) -> bool:
        """
        Update all of the model's learnable parameters.

        Args:
            None

        Return:
            updates_success (boolean): Boolean indicating success with updating
                model learnable parameters
        """
        # Make debug log for performing model update
        log._log_debug(
            "Performing model update...",
            self.log_id
        )

        updates_successful = True

        for layer in self.layers:
            # Store the boolean result of updating learnable parameters
            # All layers should update successfully, otherwise return False
            if not layer._update():
                updates_successful = False

        # Check if updating all of the layers succeeded
        if updates_successful:
            # Make debug log for successfully performing model update
            log._log_debug(
                f"Successfully performed model update!",
                self.log_id
            )

        # Return boolean indicating successfully updating all the learnable
        #   parameters
        return updates_successful
    
    def _get_pred_accuracy(self,
        probabilities: Tensor,
        true_labels: Tensor,
        predictions: Optional[Tensor]=None,
        num_predictions=NUM_PREDICTIONS,
        probability_threshold=PROBABILITY_THRESHOLD
    ) -> Optional[dict]:
        """
        Get accuracy metrics for predictions.

        Args:
            probabilities (Tensor): The probabilities tensor
            true_labels (Tensor): The true labels tensor
            predictions (Tensor): The predictions tensor
            num_predictions (int): The number of predictions to make
            probability_threshold (float): The threshold for a probability to
                be considered true

        Return:
            accuracy_metrics (dict[str, int]): Dict of accuracy metrics
        """
        # Make debug log for getting model prediction accuracy metrics
        log._log_debug(
            "Getting model prediction accuracy metrics...",
            self.log_id
        )

        # Check if the model has a loss function name
        if self.loss_function is not None:

            # Get the predictions if not provided
            if predictions is None:
                model_prediction = self.predict(
                    probabilities=probabilities,
                    true_labels=true_labels
                )
                assert(isinstance(model_prediction, Tensor))
            predictions = model_prediction
            
            # Initialize the accuracy metrics dict
            accuracy_metrics = {}

            # Get the loss function name
            loss_function_name = self.loss_function[0]

            # Check if using binary cross entropy loss
            if loss_function_name == 'binary_cross_entropy_loss':
                
                # Get the number of correct and total labels
                accuracy_metrics['correct labels'] = \
                                (predictions == true_labels).sum().item()
                accuracy_metrics['total labels'] = probabilities.numel()

                # Get the true positives, false positives, true negatives, and
                #   false negatives
                accuracy_metrics['true positives'] = (
                    (predictions == 1) & (true_labels == 1)
                ).sum().item()
                accuracy_metrics['false positives'] = (
                    (predictions == 1) & (true_labels == 0)
                ).sum().item()
                accuracy_metrics['true negatives'] = (
                    (predictions == 0) & (true_labels == 0)
                ).sum().item()
                accuracy_metrics['false negatives'] = (
                    (predictions == 0) & (true_labels == 1)
                ).sum().item()

            # Else, the model is using cross entropy loss
            else:
                
                # Get the predictions if not provided
                if predictions is None:
                    top_preds_indices = probabilities.topk(
                        num_predictions, dim=-1
                    ).indices

                    # Get the predictions
                    predictions = \
                        (top_preds_indices == true_labels.unsqueeze(-1)).any(dim=-1)

            # Get the number of correct and total predictions
            accuracy_metrics['correct predictions'] = \
                (predictions == true_labels).all(dim=-1).sum().item()
            accuracy_metrics['total predictions'] = predictions.shape[0]
        
            # Make debug log for successfully getting model prediction accuracy metrics
            log._log_debug(
                "Successfully retrieved model prediction accuracy metrics!",
                self.log_id
            )

            # Return the accuracy metrics
            return accuracy_metrics

        # Else, make error log for failure to get prediction accuracy since the
        #   loss function was not initialized, and return None
        log._log_error(
            "Could not retrieve model prediction accuracy since the loss function "
            "was not initialized!",
            self.log_id
        )
        

# ========================== STATIC MODEL METHODS =============================

MODEL_SETTINGS_FILENAME = f"{Path(__file__).parent}/model_settings.json"

def _get_model_settings(
    settings_jsonfile: Optional[str]=None,
    model_type: Optional[str]=None,
) -> Optional[Any]:
    """
    Load model settings for the specified model type.

    Args:
        settings_jsonfile (str): The filename of the settings JSON file
        model_type (str): The model type

    Return:
        The settings file contents
    """
    # Initialize the settings JSON file if not provided
    if settings_jsonfile is None:
        settings_jsonfile = MODEL_SETTINGS_FILENAME

    # Make info log for getting model settings from the specified JSON file
    log._log_info(
        f"Retrieving model settings from {settings_jsonfile}...",
        log.MODEL_MODULE
    )

    # Load model settings from the specified JSON file
    model_settings = util._load_json(settings_jsonfile)

    # Check if the model settings exist
    if model_settings is not None:
        # If a setting name was provided, return only the model settings
        #   associated with the specified model type
        if model_type is not None:
            if model_type in model_settings:
                # Make info log for successfully getting model settings for the
                #   specified model type
                log._log_info(
                    f"Successfully retrieved model settings for {model_type}!",
                    log.MODEL_MODULE
                )
                return model_settings[model_type]
            
            else:
                # Make warning log for failure to get model settings for the
                #   specified model type, and return None
                log._log_warning(
                    f"Failed to retrieve model settings for {model_type}!",
                    log.MODEL_MODULE
                )

        # Else, make info log for successfully getting model settings
        log._log_info(
            f"Succesfully retrieved model settings!",
            log.MODEL_MODULE
        )

        # Return all of the model settings
        return model_settings
    
    # Else, make warning log for failure to load model settings from the settings
    #   JSON file, and return None
    log._log_warning(
        f"Could not load model settings from {settings_jsonfile}!",
        log.MODEL_MODULE
    )


# =========================== TRANSFORMER MODEL ===============================

TRANSFORMER_ATTN_EMBEDDING_SIZE = 256
TRANSFORMER_FINAL_EMBEDDING_SIZE = 768

transformer_model_default_hyperparameters = {
    # Base model hyperparameters
    'batch_size': BATCH_SIZE,
    'num_predictions': NUM_PREDICTIONS,
    'num_folds': NUM_FOLDS,
    'num_epochs': NUM_EPOCHS,
    'eps': loss.EPS,
    'patience': loss.PATIENCE,
    'reg_type': reg.REG_TYPE,
    'reg_strength': reg.REG_STRENGTH,
    'learning_rate': update.LEARNING_RATE,

    # Transformer model hyperparameters
    'max_seq_len': layer.MAX_SEQ_LEN,
    'num_attn_heads': attn.NUM_ATTN_HEADS,
    'embedding_size': TRANSFORMER_ATTN_EMBEDDING_SIZE,
    'feed_fwd_size': layer.FEED_FWD_SIZE,
    'dropout': reg.DROPOUT
}

class Transformer(Model):
    """
    This is the transformer class, which applys attention and masking to its
        encoder (training) and decoder (prediction) blocks.
    """
    def __init__(self,
        # Base model hyperparameters
        batch_size: Optional[int]=None,
        num_pred_labels: Optional[int]=None,
        num_folds: Optional[int]=None,
        num_epochs: Optional[int]=None,
        eps: Optional[float]=None,
        patience: Optional[int]=None,
        reg_type: Optional[str]=None,
        reg_strength: Optional[float]=None,
        learning_rate: Optional[float]=None,

        # Transformer model hyperparameters
        attn_embedding_size: Optional[int]=None,
        feed_fwd_size: Optional[int]=None,
        max_seq_len: Optional[int]=None,
        num_attn_heads: Optional[int]=None,
        dropout: Optional[float]=None,
        final_embedding_size: Optional[int]=None,

        # Model parameters
        model_sequence_parameters: Optional[dict[str, Any]]=None,
        model_loss_function_name: Optional[str]=None,
        model_update_function_name: Optional[str]=None,
        dataset: Optional[DataSet]=None,
        model_data_filename: Optional[str]=None,
        transformer_model_settings_filename: Optional[str]=None,
        override_model_settings=False,
        object_name=None, has_log_id=False,
        do_print_layer_attr=False,
        do_print_tensor_function_attr=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.TRANSFORMER)

        # Make debug log of loading the Transformer model
        log._log_debug(
            "Loading the Transformer model...",
            log.MODEL_MODULE
        )

        # Set the transformer model attributes if no model data filename was
        #   provided or the model data fails to load
        if model_data_filename is None or not self.load(model_data_filename):

            # Get the transformer model hyperparameters
            model_hyperparameters = util._get_update_dict(
                locals(), transformer_model_default_hyperparameters
            )

            # Initialize missing transfomer model hyperparameters with default values
            model_hyperparameters |= util._get_update_dict(
                transformer_model_default_hyperparameters, model_hyperparameters,
                update_none_only=True
            )

            # Get the transformer model init hyperparameters
            model_init_hyperparameters = {
                'embedding_size': attn_embedding_size
            }

            # Get the transformer model final hyperparameters
            model_final_hyperparameters = {
                'pre_embedding_size': attn_embedding_size,
                'embedding_size': final_embedding_size
            }

            # Set the transformer model forward pass function
            model_forward_pass_functions = {
                'encode': self._encode,
                'decode': self._decode,
                'projection': self._project
            }

            # Set the transformer model backpropagation functions
            model_backpropagation_functions = {
                'project_backward': self._project_backward,
                'decode_backward': self._decode_backward,
                'encode_backward': self._encode_backward
            }

            # Get the Transformer model settings
            transformer_settings = _get_model_settings(
                settings_jsonfile=transformer_model_settings_filename,
                model_type='transformer'
            )

            # Check if the Transformer model settings exist
            if transformer_settings is not None:

                # Get the loss and update function names
                # If no loss and update function names were provided, get them from the
                #   Transformer model settings
                if (model_loss_function_name is None or not override_model_settings):
                    if 'loss_function' in transformer_settings:
                        model_loss_function_name = \
                                        transformer_settings['loss_function']
                    if 'update_function' in transformer_settings:
                        model_update_function_name = \
                                        transformer_settings['update_function']
                    
                # Get the model settings (hyperparameters)
                model_settings = transformer_settings['hyperparameters']

                if 'reg_type' in model_hyperparameters:
                    reg_type = model_hyperparameters['reg_type']

                # Update the model update function name if using regularization
                if reg_type is not None:
                    model_update_function_name = tf._get_update_function_name(reg_type)

                # Get the sequence parameters
                # If no sequence parameters were provided, get them from the
                #   Transformer model settings
                if model_sequence_parameters is None:
                    model_sequence_parameters = transformer_settings['sequences']
                    assert(model_sequence_parameters is not None)

                # Set the encoder sequence
                self.encoder_blocks = self._get_seq_layers(
                    sequence_type='encoding',
                    model_sequence_parameters=model_sequence_parameters,
                    model_update_function_name=model_update_function_name,
                    model_init_hyperparameters=model_init_hyperparameters,
                    model_hyperparameters=model_hyperparameters,
                    model_settings=model_settings,
                    override_model_settings=override_model_settings,
                    do_print_layer_attr=do_print_layer_attr,
                    do_print_tensor_function_attr=do_print_tensor_function_attr
                )

                # Set the decoder sequence
                self.decoder_blocks = self._get_seq_layers(
                    sequence_type='decoding',
                    model_sequence_parameters=model_sequence_parameters,
                    model_update_function_name=model_update_function_name,
                    model_hyperparameters=model_hyperparameters,
                    model_settings=model_settings,
                    override_model_settings=override_model_settings,
                    do_print_layer_attr=do_print_layer_attr,
                    do_print_tensor_function_attr=do_print_tensor_function_attr
                )

                # Set the projection sequence
                self.projection_layers = self._get_seq_layers(
                    sequence_type='projection',
                    model_sequence_parameters=model_sequence_parameters,
                    model_update_function_name=model_update_function_name,
                    model_hyperparameters=model_hyperparameters,
                    model_final_hyperparameters=model_final_hyperparameters,
                    model_settings=model_settings,
                    override_model_settings=override_model_settings,
                    do_print_layer_attr=do_print_layer_attr,
                    do_print_tensor_function_attr=do_print_tensor_function_attr
                )

            else:
                # Make warning log for Transformer model settings not loading
                log._log_warning(
                    "No Transformer model settings loaded!",
                    self.log_id
                )

                # Set empty layer lists for the Transformer model
                self.encoder_blocks = []
                self.decoder_blocks = []
                self.projection_layers = []

            # Set the encoder reverse sequence
            self.encoder_blocks_reverse = self.encoder_blocks.copy()
            self.encoder_blocks_reverse.reverse()

            # Set the decoder reverse sequence
            self.decoder_blocks_reverse = self.decoder_blocks.copy()
            self.decoder_blocks_reverse.reverse()

            # Set the projection reverse sequence
            self.projection_layers_reverse = self.projection_layers.copy()
            self.projection_layers_reverse.reverse()

            # Get the model layers
            model_layers = self.encoder_blocks \
                + self.decoder_blocks \
                + self.projection_layers

            # Initialize the base model attributes
            assert(model_update_function_name is not None)
            super().__init__(
                model_layers=model_layers,
                model_forward_pass_functions=model_forward_pass_functions,
                model_backpropagation_functions=model_backpropagation_functions,
                model_loss_function_name=model_loss_function_name,
                model_update_function_name=model_update_function_name,
                dataset=dataset,
                model_hyperparameters=model_hyperparameters,
                object_name=object_name, has_log_id=True,
                do_print_tensor_function_attr=do_print_tensor_function_attr
            )

        # If a dataset was provided, use that dataset
        # NOTE: This overrides the dataset from the model data
        elif dataset is not None:
            self.set_dataset(dataset)

        # Check if the Transformer model has layers
        if len(self.layers) > 0:
            # Make debug log for successfully loading the Transformer model
            log._log_debug(
                "Successfully loaded the Transformer model!",
                log.MODEL_MODULE
            )

        else:
            # Make warning log for loading the Transformer model
            log._log_warning(
                "Partially loaded the Transformer model! DO NOT USE!",
                log.MODEL_MODULE
            )
    
    def _encode(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Encode the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The encode output values tuple or dict
        """
        return self._run_pass(
            layers=self.encoder_blocks,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    
    def _encode_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform encode backward on the upstream gradient.

            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The encode backward output values tuple or dict
        """
        return self._run_pass(
            layers=self.encoder_blocks_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    
    def _decode(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Decode the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The decode output values tuple or dict
        """
        return self._run_pass(
            layers=self.decoder_blocks,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    
    def _decode_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform decode backward on the upstream gradient.

            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The decode backward output values tuple or dict
        """
        return self._run_pass(
            layers=self.decoder_blocks_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    
    def _project(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform projection on the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The projection output values tuple or dict
        """
        return self._run_pass(
            layers=self.projection_layers,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    
    def _project_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform projection backward on the input.

            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The projection backward output values tuple or dict
        """
        return self._run_pass(
            layers=self.projection_layers_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    

def get_transformer(
    transformer_model_args: Any
) -> Optional[Transformer]:
    """
    Return a Transformer model using the specified arguments.

    Args:
        transformer_model_args (Any): Arguments for the Transformer model

    Return:
        A Transformer model
    """
    # Make info log for getting the Transformer model
    log._log_info(
        "Getting the Transformer model...",
        log.MODEL_MODULE
    )

    # Get the Transformer model
    transformer_model = Transformer(**transformer_model_args)

    # Check if the Transformer model has any layers
    if len(transformer_model.layers) > 0:
        # Make info log for successfully retrieving the Transformer model
        log._log_info(
            "Successfully retrieved the Transformer model!",
            log.MODEL_MODULE
        )
        return transformer_model

    # Else, make error log for failure to retrieve the Transformer model, and
    #   return None
    log._log_error(
        "Failed to retrieve the Transformer model!",
        log.MODEL_MODULE
    )

    
    
# ======================== CONVOLUTION NEURAL NETWORK =========================

NUM_IN_CHANNELS = 3
NUM_OUT_FEATURES = 64
CNN_EMBEDDING_SIZE = 768

cnn_model_default_hyperparameters = {
    # Base model hyperparameters
    'batch_size': BATCH_SIZE,
    'num_pred_labels': NUM_PREDICTIONS,
    'num_folds': NUM_FOLDS,
    'num_epochs': NUM_EPOCHS,
    'eps': loss.EPS,
    'patience': loss.PATIENCE,
    'reg_type': reg.REG_TYPE,
    'reg_strength': reg.REG_STRENGTH,
    'learning_rate': update.LEARNING_RATE,

    # CNN model hyperparameters
    'kernel_size': conv.KERNEL_SIZE,
    'stride': conv.STRIDE,
    'padding': conv.PADDING,
    'pool_size': pool.KERNEL_SIZE,
    'pool_stride': pool.STRIDE,
    'pool_type': pool.POOL_TYPE,
}

class CNN(Model):
    """
    This is the convolution neural network class, which applies convolution
        and linear projection to an input to produce a prediction.
    """
    def __init__(self,
        # Base model hyperparameters
        batch_size: Optional[int]=None,
        num_pred_labels: Optional[int]=None,
        num_folds: Optional[int]=None,
        num_epochs: Optional[int]=None,
        eps: Optional[float]=None,
        patience: Optional[int]=None,
        reg_type: Optional[str]=None,
        reg_strength: Optional[float]=None,
        learning_rate: Optional[float]=None,

        # CNN model hyperparameters
        num_in_channels: Optional[int]=None,
        num_out_features: Optional[int]=None,
        kernel_size: Union[int, tuple[int, ...], None]=None,
        stride: Optional[int]=None,
        padding: Union[int, tuple[int, ...], None]=None,
        pool_size: Union[int, tuple[int, ...], None]=None,
        pool_stride: Union[int, tuple[int, ...], None]=None,
        pool_type: Optional[str]=None,
        embedding_size: Optional[int]=None,

        # Model parameters
        model_sequence_parameters: Optional[dict[str, Any]]=None,
        model_loss_function_name: Optional[str]=None,
        model_update_function_name: Optional[str]=None,
        dataset: Optional[DataSet]=None,
        model_data_filename: Optional[str]=None,
        cnn_model_settings_filename: Optional[str]=None,
        override_model_settings=False,
        object_name=None, has_log_id=False,
        do_print_layer_attr=False,
        do_print_tensor_function_attr=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log._set_log_id(object_name, log.CNN)

        # Make debug log of loading the CNN model
        log._log_debug(
            "Loading the CNN model...",
            log.MODEL_MODULE
        )

        # Set the CNN model attributes if no model data filename was provided
        #   or the model data fails to load
        if model_data_filename is None or not self.load(model_data_filename):

            # Get the CNN model hyperparameters
            model_hyperparameters = util._get_update_dict(
                locals(), cnn_model_default_hyperparameters
            )

            # Initialize missing cnn model hyperparameters with default values
            model_hyperparameters |= util._get_update_dict(
                cnn_model_default_hyperparameters, model_hyperparameters,
                update_none_only=True
            )

            # Get the CNN model first layer hyperparameters
            model_init_parameters = {
                'num_in_channels': num_in_channels,
                'num_out_features': num_out_features
            }

            # Get the CNN model final layer hyperparameters
            model_final_hyperparameters = {
                'embedding_size': embedding_size
            }

            # Get the CNN model forward pass function
            model_forward_pass_functions = {
                'encode': self._encode,
                'project': self._project
            }

            # Get the CNN model backpropagation functions
            model_backpropagation_functions = {
                'project_backward': self._project_backward,
                'encode_backward': self._encode_backward
            }

            # Get the CNN model settings
            cnn_settings = _get_model_settings(
                settings_jsonfile=cnn_model_settings_filename,
                model_type='cnn'
            )

            # Check if the CNN model settings exist
            if cnn_settings is not None:

                # Get the loss and update function names
                # If no loss and update function names were provided, get them from the
                #   CNN model settings
                if (model_loss_function_name is None or not override_model_settings):
                    if 'loss_function' in cnn_settings:
                        model_loss_function_name = cnn_settings['loss_function']
                    if 'update_function' in cnn_settings:
                        model_update_function_name = cnn_settings['update_function']

                # Get the model settings (hyperparameters)
                model_settings = cnn_settings['hyperparameters']

                if 'reg_type' in model_hyperparameters:
                    reg_type = model_hyperparameters['reg_type']

                # Update the model update function name if using regularization
                if reg_type is not None:
                    model_update_function_name = tf._get_update_function_name(reg_type)

                # Get the sequence parameters
                # If no sequence parameters were provided, get them from the
                #   CNN model settings
                if model_sequence_parameters is None:
                    model_sequence_parameters = cnn_settings['sequences']
                assert(model_sequence_parameters is not None)

                # Set the encoder layers
                self.encode_layers = self._get_seq_layers(
                    sequence_type='convolution',
                    model_sequence_parameters=model_sequence_parameters,
                    model_update_function_name=model_update_function_name,
                    model_init_hyperparameters=model_init_parameters,
                    model_hyperparameters=model_hyperparameters,
                    model_settings=model_settings,
                    override_model_settings=override_model_settings,
                    do_print_layer_attr=do_print_layer_attr,
                    do_print_tensor_function_attr=do_print_tensor_function_attr
                )

                # Set the projection layers
                self.projection_layers = self._get_seq_layers(
                    sequence_type='projection',
                    model_sequence_parameters=model_sequence_parameters,
                    model_update_function_name=model_update_function_name,
                    model_hyperparameters=model_hyperparameters,
                    model_settings=model_settings,
                    override_model_settings=override_model_settings,
                    do_print_layer_attr=do_print_layer_attr,
                    do_print_tensor_function_attr=do_print_tensor_function_attr
                )

            else:
                # Make warning log for CNN model settings not loading
                log._log_warning(
                    "No CNN model settings loaded!",
                    self.log_id
                )

                # Set empty layer lists for the CNN model
                self.encoder_blocks = []
                self.decoder_blocks = []
                self.projection_layers = []

            # Set the encoder reverse sequence
            self.encode_layers_reverse = self.encode_layers.copy()
            self.encode_layers_reverse.reverse()

            # Set the projection reverse sequence
            self.projection_layers_reverse = self.projection_layers.copy()
            self.projection_layers_reverse.reverse()

            # Get the model layers
            model_layers = self.encode_layers + self.projection_layers

            # Initialize the base model attributes
            assert(model_update_function_name is not None)
            super().__init__(
                model_layers=model_layers,
                model_forward_pass_functions=model_forward_pass_functions,
                model_backpropagation_functions=model_backpropagation_functions,
                model_loss_function_name=model_loss_function_name,
                model_update_function_name=model_update_function_name,
                dataset=dataset,
                model_hyperparameters=model_hyperparameters,
                object_name=object_name, has_log_id=True,
                do_print_tensor_function_attr=do_print_tensor_function_attr
            )

        # If a dataset was provided, use that dataset
        # NOTE: This overrides the dataset from the model data
        elif dataset is not None:
            self.set_dataset(dataset)

        # Check if the CNN model has layers
        if len(self.layers) > 0:
            # Make debug log for successfully loading the CNN model
            log._log_debug(
                "Successfully loaded the CNN model!",
                log.MODEL_MODULE
            )

        else:
            # Make warning log for loading the Transformer model
            log._log_warning(
                "Partially loaded the CNN model! DO NOT USE!",
                log.MODEL_MODULE
            )

    def _encode(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Encode the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The encode output values tuple or dict
        """
        return self._run_pass(
            layers=self.encode_layers,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    
    def _encode_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform encode backward on the upstream gradient.

            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The encode backward output values tuple or dict
        """
        return self._run_pass(
            layers=self.encode_layers_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    
    def _project(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform projection on the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The projection output values tuple or dict
        """
        return self._run_pass(
            layers=self.projection_layers,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    
    def _project_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None,
        do_return_dict=False
    ) -> Union[
        tuple[Any, ...],
        dict[str, Any]
    ]:
        """
        Perform projection backward on the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple
            do_return_dict (bool): Boolean indicating whether or not to return
                a dict of return values

        Return:
            The projection backward output values tuple or dict
        """
        return self._run_pass(
            layers=self.projection_layers_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys,
            do_return_dict=do_return_dict
        )
    

def get_cnn(
    cnn_model_args: Any
) -> Optional[CNN]:
    """
    Return a CNN model using the specified arguments.

    Args:
        cnn_model_args (Any): Arguments for the CNN model

    Return:
        A CNN model
    """
    # Make info log for getting the CNN model
    log._log_info(
        "Getting the CNN model...",
        log.MODEL_MODULE
    )

    # Get the CNN model
    cnn_model = CNN(**cnn_model_args)

    # Check if the CNN model has any layers
    if len(cnn_model.layers) > 0:
        # Make info log for successfully retrieving the CNN model
        log._log_info(
            "Successfully retrieved the CNN model!",
            log.MODEL_MODULE
        )
        return cnn_model

    # Else, make error log for failure to retrieve the CNN model, and
    #   return None
    log._log_error(
        "Failed to retrieve the CNN model!",
        log.MODEL_MODULE
    )
