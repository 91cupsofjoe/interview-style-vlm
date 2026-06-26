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
from model.layer import Layer, get_layer
from function import \
    attention as attn, convolution as conv, loss, pool, regularization as reg, \
        update, util
from function.tensor_function import get_tensor_function
from log import logger as log


# ================================= MODEL =====================================

BATCH_SIZE = 32
NUM_PRED_LABELS = 1
NUM_FOLDS = 5
NUM_EPOCHS = 100

default_model_hyperparams = {
    'batch_size': BATCH_SIZE,
    'num_pred_labels': NUM_PRED_LABELS,
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
        model_forward_pass_functions: list[Callable],
        model_backpropagation_functions: list[Callable],
        model_loss_function_name: str,
        model_hyperparameters: Optional[dict]=None,
        base_model_hyperparameters: Optional[dict]=None,
        dataset: Optional[DataSet]=None,
        model_data_filename: Optional[str]=None,
        object_name :Optional[str]=None, has_log_id: Optional[bool]=False
    ) -> None:
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.MODEL)

        # Set the model attributes if no model data filename was provided or
        #   the model data fails to load
        if model_data_filename is None or not self.load(model_data_filename):

            # Set the model layers
            self.layers = model_layers

            # Initialize the model hyperparameters if not provided
            if model_hyperparameters is None:
                model_hyperparameters = {}

            # Initialize the base model hyperparameters if not provided
            if base_model_hyperparameters is None:
                base_model_hyperparameters = {}

            # Initialize all base model hyperparameters
            for key, value in default_model_hyperparams:
                if key not in base_model_hyperparameters:
                    base_model_hyperparameters[key] = value
            
            # Fit both sets of hyperparameters as the model hyperparameters
            self.hyperparameters = base_model_hyperparameters | model_hyperparameters

            # Set the model forward pass and backpropagation functions
            self.forward_pass_functions = model_forward_pass_functions
            self.backpropagation_functions = model_backpropagation_functions

            # Set the model loss and update functions
            self.loss_function = get_tensor_function(
                tensor_function_name=model_loss_function_name,
                tensor_function_cache_parameters=self.hyperparameters
            )

        # If a dataset was provided, use that dataset
        # NOTE: This overrides the dataset from the model data
        if dataset is not None:
            self.set_dataset(dataset)

    model_attribute_names = {
        'layers',
        'dataset',
        'hyperparameters',
        'forward_pass_functions',
        'backpropagation_functions',
        'loss_function'
    }

    MODEL_DATA_DIR = 'data/'

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

                log.log_error(
                    "Failed to load the following from the model data:\n"
                    +missing_keys_str[:-2],
                    self.log_id
                )
                return False
            
            # Else, log success and return True since all model data loaded successfully
            log.log_success(
                "Model data loaded successfully!",
                self.log_id
            )
            return True

        except:
            # Log error and return False since loading the model data file failed
            log.log_error(
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
        # Initialize the model data file if filename not provided
        if model_data_filename is None:
            model_data_filename = log.get_object_name(self.log_id)
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
                # Log success and return True since the model data saved successfully
                log.log_success(
                    "Saved the model data!",
                    self.log_id
                )

            except:
                # Log error and return False since the model data failed to save
                log.log_error(
                    "Could not save the model data to the specified file!",
                    self.log_id
                )
                return False

        # Else, log error and return False since the model data filename
        #   was not provided
        log.log_error(
            "Could not load the model data since the model data filename was "
            "not provided!",
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
    
    sequence_to_layer_types = {
        'convolution' : 'convolution_layer',
        'transformer_encoder' : 'transformer_block',
        'transformer_decoder' : 'transformer_block',
        'projection' : 'projection_layer'
    }

    def get_seq_layers(self,
        sequence_type: str,
        sequence_parameters: dict[str, Any],
        layer_update_function_name: Optional[str]=None,
        model_init_hyperparameters: Optional[dict[str, Any]]=None,
        model_hyperparameters: Optional[dict[str, Any]]=None,
        model_final_hyperparameters: Optional[dict[str, Any]]=None,
        override_layer_parameters=False
    ) -> list[Layer]:
        """
        Return a list of the specified sequence layers

        Args:
            sequence_type (str): The sequence type
            sequence_parameters (dict[str, Any]): The sequence parameters dict

        Return:
            List of Layer objects
        """
        # Create the sequence layers
        sequence_layers = []

        # Get layer parameter updates from the model hyperparameters
        # Initialize the model hyperparameters if not provided
        if model_hyperparameters is None:
            model_hyperparameters = {}
        layer_parameter_updates = model_hyperparameters

        # Check if the sequence type is valid
        if sequence_type in self.sequence_to_layer_types:
            # Get layer type, forward functions list, backward functions list,
            #   and layers list
            layer_type = self.sequence_to_layer_types[sequence_type]
            layer_pass_function_names = sequence_parameters['pass_functions']
            layers = sequence_parameters['layers']
            # Iterate through the list of layers to append layers
            for i in range(len(layers)):
                # Get the layer parameters
                layer_parameters = layers[i]

                # Initialize the layer parameter updates
                layer_parameter_updates = {}

                # Update the first layer of the sequence with the model init parameters
                if i == 0 and model_init_hyperparameters is not None:
                    layer_parameter_updates = model_init_hyperparameters

                # Update the final layer of the sequence with the model final parameters
                if i - 1 == len(layers) and model_final_hyperparameters is not None:
                    layer_parameter_updates = \
                            layer_parameter_updates | model_final_hyperparameters
                    
                # Check if overriding layer parameters with model hyperparameters
                if override_layer_parameters:
                    # Override the layer parameters with the parameter updates
                    layer_parameters = layer_parameters | layer_parameter_updates
                else:
                    # Layer parametes have the final say on its values
                    layer_parameters = layer_parameter_updates | layer_parameters

                # Append layer with the specified layer type and parameters
                sequence_layers.append(get_layer(
                    layer_type=layer_type,
                    layer_parameters=layer_parameters | layer_parameter_updates,
                    layer_pass_function_names=layer_pass_function_names,
                    layer_update_function_name=layer_update_function_name
                ))

        # Return the sequence layers
        return sequence_layers

    def run_pass(self,
        layers: list, # List of Layer, ConvolutionLayer, or TransformerBlock
        x: Optional[Tensor]=None,
        upstream_grad: Optional[Tensor]=None,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[Any, ...]]=None
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

        Return:
            output_values (Tensor): The output values tuple
        """
        # Iterate through the layers
        for layer in layers:
            # Run the forward pass if x was provided
            if x is not None:
                output_values = layer.forward(
                    x=x,
                    kwargs=kwargs,
                    output_keys=output_keys
                )
                x = output_values[0]

            # Else, run backpropagation if the upstream gradient was provided
            elif upstream_grad is not None:
                output_values = layer.forward(
                    upstream_grad=upstream_grad,
                    kwargs=kwargs,
                    output_keys=output_keys
                )
                x = output_values[0]

            # Else, log error and return an empty tuple since neither an input
            #   nor upstream gradient tensor was provided
            else:
                log.log_error(
                    f"Could not run forward pass nor backpropagation since no "
                    f"input or upstream gradient tensor was provided!",
                    self.log_id
                )
                return tuple()

        # Return the output values
        return output_values

    def train(self,
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
        use_test_labels=False,

        # If using preloaded training data
        training_data: Optional[Tensor]=None,
        training_examples_tensor: Optional[Tensor]=None,
        target_labels_tensor: Optional[Tensor]=None,

        # General data loading params
        training_test_split=-1.0,

        # Prediction and accuracy params
        num_pred_labels=-1,
        do_measure_accuracy=False,

        # Crossvalidation
        num_folds=-1,

        # General params
        num_epochs=-1,
        batch_size=-1,
        eps=-1,
        patience=-1,
        use_patience=False,
        do_print_messages: Optional[bool]=False
    ) -> bool:
        """
        Train the model on the given data.

        Args:
            NOTE: See DataSet.load_data for more info on data loading args
            training_data (Tensor): The training data tensor (if using a
                single tensor for both training examples and training labels)
            training_examples (Tensor): The training examples tensor
            training_labels (Tensor): The training labels tensor
            num_pred_labels (int): The number of labels to predict per loop
            do_accuracy_test (bool): Measure accuracy per loop and (optionally)
                report accuracy metrics per epoch
            do_print_messages (bool): Print training messages

        Return:
            Boolean indicating training success
        """
        # If training data is provided as a single tensor, parse out the
            # training examples and training labels
        if training_data is not None:
            training_examples_tensor = training_data[:-1]
            target_labels_tensor = training_data[-1]

        # If training examples and training labels are not provided, load them
        #   from the model's dataset
        if self.dataset is not None and \
                (training_examples_tensor is None or target_labels_tensor is None):
            training_tensors = \
                self.dataset.get_training_tensors(
                    data_sources=data_sources,
                    examples_data=examples_data,
                    examples_data_name=examples_data_name,
                    examples_data_tensor_function_name=\
                                    examples_data_tensor_function_name,
                    examples_data_tensor_function_arg_keys=\
                                    examples_data_tensor_function_arg_keys,
                    labels_data=labels_data,
                    labels_data_name=labels_data_name,
                    labels_data_tensor_function_name=labels_data_tensor_function_name,
                    labels_data_tensor_function_arg_keys=\
                                    labels_data_tensor_function_arg_keys,
                    training_test_split=training_test_split
                )
            
            # Check if training tensors exist
            if training_tensors is not None:
                training_examples_tensor, target_labels_tensor = training_tensors

            # Check if using test labels
            if use_test_labels:
                # Use test labels instead of training labels
                test_tensors = self.dataset.get_test_tensors()
                if test_tensors is not None:
                    target_labels_tensor = test_tensors[1]
        
        # Only run the training loop if training examples and training labels
        #   tensors are provided
        if training_examples_tensor is not None and target_labels_tensor is not None:
            # Get the total number of training examples
            num_training_examples = training_examples_tensor.shape[0]

            # Get the model training hyperparameters
            if num_epochs < 0: num_epochs = self.hyperparameters['num_epochs']
            if batch_size < 0: batch_size = self.hyperparameters['batch_size']
            if num_pred_labels < 0:
                num_pred_labels = self.hyperparameters['num_pred_labels']
            if num_folds < 0: num_folds = self.hyperparameters['num_folds']
            if eps == float('-inf'): eps = self.hyperparameters['eps']
            if patience < 0: patience = self.hyperparameters['patience']

            # Get the crossvalidation and non-crossvalidation training examples
            #   and target labels tensor sets
            cv_sets = self.get_cross_validation_sets(
                training_examples=training_examples_tensor,
                target_labels=target_labels_tensor,
                num_folds=num_folds
            )
            base_training_examples = cv_sets[0]
            base_target_labels = cv_sets[1]
            cv_training_examples = None
            cv_target_labels = None

            # Check if there are more than 2 cv sets
            if len(cv_sets) > 2:
                cv_training_examples = cv_sets[2]
                cv_target_labels = cv_sets[3]

            # Run the model training loop for the specified number of epochs
            e = 0 # epoch index
            b = 0 # batch index
            p = 0 # Patience count
            scalar_loss = float('inf')

            while e < num_epochs:
                    
                while b < num_training_examples:

                    # Get the current training and target label batches
                    start = b
                    stop = min(b + batch_size, num_training_examples)
                    training_batch = base_training_examples[start:stop]
                    target_batch = base_target_labels[start:stop]

                    # Get the training loop output values
                    training_loop_output_values = \
                        self.run_training_loop(
                            training_batch=training_batch,
                            target_batch=target_batch,
                            cv_training_examples=cv_training_examples,
                            cv_target_labels=cv_target_labels,
                            num_pred_labels=num_pred_labels,
                            do_measure_accuracy=do_measure_accuracy
                        )
                    
                    # Make sure the training loop output values exist
                    if training_loop_output_values is not None:
                        # Get the scalar loss, prediction labels, and accuracy metrics
                        scalar_loss = training_loop_output_values[0]
                        prediction_labels = training_loop_output_values[1]
                        accuracy_metrics = training_loop_output_values[2]
                        cv_prediction_labels = None
                        cv_accuracy_metrics = None

                        # Check if the training loop output values include the
                        #   crossvalidation items
                        if len(training_loop_output_values) > 3:
                            cv_prediction_labels = training_loop_output_values[3]
                            cv_accuracy_metrics = training_loop_output_values[4]

                    # Print messages if specified
                    if do_print_messages:
                        print(f"\nEpoch #{e+1}: ", end='')

                        # Print accuracy metrics
                        if do_measure_accuracy:
                            print(f"Training labels accuracy = "
                                  f"{accuracy_metrics['correct']}, ", end='')
                            
                            # Check if the crossvalidation metrics exist
                            if cv_accuracy_metrics is not None:
                                print(f"Cross validation accuracy = "
                                      f"{cv_accuracy_metrics['correct']}", end='')
                                
                    # Check if the scalar loss is below epsilon
                    if scalar_loss < eps:
                        # If using patience, update and check patience index
                        if use_patience:
                            p += 1
                            # Stop training if the patience threshold is reached
                            if p >= patience:
                                # Log status and return True
                                log.log_status(
                                    "Training stopped due to reaching the "
                                    "patience threshold!",
                                    self.log_id
                                )
                                return True
                        
                        else:
                            # Stop training since the loss threshold was reached
                            # Log status and return True
                            log.log_status(
                                "Training stopped due to reaching the "
                                "loss threshold!",
                                self.log_id
                            )
                            return True
                    else:
                        # Reset the patience level since the scalar loss was above
                        #   the loss threshold
                        p = 0

                    b += batch_size
                
                e += 1

                # Randomize the training data if running another epoch
                if e < num_epochs:
                    rand_nums = torch.randperm(num_training_examples)
                    base_training_examples = base_training_examples[rand_nums]
                    base_target_labels = base_target_labels[rand_nums]

        # Else, log error and return False since the training data wasn't provided
        log.log_error(
            "Could not train the model since the training data wasn't provided!",
            self.log_id
        )
        return False

    def get_cross_validation_sets(self,
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
            cv_training_examples (Tensor): The crossvalidation training examples tensor
            cv_target_labels (Tensor): The crossvalidation target labels tensor
            non_cv_examples (Tensor): The non-crossvalidation training examples tensor
            non_cv_target_labels (Tensor): The non-crossvalidation target labels tensor
        """
        # Check if the number of folds is less than or equal 1
        if num_folds <= 1:
            # Return the training examples and target labels
            return training_examples, target_labels

        # Get the number of training examples / target labels
        num_training_examples = training_examples.shape[0]

        # Randomize the training examples and target labels
        if do_random:
            rand_nums = torch.randperm(num_training_examples)
            base_training_examples = training_examples[rand_nums]
            base_target_labels = target_labels[rand_nums]
        else:
            base_training_examples = training_examples.clone()
            base_target_labels = target_labels.clone()

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
        cv_training_examples = base_training_examples[cv_start:cv_stop]
        cv_target_labels = base_target_labels[cv_start:cv_stop]
        
        non_cv_training_examples = base_training_examples[0:cv_start]
        non_cv_target_labels = base_target_labels[0:cv_start]

        if cv_stop < num_training_examples:
            non_cv_examples = torch.cat(
                (non_cv_training_examples,
                 base_training_examples[cv_stop:num_training_examples])
            )
            non_cv_target_labels = torch.cat(
                (non_cv_target_labels,
                 base_target_labels[cv_stop:num_training_examples])
            )

        # Return the crossvalidation and non-cv sets
        return non_cv_examples, non_cv_target_labels, \
            cv_training_examples, cv_target_labels

    def run_training_loop(self,
        training_batch: Tensor,
        target_batch: Tensor,
        cv_training_examples: Optional[Tensor],
        cv_target_labels: Optional[Tensor],
        num_pred_labels=NUM_PRED_LABELS,
        do_measure_accuracy=False
    ) -> Optional[tuple[Any, ...]]:
        """
        Run the model training loop on the given data, performing these steps:
            forward pass --> loss --> backpropagation --> update.

        Args:
            training_batch (Tensor): The training examples batch tensor
            target_batch (Tensor): The target labels batch tensor
            cv_training_examples (Tensor): The crossvalidation training examples tensor
            cv_target_labels (Tensor): The crossvalidation target labels tensor
            num_pred_labels (int): The number of prediction labels
            do_measure_accuracy (bool): The boolean indicating to measure
                training accuracy

        Return:
            Tuple of float (scalar loss) and boolean to indicate training
                loop success
        """
        # Feed the training examples through the forward pass
        forward_pass_output_values = self.forward_pass(training_batch)

        # Get the logits and probabilities
        logits = forward_pass_output_values[0]
        probabilities = forward_pass_output_values[1]

        # Perform backpropagation on the logits
        self.backpropagation(logits.clone())

        # Make sure the update is successful, otherwise return None
        if not self.update():
            return None

        # Initialize the scalar loss
        scalar_loss = None
        # Perform the loss calculation
        loss_output_values = self.calculate_loss(logits, target_batch)
        # Make sure the loss output values exist
        if loss_output_values is not None:
            # Get the scalar loss
            scalar_loss = loss_output_values[0]

        # Get the model prediction labels
        prediction_labels = self.predict(
            probabilities=probabilities,
            num_pred_labels=num_pred_labels
        )

        # Make sure the prediction tensor exists and measuring accuracy is specified
        if prediction_labels is not None and do_measure_accuracy:
            accuracy_metrics = self.get_pred_accuracy(
                prediction_labels=prediction_labels[:, 0],
                target_labels=target_batch
            )

        # Check if crossvalidation training examples and target labels were provided
        if cv_training_examples is not None and cv_target_labels is not None:
            # Get the model prediction labels
            cv_prediction_labels = self.predict(
                query_examples_tensor=cv_training_examples,
                probabilities=probabilities,
                num_pred_labels=num_pred_labels
            )

            # Make sure the prediction tensor exists and measuring accuracy is specified
            if cv_prediction_labels is not None and do_measure_accuracy:
               cv_accuracy_metrics = self.get_pred_accuracy(
                    prediction_labels=cv_prediction_labels[:, 0],
                    target_labels=target_batch
                )

        # Return the scalar loss, prediction labels, accuracy metrics,
        # Only return the crossvalidation prediction labels and crossvalidation
        #   accuracy metrics if they exist
        if cv_training_examples is not None and cv_target_labels is not None:
            return scalar_loss, prediction_labels, accuracy_metrics, \
                cv_prediction_labels, cv_accuracy_metrics
        
        else:
            return scalar_loss, prediction_labels, accuracy_metrics

    # ---------------------- MODEL TRAINING LOOP METHODS ----------------------

    def forward_pass(self,
        x: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform the forward pass on the input.

        Args:
            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (tuple[Any, ...]): The model output values tuple.
        """
        for function in self.forward_pass_functions:
            output_values = function.forward(
                x=x,
                kwargs=kwargs,
                output_keys=output_keys
            )
        
        return output_values
    
    def calculate_loss(self,
        prediction: Tensor, target_labels: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> Optional[tuple[Any, ...]]:
        """
        Calculate the loss between the model prediction (output) and target labels

        Args:
            prediction (Tensor): The model prediction tensor
            target_labels (Tensor): The target labels tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            The scalar loss float
        """
        # Return the loss calculation from the prediction and target_labels
        if self.loss_function is not None:
            # Initialize the keyword arguments if not provided
            if kwargs is None:
                kwargs = {}

            # Get the loss output values
            return self.loss_function.run(
                kwargs=(kwargs | locals()),
                output_keys=output_keys
            )
        
        # Else log error and return None since the loss function wasn't set
        log.log_error(
            "Could not calculate loss since loss function was not set!",
            self.log_id
        )
    
    def backpropagation(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform the backpropagation on the upstream gradient.

        Args:
            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (tuple[Any, ...]): The model output values tuple.
        """
        for function in self.forward_pass_functions:
            output_values = function.forward(
                upstream_grad=upstream_grad,
                kwargs=kwargs,
                output_keys=output_keys
            )
        
        return output_values
    
    def update(self) -> bool:
        """
        Update all of the model's learnable parameters.

        Args:
            None

        Return:
            update_success (boolean): Boolean indicating success with updating
        """
        updates_successful = True

        for layer in self.layers:
            # Store the boolean result of updating learnable parameters
            # All layers should update successfully, otherwise return False
            if not layer.update():
                updates_successful = False

        # Return boolean indicating successfully updating all the learnable
        #   parameters
        return updates_successful
    
    def predict(self,
        query_examples: Optional[list[Any]]=None,
        query_examples_tensor: Optional[Tensor]=None,
        probabilities: Optional[Tensor]=None,
        target_labels_data: Optional[list[Any]]=None,
        use_test_labels: Optional[bool]=False,
        num_pred_labels: Optional[int]=None,
        kwargs: Optional[dict[str, Any]]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> Optional[Tensor]:
        """
        Predict the label for the provided example or set of probabilities.

        Args:
            query_examples (Any): The query examples data
            query_examples_tensor (Tensor): The query examples tensor
            probabilities (Tensor): The probabilities tensors
            labels_data (list[Any]): The labels data list
            use_test_labels (bool): The boolean indicating if using test labels
            num_pred_labels (Tensor): The number of labels to predict
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            pred_labels_tensor (Tensor): The predicted labels tensor
        """
        # Set the labels data if not provided
        if target_labels_data is None:
            # Make sure the model dataset exists
            if self.dataset is not None:
                # If using test labels, set the labels data to the dataset
                #   test labels
                if use_test_labels:
                    test_data = self.dataset.get_test_data()

                    # Make sure the test data and labels data exists
                    if test_data is not None and test_data[1] is not None:
                        target_labels_data = test_data[1]
                    
                    # Make sure the labels data exists
                    if target_labels_data is None:
                        # Log error and return None since the test data for
                        #   the model dataset was not set
                        log.log_error(
                            "Could not make a prediction since the test data "
                            "was not set!",
                            self.log_id
                        )
                        return None

                # Else, use the training labels
                else:
                    training_data = self.dataset.get_training_data()

                    # Make sure the test data and labels data exists
                    if training_data is not None and training_data[1] is not None:
                        labels_data = training_data[1]

                    # Make sure the labels data exists
                    if labels_data is None:
                        # Log error and return None since the training data for
                        #   the model dataset was not set
                        log.log_error(
                            "Could not make a prediction since the training data "
                            "was not set!",
                            self.log_id
                        )
                        return None

            else:
                # Log error and return None since the model dataset was not set
                log.log_error(
                    "Could not make a prediction since the model dataset was not set!",
                    self.log_id
                )
                return None

        # If probabilities are not provided but query examples are, run the
        #   query examples through the forward pass to get the probabilities
        if probabilities is None and not \
            (query_examples is None and query_examples_tensor is None):
            # If the query example tensor is not provided, get it from the
            #   query example data
            if query_examples_tensor is None and self.dataset is not None:
                query_example_tensor = self.dataset.item_to_tensor(
                    example=query_examples
                )

            # Make sure the query examples tensor exists
            if query_example_tensor is not None:
                forward_pass_output_values = self.forward_pass(
                    x=query_example_tensor,
                    kwargs=kwargs,
                    output_keys=output_keys
                )

                # The probabilities are the second element of the output values tuple
                probabilities = forward_pass_output_values[1]

            else:
                # Log error and return None since the query examples couldn't be
                #   converted into a tensor
                log.log_error(
                    f"Could not make the prediction since the query examples "
                    f"could not be converted into a tensor!",
                    self.log_id
                )

        if probabilities is None:
            # Both the probabilities and query examples are not provided, so
            #   log error and return None
            log.log_error(
                f"Could not make a prediction since neither the probabilties "
                f"nor query examples were provided!",
                self.log_id
            )
            return None
        
        # Else, get the specified number of most probable labels

        # Flatten the probabilities tensor
        probs_dim = probabilities.shape[-1]
        probs_flat = probabilities.reshape(-1, probs_dim)
        # Convert the flattened probabilties tensor into a list
        probs_list = probs_flat.tolist()

        # Get the indices for the most probable labels
        # Initialize the number of predicted labels to 1 if not provided
        if num_pred_labels is None:
            num_labels = 1

        # Initialize the label indices list
        pred_label_indices = [[]]

        # Iterate through the rows of the probabilities list
        for row in probs_list:
            j = 0
            while j < num_labels:
                # Get the max probability
                max_prob_index = row.find(max(row))
                # Pop and append the max probabilily to the predicted label indices
                pred_label_indices[row].append(row.pop(index=max_prob_index))
                j += 1

        # Convert the predicted label indices to a tensor
        pred_labels_tensor = torch.tensor(pred_label_indices)
        # Convert each index in the predicted label indices to their respective label
        pred_labels_tensor[target_labels_data]

        # Return the predicted labels
        return pred_labels_tensor
    
    def get_pred_accuracy(self,
        prediction_labels: Tensor,
        target_labels: Tensor,
        do_F1=False
    ) -> dict[str, float]:
        """
        Get accuracy metrics for predictions.

        Args:
            predictions (Tensor): The predictions tensor
            target_labels (Tensor): The target labels tensor

        Return:
            accuracy_metrics (dict[str, float]): Dict of accuracy metrics
        """
        # Initialize the accuracy metrics dict
        accuracy_metrics = {}

        # Get the measurement for correctly predicted examples
        accuracy_metrics['correct'] = \
            (prediction_labels == target_labels).float().sum() \
                            / prediction_labels.shape[0]

        # Return the accuracy metrics
        return accuracy_metrics
        

# ========================== STATIC MODEL METHODS =============================

def get_model_settings(
    settings_jsonfile: Optional[str]=None,
    model_type: Optional[str]=None,
) -> Any:
    """
    Load model settings for the specified model type.

    Args:
        settings_jsonfile (str): The filename of the settings JSON file
        model_type (str): The model type

    Return:
        The settings file contents
    """
    # Load from the model settings file, if provided
    if settings_jsonfile is not None:
        model_settings = util.load_json(settings_jsonfile)
    else:
        settings_jsonfile = "NO_MODEL_SETTINGS_JSONFILE"

    # Make sure that the model settings exist
    if model_settings is not None:
        # If a setting name was provided, return only the model settings
        #   associated with the specified model type
        if model_type is not None:
            if model_type in model_settings:
                return model_settings[model_type]
            # Else, return None

        # Else, return all of the model settings
        return model_settings
    
    # Else, log error and return None since the model settings couldn't load
    log.log_error(
        f"Couldn't load the model settings from {settings_jsonfile}!",
        log.MODEL_MODULE
    )


MODEL_SETTINGS_FILENAME = f"{Path(__file__).parent}/model_settings.json"
model_settings = get_model_settings(MODEL_SETTINGS_FILENAME)


# =========================== TRANSFORMER MODEL ===============================

NUM_IN_TOKENS = 64
NUM_OUT_CLASSES = 128
MAX_SEQ_LEN = 12
TRANSFORMER_EMBEDDING_SIZE = 256
FEED_FWD_SIZE = 1024
TRANSFORMER_FINAL_EMBEDDING_SIZE = 768

class Transformer(Model):
    """
    This is the transformer class, which applys attention and masking to its
        encoder (training) and decoder (prediction) blocks.
    """
    def __init__(self,
        embedding_size=TRANSFORMER_EMBEDDING_SIZE,
        feed_fwd_size=FEED_FWD_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_attn_heads=attn.NUM_ATTN_HEADS, dropout=reg.DROPOUT,
        final_embedding_size=TRANSFORMER_FINAL_EMBEDDING_SIZE,
        base_model_hyperparameters: Optional[dict]=None,
        model_sequence_parameters: Optional[dict[str, Any]]=None,
        model_loss_function_name: Optional[str]=None,
        model_update_function_name: Optional[str]=None,
        override_model_settings=True,
        dataset: Optional[DataSet]=None,
        model_data_filename: Optional[str]=None,
        object_name=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.TRANSFORMER)

        # Set the transformer model attributes if no model data filename was
        #   provided or the model data fails to load
        if model_data_filename is None or not self.load(model_data_filename):

            # Get the transformer model init hyperparameters
            model_init_hyperparameters = {
                'embedding_size': embedding_size
            }
            
            # Get the transformer model hyperparameters
            model_hyperparameters = {
                'max_seq_len': max_seq_len,
                'num_attn_heads': num_attn_heads,
                'embedding_size': embedding_size,
                'feed_fwd_size': feed_fwd_size,
                'dropout': dropout
            }

            # Get the transformer model final hyperparameters
            model_final_hyperparameters = {
                'pre_embedding_size': embedding_size,
                'embedding_size': final_embedding_size
            }

            # Set the transformer model forward pass function
            model_forward_pass_functions = [
                self.encode, self.decode, self.project
            ]

            # Set the transformer model backpropagation functions
            model_backpropagation_functions = [
                self.project_backward, self.decode_backward, self.encode_backward
            ]

            # Get the Transformer model settings
            assert(model_settings is not None)
            transformer_settings = model_settings['transformer']

            # Set the loss and update function names
            # If no loss and update function names were provided, get them from the
            #   Transformer model settings
            if model_loss_function_name is None:
                model_loss_function_name = transformer_settings['loss_function']
                assert(model_loss_function_name is not None)

            if model_update_function_name is None:
                model_update_function_name = transformer_settings['update_function']
                assert(model_update_function_name is not None)

            # Set the sequence parameters
            # If no sequence parameters were provided, get them from the
            #   Transformer model settings
            if model_sequence_parameters is None:
                model_sequence_parameters = transformer_settings['sequence_blocks']
                assert(model_sequence_parameters is not None)

            # Set the encoder sequence and its reverse sequence
            self.encoder_blocks = self.get_seq_layers(
                sequence_type='encoder',
                sequence_parameters=model_sequence_parameters,
                layer_update_function_name=model_update_function_name,
                model_init_hyperparameters=model_init_hyperparameters,
                model_hyperparameters=model_hyperparameters,
                override_layer_parameters=override_model_settings
            )
            self.encoder_blocks_reverse = self.encoder_blocks.copy()
            self.encoder_blocks_reverse.reverse()

            # Set the decoder sequence and its reverse sequence
            self.decoder_blocks = self.get_seq_layers(
                sequence_type='decoder',
                sequence_parameters=model_sequence_parameters,
                layer_update_function_name=model_update_function_name,
                model_hyperparameters=model_hyperparameters,
                override_layer_parameters=override_model_settings
            )
            self.decoder_blocks_reverse = self.decoder_blocks.copy()
            self.decoder_blocks_reverse.reverse()

            # Set the projection sequence and its reverse sequence
            self.projection_layers = self.get_seq_layers(
                sequence_type='projection',
                sequence_parameters=model_sequence_parameters,
                layer_update_function_name=model_update_function_name,
                model_hyperparameters=model_hyperparameters,
                model_final_hyperparameters=model_final_hyperparameters,
                override_layer_parameters=override_model_settings
            )
            self.projection_layers_reverse = self.projection_layers.copy()
            self.projection_layers.reverse()

            # Get the model layers
            model_layers = self.encoder_blocks \
                + self.decoder_blocks \
                + self.projection_layers

            # Initialize the base model attributes
            super().__init__(
                model_layers=model_layers,
                model_forward_pass_functions=model_forward_pass_functions,
                model_backpropagation_functions=model_backpropagation_functions,
                model_loss_function_name=model_loss_function_name,
                dataset=dataset,
                model_hyperparameters=model_hyperparameters,
                base_model_hyperparameters=base_model_hyperparameters,
                object_name=object_name, has_log_id=True
            )

        # If a dataset was provided, use that dataset
        # NOTE: This overrides the dataset from the model data
        elif dataset is not None:
            self.set_dataset(dataset)
    
    def encode(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Encode the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (tuple[Any, ...]): The encode output values tuple
        """
        return self.run_pass(
            layers=self.encoder_blocks,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys
        )
    
    def encode_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform encode backward on the upstream gradient.

            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (tuple[Any, ...]): The oencode backwardutput values tuple
        """
        return self.run_pass(
            layers=self.encoder_blocks_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys
        )
    
    def decode(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Decode the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (tuple[Any, ...]): The decode output values tuple
        """
        return self.run_pass(
            layers=self.decoder_blocks,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys
        )
    
    def decode_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform decode backward on the upstream gradient.

            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (tuple[Any, ...]): The decode backward output values tuple
        """
        return self.run_pass(
            layers=self.decoder_blocks_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys
        )
    

    def project(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform projection on the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            The projection output values tuple
        """
        return self.run_pass(
            layers=self.projection_layers,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys
        )
    

    def project_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform projection backward on the input.

            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            The projection backward output values tuple
        """
        return self.run_pass(
            layers=self.projection_layers_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys
        )
    
    
# ======================== CONVOLUTION NEURAL NETWORK =========================

NUM_IN_CHANNELS = 3
NUM_OUT_FEATURES = 64
CNN_EMBEDDING_SIZE = 768

class CNN(Model):
    """
    This is the convolution neural network class, which applies convolution
        and linear projection to an input to produce a prediction.
    """
    def __init__(self,
        num_in_channels=NUM_IN_CHANNELS, num_out_features=NUM_OUT_FEATURES,
        kernel_size=conv.KERNEL_SIZE,
        stride=conv.STRIDE, padding=conv.PADDING,
        pool_size=pool.KERNEL_SIZE,
        pool_stride=pool.STRIDE, pool_type=pool.POOL_TYPE,
        embedding_size=CNN_EMBEDDING_SIZE,
        base_model_hyperparameters: Optional[dict]=None,
        model_sequence_parameters: Optional[dict[str, Any]]=None,
        model_loss_function_name: Optional[str]=None,
        model_update_function_name: Optional[str]=None,
        override_model_settings=True,
        dataset: Optional[DataSet]=None,
        model_data_filename: Optional[str]=None,
        object_name=None, has_log_id=False
    ):
        # Set the log id if none is provided
        if not has_log_id:
            self.log_id = log.set_log_id(object_name, log.TRANSFORMER)

        # Set the CNN model attributes if no model data filename was provided
        #   or the model data fails to load
        if model_data_filename is None or not self.load(model_data_filename):

            # Set the CNN model init parameters
            model_init_parameters = {
                'num_in_channels': num_in_channels,
                'num_out_channels': num_out_features
            }
            
            # Set the CNN model hyperparameters
            model_hyperparameters = {
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'pool_size': pool_size,
                'pool_stride': pool_stride,
                'pool_type': pool_type
            }

            model_final_hyperparameters = {
                'pre_embedding_size': num_out_features,
                'embedding_size': embedding_size
            }

            # Set the CNN model forward pass function
            model_forward_pass_functions = [
                self.encode, self.project
            ]

            # Set the CNN model backpropagation functions
            model_backpropagation_functions = [
                self.project_backward, self.encode_backward
            ]

            # Get the CNN model settings
            assert(model_settings is not None)
            cnn_settings = model_settings['cnn']

            # Set the loss and update function names
            # If no loss and update function names were provided, get them from the
            #   CNN model settings
            if model_loss_function_name is None:
                model_loss_function_name = cnn_settings['loss_function']
                assert(model_loss_function_name is not None)

            if model_update_function_name is None:
                model_update_function_name = cnn_settings['update_function']
                assert(model_update_function_name is not None)

            # Set the sequence parameters
            # If no sequence parameters were provided, get them from the
            #   CNN model settings
            if model_sequence_parameters is None:
                model_sequence_parameters = cnn_settings['sequence_blocks']
                assert(model_sequence_parameters is not None)

            # Set the encoder layers and its reverse sequence
            self.encode_layers = self.get_seq_layers(
                sequence_type='encoder',
                sequence_parameters=model_sequence_parameters,
                layer_update_function_name=model_update_function_name,
                model_init_hyperparameters=model_init_parameters,
                model_hyperparameters=model_hyperparameters,
                override_layer_parameters=override_model_settings
            )
            self.encode_blocks_reverse = self.encode_layers.copy()
            self.encode_layers.reverse()

            # Set the projection layers and its reverse sequence
            self.projection_layers = self.get_seq_layers(
                sequence_type='projection',
                sequence_parameters=model_sequence_parameters,
                layer_update_function_name=model_update_function_name,
                model_hyperparameters=model_hyperparameters,
                model_final_hyperparameters=model_final_hyperparameters,
                override_layer_parameters=override_model_settings
            )
            self.projection_layers_reverse = self.projection_layers.copy()
            self.projection_layers.reverse()

            # Get the model layers
            model_layers = self.encode_layers + self.projection_layers

            # Initialize the base model attributes
            super().__init__(
                model_layers=model_layers,
                model_forward_pass_functions=model_forward_pass_functions,
                model_backpropagation_functions=model_backpropagation_functions,
                model_loss_function_name=model_loss_function_name,
                dataset=dataset,
                model_hyperparameters=model_hyperparameters,
                base_model_hyperparameters=base_model_hyperparameters,
                object_name=object_name, has_log_id=True
            )

        # If a dataset was provided, use that dataset
        # NOTE: This overrides the dataset from the model data
        elif dataset is not None:
            self.set_dataset(dataset)

    def encode(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Encode the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (tuple[Any, ...]): The encode output values tuple
        """
        return self.run_pass(
            layers=self.encode_layers,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys
        )
    
    def encode_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform encode backward on the upstream gradient.

            upstream_grad (Tensor): The upstream gradient tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            output_values (tuple[Any, ...]): The encode backwardutput values tuple
        """
        return self.run_pass(
            layers=self.encode_layers,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys
        )
    
    def project(self,
        x: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform projection on the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            The projection output values tuple
        """
        return self.run_pass(
            layers=self.projection_layers,
            x=x,
            kwargs=kwargs,
            output_keys=output_keys
        )
    
    def project_backward(self,
        upstream_grad: Tensor,
        kwargs: Optional[dict]=None,
        output_keys: Optional[tuple[str, ...]]=None
    ) -> tuple[Any, ...]:
        """
        Perform projection backward on the input.

            x (Tensor): The input tensor
            kwargs (dict[str, Any]): The keyword arguments dict
            output_keys (tuple[Any, ...]) The output keys tuple

        Return:
            The projection backward output values tuple
        """
        return self.run_pass(
            layers=self.projection_layers_reverse,
            upstream_grad=upstream_grad,
            kwargs=kwargs,
            output_keys=output_keys
        )