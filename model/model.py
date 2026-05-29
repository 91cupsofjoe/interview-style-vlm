import torch
from torch import Tensor
from typing import Optional, Union, Callable, Any

from data.dataset import DataSet
from model import layer as lyr
from log import logger as log


# ================================= MODEL =====================================

LEARNING_RATE = 1 # Default learning rate
NUM_EPOCHS = 100 # Default number of training epochs
REG_PARAMS = ('ridge', 1.0) # Default regularization type and strength
NUM_FOLDS = 5 # Default number of folds for crossvalidation

default_model_hyperparams = {
    'learning_rate': LEARNING_RATE,
    'num_epochs': NUM_EPOCHS,
    'reg_type': REG_PARAMS[0],
    'reg_strength': REG_PARAMS[1],
    'num_folds': NUM_FOLDS
}

class Model:
    """
    This is the base model class, which performs training and prediction.
    """
    def __init__(self,
        model_sequences: dict,
        model_update_function: Callable,
        model_loss_function: Optional[Callable],
        model_hyperparams: Optional[dict[str, Any]],
        object_name=None
    ):
        # Set the log id for the Model object
        self.log_id = log.set_log_id(object_name)

        # Set model hyperparameters
        if model_hyperparams is None:
            model_hyperparams = {}

        # Check for provided model hyperparameters
        for key, value in default_model_hyperparams:
            if key not in model_hyperparams:
                model_hyperparams[key] = value
        
        self.hyperparams = model_hyperparams

        # Set the model loss and update functions
        self.loss_function = model_loss_function
        self.update_function = model_update_function

        # Create separate forward pass and backpropagation lists of sequences,
        #   a unified dict of sequences, and a list of layers
        self.forward_pass_seqs = []
        self.backpropagation_seqs = []
        self.sequences = {}
        self.layers = []
    
        # Iterate through the list of sequences, adding sequences of layers
        #   in the appropriate pass.
        for seq_name, seq_params in model_sequences:
            # Get the forward functions
            forward_functions = None
            if 'forward_functions' in seq_params:
                forward_functions = seq_params['forward functions']

            # Get the backward functions
            backward_functions = None
            if 'backward_functions' in seq_params:
                backward_functions = seq_params['forward functions']

            # Get the sequences layers
            layer_params_sets = seq_params['layers']
            sequence = []

            # Iterate through sets of layer parameters to create the layers
            for layer_params in layer_params_sets:
                # Get layer type and set forward and backward functions
                #   in the layer parameters
                layer_type = ''
                if 'layer_type' in layer_params:
                    layer_type = layer_params.pop('layer_type')

                layer_params['forward_functions'] = forward_functions
                layer_params['backward_functions'] = backward_functions

                # Add the layer to the model sequences dict and layers list
                layer = lyr.get_layer(
                    layer_type=layer_type,
                    layer_parameters=layer_params
                )

                # Only append valid layers
                if layer is not None:
                    sequence.append(layer)
                    self.layers.append(layer)

            # For the forward pass, add the sequence to the front of the
            #   forward pass sequence list and to the sequences dict
            forward_seq_name = seq_name + '_forward'
            forward_seq = sequence.copy()

            self.forward_pass_seqs.append(forward_seq)
            self.sequences[forward_seq_name] = forward_seq


            # For backpropagation, add the sequence to the back of the
            #   backpropagation sequence list and to the sequences dict
            backward_seq_name = seq_name + '_backward'
            backward_seq = sequence.reverse()

            self.backpropagation_seqs.insert(0, backward_seq)
            self.sequences[backward_seq_name] = backward_seq

        # Initialize the dataset for the model
        self.dataset: Optional[DataSet] = None

    def set_dataset(self, dataset: DataSet):
        """
        Set the dataset for the model.

        Args:
            dataset (DataSet): The dataset for the model

        Return:
            None
        """
        self.dataset = dataset

    def train(self,
        training_data=None,
        training_examples=None, training_labels=None
    ) -> tuple[Optional[float], Optional[Tensor]]:
        """
        Train the model on the given dataset, performing these steps:
            forward pass --> loss calculation --> backpropagation
                --> update learnable parameters

        Args:
            None

        Return:
            None
        """
        # If training data is provided as a single tensor, parse out the
            # training examples and training labels
        if training_data is not None:
            training_examples = training_data[:-1]
            training_labels = training_data[-1]

        # If training examples and training labels are not provided, load them
        #   from the model's dataset
        if self.dataset is not None and \
                        (training_examples is None or training_labels is None):
            training_data = self.dataset.get_training_data(
                sep_examples_labels=True
            )
            assert(isinstance(training_data, tuple))
            training_examples, training_labels = training_data
        
        # Only run the training loop if training examples and training labels
        #   are provided
        if training_examples is not None and training_labels is not None:

            prediction = self.forward_pass(training_examples)

            loss = self.calculate_loss(prediction, training_labels)

            x_grad = self.backpropagation(prediction)

            self.update_learnable_parameters()

            # Return the scalar loss value and the input patches gradient
            #   NOTE: For debugging
            return loss, x_grad
        
        # Else, return None for the loss calculation and the input patches gradient
        return None, None

    # ----------------------- MODEL TRAINING METHODS --------------------------

    def forward_pass(self, x: Tensor):
        """
        Perform the forward pass on the input patches tensor to get the output
            prediction tensor

        Args:
            x (Tensor): The input patches tensor

        Return:
            The output prediction tensor
        """
        # Iterate through the forward pass sequences
        for seq in self.forward_pass_seqs:
            # Iterate through the layers in the sequences to update the
            #   input tensor
            for layer in seq:
                x = layer.forward(x)

        # Return the output prediction tensor
        return x
    
    def calculate_loss(self,
        prediction: Tensor, target_labels: Tensor
    ) -> Optional[float]:
        """
        Calculate the loss between the output prediction and target labels

        Args:
            prediction (Tensor): The output prediction tensor
            target_labels (Tensor): The target labels tensor

        Return:
            The loss function output
        """
        # Check that the loss function was set
        if self.loss_function is None:
            # Log error and return None since the loss function was not set
            log.log_error(
                "Could not calculate loss since the loss function was not set!",
                self.log_id
            )
            return None
        
        # Return the loss calculation from the prediction and target_labels
        return self.loss_function(prediction, target_labels, self.hyperparams)
    
    def backpropagation(self, x_grad: Tensor):
        """
        Perform backpropagation on the output prediction tensor to get the
            various loss gradients, returning the gradient of the loss wrt the
            input patches

        Args:
            x (Tensor): The output prediction tensor

        Return:
            The input patches gradient tensor
        """
        # Iterate through the backpropagation sequences
        for seq in self.backpropagation_seqs:
            # Iterate through the layers in the sequences to update the
            #   input gradient tensor
            for layer in seq:
                x_grad = layer.backward(x_grad)

        # Return the input gradient tensor
        return x_grad
    
    def update_learnable_parameters(self):
        """
        Iterate through all of the layers in the model, updating each layer's
            learning parameters with the layer's gradients, using the model's
            update function
        """
        # Check that the update function was set
        if self.update_function is None:
            # Log error and return None since the update function was not set
            log.log_error(
                f"Could not update learnable parameters since the update "
                f"function was not set!",
                self.log_id
            )

        assert(self.update_function)
        # Iterate through the model's layers
        for layer in self.layers:
            #Iterate through each learnable parameter in the layer
            for learnable_parameter, gradient in layer.learnable_parameter_pairs:
                # Update the learnable parameter using the gradient
                self.update_function(
                    learnable_parameter, gradient, self.hyperparams
                )


# =========================== TRANSFORMER MODEL ===============================

NUM_INPUT_TOKENS = 20 # Default number of input tokens
NUM_OUTPUT_CLASSES = 64 # Default number of output classes
NUM_ATTN_HEADS = 4 # Default number of attention heads
DROPOUT = 0.1 # Default dropout value

class Transformer(Model):
    """
    This is the transformer class, which applys attention and masking to its
        encoder (training) and decoder (prediction) blocks.
    """
    def __init__(self,
        model_sequences: dict[str, Any],
        model_update_function: Callable,
        model_loss_function=None, 
        num_input_tokens=NUM_INPUT_TOKENS, num_output_classes=NUM_OUTPUT_CLASSES,
        num_attn_heads=NUM_ATTN_HEADS, dropout=DROPOUT,
        object_name=None
    ):
        # Set logger id for the Transformer object
        self.log_id = log.set_log_id(object_name)
        
        # Get the transformer hyperparameters
        model_hyperparams = {
            'num_input_tokens': num_input_tokens,
            'num_output_classes': num_output_classes,
            'num_attn_heads': num_attn_heads,
            'dropout': dropout
        }

        # Initialize the transformer model
        super().__init__(
            model_sequences=model_sequences,
            model_loss_function=model_loss_function,
            model_update_function=model_update_function,
            model_hyperparams=model_hyperparams
        )

    def encode(self, input):
        """
        Encode the input Tensor by feeding it through the encoder sequence

        Args:
            input (Tensor): The Tensor to be encoded

        Return:
            The encoded Tensor
        """
        for encoder in self.sequences['encoder_forward']:
            input = encoder.forward(input)

        return input
    
    def decode(self, input):
        """
        Decode the input Tensor by feeding it through the decoder sequence

        Args:
            input (Tensor): The Tensor to be decoded

        Return:
            The decoded Tensor
        """
        for decoder in self.sequences['decoder_forward']:
            input = decoder.forward(input)

        return input
    
    
# ======================== CONVOLUTION NEURAL NETWORK =========================

BATCH_SIZE = 32 # Default number of input examples for the input patches

class CNN(Model):
    """
    This is the convolution neural network class, which applies convolution
        and linear projection to an input to produce a prediction.
    """
    def __init__(self,
        model_sequences: dict[str, Any],
        model_update_function: Callable,
        model_loss_function=None,
        batch_size=BATCH_SIZE,
        object_name=None
    ):
        # Set the logger id for the CNN
        # Set the logger id for the Model object
        self.log_id = log.set_log_id(object_name)

        # Get the transformer hyperparameters
        model_hyperparams = {
            'batch_size': batch_size
        }

        # Initialize the CNN model
        super().__init__(
            model_sequences=model_sequences,
            model_update_function=model_update_function,
            model_loss_function=model_loss_function,
            model_hyperparams=model_hyperparams,
            object_name=object_name
        )

    def encode(self, input_patches: Tensor):
        """
        Encode the input patches tensor to the convolution output tensor

        Args:
            input_patches (Tensor): The input patches tensor

        Return:
            The convolution output tensor
        """
        return self.forward_pass(input_patches)