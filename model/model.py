from pydantic import BaseModel

from model import layer as ly
LEARNING_RATE = 1 # Default learning rate for the model
NUM_EPOCHS = 1 # Default number of epochs
# The rest of the hyperparameters for the model are defined in the layer module

class Model(BaseModel):

    # Establish the valid hyperparameters for each of the layers
    valid_conv_layer_args = {
        'num channels', 'num out features',
        'kernel size', 'stride', 'padding',
    }
    valid_proj_layer_args = {
        'embedding size'
    }

    # Establish the valid post-convolution function categories
    valid_function_categories = {
        'forward', 'loss', 'backward'
    }

    def __init__(self,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        conv_layers: dict = {},
        proj_layers: dict = {},
        functions = {
            # The forward and backward passes each have a dict[str, tuples], where
            #   each tuple has a function pointer and a list of parameter keys
            'conv_forward': {},
            'conv_backward': {},
            'proj_forward': {},
            'proj_backward': {},

            # The loss and loss derivative functions entry values are only a single
            #   tuple, each with a function pointer and a list of parameter keys
            'loss': (),
            'loss_derivative': ()
        }
    ):
        self.attr = {
            'learning_rate': learning_rate, # The learning rate of the model
            'num_epochs': num_epochs # The number of training epochs to run
        }

        # Read in a dict of convolution layer hyperparams to create one or more
        #   convolution layers
        self.conv_layers = {}
        for conv_layer_params in conv_layers:
            self.create_conv_layer(conv_layer_params)

        # Do the same for projection layers and their hyperparameters
        for proj_layer_params in proj_layers:
            self.create_proj_layer(proj_layer_params)
        self.proj_layers = {}
        
        # Store list of functions to apply between convolution ans projection
        self.functions = {}
        # Check if functions are provided as a dictionary and parse accordingly
        if isinstance(functions, dict):
            for func_cat, _ in functions:
                if func_cat in self.valid_function_categories:
                    self.functions[func_cat] = functions[func_cat]

    def __setfunc__(self, cat, func, params={}):
        if cat in self.valid_function_categories:
            self.functions[cat] = (func, params)

    def create_conv_layer(self,
        conv_layer_hyperparams: dict = {}
    ):
        """
        Create and store a convolution layer based on specified hyperparameters

        Args:
            conv_layer_hyperparams (dict): The dictionary of hyperparameters

        Return:
            None
        """
        # Set default sequence id for the convolution layer to be created
        seq_id = len(conv_layer_hyperparams)
        # If a sequence id is provided, use that for the, but check for
        #   duplicate keys
        if 'seq id' in conv_layer_hyperparams.keys():
            temp_id = conv_layer_hyperparams['seq id']
            if temp_id not in self.conv_layers.keys():
                seq_id = temp_id
        
        # Add the convolution layer to the model
        self.conv_layers[seq_id] = ly.ConvLayer(
            {k: v for k, v in conv_layer_hyperparams.items()
             if k in self.valid_conv_layer_args}
        )

    def create_proj_layer(self,
        proj_layer_hyperparams: dict = {}
    ):
        """
        Create and store a linear projection layer based on specified hyperparameters

        Args:
            proj_layer_hyperparams (dict): The dictionary of hyperparameters

        Return:
            None
        """
        # Set default sequence id for the convolution layer to be created
        seq_id = len(proj_layer_hyperparams)
        # If a sequence id is provided, use that for the, but check for
        #   duplicate keys
        if 'seq id' in proj_layer_hyperparams.keys():
            temp_id = proj_layer_hyperparams['seq id']
            if temp_id not in self.conv_layers.keys():
                seq_id = temp_id

        # Add the projection layer to the model
        self.proj_layers[seq_id] = ly.ProjLayer(
            {k: v for k, v in proj_layer_hyperparams.items()
             if k in self.valid_proj_layer_args}
        )

    def forward(self, x):
        """
        Feed input patches through convolution layers and projection layers to
            calculate the final projection output

        Args:
            x (Tensor): input patches

        Return:
            None
        """
        # Run the sequence of convolution layers on the input patches
        #   to get the convolution output
        for conv_layer in self.conv_layers:
            x = conv_layer.forward(x)

        # Get the last convolution layer
        last_conv_layer = self.conv_layers[len(self.conv_layers) - 1]

        # Articulate forward pass functions and respective parameters to be
        #   applied to the convolution output Tensor before linear projection
        flatten = ('flatten', {})
        relu_forward = ('relu_forward', {})
        pool_forward = ('pool_forward', {
            'kernel_size': self.attr['pool_size'],
            'pool_stride': self.attr['pool_stride'],
            'pool_type': self.attr['pool_type']
        })

        # Apply the forward pass functions to the convolution output Tensor
        h = self.apply_pass_functions(
            x=x,
            layer=last_conv_layer,
            direction='forward',
            functions_list=[
                flatten,
                relu_forward,
                pool_forward
            ]
        )

        # Run the sequence of projection layers on the convolution output
        #   to get the projection output
        for proj_layer in self.proj_layers:
            h = proj_layer.forward(h)

    def get_loss(self,
        predictions, true_labels
    ):
        """
        Calculate a scalar value (loss) representing how off model predictions
            are from their respective true labels

        Args:
            predictions (Tensor): predictions Tensor
            true_labels (Tensor): true Labels Tensor
            predictions_dim (int): Dimension along the predictions Tensor
                containing the logits
        """
        # First get the list of weights from all the layers
        weights = [
            weight for weight, _ in
                [layer for _, layer in
                    [set(self.conv_layers.items()).union(
                        set(self.proj_layers.items()))
                    ]
                ]
            ]
        
        # Create the parameters for the model's loss function

        # Return the result from the model's loss function
        # First check that the loss function is set
        try:
            return self.functions['loss'](predictions, true_labels, weights)
        except:
            raise RuntimeError(f"'No loss derivative function set!"
                               f"Update the models loss")

    def backprop(self, predictions, true_labels):
        """
        Move backward from the loss function to the projection and convolution
            layers to determine how each weight and bias contributed to the total
            prediction error, calculating the respective gradients

        Args:
            predictions (Tensor): The predictions Tensor
            true_labels (Tensor): The true labels Tensor

        Return:
            None
        """
        # First get the derivative of the loss wrt to the projection
        #   output using the model's loss derivative function
        # Raise error if the loss function isn't set
        try:
            d_zproj = self.functions['loss derivative'](predictions, true_labels)
        except:
            raise RuntimeError(f"'No loss derivative function set!"
                               f"Update the models loss")

        # Iterate through the projection layers to update the derivative of the
        #   loss function wrt to the projection output
        for i in range(len(self.proj_layers) - 1, -1, -1):
            proj_layer = self.proj_layers[i]
            d_zproj = proj_layer.backward(d_zproj)

        # Get the last convolution layer
        last_conv_layer = self.conv_layers[len(self.conv_layers) - 1]

        # Articulate backpropagation functions and respective parameters to be
        #   applied to the derivative of the loss wrt the projection output
        unflatten = ('unflatten', {
            'layer_params': ['z_conv']
        })
        relu_backward = ('relu_backward', {
            'layer_params': ['z_conv']
        })
        pool_backward = ('pool_backward', {
            'layer_params':
                ['a_relu', 'pool_type', 'pool_size', 'pool_stride']
        })

        # Apply the backpropagation functions to the derivative of the loss wrt
        #   to the convolution output
        d_zconv = self.apply_pass_functions(
            x=d_zproj,
            layer=last_conv_layer,
            direction='forward',
            functions_list=[
                unflatten,
                relu_backward,
                pool_backward
            ]
        )

        # Iterate through the convolution layers to update the derivative of
        #   the loss function wrt to the convolution output
        for i in range(len(self.conv_layers) - 1, -1, -1):
            conv_layer = self.conv_layers[i]
            d_zconv = conv_layer.backward(d_zconv)

    def update(self):
        """
        For each layer, update the weight matrix and bias vector based on the
            layer's gradients and model hyperparameters

        Args:
            None

        Return:
            None
        """
        # Iterate through the convolution layers, updating each layer's
        #   convolution weight and bias
        learning_rate = self.attr['learning_rate']
        for conv_layer in self.conv_layers:
            W_new = conv_layer.__get__('W_conv') \
                - learning_rate * conv_layer.__get__('d_Wconv')
            conv_layer.__set__('W_conv', W_new)

            b_new = conv_layer.__get__('b_conv') \
                 - learning_rate * conv_layer.__get__('d_bconv')
            conv_layer.__set__('b_conv', b_new)
            
        # Do the same for the projection layers
        for proj_layer in self.proj_layers:
            W_new = proj_layer.__get__('W_proj') \
                - learning_rate * proj_layer.__get__('d_Wproj')
            proj_layer.__set__('W_proj', W_new)
            
            b_new = proj_layer.__get__('b_proj') \
                 - learning_rate * proj_layer.__get__('d_bproj')
            proj_layer.__set__('b_proj', b_new)

    # ========================== HELPER FUNCTIONS =============================

    def apply_pass_functions(self, x, layer,
                    direction='forward', functions_list=None):
        """
        Apply forward or backpropagation function(s) to an input Tensor

        Args:
            x (Tensor): The input Tensor
            category (str): The direction of the pass
            func_names_params (list[ tuple[ str, dict[str, any] ] ]): A list of
                tuples = (function name, parameter dictionary)

        Return:
            The transformed input Tensor
        """
        if not functions_list:
            functions_list = []

        # Iterate through function entries, each =
        #   tuple(function pointer, function parameter keys)
        for func_name, func_params in functions_list:
            # Get function from function name but firs check that it exists
            #   as a function for the specified pass direction
            if func_name in self.functions[direction]:
                function = self.functions[direction][func_name]

                # If there are layer params, get layer attributes
                if 'layer_params' in func_params:
                    layer_params = func_params['layer_params']
                    for layer_key in layer_params:
                        func_params[layer_key] = layer.__get__(layer_key)

                # Apply the function with the params to the input Tensor
                x = function(x, func_params)

        # Return the transformed input Tensor
        return x