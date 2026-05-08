from pydantic import BaseModel

from model import layer as ly
from model import loss

CONV_SEQ_LEN = 8 # Default sequence length for convolution
PROJ_SEQ_LEN = 8 # Default sequence length for projection
# The rest of the hyperparameters for the model are defined in the layer module

class Model(BaseModel):

    # Establish the valid hyperparameters for each of the layers
    conv_layer_args = {
        'num channels', 'num out features',
        'kernel size', 'stride', 'padding',
        'pool size', 'pool stride', 'pool type'
    }
    proj_layer_args = {
        'embedding size'
    }

    # Establish the valid post-convolution function categories
    function_categories = {
        'apply', 'unapply'
    }

    def __init__(self,
        conv_layers: dict = {},
        proj_layers: dict = {},
        functions = {
            'apply': [],
            'unapply': []
        }
    ):
        # Read in a dict of convolution layer hyperparams to create one or more
        #   convolution layers
        self.conv_layers = {}
        for conv_layer_hyperparams in conv_layers:
            self.create_conv_layer(conv_layer_hyperparams)

        # Do the same for projection layers and their hyperparameters
        for proj_layer_hyperparams in proj_layers:
            self.create_proj_layer(proj_layer_hyperparams)
        self.proj_layers = {}
        
        # Store list of functions to apply between convolution ans projection
        self.functions = {}
        # Check if functions are provided as a dictionary and parse accordingly
        if isinstance(functions, dict):
            for func_cat in functions:
                if func_cat in self.function_categories:
                    self.functions[func_cat] = functions[func_cat]

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
             if k in self.conv_layer_args}
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
             if k in self.proj_layer_args}
        )

    def forward(self, x):
        """
        Feed input patches through convolution layers and projection layers to
            get the final projection output

        Args:
            x (Tensor): input patches

        Return:
            A transformed projection output Tensor
        """
        # Run the sequence of convolution layers on the input patches
        #   to get the convolution output
        for conv_layer in self.conv_layers:
            x = conv_layer.forward(x)

        # Apply the post convolution functions
        #   Ex: For image captioning, flatten the convolution output
        for f in self.functions['apply']:
            x = f(x)

        # Run the sequence of projection layers on the convolution output
        #   to get the projection output
        for proj_layer in self.proj_layers:
            x = proj_layer.forward(x)

        # Return the projection output
        return x

    def get_loss(self,
        predictions, true_labels, predictions_dim=0
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
        return loss.cross_entropy_loss(
            predictions=predictions,
            predictions_dim=predictions_dim,
            true_labels=true_labels,
            weights = [
                weight for weight, _ in
                    [layer for _, layer in
                        [set(self.conv_layers.items()).union(
                            set(self.proj_layers.items()))
                        ]
                    ]
            ]
        )

    def backprop(self, predictions, true_labels):
        """
        Move backward from the loss function to the projection and convolution
            layers to determine how each weight and bias contributed to the total
            prediction error

        Args:
            input_patches (Tensor): the input patches

        Return:
            None
        """
        # First get the derivative of the loss function wrt to the projection
        #   output = 2 * (z_proj - y)
        d_zproj = 2 * (predictions - true_labels)

        # Iterate through the projection layers to update the derivative of the
        #   loss function wrt to the projection output
        for i in range(len(self.proj_layers) - 1, -1):
            proj_layer = self.proj_layers[i]
            d_zproj = proj_layer.backward(d_zproj)

        # Get the last convolution layer output
        last_conv_layer = self.conv_layers[len(self.conv_layers) - 1]
        _, _, z_conv, _, _, _ = last_conv_layer.__get__()

        # Unapply post-convolution functions to get the derivative of the loss
        #   function wrt to the convolution output
        for f in self.functions['unapply']:
            d_zconv = f(d_zproj, z_conv)

        # Iterate through the convolution layers to update the derivative of
        #   the loss function wrt to the convolution output
        for i in range(len(self.conv_layers) - 1, -1):
            conv_layer = self.conv_layers[i]
            d_zconv = conv_layer.backward(d_zconv)

        # Return the updated derivative of the loss function wrt to the
        #   convolution output = the loss derivative wrt to the convolution input
        return d_zconv # d_input_patches

    def update(self,
        learning_rate,
        convolution_weight, convolution_bias,
        projection_weight, projection_bias,
        conv_weight_change, conv_bias_change,
        proj_weight_change, proj_bias_change
    ):
        return self.get_updates(
            learning_rate=learning_rate,
            W_conv=convolution_weight, b_conv=convolution_bias,
            W_proj=projection_weight, b_proj=projection_bias,
            grad_Wconv=conv_weight_change, grad_bconv=conv_bias_change,
            grad_Wproj=proj_weight_change, grad_bproj=proj_bias_change
        )

    # ========================== HELPER FUNCTIONS =============================

    def get_gradients(self,
        x, z_conv, # W_conv and b_conv not used
        h, W_proj, z_proj, # b_proj not used
        true_labels
    ):
        """
        Perform backpropagation to get the derivative of the loss function w.r.t:
            1. the convolution weight
            2. the convolution bias
            3. the projection weight
            4. the projection bias

        Args:
            x (Tensor): the input patches
            W_conv (Tensor): the convolution weight -- NOT USED
            b_conv (Tensor): the convolution bias -- NOT USED
            z_conv (Tensor): the convolution output
            h (Tensor): the flattened convolution output = the projection input
            W_proj (Tensor): the projection weight
            b_proj (Tensor): the projection bias -- NOT USED
            z_proj (Tensor): the projection output = the prediction
            true_label (float): the true labels for the batch of images

        Return:
            grad_Wconv (Tensor): The gradient for the convolution weight
            grad_bconv (Tensor): The gradient for the convolution bias
            grad_Wproj (Tensor): The gradient for the projection weight
            grad_bproj (Tensor): The gradient for the projection bias

        Using d(x) = the derivative of x,
            d(y)/d(x) = the derivative of y w.r.t. x
            L = the loss function (we sum over the batch differences),
            x = the convolution input (input patches matrix),
            W_conv = the convolution weight,
            b_conv = the convolution bias,
            z_conv = the convolution output
            h = z_flat = the flattened convolution output,
            W_proj = the projection weight,
            b_proj = the projection bias, and
            z_proj = the projection output:

        (1) d(L)/d(W_conv) = d(L)/d(z_conv) * d(z_conv)/d(W_conv)
            a. To compute d(L)/d(z_conv), we backpropagate the gradient from the
               projection layer to h, then reshape it back to the shape of z_conv.

               i. First we unflatten d(L)/d(h). Then for d(L)/d(h), we backpropagate
               through the linear layer, giving us
               d(L)/d(h) = d(L)/d(z_proj) * (W_proj)^T. Then we have:

                    d(L)/d(z_conv)
                    = unflatten( d(L)/d(h) )
                    = d(L)/d(z_conv)
                    = unflatten( d(L)/d(z_proj) * (W_proj)^T )
                    = reshape( d(L)/d(z_proj) * (W_proj)^T, shape(z_conv) )
                    = reshape( d( (z_proj - y)^2 - lambda||W||^2 )/d(z_proj)
                                  * (W_proj)^T, shape(z_conv) )
                    = reshape( 2 * (z_proj - y) * (W_proj)^T, shape(z_conv) )

                    *** d(L)/d(z_proj) = 2 * (z_proj - y) and not
                    2/N * (z_proj - y) since we sum over batches for the loss
                    (as opposed to getting the mean over batches for the loss).
                    
                ii. d(z_conv)/d(W_conv) = d(x * W_conv + b_conv)/d(W_conv) = x
            
            b. d(z_conv)/d(W_conv) = d(x * W_conv + b_conv)/d(W_conv) = x

            c. Taking the results of a. and b., we get d(L)/d(W_conv)
               = reshape(2 * (z_proj - y) * (W_proj)^T, shape(z_conv)) * x
               = x^T * reshape(2 * (z_proj - y) * (W_proj)^T, shape(z_conv))
               
               (i) We move x in front of the reshape function. We do this not
                  just to get the correct dimensions from the result of the matrix
                  multiplication, but because the derivative of the loss function
                  w.r.t. to either of the weights describes the following: each
                  W_conv[i,j] learns from how much of the input feature i was
                  present times how much the output feature j was off by.

            d. Therefore, the derivative of the loss function w.r.t. the
               convolution weight is equal to product between (i) the convolution
               input = the input patches matrix, transposed, and (ii) the reshape
               function (which takes in the result of both doubling the difference
               between the prediction output and the target label AND then
               multiplying that with the transpose of the projection weight, in
               the shape of the convolution output)

        (2) d(L)/d(b_conv) = d(L)/d(z_conv) * d(z_conv)/d(b_conv)
            a. We already know that d(L)/d(z_conv)
               = reshape( 2 * (z_proj - y) * (W_proj)^T, shape(z_conv) )

            b. d(z_conv)/d(b_conv) = d(x * W_conv + b_conv)/d(b_conv) = 1

            c. Taking the results of a. and b. we get d(L)/d(b_conv)
               = reshape(2 * (z_proj - y) * (W_proj)^T, shape(z_conv)) * 1
               = reshape(2 * (z_proj - y) * (W_proj)^T, shape(z_conv))

            d. Therefore, the derivative of the loss function w.r.t. the
               convolution bias is simply the reshape function, which takes in
               the result of both doubling the difference between the prediction
               output and the target label AND then multiplying that with the
               transpose of the projection weight, in the shape of the convolution
               output

        (3) d(L)/d(W_proj) = d(L)/d(z_proj) * d(z_proj)/d(W_proj)
            a. From (1)a., we have d(L)/d(z_proj) = 2 * (z_proj - y)

            b. d(z_proj)/d(W_proj) = d(h * W_proj + b_proj)d(W_proj) = h

            c. Taking the results of a. and b., we get
               d(L)/d(W_proj) = 2 * (z_proj - y) * h
               = h^T * 2 * (z_proj - y)

               i. Similar to (1)c.(i), we move h (= the flattened convolution
                  output = the projection input) in front because each
                  W_proj[i, j] learns from how much input feature i correlates
                  to output error j.

            d. Therefore, the derivative of the loss function w.r.t. the
               projection weight is the product of (i) the flattened convolution
               input = the projection input, transposed, and (ii) twice the
               difference between the projection output and the target label.

        (4) d(L)/d(b_proj) = d(L)/d(z_proj) * d(z_proj)/d(b_proj)
            a. We already know that d(L)/d(z_proj) = 2 * (z_proj - y)

            b. d(z_proj)/d(b_proj) = d(x * W_proj + b_proj)/d(b_proj) = 1

            c. Taking the results of a. and b., we get d(L)/d(b_conv)
               = 2 * (z_proj - y) * 1 = 2 * (z_proj - y)

            d. Therefore, the derivative of the loss function w.r.t. the
               projection bias is simply twice the difference between the
               projection output and the target label.
        """
        # d(L)/d(z_proj) = 2 * (z_proj - y)
        dL_dz_proj = 2 * (z_proj - true_labels)

        # d(L)/d(W_conv) = x^T * reshape(2 * (z_proj - y) * (W_proj)^T, shape(z_conv))
        grad_Wconv = x.T @ (dL_dz_proj @ W_proj.T)
        # d(L)/d(b_conv) = reshape(2 * (z_proj - y) * (W_proj)^T, shape(z_conv))
        #   Sum over all patches to get a 1-d vector (same shape as conv bias)
        unflatten = (dL_dz_proj @ W_proj.T).reshape(z_conv.shape)
        grad_bconv = (unflatten).sum(dim=
                        tuple([i for i in unflatten.shape if i != feature_dim]))
        # d(L)/d(W_proj) = h^T * 2 * (z_proj - y)
        grad_Wproj = h.T @ dL_dz_proj
        # d(L)/d(b_proj) = 2 * (z_proj - y)
        #   Sum over all patches to get a 1-d vector (same shape as proj bias)
        grad_bproj = dL_dz_proj.sum(axis=
                        tuple([i for i in dL_dz_proj.shape if i != embedding_dim]))

        # Return the gradients
        return grad_Wconv, grad_bconv, grad_Wproj, grad_bproj
    
    def get_updates(self,
        learning_rate,
        W_conv, b_conv, W_proj, b_proj,
        grad_Wconv, grad_bconv, grad_Wproj, grad_bproj
    ):
        """
        Return the updated convolution weight, convolution bias, projection
            weight, and projection bias based on their respective gradients

        Args:
            learning_rate (float): the model's learning weight
            W_conv (Tensor): the convolution weight
            b_conv (Tensor): the convolution bias
            W_proj (Tensor): the projection weight
            b_proj (Tensor): the projection bias
            grad_Wconv (Tensor): The gradient for the convolution weight
            grad_bconv (Tensor): The gradient for the convolution bias
            grad_Wproj (Tensor): The gradient for the projection weight
            grad_bproj (Tensor): The gradient for the projection bias

        Return:
            Wconv_new (Tensor): the updated convolution weight
            bconv_new (Tensor): the updated convolution bias
            Wproj_new (Tensor): the updated projection weight
            bproj_new (Tensor): the updated projection bias
        """
        
        # Get the updated weights and biases
        Wconv_new = W_conv - learning_rate * grad_Wconv
        bconv_new = b_conv - learning_rate * grad_bconv
        Wproj_new = W_proj - learning_rate * grad_Wproj
        bproj_new = b_proj - learning_rate * grad_bproj

        # Return the updated weights and biases
        return Wconv_new, bconv_new, Wproj_new, bproj_new