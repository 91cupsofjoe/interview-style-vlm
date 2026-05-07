import math
from pydantic import BaseModel

import torch
import torch.nn.functional as nnf

from dataset import dataset as ds

BATCH_SIZE = 32 # Default batch size for training
NUM_CHANNELS = 3 # Default number of input channels for convolution
NUM_OUT_FEATURES = 64 # Default number of output features for convolution
KERNEL_SIZE = 3 # Default kernel size for convolution = (3, 3)
EMBEDDING_SIZE = 128 # Default embedding size for projection output
CONV_SEQ_LEN= 8 # Default sequence length for convolution
PROJ_SEQ_LEN= 8 # Default sequence length for projection
POOLING_DIMS = 2 # Default pooling dimensions = (2, 2)
POOLING_TYPE = "max" # Default pooling type
STRIDE = 1 # Default stride for convolution and pooling
PADDING = 0 # Default padding for convolution

class Model(BaseModel):
    def __init__(self):
        self.model = None
        self.convolution_weights_biases = None
        self.projection_weights_biases = None

    def forward_conv(
            self,
            # Tensor of batch size x RGB channels x height x width
            input_patches, batch_size=BATCH_SIZE,
            num_channels=NUM_CHANNELS, num_out_features=NUM_OUT_FEATURES,
            kernel_size=KERNEL_SIZE, # embedding_size=EMBEDDING_SIZE,
            seq_len=CONV_SEQ_LEN, # proj_seq_len=PROJ_SEQ_LEN,
            stride=STRIDE, padding=PADDING,
            pooling_dims=POOLING_DIMS, pooling_type=POOLING_TYPE,
            convolution_weights_biases=[], # projection_weights_biases=[]
        ):
        """
        Perform the forward pass
        
        Args:
            input_patches (Tensor): The input patches Tensor
            
        Return:
            The output projection (prediction) Tensor
        """
        # NOTE: Use the model's internal weights and biases if not provided
        while len(convolution_weights_biases) < seq_len:
            convolution_weights_biases.append((None, None))

        # For each convolution layer, perform
        #   convolution, ReLU activation, and pooling
        for i in range(seq_len):
            W_conv, b_conv = convolution_weights_biases[i]
            # Randomize the convolution weight and bias is not already set
            if W_conv is None or b_conv is None:
                W_conv = torch.randn(num_out_features, num_channels, kernel_size)
                b_conv = torch.randn(num_out_features)

            # Convolution and pooling for the sequence
            z_conv = self.get_conv2d(input_patches, W_conv, b_conv,
                            stride=stride, padding=padding)
            
            # ReLU activation
            h = self.get_ReLU(z_conv)

            # Pooling
            input_patches = self.get_pooled(h, kernel_size=pooling_dims,
                            stride=stride, type=pooling_type)
            
        # Return the convolution output of the final convolution layer
        return input_patches

    def forward_proj(self,
            # Tensor of batch size x RGB channels x height x width
            input_patches, batch_size=BATCH_SIZE,
            num_channels=NUM_CHANNELS, num_out_features=NUM_OUT_FEATURES,
            kernel_size=KERNEL_SIZE, # embedding_size=EMBEDDING_SIZE,
            seq_len=CONV_SEQ_LEN, # proj_seq_len=PROJ_SEQ_LEN,
            stride=STRIDE, padding=PADDING,
            projection_weights_biases=[]
        ):
        # Randomize the projection weight and bias if not already set
        if W_proj is None or b_proj is None:
            # Num patches is the 2nd dimension (index 1) for h
            W_proj = torch.randn(embedding_size, h.shape[1])
            b_proj = torch.randn(embedding_size)

    def predict(self, preprocessed_image):
        """
        Take a preprocessed image as input and return a predicted caption

        Args:
            preprocessed_image (Tensor): Tensor for the preprocessed image

        Return:
            The predicted caption for the query image
        """
        # Open 

    # =========================== HELPER FUNCTIONS ============================

    def get_conv2d(self, x, W, b, stride=STRIDE, padding=PADDING):
        batch_size, channels, image_h, image_w = x.shape
        num_out_features, _, kernel_h, kernel_w = W.shape

        # Calculate the spatial grid dimensions from subtracting the kernel
        #   dimensions and factoring in jump size (stride) and border (padding)
        h_out = math.floor( (image_h + 2 * padding - kernel_h) / stride) + 1
        w_out = math.floor( (image_w + 2 * padding - kernel_w) / stride) + 1

        # Calculate patch size and num patches
        patch_size = channels * kernel_h * kernel_w
        num_patches = h_out * w_out

        # Get input tensor with dims:
        #   batch size, patch size, num patches
        input_tensor = nnf.unfold(x, kernel_size=(kernel_h, kernel_w),
                        padding=padding, stride=stride)
        
        # Transpose input tensor, switching its dimensions from [batch size,
        #   patch size, num patches] to [batch size, num patches, patch size]
        input_tensor = input_tensor.transpose(1, 2)

        # Make sure the shape of the output tensor =
        #   (batch size, patch size, num patches)
        assert(input_tensor.shape == (batch_size, patch_size, num_patches))
        
        # Get flattened weight matrix with dims:
        #   num output features, num channels * kernel h * kernel w
        W_flat = W.view(num_out_features, -1)

        # Get output tensor with dims:
        #   batch size, num output features, num patches
        output_tensor = input_tensor @ W_flat.T

        # Swap the num output features and num patches dimensions
        output_tensor.transpose(1, 2)

        # For each spatial position in each batch image in a feature map,
        #   add the scalar bias value associated with that feature.
        output_tensor += b.view(1, 1, -1) # b.view(1, 1, num out features)

        # Reshape the output tensor's spatial dimensions (num_patches)
        #   according to calculated height and width from stride and padding
        output_tensor = output_tensor.reshape(
                        batch_size, num_out_features, h_out, w_out)
        
        # Finally return the output tensor
        return output_tensor
    
    def get_ReLU(self, x):
        # ReLU activation function (per element e): max(e, 0)
        return torch.clamp(x, min=0)
    
    def get_pooled(self, x, kernel_size=(2, 2), stride=1, type="max"):
        if type == "average":
            return nnf.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
        # Else, return average pooling -- the default
        return nnf.max_pool2d(x, kernel_size=kernel_size, stride=stride)
    
    def get_projection(self, x, W, b):
        # The projection output should have dimensions
        #   batch size x num patches x embedding size
        return x @ W.T + b
    
    def get_gradients(self,
                 x, z_conv, # W_conv and b_conv not used
                 h, W_proj, z_proj, # b_proj not used
                 true_label):
        """
        Perform backpropagation to get the derivative of the loss function w.r.t:
            1. the convolution weight
            2. the convolution bias
            3. the projection weight
            4. the projection bias

        Args:
            x (tensor): the input patches
            W_conv (tensor): the convolution weight -- NOT USED
            b_conv (tensor): the convolution bias -- NOT USED
            z_conv (tensor): the convolution output
            h (tensor): the flattened convolution output = the projection input
            W_proj (tensor): the projection weight
            b_proj (tensor): the projection bias -- NOT USED
            z_proj (tensor): the projection output = the prediction
            true_label (float): the true labels for the batch of images

        Return:
            grad_Wconv (tensor): The gradient for the convolution weight
            grad_bconv (tensor): The gradient for the convolution bias
            grad_Wproj (tensor): The gradient for the projection weight
            grad_bproj (tensor): The gradient for the projection bias

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
        dL_dz_proj = 2 * (z_proj - true_label)

        # d(L)/d(W_conv) = x^T * reshape(2 * (z_proj - y) * (W_proj)^T, shape(z_conv))
        grad_Wconv = x.T @ (dL_dz_proj @ W_proj.T).reshape(z_conv.shape)
        # d(L)/d(b_conv) = reshape(2 * (z_proj - y) * (W_proj)^T, shape(z_conv))
        #   Sum over all patches to get a 1-d vector (same shape as conv bias)
        grad_bconv = ((dL_dz_proj @ W_proj.T).reshape(z_conv.shape)).sum(dim=(0, 2, 3))
        # d(L)/d(W_proj) = h^T * 2 * (z_proj - y)
        grad_Wproj = h.T @ dL_dz_proj
        # d(L)/d(b_proj) = 2 * (z_proj - y)
        #   Sum over all patches to get a 1-d vector (same shape as proj bias)
        grad_bproj = dL_dz_proj.sum(axis=0)

        # Return the gradients
        return grad_Wconv, grad_bconv, grad_Wproj, grad_bproj
    
    def get_updates(self, learning_rate,
                    W_conv, b_conv, W_proj, b_proj,
                    grad_Wconv, grad_bconv, grad_Wproj, grad_bproj,
            ):
        """
        Return the updated convolution weight, convolution bias, projection
            weight, and projection bias based on their respective gradients

        Args:
            W_conv (tensor): the convolution weight
            b_conv (tensor): the convolution bias
            W_proj (tensor): the projection weight
            b_proj (tensor): the projection bias
            grad_Wconv (tensor): The gradient for the convolution weight
            grad_bconv (tensor): The gradient for the convolution bias
            grad_Wproj (tensor): The gradient for the projection weight
            grad_bproj (tensor): The gradient for the projection bias

        Return:
            Wconv_new (tensor): the updated convolution weight
            bconv_new (tensor): the updated convolution bias
            Wproj_new (tensor): the updated projection weight
            bproj_new (tensor): the updated projection bias
        """
        
        # Get the updated weights and biases
        Wconv_new = W_conv - learning_rate * grad_Wconv
        bconv_new = b_conv - learning_rate * grad_bconv
        Wproj_new = W_proj - learning_rate * grad_Wproj
        bproj_new = b_proj - learning_rate * grad_bproj

        # Return the updated weights and biases
        return Wconv_new, bconv_new, Wproj_new, bproj_new