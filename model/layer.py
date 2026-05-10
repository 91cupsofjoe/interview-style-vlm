import math

from pydantic import BaseModel
import torch


class Layer(BaseModel):
    def __init__(self):
        # Attributes will store weight, bias, derivatives, and hyperparameters
        self.attr = {}

    def __get__(self, attribute = None):
        # Check if a valid attribute key was provided
        if attribute and attribute in self.attr.keys:
            return self.attr[attribute]
        # Else, return the entire attributes dictionary
        return self.attr
    
    def __set__(self, key, value):
        # Set the attribute key and value to attributes
        self.attr[key] = value


# =========================== CONVOLUTION LAYER ===============================

NUM_CHANNELS = 3 # Default number of input channels for convolution
NUM_OUT_FEATURES = 64 # Default number of output features for convolution
KERNEL_SIZE = 3 # Default kernel size for convolution = (3, 3)
STRIDE = 1 # Default stride for convolution and pooling
PADDING = 0 # Default padding for convolution
POOL_SIZE = 2 # Default pooling dimensions = (2, 2)
POOL_STRIDE = 2 # Default pooling stride
POOL_TYPE = "max" # Default pooling type

class ConvLayer(Layer):
    def __init__(self,
        W=None,b=None, x=None,
        num_channels=NUM_CHANNELS, num_out_features=NUM_OUT_FEATURES,
        kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING,
        pool_size=POOL_SIZE, pool_stride=POOL_STRIDE, pool_type=POOL_TYPE
    ):
        super().__init__()

        # Convolution layer attributes
        self.attr = {
            # Tensors
            'W_conv': W, # Convolution weight
            'b_conv': b, # Convolution bias
            'x': x, # Input patches
            'z_conv': None, # Convolution output
            'd_Wconv': None, # Derivative of loss wrt convolution weight
            'd_bconv': None, # Derivative of loss wrt bias

            # Hyperparameters
            'num_channels': num_channels, # Number of input channels
            'num_out_features': num_out_features, # Number of output features
            'kernel_size': kernel_size, # Dimensions of the filter
            'stride': stride, # Number of pixels the filter moves
            'padding': padding, # Number of pixels added around filter
            'pool_size': pool_size, # The dimensions of the pool window
            'pool_stride': pool_stride, # The number of pixels per window move
            'pool_type': pool_type # The type of pool operation (max or average)
        }

    def forward(self, x, conv_func):
        """
        Feed the input patches Tensor through the convolution layer

        Args:
            x (Tensor): The input patches Tensor

        Return:
            x (Tensor): The ReLU activated, pooled convolution output Tensor
        """
        # First store a copy of the input patches Tensor
        self.attr['x'] = x.clone()

        # Check convolution layer's weight and bias have been initialized
        #   If not, use the num out features, num channels, and kernel size
        #   to initialize the weight and bias
        if self.W is None or self.b is None:
            self.W = torch.randn(
                self.attr['num_out_features'], self.attr['num_channels'],
                self.attr['kernel_size'][0], self.attr['kernel_size'][1]
            )
            self.b = torch.randn(self.attr['num_out_features'])
        
        # Apply convolution to the input patches Tensor
        try:
            conv_func = self.attr['conv_func']
            conv_func_params = self.attr['conv_func_params']
            z_conv = conv_func(x, conv_func_params)
        except:
            raise RuntimeError("Could not run convolution!")

        # Store a copy of the pooled, ReLU activated convolution output Tensor
        #   and return the original
        self.z_conv = z_conv.clone()
        return z_conv
    
    def backward(self, d_zconv):
        """
        Feed the convolution output gradient Tensor backward through the
            convolution layer, to update the convolution weight and bias
            gradient Tensors and get the input patches gradient

        Args:
            d_zproj (Tensor): The convolution output gradient Tensor

        Return:
            The input patches gradient
        """
        # First store a copy of the input patches Tensor
        self.attr['x'] = x.clone()

        # Check convolution layer's weight and bias have been initialized
        #   If not, use the num out features, num channels, and kernel size
        #   to initialize the weight and bias
        if self.W is None or self.b is None:
            self.W = torch.randn(
                self.attr['num_out_features'], self.attr['num_channels'],
                self.attr['kernel_size'][0], self.attr['kernel_size'][1]
            )
            self.b = torch.randn(self.attr['num_out_features'])
        
        # Apply convolution to the input patches Tensor
        try:
            conv_func = self.attr['conv_func']
            conv_func_params = self.attr['conv_func_params']
            z_conv = conv_func(x, conv_func_params)
        except:
            raise RuntimeError("Could not run convolution!")

        # Store a copy of the pooled, ReLU activated convolution output Tensor
        #   and return the original
        self.z_conv = z_conv.clone()
        return z_conv
    
    def transform(self, tensor, transform):
        """
        Perform a forward pass or backpropagation transformation on the input
            Tensor

        Args:

        """


# ============================ PROJECTION LAYER ===============================

EMBEDDING_SIZE = 128 # Default embedding size for projection

class ProjLayer(Layer):
    def __init__(self,
        W=None, b=None, h=None,
        proj_func=None, proj_func_params=None,
        embedding_size=EMBEDDING_SIZE
    ):
        super().__init__()

        if proj_func_params is None:
            proj_func_params = {}

        # Projection layer attributes
        self.attributes = {
            # Tensors
            'W_proj': W, # Convolution weight
            'b_proj': b, # Convolution bias
            'x': h, # Input patches
            'z_proj': None, # Convolution output
            'd_Wproj': None, # Derivative of loss wrt convolution weight
            'd_bproj': None, # Derivative of loss wrt bias

            # Hyperparameters
            'embedding_size': embedding_size # The length of an embedding
        }

    def forward(self, h, proj_func):
        """
        Feed the convolution output Tensor through the linear projection layer

        Args:
            x (Tensor): The convolution output / projection input Tensor

        Return:
            x (Tensor): The projection output Tensor
        """
        # First store a copy of the projection input Tensor
        self.attr['h'] = h.clone()

        # Check if the projection layer's weight and bias have been initialized
        #   If not, use the projection input's patch size and the projection
        #   layer's embedding size to initialize the weight and bias
        if self.W is None or self.b is None:
            patch_size = h.shape[-1]
            self.W = torch.randn(patch_size, self.attr['embedding_size'])
            self.b = torch.randn(self.attr['embeddings_size'])

        # Apply projection to the input Tensor
        try:
            proj_func = self.attr['proj_func']
            proj_func_params = self.attr['proj_func_params']
            z_proj = proj_func(h, proj_func_params)
        except:
            raise RuntimeError("Could not run projection!")

        # Store a copy of the projection output Tensor and return the original
        self.attr['z_proj'] = z_proj.clone()
        return z_proj
    
    def backward(self, d_zproj):
        """
        Feed the derivative of the loss wrt the projection output
            backward through the linear projection layer, updating the
            derivatives of the loss function wrt both the projection weight
            and the projection bias

        Args:
            d_zproj (Tensor): The derivative of the loss
                wrt the projection output

        Return:
            The updated derivative of the loss wrt the projection output
        """
        # Apply projection backward to the projection output gradient
        try:
            proj_func = self.attr['proj_func']
            proj_func_params = self.attr['proj_func_params']
            z_proj = proj_func(h, proj_func_params)
        except:
            raise RuntimeError("Could not run projection!")

        # Store a copy of the projection output Tensor and return the original
        self.attr['z_proj'] = z_proj.clone()
        return z_proj