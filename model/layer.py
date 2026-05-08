import math

from pydantic import BaseModel
import torch
import torch.nn.functional as nnf

from util import util
from model import image_encoder as ie

BATCH_SIZE = 32 # Default batch size for training
NUM_CHANNELS = 3 # Default number of input channels for convolution
NUM_OUT_FEATURES = 64 # Default number of output features for convolution
FEATURES_DIM = 1 # Default dimension for output features for convolution
KERNEL_SIZE = 3 # Default kernel size for convolution = (3, 3)
STRIDE = 1 # Default stride for convolution and pooling
PADDING = 0 # Default padding for convolution
POOL_SIZE = 2 # Default pooling dimensions = (2, 2)
POOL_STRIDE = 2 # Default pooling stride
POOL_TYPE = "max" # Default pooling type
EMBEDDING_SIZE = 128 # Default embedding size for projection
EMBEDDINGS_DIM = 1 # Default embedding dimension for projection

class Layer(BaseModel):
    def __init__(self,
        W=None,b=None,
    ):
        self.W = W
        self.b = b
        self.dW = None # the derivative w.r.t. the weight matrix
        self.db = None # the derivative w.r.t. the bias vector

    def __output__(self):
        # Return the layer's weight matrix and bias vector
        return self.W, self.b

class ConvLayer(Layer):
    def __init__(self,
        W=None,b=None,
        num_channels=NUM_CHANNELS,
        num_out_features=NUM_OUT_FEATURES, features_dim=FEATURES_DIM,
        kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING,
        pool_size=POOL_SIZE, pool_stride=POOL_STRIDE, pool_type=POOL_TYPE
    ):
        super().__init__(W=W, b=b)

        self.num_channels = num_channels
        self.num_out_features = num_out_features
        self.features_dim = features_dim
        self.kernel_size = util.get_tuple(kernel_size)
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_type = pool_type

        self.x = None
        self.z_conv = None

    def __get__(self):
        return self.W, self.b, self.x, self.z_conv, self.dW, self.db

    def forward(self, x):
        """
        Feed the input patches Tensor through the convolution layer

        Args:
            x (Tensor): The input patches Tensor

        Return:
            x (Tensor): The ReLU activated, pooled convolution output Tensor
        """
        # First store a copy of the input patches Tensor
        self.x = x.clone()

        # Check convolution layer's weight and bias have been initialized
        #   If not, use the num out features, num channels, and kernel size
        #   to initialize the weight and bias
        if self.W is None or self.b is None:
            self.W = torch.randn(
                self.num_out_features, self.num_channels,
                self.kernel_size[0], self.kernel_size[1]
            )
            self.b = torch.randn(self.num_out_features)
        
        # Apply convolution to the input Tensor
        x = ie.get_conv2d(
            x, self.W, self.b,
            stride=self.stride, padding=self.padding
        )
        
        # Apply ReLU activation to the convolution output Tensor
        x = util.get_ReLU(x)
        
        # Apply pooling to the ReLU activated convolution output Tensor
        x = ie.get_pool(
            x,
            kernel_size=self.pool_size,
            stride=self.pool_stride,
            pool_type=self.pool_type
        )

        # Store a copy of the pooled, ReLU activated convolution output Tensor
        #   and return the original
        self.z_conv = x.clone()
        return x
    
    def backward(self, d_zconv):
        """
        Feed the derivative of the loss function wrt the convolution output
            backward through the convolution layer, updating the derivatives
            of the loss function wrt both the convolution weight and the
            convolution bias

        Args:
            d_zproj (Tensor): The derivative of the loss function
                wrt the projection output

        Return:
            The updated derivative of the loss function wrt the projection output
        """
        self.dW = self.x.T @ d_zconv
        self.db = d_zconv.sum(axis=
                        tuple([i for i in d_zconv.shape
                               if i != self.features_dim])
        )
        return d_zconv @ self.W.T

class ProjLayer(Layer):
    def __init__(self,
        W=None, b=None,
        embedding_size=EMBEDDING_SIZE,
        embeddings_dim=EMBEDDINGS_DIM
    ):
        super().__init__(W=W, b=b)

        self.embedding_size = embedding_size
        self.embeddings_dim = embeddings_dim

    def __get__(self):
        return self.W, self.b, self.h, self.z_proj, self.dW, self.db

    def forward(self, x):
        """
        Feed the convolution output Tensor through the linear projection layer

        Args:
            x (Tensor): The convolution output / projection input Tensor

        Return:
            x (Tensor): The projection output Tensor
        """
        # First store a copy of the projection input Tensor
        self.h = x.clone()

        # Check if the projection layer's weight and bias have been initialized
        #   If not, use the input's patch size and the projection layer's
        #   embedding size to initialize the weight and bias
        if self.W is None or self.b is None:
            patch_size = x.shape[-1]
            self.W = torch.randn(patch_size, self.embedding_size)
            self.b = torch.randn(self.embedding_size)

        # Apply projection to the input Tensor
        x = util.get_projection(x, self.W, self.b)

        # Store a copy of the projection output Tensor and return the original
        self.z_proj = x.clone()
        return x
    
    def backward(self, d_zproj):
        """
        Feed the derivative of the loss function wrt the projection output
            backward through the linear projection layer, updating the
            derivatives of the loss function wrt both the projection weight
            and the projection bias

        Args:
            d_zproj (Tensor): The derivative of the loss function
                wrt the projection output

        Return:
            The updated derivative of the loss function wrt the projection output
        """
        self.dW = self.h.T @ d_zproj
        self.db = d_zproj.sum(axis=
                        tuple([i for i in d_zproj.shape
                               if i != self.embeddings_dim]))
        return d_zproj @ self.W.T