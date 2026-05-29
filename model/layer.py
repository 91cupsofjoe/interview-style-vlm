from typing import Union, Optional, Callable, Any

import torch
from torch import Tensor, Size


# ================================== LAYER ====================================

class Layer:
    """
    This is the base layer class. Each Layer object contains its own attributes,
        learnable parameters, and both forward and backward function sets.
    """
    def __init__(self, 
        static_parameters: Optional[dict[str, Any]],
        learnable_parameters: Optional[dict[str, Tensor]],
        forward_func_ptrs: Optional[list[Callable]],
        backward_func_ptrs: Optional[list[Callable]]
    ):
        # Create layer parameters
        if static_parameters is None:
            static_parameters = {}
        self.static_parameters = static_parameters

        # Create learnable parameter pairs
        if learnable_parameters is None:
            learnable_parameters = {}
        self.learnable_parameter_pairs = {}
        self.set_learnable_parameter_pairs(learnable_parameters)

        # Each of the forward and backward function lists are a list of
        #   function pointers.
        # Create empty lists for the function pointer lists if they don't exist.
        if forward_func_ptrs is None:
            forward_func_ptrs = []
        if backward_func_ptrs is None:
            backward_func_ptrs = []

        self.forward_func_ptrs = forward_func_ptrs
        self.backward_func_ptrs = backward_func_ptrs


    def forward(self, x: Tensor) -> Tensor:
        # Store the input tensor in the layer attributes.
        self.static_parameters['forward_input'] = x.clone()
        
        # Run the forward pass functions on the input x.
        for func in self.forward_func_ptrs:
            x = func(x, self.static_parameters)

        # Store the output tensor in the layer attributes and return it.
        self.static_parameters['forward_output'] = x.clone()
        return x
    

    def backward(self, x: Tensor) -> Tensor:
        # Store the input tensor in the layer attributes.
        self.static_parameters['backward_input'] = x.clone()

        # Run the backpropagation functions on the input tensor.
        for func in self.backward_func_ptrs:
            x = func(x, self.static_parameters)

        # Store the output tensor in the layer attributes and return it.
        self.static_parameters['backward_output'] = x.clone()
        return x
    
    
    def get_init_learnable_parameter_sets(self,
        learnable_weights, learnable_biases,
        weight_shapes, bias_shapes,
        use_learnable_bias, num_learnable_sets
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Initialize the sets of learnable parameters if valid sets of them
            are not already provided.

        Args:
            learnable_weights (list[Tensor]): List of learnable weights
            learnable_biases (list[Tensor]): List of learnable biases
            use_learnable_bias (bool): Flag to enable/disable biases
            weight_shapes (list[Size]): Valid shapes for learnable weights
            bias_shapes (list[Size]): Valid shapes for learnable biases
            num_learnable_sets (int): The number of learnable sets

        Return:
            Tensors for the sets of learnable parameters
        """
        # Check if the provided learnable weights and biases (if applicable)
        #   exist and are valid
        if not self.check_valid_learnable_parameter_sets(
            learnable_weights, learnable_biases,
            weight_shapes, bias_shapes,
            use_learnable_bias, num_learnable_sets
        ):
            # Initialize the learnable weights and biases
            learnable_weights = []
            learnable_biases = []
            for i in range(num_learnable_sets):
                learnable_weights.append(torch.randn(weight_shapes[i]))
                learnable_biases.append(torch.randn(weight_shapes[i]))

        # Return the learnable parameters
        return learnable_weights, learnable_biases
    
    
    def get_init_learnable_parameters(self,
        learnable_weights, learnable_biases,
        weight_shapes, bias_shapes,
        use_learnable_bias
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        Initialize the learnable parameters if valid ones are not
            already provided.

        Args:
            learnable_weights (list[Tensor]): List of learnable weights
            learnable_biases (list[Tensor]): List of learnable biases
            use_learnable_bias (bool): Flag to enable/disable biases
            weight_shapes (list[Size]): Valid shapes for learnable weights
            bias_shapes (list[Size]): Valid shapes for learnable biases
            num_learnable_sets (int): The number of learnable sets

        Return:
            Tensors for the learnable parameters
        """
        return self.get_init_learnable_parameter_sets(
            learnable_weights, learnable_biases,
            weight_shapes, bias_shapes,
            use_learnable_bias, num_learnable_sets=1
        )
        
    
    def set_learnable_parameter_pairs(self, learnable_params) -> None:
        # Each tuple in the layer's learnable parameters dictionary contains
        #   the learnable parameter and its respective gradient

        # Iterate through the learnable parameters to create tuples
        #   containing the learnable parameter and its gradient
        for param_name, learnable_param in learnable_params:
            gradient = torch.zeros(learnable_param.shape)
            self.learnable_parameter_pairs[param_name] = (
                learnable_param, gradient
            )

    def get_learnable_parameter_pair(self, param_name=None) \
                    -> Union[tuple[Tensor, Tensor], 
                        dict[str, tuple[Tensor, Tensor]], None]:
        if param_name:
            # Check if the learning parameter exists
            if param_name in self.learnable_parameter_pairs:
                return self.learnable_parameter_pairs[param_name]
        else:
            # Return all the learning parameter pairs in the layer
            return self.learnable_parameter_pairs
        
        # Invalid parameter name provided, return None
        return None

    def check_valid_learnable_parameter_sets(self,
        learnable_weights, learnable_biases, use_learnable_bias,
        weight_shapes, bias_shapes, num_learnable_sets
    ) -> bool:
        """
        Check if the provided sets of learnable parameters are valid.

        Args:
            learnable_weights (list[Tensor]): List of learnable weights
            learnable_biases (list[Tensor]): List of learnable biases
            use_learnable_bias (bool): Flag to enable/disable biases
            weight_shapes (list[Size]): Valid shapes for learnable weights
            bias_shapes (list[Size]): Valid shapes for learnable biases
            num_learnable_sets (int): The number of learnable sets

        Return:
            Boolean indicating if the sets of learnable parameters are valid
        """
        # Check if learnable weights and biases (if applicable) exist
        if learnable_weights is None \
            or (learnable_biases is None and use_learnable_bias) \
                                                                 \
                or (learnable_weights   # Check valid weights
                    and (len(learnable_weights) != num_learnable_sets
                        or any([learnable_weights[i].shape != weight_shapes[i]]
                                        for i in range(num_learnable_sets))

                    or (learnable_weights and learnable_biases) # Check valid biases
                        and(len(learnable_biases) != num_learnable_sets
                        or any([learnable_biases[i].shape != bias_shapes[i]
                                        for i in range(num_learnable_sets)])
                        )
                    )
                ):
            # The provided learnable parameters are not valid
            return False
        
        # Else, the provided learnable parameters are valid
        return True
    
    def check_valid_learnable_parameters(self,
        learnable_weight, learnable_bias, use_learnable_bias,
        weight_shape, bias_shape
    ) -> bool:
        """
        Check if the provided learnable parameters are valid.

        Args:
            learnable_weight (Tensor): Learnable weight
            learnable_bias (Tensor): Learnable bias
            use_learnable_bias (bool): Flag to enable/disable bias
            weight_shape (Size): Valid shapes for learnable weight
            bias_shape (Size): Valid shapes for learnable bias

        Return:
            Boolean indicating if the learnable parameters are valid
        """
        return self.check_valid_learnable_parameter_sets(
            learnable_weights=[learnable_weight],
            learnable_biases=[learnable_bias],
            use_learnable_bias=use_learnable_bias,
            weight_shapes=[weight_shape],
            bias_shapes=[bias_shape],
            num_learnable_sets=1
        )
    
    
# ============================ CONVOLUTION LAYER ==============================

BATCH_SIZE = 4 # Default number of input examples for the input patches
NUM_IN_CHANNELS = 3 # Default number of input channels for convolution
NUM_OUT_FEATURES = 64 # Default number of output features for convolution
KERNEL_SIZE = 3 # Default kernel size for convolution = (3, 3)
STRIDE = 1 # Default stride for convolution and pooling
PADDING = 0 # Default padding for convolution
POOL_SIZE = 2 # Default pooling dimensions = (2, 2)
POOL_STRIDE = 2 # Default pooling stride
POOL_TYPE = "max" # Default pooling type
    
class ConvolutionLayer(Layer):
    """
    This is the convolution layer class.
    """
    def __init__(self,
        conv_weight=None, conv_bias=None,
        use_conv_bias=True,
        batch_size=BATCH_SIZE,
        num_in_channels=NUM_IN_CHANNELS, num_out_features=NUM_OUT_FEATURES,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE, padding=PADDING,
        pool_type=POOL_TYPE,
        pool_size=POOL_SIZE, pool_stride=POOL_STRIDE,
        forward_func_ptrs=None,
        backward_func_ptrs=None
    ):
        # Get the static parameters for the convolution layer
        static_parameters = {
            'use_bias': use_conv_bias,
            'num_in_channels': num_in_channels, 'num_out_features': num_out_features,
            'kernel_size': kernel_size, 'stride': stride, 'padding': padding,
            'pool_size': pool_size, 'pool_stride': pool_stride, 'pool_type': pool_type
        }

        # Get the learnable parameters for the convolution layer
        learnable_parameters = {}

        weight_shape = Size(
            [batch_size, num_in_channels * kernel_size, num_out_features])
        bias_shape = Size([num_out_features])

        # Initialize the convolution weight and bias (if applicable),
        #   along with the respective gradients
        conv_weight, conv_bias = self.get_init_learnable_parameters(
            conv_weight, conv_bias, use_conv_bias,
            weight_shape, bias_shape
        )
        learnable_parameters['W_conv'] = conv_weight,
        learnable_parameters['b_conv'] = conv_bias

        super().__init__(
            static_parameters=static_parameters,
            learnable_parameters=learnable_parameters,
            forward_func_ptrs=forward_func_ptrs,
            backward_func_ptrs=backward_func_ptrs
        )


# ============================= PROJECTION LAYER ==============================

NUM_PATCHES = 512 # Default number of patches/feature maps
EMBEDDING_SIZE = 756 # Default embedding size for projection

class ProjectionLayer(Layer):
    """
    This is the projection layer class.
    """
    def __init__(self,
        proj_weight=None, proj_bias=None,
        use_proj_bias=True,
        num_patches=NUM_PATCHES, embedding_size=EMBEDDING_SIZE,
        forward_func_ptrs=None,
        backward_func_ptrs=None
    ):
        # Get the static parameters for the projection layer
        static_parameters = {
            'use_bias': use_proj_bias,
            'embedding_size': embedding_size
        }

        # Get the learnable parameters for the projection layer
        learnable_parameters = {}

        weight_shape = Size([num_patches, embedding_size])
        bias_shape = Size([embedding_size])

        # Initialize the projection weight and bias (if applicable),
        #   along with the respective gradients
        proj_weight, proj_bias = self.get_init_learnable_parameters(
            proj_weight, proj_bias, use_proj_bias,
            weight_shape, bias_shape
        )
        learnable_parameters['W_proj'] = proj_weight,
        learnable_parameters['b_proj'] = proj_bias

        super().__init__(
            static_parameters=static_parameters,
            learnable_parameters=learnable_parameters,
            forward_func_ptrs=forward_func_ptrs,
            backward_func_ptrs=backward_func_ptrs
        )


# ==================== TRANSFORMER ENCODER/DECODER BLOCKS =====================

BATCH_SIZE = 64 # Default number of input examples per input patches
MAX_SEQ_LEN = 12 # Default max sequence length
NUM_ATTN_HEADS = 2 # Default number of parallel attention heads
EMBEDDING_SIZE = 64 # Default embedding size for encoding/decoding
FEED_FWD_SIZE = 128 # Default size for feed forward network output
DROPOUT = 0.1 # Default dropout value

NUM_LAYER_NORM_OPS = 2 # Default number of layer normalization operations
NUM_FEED_FORWARD_OPS = 2 # Default number of feed forward operations

learnable_parameter_subscripts = [
    'q', 'k', 'v', 'o'
]

class TransformerBlock(Layer):
    """
    This is the transformer encoder/decoder block (layer) class.
    """
    def __init__(self,
        proj_weights=None, proj_biases=None,
        use_proj_bias=True,
        layer_norm_weights=None, layer_norm_biases=None,
        use_layer_norm_bias=True, num_layer_norm_ops=NUM_LAYER_NORM_OPS,
        feed_fwd_weights=None, feed_fwd_biases=None,
        use_feed_fwd_bias=True, num_feed_fwd_ops=NUM_FEED_FORWARD_OPS,
        batch_size=BATCH_SIZE,
        num_attn_heads=NUM_ATTN_HEADS, max_seq_len=MAX_SEQ_LEN,
        embedding_size=EMBEDDING_SIZE, feed_fwd_size=FEED_FWD_SIZE,
        forward_func_ptrs=None,
        backward_func_ptrs=None
    ):
        # First get number of learnable attention parameter sets and
        #   size of individual attention heads
        num_attn_parameter_sets = 4
        assert embedding_size % num_attn_heads == 0
        attn_head_size = embedding_size // num_attn_heads

        # Get the static parameters for the transformer block
        static_parameters = {
            'use_proj_bias' : use_proj_bias,
            'use_layer_norm_bias' : use_layer_norm_bias,
            'use_feed_fwd_bias' : use_feed_fwd_bias,
            'batch_size': batch_size, 'max_seq_len': max_seq_len,
            'embedding_size': embedding_size,
            'num_attn_heads': num_attn_heads, 'attn_head_size': attn_head_size
        }

        # Get the learnable parameters for the transformer block
        learnable_parameters = {}

        # Initialize the attention projection weight and bias
        #   (if applicable), along with the respective gradients
        proj_weight_shapes = [Size([embedding_size])
                        for i in range(num_attn_parameter_sets)]
        proj_bias_shapes = [Size([embedding_size])
                        for i in range(num_attn_parameter_sets)]
        
        proj_weights, proj_biases = self.get_init_learnable_parameter_sets(
            learnable_weights=proj_weights,
            learnable_biases=proj_biases,
            weight_shapes=proj_weight_shapes,
            bias_shapes=proj_bias_shapes,
            use_learnable_bias=use_proj_bias,
            num_learnable_sets=num_attn_parameter_sets
        )

        for i in range(len(learnable_parameter_subscripts)):
            subscript = learnable_parameter_subscripts[i]
            learnable_parameters['W'+subscript+"_proj"] = proj_weights[i]
            learnable_parameters['b'+subscript+"_proj"] = proj_biases[i]
        
        # Initialize the layer norm weights, biases (if applicable),
        #   and gradients
        layer_norm_weight_shapes = [Size([embedding_size])
                        for i in range(num_attn_parameter_sets)]
        layer_norm_bias_shapes = [Size([embedding_size])
                        for i in range(num_attn_parameter_sets)]
        
        layer_norm_weights, layer_norm_biases = \
            self.get_init_learnable_parameter_sets(
                learnable_weights=layer_norm_weights,
                learnable_biases=layer_norm_biases,
                weight_shapes=layer_norm_weight_shapes,
                bias_shapes=layer_norm_bias_shapes,
                use_learnable_bias=use_layer_norm_bias,
                num_learnable_sets=num_layer_norm_ops
            )
        
        for i in range(num_layer_norm_ops):
            learnable_parameters['gamma_'+str(i+1)] = layer_norm_weights[i]
            learnable_parameters['beta_'+str(i+1)] = layer_norm_biases[i]
                
        # Initialize the feed forward weights, biases (if applicable),
        #   and gradients
        feed_fwd_weight_shapes = [
            Size([embedding_size, feed_fwd_size]) if i % 2 == 0
            else Size([feed_fwd_size, embedding_size])
                for i in range(num_attn_parameter_sets)
        ]
        feed_fwd_bias_shapes = [
            Size([feed_fwd_size]) if i % 2 == 0
            else Size([embedding_size])
                for i in range(num_attn_parameter_sets)
        ]
        
        feed_fwd_weights, feed_fwd_biases = \
            self.get_init_learnable_parameter_sets(
                learnable_weights=feed_fwd_weights,
                learnable_biases=feed_fwd_biases,
                weight_shapes=feed_fwd_weight_shapes,
                bias_shapes=feed_fwd_bias_shapes,
                use_learnable_bias=use_feed_fwd_bias,
                num_learnable_sets=num_feed_fwd_ops
            )
        
        for i in range(num_layer_norm_ops):
            learnable_parameters['Wff_'+str(i+1)+'_proj'] = feed_fwd_weights[i]
            learnable_parameters['bff_'+str(i+1)+'_proj'] = feed_fwd_biases[i]

        # Take all the learnable parameters and create a dictionary containing
        #   tuples of learnable parameters and their respective gradient
        self.set_learnable_parameter_pairs(learnable_parameters)

        super().__init__(
            static_parameters=static_parameters,
            learnable_parameters=learnable_parameters,
            forward_func_ptrs=forward_func_ptrs,
            backward_func_ptrs=backward_func_ptrs
        )


# ============================= HELPER METHODS ================================

layer_subclasses = {
    'convolution_layer' : ConvolutionLayer,
    'projection_layer' : ProjectionLayer,
    'transformer_block' : TransformerBlock
}

def get_layer(layer_type: str, layer_parameters: dict[str, Any]) -> Layer:
    """
    Return a Layer subclass object with the specified layer parameters.

    Args:
        layer_type (str): The type of layer
        layer_parameters (dict[str, Any]): The layer parameters

    Return:
        The Layer subclass object
    """
    layer_subclass = layer_subclasses[layer_type]

    return layer_subclass(**layer_parameters)