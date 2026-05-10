import math
import torch
import torch.nn.functional as nnf

from util import util


# ========================= CONVOLUTION FUNCTIONS =============================

# ---------------- FORWARD PASS ----------------
# NOTE: The convolution forward pipeline =
#   conv2d --> ReLU --> pool --> flatten

def conv2d(x, W, b, stride, padding):
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
    assert(input_tensor.shape == (batch_size, num_patches, patch_size))
        
    # Get flattened weight matrix with dims:
    #   num output features, num channels * kernel h * kernel w
    W_flat = W.view(num_out_features, -1)

    # Get output tensor with dims:
    #   batch size, num output features, num patches
    output_tensor = input_tensor @ W_flat.T

    # Swap the num output features and num patches dimensions
    output_tensor = output_tensor.transpose(1, 2)

    # For each spatial position in each batch image in a feature map,
    #   add the scalar bias value associated with that feature.
    output_tensor += b.view(1, 1, -1) # b.view(1, 1, num out features)

    # Reshape the output tensor's spatial dimensions (num_patches)
    #   according to calculated height and width from stride and padding
    output_tensor = output_tensor.reshape(
                    batch_size, num_out_features, h_out, w_out)
        
    # Finally return the output tensor
    return output_tensor

def relu(z_conv, params=None):
    """
    Apply forward pass ReLU activation on the convolution output

    Args:
        a_relu (Tensor): The ReLU activated convolution output Tensor
        params (dict): The parameters for ReLU activation (none required,
            passed in for consistency with other forward pass functions)

    Return:
        The pooled, ReLU activated convolution output Tensor
    """
    return util.get_ReLU(z_conv)
    
def pool(a_relu, params):
    """
    Apply forward pass pooling to the ReLU activated convolution output

    Args:
        a_relu (Tensor): The ReLU activated convolution output Tensor
        params (dict): The parameters for applying pooling

    Return:
        The pooled, ReLU activated convolution output Tensor
    """
    # Get pool forward params
    kernel_size = params['pool_size']
    stride = params['pool_stride']
    pool_type = params['pool_type']

    if pool_type == "average":
        return nnf.avg_pool2d(a_relu, kernel_size=kernel_size, stride=stride)
    # Else, return max pooling -- the default
    return nnf.max_pool2d(a_relu, kernel_size=kernel_size, stride=stride)

def flatten(x):
    """
    Take an image Tensor and flatten its patches and spatial dimensions

    Args:
        x (Tensor): The image Tensor to flatten

    Return:
        The flattened image Tensor
    """
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)

# -------------- BACKPROPAGATION ---------------
# NOTE: The convolution backward pipeline =
#   unflatten --> pool backward --> ReLU backward --> conv2d backward

def unflatten(d_zproj, params):
    """
    Unflatten the projection output derivative Tensor to get the pooled, ReLu
        activated convolution output derivate Tensor

    Args:
        d_zproj (Tensor): The projection output derivate Tensor
        params (dict): The parameters for unflattening

    Return:
        The derivative of the loss wrt the pooled, ReLU activated convolution output
    """
    pooled = params['z_pool']
    return d_zproj.reshape(pooled.shape)

def pool_backward(d_pool, params):
    """
    Apply pooling backward on the pooled, ReLU activated convolution output
        derivative Tensor to get the derivative of the loss wrt the ReLU
        activated convolution output

    Args:
        d_pool (Tensor): The pooled, ReLU activated convolution output
            derivative Tensor
        params (dict): The parameters for applying pool backward

    Return:
        The derivative of the loss wrt the ReLU activated convolution output
    """
    # Get pool backward parameters
    a_relu = params['a_relu'] # ReLU activated convolution output
    pool_type = params['pool_type']
    pool_size = params['pool_size'] # Pool window dimensions
    stride = params['pool_stride'] # Number of pixels to jump per move

    # The derivative of the loss wrt to the ReLU activated convolution output
    d_relu = torch.zeros_like(a_relu)

    # The 3rd and 4th dimensions of the ReLU activated convolution output are
    #   the height and width of the prepooled Tensor
    h_dim, w_dim = (2, 3)
    # Get pool window height and width
    pool_h, pool_w = (pool_size)

    # Iterate over each position in the pooled spatial grid
    for i in range(d_pool.shape[h_dim]):
        for j in range(d_pool.shape[w_dim]):
            h_start = i * stride
            h_stop = h_start + pool_size[0]
            w_start = j * stride
            w_stop = w_start + pool_size[1]

            # Get the pool window
            window = a_relu[:, :, h_start:h_stop, w_start:w_stop]
            # Get the upstream gradient
            upstream_grad = d_pool[:, :, i:i+1, j:j+1] 
            
            if pool_type == 'average':
                avg_value = upstream_grad / (pool_h * pool_w)
                d_relu[:, :, h_start:h_stop, w_start:w_stop] += avg_value

            else: # pool_type = 'max'
                max_values = window.amax(dim=(h_dim, w_dim), keepdim=True)
                mask = (window == max_values)

                d_relu[:, :, h_start:h_stop, w_start:w_stop] += \
                mask * upstream_grad

    # Return the derivative of the loss wrt the ReLU activated convolution output
    return d_relu

def relu_backward(d_relu, params):
    """
    Apply ReLU backward on the ReLU activated convolution output derivative Tensor
        to get the derivative of the loss wrt the convolution output

    Args:
        d_relu (Tensor): The  ReLU activated convolution output derivative Tensor

    Return:
        The derivative of the loss wrt the convolution output
    """
    # Get the convolution output Tensor from the params
    z_conv = params['z_conv']
    return d_relu * (z_conv > 0)

def conv2d_backward(d_zconv, params):
    """
    Apply convolution2d backward on the convolution output derivative Tensor to
        get the derivative of the loss wrt the input patches

    Args:
        d_zconv (Tensor): The derivative

    Return:
        The updated derivate of the loss wrt the input patches
    """
    # Get convolution 2d params
    x = params['x']
    W_conv = params['W_conv']
    kernel_size = params['kernel_size']
    stride = params['stride']
    padding = params['padding']
    
    # Flatten the convolution output derivative and the input patches,
    #   patch_size --> patch_height * patch_width
    # d_zconv dims = [batch_size, num_out_features, num_patch_rows, num_patch_cols
    batch_dim, features_dim, h_dim, w_dim = (0, 1, 2, 3)
    d_zconv_3d = d_zconv.flatten(start_dim=h_dim)
    # x dims = [batch_size, num_patches, patch_height, patch_width]
    x_3d = x.flatten(start_dim=2) # patch_height = 3rd dim
    x_unfold = nnf.unfold(x_3d, kernel_size=kernel_size,
                    stride=stride, padding=padding)

    # Get the input patches derivative
    #   (1) For each feature in the 3d convolution output derivative,
    #       get the sum of all elements across the batch and patch space. Do
    #       this across all features to get a vector of feature value sums
    #   (2) Do the same for all patches in the input patches to get a vector
    #       of patch value sums.
    #   (3) Multiply the two vectors to get the input patches derivative
    d_Wconv_2d = torch.einsum(
        # z_conv_3d dims = [(b)atch_size, num_out_(f)eatures, (n)um_patches]
        # x_unfold dims = [(b)atch_size, (p)atch_size, (n)um_patches]
        "bfn,bpn->fp",
        d_zconv_3d, x_unfold)

    # Get the weight and bias gradients
    d_Wconv = d_Wconv_2d.reshape(W_conv.shape)
    d_bconv = d_zconv.sum(dim=(batch_dim, h_dim, w_dim))
    
    # Get the downstream input patches gradient
    # First flatten the convolution weight matrix
    Wconv_2d = W_conv.reshape(W_conv.shape[0], -1) # num in channels is 1st dim

    # Use the flattened conv weight to get the unfolded input patches gradient
    d_x_unfold = torch.einsum(
        "fp,bfn->bpn",
        Wconv_2d, d_zconv_3d
    )

    # Fold the input patches gradient to the dimensions of the input patches
    d_x = nnf.fold(
        d_x_unfold,
        output_size=(h_dim, w_dim),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )

    # Return the convolution gradients
    return d_Wconv, d_bconv, d_x

    
# ========================== PROJECTION FUNCTIONS =============================

# ---------------- FORWARD PASS ----------------
# NOTE: The projection forward pipeline =
#   linear projection

def lin_proj(h, params):
    """
    Apply linear projection to the flattened convolution output Tensor

    Args:
        h (Tensor): The flattened convolution output Tensor

    Return:
        The projection output Tensor
    """
    # Get the linear projection parameters
    W_proj = params['W_proj']
    b_proj = params['b_proj']

    # Return the projection output Tensor
    return util.get_linear_projection(h, W_proj, b_proj)

# -------------- BACKPROPAGATION ---------------
# NOTE: The projection backward pipeline =
#   linear projection backward

def lin_proj_backward(d_loss, params):
    """
    Apply linear projection backward on the loss derivative Tensor to get the
        derivate of the loss wrt the projection output

    Args:
        d_loss (Tensor): The loss derivative Tensor

    Return:
        The derivate of the loss wrt projection output
    """
    # Get linear projection backward parameters
    h = params['h']
    W_proj = params['W_proj']

    d_Wproj = h.T @ d_loss
    d_bproj = d_loss.sum(axis=0) # Get sum of embedding values for each feature
    d_zproj = d_loss @ W_proj.T
    
    return d_Wproj, d_bproj, d_zproj