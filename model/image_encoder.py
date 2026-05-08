import math
import torch.nn.functional as nnf

def get_conv2d(x, W, b, stride, padding):
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
    
def get_pool(x, kernel_size, stride, pool_type):
    """
    Take in an image Tensor and apply pooling to it

    Args:
        x (Tensor): The image Tensor to compress
        kernel_size (tuple[int]): The dimensions of the pooling window
        stride (int): The number of pixels the pool window moves
        type (str): The type of pooling operation to perform

    Return:
        The pooled image Tensor
    """
    if pool_type == "average":
        return nnf.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
    # Else, return max pooling -- the default
    return nnf.max_pool2d(x, kernel_size=kernel_size, stride=stride)

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

def unflatten(x, unflatten_shape):
    """
    Take an image Tensor and unflatten it

    Args:
        x (Tensor): The image Tensor to unflatten

    Return:
        The unflattened image Tensor
    """
    return x.reshape(unflatten_shape)