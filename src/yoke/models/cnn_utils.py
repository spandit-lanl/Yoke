"""Collection of helper functions to facilitate constructing CNN modules.

"""

import math


####################################
## Get Conv2D Shape
####################################
def conv2d_shape(w, h, k, s_w, s_h, p_w, p_h):
    """ Function to calculate the new dimension of an image after a nn.Conv2d

        Args:
            w (int): starting width
            h (int): starting height
            k (int): kernel size
            s_w (int): stride size along the width
            s_h (int): stride size along the height
            p_w (int): padding size along the width
            p_h (int): padding size along the height

        Returns:
            new_w (int): number of pixels along the width
            new_h (int): number of pixels along the height
            total (int): total number of pixels in new image

        See Also: 
        Formula taken from 
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        Assuming a 2D input and dilation = 1

    """

    new_w = int(math.floor(((w + 2*p_w - (k-1) -1)/s_w)+1))
    new_h = int(math.floor(((h + 2*p_h - (k-1) -1)/s_h)+1))
    total = new_w * new_h

    return new_w, new_h, total


def convtranspose2d_shape(w, h, k_w, k_h, s_w, s_h, p_w, p_h, op_w, op_h, d_w, d_h):
    """Function to calculate the new dimension of an image after a
    nn.ConvTranspose2d. This assumes *groups*, *dilation*, and *ouput_padding*
    are all default values.

    Args:
        w (int): starting width
        h (int): starting height
        k_w (int): kernel width size
        k_h (int): kernel height size
        s_w (int): stride size along the width
        s_h (int): stride size along the height
        p_w (int): padding size along the width
        p_h (int): padding size along the height
        op_w (int): output padding size along the width
        op_h (int): output padding size along the height
        d_w (int): dilation size along the width
        d_h (int): dilation size along the height

    Returns:
        new_w (int): number of pixels along the width
        new_h (int): number of pixels along the height
        total (int): total number of pixels in new image

    See Also: 
    Formula taken from 
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    """

    new_w = (w - 1)*s_w - 2*p_w + d_w*(k_w - 1) + op_w + 1
    new_h = (h - 1)*s_h - 2*p_h + d_h*(k_h - 1) + op_h + 1
    total = new_w * new_h

    return new_w, new_h, total


def count_parameters(model):
    """Counts trainable parameters in an instantiated pytorch model. Returns the
    count and prints a statement of the count.

    """
    
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Model has this many trainable parameters:', num_trainable_params)

    return num_trainable_params


if __name__ == '__main__':
    """For testing and debugging.

    """

    # Excercise conv2d_shape function
    input_width = 100
    input_height = 150
    symmetric_kernel_size = 3
    stride_width = 2
    stride_height = 1
    padding_width = 2
    padding_height = 1
    new_w, new_h, total_pixels = conv2d_shape(input_width,
                                              input_height,
                                              symmetric_kernel_size,
                                              stride_width,
                                              stride_height,
                                              padding_width,
                                              padding_height)
    print('New conv-image size:', new_w, new_h, total_pixels)

    kernel_width = 2
    kernel_height = 3
    new_w, new_h, total_pixels = convtranspose2d_shape(input_width,
                                                       input_height,
                                                       kernel_width,
                                                       kernel_height,
                                                       stride_width,
                                                       stride_height,
                                                       padding_width,
                                                       padding_height)
    print('New convtrans-image size:', new_w, new_h, total_pixels)
