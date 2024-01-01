t1 = conv(input_tensor) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = t1 - other # Subtract 'other' from the output of the convolution
