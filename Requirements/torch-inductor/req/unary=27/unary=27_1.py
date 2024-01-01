t1 = conv(input_tensor) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = torch.clamp_min(t1, min_value) # Clamp the output of the convolution to a minimum value
t3 = torch.clamp_max(t2, max_value) # Clamp the output of the previous operation to a maximum value
