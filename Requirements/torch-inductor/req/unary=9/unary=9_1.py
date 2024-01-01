t1 = conv(input_tensor) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = t1 + 3 # Add 3 to the output of the convolution
t3 = torch.clamp_min(t2, 0) # Clamp the output of the addition operation to a minimum of 0
t4 = torch.clamp_max(t3, 6) # Clamp the output of the previous operation to a maximum of 6
t5 = t4 / 6 # Divide the output of the previous operation by 6
