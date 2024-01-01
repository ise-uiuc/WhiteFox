t1 = conv_transpose(input_tensor) # Apply pointwise transposed convolution to the input tensor
t2 = t1 + 3 # Add 3 to the output of the transposed convolution
t3 = torch.clamp_min(t2, 0) # Clamp the output of the addition operation at a minimum of 0
t4 = torch.clamp_max(t3, 6) # Clamp the output of the previous operation at a maximum of 6
t5 = t4 / 6 # Divide the output of the previous operation by 6
