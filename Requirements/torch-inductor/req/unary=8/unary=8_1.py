t1 = conv_transpose(input_tensor) # Apply pointwise transposed convolution to the input tensor
t2 = t1 + 3 # Add 3 to the output of the transposed convolution
t3 = torch.clamp(t2, min=0) # Clamp the output of the addition operation to a minimum of 0
t4 = torch.clamp(t3, max=6) # Clamp the output of the previous clamp operation to a maximum of 6
t5 = t1 * t4 # Multiply the output of the transposed convolution by the output of the clamp operation
t6 = t5 / 6 # Divide the output of the multiplication operation by 6
