t1 = conv_transpose(input_tensor) # Apply pointwise transposed convolution to the input tensor
t2 = t1 * 0.5 # Multiply the output of the transposed convolution by 0.5
t3 = t1 * 0.7071067811865476 # Multiply the output of the transposed convolution by 0.7071067811865476
t4 = torch.erf(t3) # Apply the error function to the output of the transposed convolution
t5 = t4 + 1 # Add 1 to the output of the error function
t6 = t2 * t5 # Multiply the output of the transposed convolution by the output of the error function
